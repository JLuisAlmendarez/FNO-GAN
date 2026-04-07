import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 1. DATASET
# ═══════════════════════════════════════════════════════════

class KolmogorovDataset(Dataset):
    def __init__(self, path, seq_len=10):
        path = Path(path)
        assert path.exists(), f"No se encontró: {path}"
        w = np.load(path)
        if w.ndim == 3:
            w = w[None]
        assert w.ndim == 4, f"Esperado (N, T, H, W), got {w.shape}"
        w = w.astype(np.float32)
        N, T, H, W = w.shape
        assert T > seq_len, f"T={T} debe ser > seq_len={seq_len}"
        self.seq_len   = seq_len
        self.H, self.W = H, W
        self.N, self.T = N, T
        self.w_mean    = float(w.mean())
        raw_std        = float(w.std())
        if raw_std < 1e-6:
            import warnings
            warnings.warn(f"w_std={raw_std:.2e} — dataset casi constante.")
        self.w_std     = raw_std + 1e-8
        w              = (w - self.w_mean) / self.w_std
        self.w         = w
        self.n_windows = T - seq_len
        logger.info(
            f"Dataset: {N} trayectorias × {self.n_windows} ventanas "
            f"= {len(self):,} muestras | seq_len={seq_len} | H×W={H}×{W}"
        )

    def __len__(self):
        return self.N * self.n_windows

    def __getitem__(self, idx):
        traj_idx    = idx // self.n_windows
        t0          = idx %  self.n_windows
        traj        = self.w[traj_idx, t0 : t0 + self.seq_len + 1]
        traj_tensor = torch.from_numpy(traj.copy())
        seq_in      = traj_tensor[:-1].unsqueeze(1)
        seq_out     = traj_tensor[1:].unsqueeze(1)
        return seq_in, seq_out, traj_tensor


# ═══════════════════════════════════════════════════════════
# 2. BLOQUE ESPECTRAL
# ═══════════════════════════════════════════════════════════

class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        scale  = 1.0 / (in_ch * out_ch)
        W1_r = scale * torch.randn(in_ch, out_ch, modes1, modes2)
        W1_i = scale * torch.randn(in_ch, out_ch, modes1, modes2)
        W2_r = scale * torch.randn(in_ch, out_ch, modes1, modes2)
        W2_i = scale * torch.randn(in_ch, out_ch, modes1, modes2)
        self.W1 = nn.Parameter(torch.complex(W1_r, W1_i))
        self.W2 = nn.Parameter(torch.complex(W2_r, W2_i))

    def _mul(self, x, w):
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x):
        B, C, H, W = x.shape
        xf  = torch.fft.rfft2(x)
        out = torch.zeros(B, self.W1.shape[1], H, W // 2 + 1,
                          dtype=torch.cfloat, device=x.device)
        out[:, :,  :self.modes1, :self.modes2] = self._mul(
            xf[:, :,  :self.modes1, :self.modes2], self.W1)
        out[:, :, -self.modes1:, :self.modes2] = self._mul(
            xf[:, :, -self.modes1:, :self.modes2], self.W2)
        return torch.fft.irfft2(out, s=(H, W))


# ═══════════════════════════════════════════════════════════
# 3. CAPA FNO
# ═══════════════════════════════════════════════════════════

class FNOBlock(nn.Module):
    def __init__(self, ch, modes1, modes2):
        super().__init__()
        self.spectral = SpectralConv2d(ch, ch, modes1, modes2)
        self.local    = nn.Conv2d(ch, ch, kernel_size=1)
        self.norm     = nn.InstanceNorm2d(ch)

    def forward(self, x):
        return F.gelu(self.norm(self.spectral(x) + self.local(x)))


# ═══════════════════════════════════════════════════════════
# 4. GENERADOR — FNO puro, sustituto de simulación
# ═══════════════════════════════════════════════════════════

class FNOGenerator(nn.Module):

    def __init__(self, hidden_ch=64, modes1=12, modes2=12, n_layers=4, z_dim=4):
        super().__init__()
        self.z_dim  = z_dim
        self.lift   = nn.Conv2d(1 + z_dim, hidden_ch, kernel_size=1)
        self.layers = nn.ModuleList(
            [FNOBlock(hidden_ch, modes1, modes2) for _ in range(n_layers)]
        )
        self.proj = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_ch // 2, 1, kernel_size=1),
        )

    def forward(self, w_n, z=None):
        B, _, H, W = w_n.shape
        if self.z_dim > 0:
            if z is None:
                z = torch.randn(B, self.z_dim, H, W, device=w_n.device)
            h = self.lift(torch.cat([w_n, z], dim=1))
        else:
            h = self.lift(w_n)
        for layer in self.layers:
            h = layer(h)
        return w_n + self.proj(h)


# ═══════════════════════════════════════════════════════════
# 5. DISCRIMINADORES — [V6-1] especializados
# ═══════════════════════════════════════════════════════════

class FNODiscriminatorStat(nn.Module):

    def __init__(self, seq_len, hidden_ch=64, modes1=12, modes2=12, n_layers=4):
        super().__init__()
        self.seq_len = seq_len
        self.temporal_mix = nn.Sequential(
            nn.Conv1d(seq_len, hidden_ch // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_ch // 2, hidden_ch, kernel_size=1),
        )
        self.layers = nn.ModuleList(
            [FNOBlock(hidden_ch, modes1, modes2) for _ in range(n_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_ch, hidden_ch // 2),
            nn.GELU(),
            nn.Linear(hidden_ch // 2, 1),
        )

    def forward(self, traj):
        """traj: (B, T, H, W) → (B, 1)"""
        B, T, H, W = traj.shape
        x = traj.permute(0, 2, 3, 1).contiguous().reshape(B * H * W, T, 1)
        x = self.temporal_mix(x).squeeze(-1)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=(-2, -1))
        return self.head(x)


class FNODiscriminatorPhys(nn.Module):

    def __init__(self, seq_len, hidden_ch=64, modes1=12, modes2=12, n_layers=4):
        super().__init__()
        # seq_len - 1 pasos temporales de residuo
        self.temporal_mix = nn.Sequential(
            nn.Conv1d(seq_len - 1, hidden_ch // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_ch // 2, hidden_ch, kernel_size=1),
        )
        self.layers = nn.ModuleList(
            [FNOBlock(hidden_ch, modes1, modes2) for _ in range(n_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_ch, hidden_ch // 2),
            nn.GELU(),
            nn.Linear(hidden_ch // 2, 1),
        )

    def forward(self, residuo):
        """residuo: (B, T-1, H, W) → (B, 1)"""
        B, T, H, W = residuo.shape
        x = residuo.permute(0, 2, 3, 1).contiguous().reshape(B * H * W, T, 1)
        x = self.temporal_mix(x).squeeze(-1)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=(-2, -1))
        return self.head(x)


# ═══════════════════════════════════════════════════════════
# 6. RESIDUO NS — [V6-2] expone campo espacial completo
# ═══════════════════════════════════════════════════════════

class NavierStokesResiduo(nn.Module):

    NU_MEAN = 0.056853
    NU_STD  = 0.014952
    F_MEAN  = 0.024284
    F_STD   = 0.005773

    def __init__(self, H, W, dt=0.01, modes1=8, modes2=8, device="cpu"):
        super().__init__()
        self.dt     = dt
        self.modes1 = modes1
        self.modes2 = modes2

        kx = torch.fft.fftfreq(W, d=1.0 / W)
        ky = torch.fft.fftfreq(H, d=1.0 / H)
        KY, KX = torch.meshgrid(ky, kx, indexing="ij")
        K2     = KX ** 2 + KY ** 2
        K2_inv = K2.clone(); K2_inv[0, 0] = 1.0

        self.register_buffer("KX",     KX)
        self.register_buffer("KY",     KY)
        self.register_buffer("K2",     K2)
        self.register_buffer("K2_inv", K2_inv)

    def _sample_nu(self, device):
        nu = torch.normal(
            mean=torch.tensor(self.NU_MEAN),
            std=torch.tensor(self.NU_STD),
        ).item()
        return max(nu, 1e-4)

    def _sample_f(self, shape, device):
        return torch.normal(
            mean=self.F_MEAN * torch.ones(shape, device=device),
            std=self.F_STD  * torch.ones(shape, device=device),
        )

    def _velocity_from_vorticity(self, w):
        """w: (B, H, W) → u, v: (B, H, W)"""
        wf = torch.fft.fft2(w); wf[:, 0, 0] = 0
        psi = wf / self.K2_inv; psi[:, 0, 0] = 0
        u =  torch.fft.ifft2( 1j * self.KY * psi).real
        v =  torch.fft.ifft2(-1j * self.KX * psi).real
        return u, v

    def _spatial_terms(self, w, nu, f):
        wf = torch.fft.fft2(w); wf[:, 0, 0] = 0
        mask = torch.zeros_like(self.K2, dtype=torch.bool)
        mask[:self.modes1,  :self.modes2] = True
        mask[-self.modes1:, :self.modes2] = True
        wf_f  = wf * mask.unsqueeze(0)
        u, v  = self._velocity_from_vorticity(w)
        dwdx  = torch.fft.ifft2(1j * self.KX * wf_f).real
        dwdy  = torch.fft.ifft2(1j * self.KY * wf_f).real
        adv   = u * dwdx + v * dwdy
        lap_w = torch.fft.ifft2(-self.K2 * wf_f).real
        return adv - nu * lap_w - f

    def _residuo_campo(self, w_n, w_next):
        nu      = self._sample_nu(w_n.device)
        f       = self._sample_f(w_n.shape, w_n.device)
        dwdt    = (w_next - w_n) / self.dt
        spatial = self._spatial_terms(w_n, nu, f)
        return dwdt + spatial   # (B, H, W)

    def residuo_espacial(self, traj):

        B, T, H, W = traj.shape
        residuos = []
        for t in range(T - 1):
            r = self._residuo_campo(traj[:, t], traj[:, t + 1])
            residuos.append(r)
        return torch.stack(residuos, dim=1)   # (B, T-1, H, W)

    def forward(self, traj):

        return (self.residuo_espacial(traj) ** 2).mean()


# ═══════════════════════════════════════════════════════════
# 7. GRADIENT PENALTY — [V6-4] separada por discriminador
# ═══════════════════════════════════════════════════════════

class GradientPenalty(nn.Module):
    def __init__(self, lambda_gp=10.0):
        super().__init__()
        self.lambda_gp = lambda_gp

    def forward(self, discriminator, real_input, fake_input):
        assert real_input.shape == fake_input.shape
        B     = real_input.size(0)
        alpha = torch.rand(B, 1, 1, 1, device=real_input.device)
        interp = (alpha * real_input + (1 - alpha) * fake_input).requires_grad_(True)
        score  = discriminator(interp)
        grad   = torch.autograd.grad(
            outputs=score, inputs=interp,
            grad_outputs=torch.ones_like(score),
            create_graph=True, retain_graph=True,
        )[0]
        grad_norm = grad.flatten(1).norm(2, dim=1)
        return self.lambda_gp * ((grad_norm - 1) ** 2).mean()


# ═══════════════════════════════════════════════════════════
# 8. ROLLOUT — G puro
# ═══════════════════════════════════════════════════════════

class Rollout:
    def __init__(self, generator, device):
        self.generator = generator.to(device)
        self.device    = device
        self.metrics   = {}

    @torch.no_grad()
    def run(self, w0, n_steps, w_true=None, z=None):
        self.generator.eval()
        w_cur = w0.to(self.device)
        if z is not None:
            z = z.to(self.device)
        traj = [w_cur.squeeze(1).cpu()]
        for _ in range(n_steps):
            w_next = self.generator(w_cur, z=z)
            traj.append(w_next.squeeze(1).cpu())
            w_cur = w_next
        traj = torch.stack(traj, dim=1)
        self._compute_metrics(traj, w_true)
        self.generator.train()
        return traj

    def _compute_metrics(self, traj, w_true):
        energy = 0.5 * (traj ** 2).mean(dim=(0, 2, 3))
        self.metrics["energy"] = energy.numpy()
        if w_true is not None:
            T    = min(traj.shape[1], w_true.shape[1])
            diff = traj[:, :T] - w_true[:, :T]
            self.metrics["rmse"]      = diff.pow(2).mean(dim=(0, 2, 3)).sqrt().numpy()
            self.metrics["rel_error"] = (
                diff.norm(dim=(2, 3)) / (w_true[:, :T].norm(dim=(2, 3)) + 1e-8)
            ).mean(0).numpy()


# ═══════════════════════════════════════════════════════════
# 9. TRAINER — Dual Discriminator WGAN-GP
# ═══════════════════════════════════════════════════════════

class WGAFNOGPTrainer:

    def __init__(
        self,
        generator,
        d_stat,
        d_phys,
        ns_residuo,
        device,
        lr_G               = 1e-4,
        lr_D               = 1e-4,
        n_critic           = 5,
        lambda_gp          = 10.0,
        use_scheduler      = True,
        scheduler_patience = 5,
        scheduler_factor   = 0.5,
        log_dir            = "logs",
        vis_freq           = 5,
    ):
        self.G          = generator.to(device)
        self.D_stat     = d_stat.to(device)
        self.D_phys     = d_phys.to(device)
        self.ns_residuo = ns_residuo.to(device)
        self.device     = device
        self.n_critic   = n_critic
        self.vis_freq   = vis_freq
        self.log_dir    = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.gp = GradientPenalty(lambda_gp=lambda_gp).to(device)

        self.opt_G      = torch.optim.Adam(generator.parameters(), lr=lr_G, betas=(0.0, 0.9))
        self.opt_D_stat = torch.optim.Adam(d_stat.parameters(),    lr=lr_D, betas=(0.0, 0.9))
        self.opt_D_phys = torch.optim.Adam(d_phys.parameters(),    lr=lr_D, betas=(0.0, 0.9))

        self.use_scheduler = use_scheduler
        if use_scheduler:
            self.sched_G      = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.opt_G,      mode="min", factor=scheduler_factor, patience=scheduler_patience)
            self.sched_D_stat = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.opt_D_stat, mode="max", factor=scheduler_factor, patience=scheduler_patience)
            self.sched_D_phys = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.opt_D_phys, mode="max", factor=scheduler_factor, patience=scheduler_patience)

        self._critic_iter = None
        self.val_example  = None

        # [V6-5] History con w_dist separados
        self.history = {
            "loss_D_stat"  : [], "loss_D_phys"  : [], "loss_G": [],
            "w_dist_stat"  : [], "w_dist_phys"  : [],
            "val_mse"      : [], "val_ns_residuo": [],
            "val_energy"   : [], "val_enstrophy" : [],
            "val_spectral_correlation": [],
            "lr_G": [], "lr_D_stat": [], "lr_D_phys": [],
        }

    # ── Generación ─────────────────────────────────────────

    def _generate_sequence(self, seq_in, z=None):

        B, seq_len, _, H, W = seq_in.shape
        w_cur = seq_in[:, 0]
        fake  = [w_cur.squeeze(1)]
        for _ in range(seq_len):
            w_next = self.G(w_cur, z=z)
            fake.append(w_next.squeeze(1))
            w_cur = w_next
        return torch.stack(fake, dim=1)   # (B, seq_len+1, H, W)

    # ── Pasos de entrenamiento ─────────────────────────────

    def _step_D(self, loader):

        loss_stat_list = []
        loss_phys_list = []
        score_real_stat_last = score_fake_stat_last = None
        score_real_phys_last = score_fake_phys_last = None

        for _ in range(self.n_critic):
            try:
                seq_in, _, real_traj = next(self._critic_iter)
            except StopIteration:
                self._critic_iter = iter(loader)
                seq_in, _, real_traj = next(self._critic_iter)

            seq_in    = seq_in.to(self.device)
            real_traj = real_traj.to(self.device)

            with torch.no_grad():
                fake_traj = self._generate_sequence(seq_in)

            # Residuos espaciales para D_phys
            R_real = self.ns_residuo.residuo_espacial(real_traj)   # (B, T-1, H, W)
            R_fake = self.ns_residuo.residuo_espacial(fake_traj)   # (B, T-1, H, W)

            # D_stat — trayectorias
            score_real_stat = self.D_stat(real_traj)
            score_fake_stat = self.D_stat(fake_traj)
            gp_stat         = self.gp(self.D_stat, real_traj, fake_traj)
            loss_D_stat     = score_fake_stat.mean() - score_real_stat.mean() + gp_stat

            self.opt_D_stat.zero_grad()
            loss_D_stat.backward()
            self.opt_D_stat.step()

            # D_phys — residuos NS
            score_real_phys = self.D_phys(R_real)
            score_fake_phys = self.D_phys(R_fake)
            gp_phys         = self.gp(self.D_phys, R_real, R_fake)
            loss_D_phys     = score_fake_phys.mean() - score_real_phys.mean() + gp_phys

            self.opt_D_phys.zero_grad()
            loss_D_phys.backward()
            self.opt_D_phys.step()

            loss_stat_list.append(loss_D_stat.item())
            loss_phys_list.append(loss_D_phys.item())
            score_real_stat_last, score_fake_stat_last = score_real_stat, score_fake_stat
            score_real_phys_last, score_fake_phys_last = score_real_phys, score_fake_phys

        w_dist_stat = (score_real_stat_last.mean() - score_fake_stat_last.mean()).item()
        w_dist_phys = (score_real_phys_last.mean() - score_fake_phys_last.mean()).item()

        return (
            float(np.mean(loss_stat_list)),
            float(np.mean(loss_phys_list)),
            w_dist_stat,
            w_dist_phys,
        )

    def _step_G(self, seq_in):

        fake_traj = self._generate_sequence(seq_in)
        R_fake    = self.ns_residuo.residuo_espacial(fake_traj)

        loss_adv_stat = -self.D_stat(fake_traj).mean()
        loss_adv_phys = -self.D_phys(R_fake).mean()
        loss_G        = loss_adv_stat + loss_adv_phys

        self.opt_G.zero_grad()
        loss_G.backward()
        nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)
        self.opt_G.step()
        return loss_G.item()

    # ── Validación ─────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, loader):
        self.G.eval()
        total_mse = total_energy = total_enstrophy = total_corr = total_ns = 0.0
        n = 0

        for seq_in, seq_out, _ in loader:
            seq_in  = seq_in.to(self.device)
            seq_out = seq_out.to(self.device)

            fake_traj   = self._generate_sequence(seq_in)
            seq_out_2d  = seq_out.squeeze(2)

            total_mse += F.mse_loss(fake_traj[:, 1:], seq_out_2d).item()
            total_ns  += self.ns_residuo(fake_traj).item()

            w_last_fake = fake_traj[:, -1]
            w_last_real = seq_out_2d[:, -1]

            total_energy    += 0.5 * (w_last_fake ** 2).mean().item()
            wf               = torch.fft.fft2(w_last_fake)
            total_enstrophy += 0.5 * (self.ns_residuo.K2 * wf.abs() ** 2).mean().item()

            _, C_k = self._spectral_correlation(w_last_real.cpu(), w_last_fake.cpu())
            total_corr += float(C_k.mean())
            n += 1

        self.G.train()
        return (
            total_mse       / n,
            total_ns        / n,
            total_energy    / n,
            total_enstrophy / n,
            total_corr      / n,
        )

    # ── Espectros ──────────────────────────────────────────

    @staticmethod
    def energy_spectrum(w):
        if w.ndim == 2: w = w.unsqueeze(0)
        B, H, W = w.shape
        kx = torch.fft.fftfreq(W, d=1.0/W)
        ky = torch.fft.fftfreq(H, d=1.0/H)
        KY, KX = torch.meshgrid(ky, kx, indexing="ij")
        K_mag  = (KX**2 + KY**2).sqrt()
        E_mean = (0.5 * torch.fft.fft2(w).abs()**2).mean(dim=0)
        k_max  = int(min(H, W) // 2)
        k_bins = np.arange(1, k_max + 1)
        K_np   = K_mag.cpu().numpy()
        E_np   = E_mean.cpu().numpy()
        E_k    = np.array([E_np[(K_np >= k-.5) & (K_np < k+.5)].sum() for k in k_bins])
        return k_bins, E_k

    @staticmethod
    def _enstrophy_spectrum(w):
        if w.ndim == 2: w = w.unsqueeze(0)
        B, H, W = w.shape
        kx = torch.fft.fftfreq(W, d=1.0/W)
        ky = torch.fft.fftfreq(H, d=1.0/H)
        KY, KX = torch.meshgrid(ky, kx, indexing="ij")
        K2     = KX**2 + KY**2
        K_mag  = K2.sqrt()
        Z_mean = (0.5 * K2 * torch.fft.fft2(w).abs()**2).mean(dim=0)
        k_max  = int(min(H, W) // 2)
        k_bins = np.arange(1, k_max + 1)
        K_np   = K_mag.cpu().numpy()
        Z_np   = Z_mean.cpu().numpy()
        Z_k    = np.array([Z_np[(K_np >= k-.5) & (K_np < k+.5)].sum() for k in k_bins])
        return k_bins, Z_k

    def _transfer_spectrum(self, w):
        if w.ndim == 2: w = w.unsqueeze(0)
        w  = w.to(self.device)
        B, H, W = w.shape
        KX, KY  = self.ns_residuo.KX, self.ns_residuo.KY
        K2      = self.ns_residuo.K2
        K2_inv  = K2.clone(); K2_inv[0, 0] = 1.0
        wf      = torch.fft.fft2(w); wf[:, 0, 0] = 0
        psi     = wf / K2_inv; psi[:, 0, 0] = 0
        u       =  torch.fft.ifft2( 1j * KY * psi).real
        v       =  torch.fft.ifft2(-1j * KX * psi).real
        adv     = u * torch.fft.ifft2(1j * KX * wf).real + \
                  v * torch.fft.ifft2(1j * KY * wf).real
        T_2d    = (wf.conj() * torch.fft.fft2(adv)).real.mean(dim=0)
        K_mag   = (KX**2 + KY**2).sqrt().cpu().numpy()
        T_np    = T_2d.cpu().numpy()
        k_max   = int(min(H, W) // 2)
        k_bins  = np.arange(1, k_max + 1)
        T_k     = np.array([T_np[(K_mag >= k-.5) & (K_mag < k+.5)].sum() for k in k_bins])
        return T_k, k_bins

    @staticmethod
    def _palinstrophy_spectrum(w):
        if w.ndim == 2: w = w.unsqueeze(0)
        B, H, W = w.shape
        kx = torch.fft.fftfreq(W, d=1.0/W)
        ky = torch.fft.fftfreq(H, d=1.0/H)
        KY, KX = torch.meshgrid(ky, kx, indexing="ij")
        K2     = KX**2 + KY**2
        K_mag  = K2.sqrt()
        P_mean = (0.5 * K2**2 * torch.fft.fft2(w).abs()**2).mean(dim=0)
        k_max  = int(min(H, W) // 2)
        k_bins = np.arange(1, k_max + 1)
        K_np   = K_mag.cpu().numpy()
        P_np   = P_mean.cpu().numpy()
        P_k    = np.array([P_np[(K_np >= k-.5) & (K_np < k+.5)].sum() for k in k_bins])
        return k_bins, P_k

    @staticmethod
    def _spectral_correlation(w_real, w_fake):
        if w_real.ndim == 2: w_real = w_real.unsqueeze(0)
        if w_fake.ndim == 2: w_fake = w_fake.unsqueeze(0)
        B, H, W = w_real.shape
        kx = torch.fft.fftfreq(W, d=1.0/W)
        ky = torch.fft.fftfreq(H, d=1.0/H)
        KY, KX  = torch.meshgrid(ky, kx, indexing="ij")
        K_mag   = (KX**2 + KY**2).sqrt()
        wf_real = torch.fft.fft2(w_real)
        wf_fake = torch.fft.fft2(w_fake)
        cross   = (wf_fake * wf_real.conj()).real.mean(dim=0)
        denom   = (wf_real.abs().mean(0) * wf_fake.abs().mean(0)).clamp(min=1e-8)
        C_2d    = cross / denom
        k_max   = int(min(H, W) // 2)
        k_bins  = np.arange(1, k_max + 1)
        K_np    = K_mag.cpu().numpy()
        C_np    = C_2d.cpu().numpy()
        C_k     = np.array([
            C_np[(K_np >= k-.5) & (K_np < k+.5)].mean()
            if ((K_np >= k-.5) & (K_np < k+.5)).any() else 0.0
            for k in k_bins
        ])
        return k_bins, C_k

    # ── Checkpointing ──────────────────────────────────────

    def save_checkpoint(self, epoch, val_mse):
        ckpt = {
            "epoch"     : epoch,
            "G"         : self.G.state_dict(),
            "D_stat"    : self.D_stat.state_dict(),
            "D_phys"    : self.D_phys.state_dict(),
            "opt_G"     : self.opt_G.state_dict(),
            "opt_D_stat": self.opt_D_stat.state_dict(),
            "opt_D_phys": self.opt_D_phys.state_dict(),
            "val_mse"   : val_mse,
        }
        if self.use_scheduler:
            ckpt["sched_G"]      = self.sched_G.state_dict()
            ckpt["sched_D_stat"] = self.sched_D_stat.state_dict()
            ckpt["sched_D_phys"] = self.sched_D_phys.state_dict()
        torch.save(ckpt, self.log_dir / "best_checkpoint.pt")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["G"])
        self.D_stat.load_state_dict(ckpt["D_stat"])
        self.D_phys.load_state_dict(ckpt["D_phys"])
        self.opt_G.load_state_dict(ckpt["opt_G"])
        self.opt_D_stat.load_state_dict(ckpt["opt_D_stat"])
        self.opt_D_phys.load_state_dict(ckpt["opt_D_phys"])
        if self.use_scheduler and "sched_G" in ckpt:
            self.sched_G.load_state_dict(ckpt["sched_G"])
            self.sched_D_stat.load_state_dict(ckpt["sched_D_stat"])
            self.sched_D_phys.load_state_dict(ckpt["sched_D_phys"])
        logger.info(f"Checkpoint cargado — época {ckpt['epoch']}, val MSE {ckpt['val_mse']:.6f}")
        return ckpt["epoch"]

    # ── Visualizaciones ────────────────────────────────────

    def _get_fake_for_plot(self):
        seq_in, _, real_traj = self.val_example
        seq_in = seq_in.to(self.device)
        self.G.eval()
        with torch.no_grad():
            fake_traj = self._generate_sequence(seq_in)
        self.G.train()
        return real_traj, fake_traj

    def plot_fields(self, epoch):
        seq_in    = self.val_example[0][:2].to(self.device)
        real_traj = self.val_example[2][:2]
        self.G.eval()
        with torch.no_grad():
            fake_traj = self._generate_sequence(seq_in)
        self.G.train()
        real_np = real_traj.cpu().numpy()
        fake_np = fake_traj.cpu().numpy()
        T    = real_np.shape[1]
        vmax = max(np.abs(real_np).max(), np.abs(fake_np).max())
        fig, axes = plt.subplots(4, T, figsize=(3 * T, 10), squeeze=False)
        row_labels = ["Real (A)", "Generado (A)", "Real (B)", "Generado (B)"]
        for s, (r_row, f_row) in enumerate([(0, 1), (2, 3)]):
            for t in range(T):
                im = axes[r_row][t].imshow(
                    real_np[s, t], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    interpolation="nearest")
                axes[r_row][t].set_title(f"t={t}", fontsize=8)
                axes[r_row][t].axis("off")
                axes[f_row][t].imshow(
                    fake_np[s, t], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    interpolation="nearest")
                axes[f_row][t].axis("off")
        for i, label in enumerate(row_labels):
            axes[i][0].set_ylabel(label, fontsize=8, rotation=90, labelpad=4)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.35, label="vorticidad (norm.)")
        fig.suptitle(f"Campos de vorticidad — época {epoch}", fontsize=12)
        plt.tight_layout()
        path = self.log_dir / f"fields_epoch{epoch:04d}.png"
        plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
        logger.info(f"  Campos → {path}")

    def plot_losses(self):
        h   = self.history
        eps = range(1, len(h["loss_D_stat"]) + 1)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)

        axes[0][0].plot(eps, h["loss_D_stat"], label="L_D_stat", color="tab:red")
        axes[0][0].plot(eps, h["loss_D_phys"], label="L_D_phys", color="tab:orange")
        axes[0][0].plot(eps, h["loss_G"],      label="L_G",      color="tab:blue")
        axes[0][0].set_title("Losses adversariales"); axes[0][0].legend()
        axes[0][0].set_xlabel("Época")

        axes[0][1].plot(eps, h["w_dist_stat"], label="W_stat", color="tab:green")
        axes[0][1].plot(eps, h["w_dist_phys"], label="W_phys", color="tab:olive")
        axes[0][1].axhline(0, color="k", lw=0.8, ls="--")
        axes[0][1].set_title("Wasserstein distances"); axes[0][1].legend()
        axes[0][1].set_xlabel("Época")

        axes[0][2].plot(eps, h["lr_G"],      label="lr_G",      color="tab:purple")
        axes[0][2].plot(eps, h["lr_D_stat"], label="lr_D_stat", color="tab:red")
        axes[0][2].plot(eps, h["lr_D_phys"], label="lr_D_phys", color="tab:orange")
        axes[0][2].set_title("Learning rates"); axes[0][2].legend()
        axes[0][2].set_xlabel("Época"); axes[0][2].set_yscale("log")

        axes[1][0].plot(eps, h["val_mse"],         label="Val MSE",        color="tab:red")
        axes[1][0].plot(eps, h["val_ns_residuo"],   label="Val NS residuo", color="tab:brown", ls="--")
        axes[1][0].set_title("Métricas monitoreo"); axes[1][0].legend()
        axes[1][0].set_xlabel("Época")

        axes[1][1].plot(eps, h["val_spectral_correlation"], color="tab:purple")
        axes[1][1].set_title("Correlación espectral val"); axes[1][1].set_xlabel("Época")

        ax2 = axes[1][2].twinx()
        axes[1][2].plot(eps, h["val_energy"],    label="Energía",   color="tab:blue")
        ax2.plot(        eps, h["val_enstrophy"], label="Enstrofía", color="tab:orange", ls="--")
        axes[1][2].set_title("Métricas físicas (val)"); axes[1][2].set_xlabel("Época")
        axes[1][2].set_ylabel("Energía", color="tab:blue")
        ax2.set_ylabel("Enstrofía",      color="tab:orange")
        l1, lb1 = axes[1][2].get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        axes[1][2].legend(l1 + l2, lb1 + lb2, fontsize=8)

        plt.suptitle("Historial de entrenamiento — v6 Dual Discriminator", fontsize=13)
        plt.tight_layout()
        path = self.log_dir / "losses.png"
        plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
        logger.info(f"  Curvas → {path}")

    def _plot_spectrum_generic(self, epoch, k_real, E_real, k_fake, E_fake,
                               ylabel, title, fname, ref_slope=None,
                               ref_label=None, loglog=True):
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_fn = ax.loglog if loglog else ax.semilogx
        plot_fn(k_real, E_real, label="Real",     color="tab:blue", lw=2)
        plot_fn(k_fake, E_fake, label="Generado", color="tab:red",  lw=2, ls="--")
        if ref_slope is not None and loglog:
            idx   = np.where(k_real >= 2)[0][0]
            k_ref = k_real[idx:]
            E_ref = E_real[idx] * (k_ref / k_real[idx]) ** ref_slope
            ax.loglog(k_ref, E_ref, label=ref_label, color="k", lw=1, ls=":")
        if not loglog:
            ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("Número de onda $k$", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{title} — época {epoch}", fontsize=12)
        ax.legend(fontsize=11); ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        path = self.log_dir / fname
        plt.savefig(path, dpi=120); plt.close()
        logger.info(f"  {title} → {path}")

    def plot_spectrum(self, epoch):
        real_traj, fake_traj = self._get_fake_for_plot()
        k_r, E_r = self.energy_spectrum(real_traj[:, -1].float())
        k_f, E_f = self.energy_spectrum(fake_traj[:, -1].cpu())
        self._plot_spectrum_generic(epoch, k_r, E_r, k_f, E_f,
            "$E(k)$", "Espectro de energía cinética",
            f"spectrum_epoch{epoch:04d}.png", ref_slope=-3, ref_label=r"$k^{-3}$")

    def plot_enstrophy_spectrum(self, epoch):
        real_traj, fake_traj = self._get_fake_for_plot()
        k_r, Z_r = self._enstrophy_spectrum(real_traj[:, -1].float())
        k_f, Z_f = self._enstrophy_spectrum(fake_traj[:, -1].cpu())
        self._plot_spectrum_generic(epoch, k_r, Z_r, k_f, Z_f,
            "$Z(k)$", "Espectro de enstrofía",
            f"enstrophy_spectrum_epoch{epoch:04d}.png", ref_slope=-1, ref_label=r"$k^{-1}$")

    def plot_transfer_spectrum(self, epoch):
        real_traj, fake_traj = self._get_fake_for_plot()
        T_r, k_r = self._transfer_spectrum(real_traj[:, -1].float())
        T_f, k_f = self._transfer_spectrum(fake_traj[:, -1].cpu())
        self._plot_spectrum_generic(epoch, k_r, T_r, k_f, T_f,
            "$T(k)$", "Espectro de transferencia",
            f"transfer_spectrum_epoch{epoch:04d}.png", loglog=False)

    def plot_palinstrophy_spectrum(self, epoch):
        real_traj, fake_traj = self._get_fake_for_plot()
        k_r, P_r = self._palinstrophy_spectrum(real_traj[:, -1].float())
        k_f, P_f = self._palinstrophy_spectrum(fake_traj[:, -1].cpu())
        self._plot_spectrum_generic(epoch, k_r, P_r, k_f, P_f,
            "$P(k)$", "Espectro de palinstrofía",
            f"palinstrophy_spectrum_epoch{epoch:04d}.png", ref_slope=1, ref_label=r"$k^{+1}$")

    def plot_spectral_correlation(self, epoch):
        real_traj, fake_traj = self._get_fake_for_plot()
        k_bins, C_k = self._spectral_correlation(
            real_traj[:, -1].float(), fake_traj[:, -1].cpu())
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.semilogx(k_bins, C_k, color="tab:purple", lw=2)
        ax.axhline(1.0, color="k",       lw=0.8, ls="--", label="Correlación perfecta")
        ax.axhline(0.0, color="tab:red", lw=0.8, ls=":",  label="Sin coherencia")
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("Número de onda $k$", fontsize=12)
        ax.set_ylabel("$C(k)$", fontsize=12)
        ax.set_title(f"Correlación espectral — época {epoch}", fontsize=12)
        ax.legend(fontsize=11); ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        path = self.log_dir / f"spectral_correlation_epoch{epoch:04d}.png"
        plt.savefig(path, dpi=120); plt.close()
        logger.info(f"  Correlación espectral → {path}")

    # ── Loop principal ─────────────────────────────────────

    def fit(self, train_loader, val_loader, n_epochs, log_every=5):
        best_val          = float("inf")
        self._critic_iter = iter(train_loader)

        if self.val_example is None:
            self.val_example = next(iter(val_loader))

        for epoch in range(1, n_epochs + 1):
            self.G.train(); self.D_stat.train(); self.D_phys.train()

            ep = {"ld_stat": [], "ld_phys": [], "lg": [],
                  "wd_stat": [], "wd_phys": []}

            pbar = tqdm(train_loader, desc=f"Epoch {epoch:4d}/{n_epochs}", leave=False)
            for seq_in, seq_out, _ in pbar:
                seq_in = seq_in.to(self.device)

                ld_stat, ld_phys, wd_stat, wd_phys = self._step_D(train_loader)
                lg = self._step_G(seq_in)

                ep["ld_stat"].append(ld_stat); ep["ld_phys"].append(ld_phys)
                ep["lg"].append(lg)
                ep["wd_stat"].append(wd_stat); ep["wd_phys"].append(wd_phys)

                pbar.set_postfix({
                    "Ws": f"{wd_stat:.2f}", "Wp": f"{wd_phys:.2f}",
                    "LG": f"{lg:.3f}",
                })

            val_mse, val_ns, val_energy, val_enstrophy, val_corr = self._validate(val_loader)

            # Schedulers — step antes de append
            if self.use_scheduler:
                self.sched_G.step(val_mse)
                self.sched_D_stat.step(float(np.mean(ep["wd_stat"])))
                self.sched_D_phys.step(float(np.mean(ep["wd_phys"])))

            # History
            self.history["loss_D_stat"].append(float(np.mean(ep["ld_stat"])))
            self.history["loss_D_phys"].append(float(np.mean(ep["ld_phys"])))
            self.history["loss_G"].append(float(np.mean(ep["lg"])))
            self.history["w_dist_stat"].append(float(np.mean(ep["wd_stat"])))
            self.history["w_dist_phys"].append(float(np.mean(ep["wd_phys"])))
            self.history["val_mse"].append(val_mse)
            self.history["val_ns_residuo"].append(val_ns)
            self.history["val_energy"].append(val_energy)
            self.history["val_enstrophy"].append(val_enstrophy)
            self.history["val_spectral_correlation"].append(val_corr)
            self.history["lr_G"].append(self.opt_G.param_groups[0]["lr"])
            self.history["lr_D_stat"].append(self.opt_D_stat.param_groups[0]["lr"])
            self.history["lr_D_phys"].append(self.opt_D_phys.param_groups[0]["lr"])

            if val_mse < best_val:
                best_val = val_mse
                self.save_checkpoint(epoch, val_mse)
                logger.info(f"✅ Nuevo mejor — época {epoch}, val MSE={val_mse:.6f}")

            if epoch % 10 == 0 and str(self.device) != "cpu":
                torch.cuda.empty_cache()

            if epoch % self.vis_freq == 0:
                self.plot_fields(epoch)
                self.plot_spectrum(epoch)
                self.plot_enstrophy_spectrum(epoch)
                self.plot_transfer_spectrum(epoch)
                self.plot_palinstrophy_spectrum(epoch)
                self.plot_spectral_correlation(epoch)
                self.plot_losses()

            if epoch % log_every == 0:
                logger.info(
                    f"Época {epoch:4d} | "
                    f"Ws {np.mean(ep['wd_stat']):+.4f} | Wp {np.mean(ep['wd_phys']):+.4f} | "
                    f"L_Ds {np.mean(ep['ld_stat']):+.5f} | L_Dp {np.mean(ep['ld_phys']):+.5f} | "
                    f"L_G {np.mean(ep['lg']):+.5f} | "
                    f"Val MSE {val_mse:.5f} | Val NS {val_ns:.5f} | "
                    f"C(k) {val_corr:.4f} | E {val_energy:.4f} | Ω {val_enstrophy:.4f}"
                )

        self.plot_losses()
        logger.info(f"\nMejor val MSE: {best_val:.6f} — {self.log_dir}/best_checkpoint.pt")
        return self.history