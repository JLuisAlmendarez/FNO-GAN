"""
WGAFNOGP v3 — Rollout Autoregresivo, Flujo de Kolmogorov 2D
=============================================================
Dataset:   .npy shape (N, T, H, W)  — solo vorticidad
           e.g.  (1152, 100, 64, 64)

Generador: wₙ (1 canal) + z ~ N(0,I) → wₙ₊₁  via FNO + residual (estocástico)
Discriminador: trayectoria (B, T+1, H, W) → escalar (WGAN)

Pérdidas:
  L_D = 𝔼[D(fake)] − 𝔼[D(real)] + λ·GP
  L_G = −𝔼[D(fake)] + α·MSE(pred, vort_{n+1}) + β·L_NS

  L_NS = ‖∂w/∂t + u·∂w/∂x + v·∂w/∂y − ν·∇²w‖²
         Euler explícito puro — todo evaluado en wₙ
         u, v recuperados de w via función de corriente (Fourier exacto)

Estructura:
  1.  SpectralConv2d     — convolución en espacio de Fourier
  2.  FNOBlock           — capa FNO completa
  3.  FNOGenerator       — G_θ: (wₙ, z) → wₙ₊₁  (estocástico, z_dim configurable)
  4.  FNODiscriminator   — D_φ: trayectoria → escalar
  5.  KolmogorovDataset  — carga .npy (N,T,H,W), ventanas deslizantes
  6.  NavierStokesLoss   — residuo NS puro en Fourier
  7.  GradientPenalty    — GP L² funcional sobre trayectorias
  8.  Rollout            — generación autoregresiva con métricas
  9.  WGAFNOGPTrainer    — loop adversarial completo + visualizaciones
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
import logging
import matplotlib
matplotlib.use("Agg")   # backend sin pantalla — seguro en servidores/clusters
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
    """
    Carga vorticidad Kolmogorov desde .npy de shape (N, T, H, W).
    Construye ventanas deslizantes sobre la marcha sin duplicar datos.
    """

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

        self.w_mean = float(w.mean())

        # [F9] warning si el dataset es casi constante
        raw_std = float(w.std())
        if raw_std < 1e-6:
            import warnings
            warnings.warn(
                f"w_std={raw_std:.2e} — dataset casi constante. "
                "Normalización degenerada. Verifica el archivo .npy."
            )
        self.w_std = raw_std + 1e-8
        w = (w - self.w_mean) / self.w_std

        self.w         = w               # (N, T, H, W) float32 en RAM
        self.n_windows = T - seq_len

        logger.info(
            f"Dataset: {N} trayectorias × {self.n_windows} ventanas "
            f"= {len(self):,} muestras | seq_len={seq_len} | H×W={H}×{W}"
        )

    def __len__(self):
        return self.N * self.n_windows

    def __getitem__(self, idx):
        """
        Retorna:
          seq_in:  (seq_len, 1, H, W)
          seq_out: (seq_len, 1, H, W)
          traj:    (seq_len+1, H, W)
        """
        traj_idx = idx // self.n_windows
        t0       = idx %  self.n_windows

        traj = self.w[traj_idx, t0 : t0 + self.seq_len + 1]   # (seq_len+1, H, W)
        # [F9] .copy() obligatorio: slice en dim T → array no contiguo;
        # sin copy los workers del DataLoader producen race conditions.
        traj_tensor = torch.from_numpy(traj.copy())
        seq_in  = traj_tensor[:-1].unsqueeze(1)   # (seq_len, 1, H, W)
        seq_out = traj_tensor[1:].unsqueeze(1)    # (seq_len, 1, H, W)

        return seq_in, seq_out, traj_tensor        # traj: (seq_len+1, H, W)


# ═══════════════════════════════════════════════════════════
# 2. BLOQUE ESPECTRAL
# ═══════════════════════════════════════════════════════════

class SpectralConv2d(nn.Module):
    """
    Convolución integral en espacio de Fourier: v' = F⁻¹(R · F(v))
    R ∈ ℂ^{in_ch × out_ch × modes1 × modes2} — parámetros aprendidos
    """

    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (in_ch * out_ch)
        self.W1 = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
        )
        self.W2 = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
        )

    def _mul(self, x, w):
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x):
        B, C, H, W = x.shape
        xf  = torch.fft.rfft2(x)
        out = torch.zeros(B, self.W1.shape[1], H, W // 2 + 1,
                          dtype=torch.cfloat, device=x.device)
        out[:, :,  :self.modes1, :self.modes2] = self._mul(
            xf[:, :,  :self.modes1, :self.modes2], self.W1
        )
        out[:, :, -self.modes1:, :self.modes2] = self._mul(
            xf[:, :, -self.modes1:, :self.modes2], self.W2
        )
        return torch.fft.irfft2(out, s=(H, W))


# ═══════════════════════════════════════════════════════════
# 3. CAPA FNO
# ═══════════════════════════════════════════════════════════

class FNOBlock(nn.Module):
    """v' = σ( F⁻¹(R·F(v)) + W·v )"""

    def __init__(self, ch, modes1, modes2):
        super().__init__()
        self.spectral = SpectralConv2d(ch, ch, modes1, modes2)
        self.local    = nn.Conv2d(ch, ch, kernel_size=1)
        self.norm     = nn.InstanceNorm2d(ch)

    def forward(self, x):
        return F.gelu(self.norm(self.spectral(x) + self.local(x)))


# ═══════════════════════════════════════════════════════════
# 4. GENERADOR
# ═══════════════════════════════════════════════════════════

class FNOGenerator(nn.Module):
    """
    G_θ: (wₙ, z) → wₙ₊₁   [M1]

    Input:  (B, 1, H, W)       vorticidad en t=n
            z ~ N(0,I)          (B, z_dim, H, W) — ruido espacial
    Output: (B, 1, H, W)       vorticidad en t=n+1

    La estocasticidad captura la degeneración del flujo turbulento:
    múltiples wₙ₊₁ físicamente válidos desde la misma condición wₙ.
    z_dim=0 recupera el generador determinista.
    """

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
        """
        w_n: (B, 1, H, W)
        z:   (B, z_dim, H, W) — si None se muestrea N(0,I)
        """
        B, _, H, W = w_n.shape
        if self.z_dim > 0:
            if z is None:
                z = torch.randn(B, self.z_dim, H, W, device=w_n.device)
            h = self.lift(torch.cat([w_n, z], dim=1))
        else:
            h = self.lift(w_n)
        for layer in self.layers:
            h = layer(h)
        return w_n + self.proj(h)   # residual: wₙ + Δw


# ═══════════════════════════════════════════════════════════
# 5. DISCRIMINADOR
# ═══════════════════════════════════════════════════════════

class FNODiscriminator(nn.Module):
    """
    D_φ: trayectoria (B, T, H, W) → score escalar (B, 1)
    Juzga coherencia dinámica de secuencias completas (no frames sueltos).
    """

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
        # [F10] .contiguous() garantiza que reshape no falle tras permute
        x = traj.permute(0, 2, 3, 1).contiguous().reshape(B * H * W, T, 1)
        x = self.temporal_mix(x).squeeze(-1)              # (B·H·W, hidden_ch)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, hidden_ch, H, W)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=(-2, -1))   # (B, hidden_ch)
        return self.head(x)         # (B, 1)


# ═══════════════════════════════════════════════════════════
# 6. PÉRDIDA NS EN FOURIER
# ═══════════════════════════════════════════════════════════

class NavierStokesLoss(nn.Module):
    """
    Residuo NS 2D periódico con Euler explícito puro: [F6]
      R = ∂w/∂t + u·∂w/∂x + v·∂w/∂y − ν·∇²w
    Todo evaluado en wₙ → integrador primer orden consistente y justificable.

    Conservación de masa: modo k=(0,0) anulado antes de calcular ψ. [F7]
    """

    def __init__(self, H, W, nu=1e-3, dt=0.01, device="cpu"):
        super().__init__()
        self.nu = nu
        self.dt = dt

        kx = torch.fft.fftfreq(W, d=1.0 / W)
        ky = torch.fft.fftfreq(H, d=1.0 / H)
        KY, KX = torch.meshgrid(ky, kx, indexing="ij")
        K2     = KX ** 2 + KY ** 2
        K2_inv = K2.clone();  K2_inv[0, 0] = 1.0

        self.register_buffer("KX",     KX)
        self.register_buffer("KY",     KY)
        self.register_buffer("K2",     K2)
        self.register_buffer("K2_inv", K2_inv)

    def _velocity_from_vorticity(self, w):
        """w: (B, H, W) → u, v: (B, H, W)"""
        wf = torch.fft.fft2(w)
        wf[:, 0, 0] = 0             # [F7] conservación de masa
        psi = wf / self.K2_inv
        psi[:, 0, 0] = 0
        u =  torch.fft.ifft2( 1j * self.KY * psi).real
        v =  torch.fft.ifft2(-1j * self.KX * psi).real
        return u, v

    def forward(self, w_n, w_pred):
        """
        w_n:    (B, 1, H, W)
        w_pred: (B, 1, H, W)
        → escalar: ‖R‖² promediado
        """
        # [F8] detección temprana de NaN/Inf
        if torch.isnan(w_pred).any() or torch.isinf(w_pred).any():
            raise RuntimeError(
                "NavierStokesLoss: w_pred contiene NaN/Inf — "
                "el generador divergió. Reduce lr_G o clip_grad_norm."
            )
        w_n    = w_n.squeeze(1)
        w_pred = w_pred.squeeze(1)

        dwdt  = (w_pred - w_n) / self.dt
        u, v  = self._velocity_from_vorticity(w_n)

        wn_f  = torch.fft.fft2(w_n)
        dwdx  = torch.fft.ifft2(1j * self.KX * wn_f).real
        dwdy  = torch.fft.ifft2(1j * self.KY * wn_f).real
        adv   = u * dwdx + v * dwdy
        lap_w = torch.fft.ifft2(-self.K2 * wn_f).real

        residual = dwdt + adv - self.nu * lap_w
        return (residual ** 2).mean()


# ═══════════════════════════════════════════════════════════
# 7. GRADIENT PENALTY
# ═══════════════════════════════════════════════════════════

class GradientPenalty(nn.Module):
    """GP = λ · 𝔼[(‖∇D(û)‖₂ − 1)²] sobre interpolaciones reales/falsas."""

    def __init__(self, lambda_gp=10.0):
        super().__init__()
        self.lambda_gp = lambda_gp

    def forward(self, discriminator, real_traj, fake_traj):
        # [F5] detectar mismatches silenciosos
        assert real_traj.shape == fake_traj.shape, (
            f"GP shape mismatch: real {real_traj.shape} vs fake {fake_traj.shape}"
        )
        B     = real_traj.size(0)
        alpha = torch.rand(B, 1, 1, 1, device=real_traj.device)
        interp = (alpha * real_traj + (1 - alpha) * fake_traj).requires_grad_(True)
        score  = discriminator(interp)
        grad   = torch.autograd.grad(
            outputs=score, inputs=interp,
            grad_outputs=torch.ones_like(score),
            create_graph=True, retain_graph=True,
        )[0]
        grad_norm = grad.flatten(1).norm(2, dim=1)
        return self.lambda_gp * ((grad_norm - 1) ** 2).mean()


# ═══════════════════════════════════════════════════════════
# 8. ROLLOUT
# ═══════════════════════════════════════════════════════════

class Rollout:
    """
    Genera trayectorias autoregresivas con G_θ entrenado.

    Métricas tras run():
      energy:    ½·‖w‖²  promedio por paso  (n_steps+1,)
      rmse:      RMSE vs ground truth        (si w_true dado)
      rel_error: error relativo L²           (si w_true dado)
    """

    def __init__(self, generator, device):
        self.generator = generator.to(device)
        self.device    = device
        self.metrics   = {}

    @torch.no_grad()
    def run(self, w0, n_steps, w_true=None, z=None):
        """
        w0:      (B, 1, H, W)
        n_steps: pasos hacia adelante
        w_true:  (B, T, H, W) opcional — ground truth para métricas
        z:       (B, z_dim, H, W) opcional
                   None  → ruido fresco N(0,I) en cada paso (máxima diversidad)
                   tensor → mismo z en todos los pasos (trayectoria reproducible)

        Retorna traj: (B, n_steps+1, H, W) en CPU
        """
        self.generator.eval()
        w_cur = w0.to(self.device)
        if z is not None:
            z = z.to(self.device)
        traj = [w_cur.squeeze(1).cpu()]

        for _ in range(n_steps):
            w_next = self.generator(w_cur, z=z)
            traj.append(w_next.squeeze(1).cpu())
            w_cur = w_next

        traj = torch.stack(traj, dim=1)   # (B, n_steps+1, H, W)
        self._compute_metrics(traj, w_true)
        self.generator.train()   # [F11] restaurar modo train
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
# 9. TRAINER ADVERSARIAL
# ═══════════════════════════════════════════════════════════

class WGAFNOGPTrainer:
    """
    Loop WGAN-GP para rollout de vorticidad de Kolmogorov.

    Pérdidas:
      L_D = 𝔼[D(fake)] − 𝔼[D(real)] + λ·GP
      L_G = −𝔼[D(fake)] + α·MSE + β·L_NS

    Incluye: tqdm, logging, scheduler LR, visualizaciones periódicas,
             espectro E(k), métricas físicas (energía, enstrofía).
    """

    def __init__(
        self,
        generator,
        discriminator,
        ns_loss,
        device,
        lr_G               = 1e-4,
        lr_D               = 1e-4,
        n_critic           = 5,
        lambda_gp          = 10.0,
        alpha_mse          = 1.0,
        beta_phys          = 0.1,
        use_scheduler      = True,    # [M2]
        scheduler_patience = 5,
        scheduler_factor   = 0.5,
        log_dir            = "logs",
        vis_freq           = 10,      # épocas entre visualizaciones
    ):
        self.G         = generator.to(device)
        self.D         = discriminator.to(device)
        self.ns_loss   = ns_loss.to(device)
        self.device    = device
        self.n_critic  = n_critic
        self.alpha_mse = alpha_mse
        self.beta_phys = beta_phys
        self.vis_freq  = vis_freq
        self.log_dir   = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.gp    = GradientPenalty(lambda_gp=lambda_gp).to(device)
        self.opt_G = torch.optim.Adam(generator.parameters(),     lr=lr_G, betas=(0.0, 0.9))
        self.opt_D = torch.optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.0, 0.9))

        # [M2] Schedulers
        self.use_scheduler = use_scheduler
        if use_scheduler:
            self.sched_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.opt_G, mode="min", factor=scheduler_factor, patience=scheduler_patience,
            )
            self.sched_D = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.opt_D, mode="min", factor=scheduler_factor, patience=scheduler_patience,
            )

        self._critic_iter = None   # [F14]
        self.val_example  = None   # [M9] fijado en primer fit()

        self.history = {
            "loss_D": [], "loss_G": [], "w_dist": [],
            "loss_mse": [], "loss_phys": [],
            "val_mse": [], "val_energy": [], "val_enstrophy": [],
            "lr_G": [], "lr_D": [],
        }

    # ── Generación de secuencias ───────────────────────────

    def _generate_sequence(self, seq_in, z=None):
        """
        seq_in: (B, seq_len, 1, H, W)
        z:      None → ruido fresco en cada paso | tensor → z fijo
        Retorna: fake_traj (B, seq_len+1, H, W), w_preds lista de (B,1,H,W)
        """
        B, seq_len, _, H, W = seq_in.shape
        w_cur   = seq_in[:, 0]
        fake    = [w_cur.squeeze(1)]
        w_preds = []
        for _ in range(seq_len):
            w_next = self.G(w_cur, z=z)
            fake.append(w_next.squeeze(1))
            w_preds.append(w_next)
            w_cur = w_next
        return torch.stack(fake, dim=1), w_preds

    def _physics_loss(self, seq_in, w_preds):
        total = sum(self.ns_loss(seq_in[:, t], wp) for t, wp in enumerate(w_preds))
        return total / len(w_preds)

    # ── Pasos de entrenamiento ─────────────────────────────

    def _step_D(self, loader):
        """n_critic pasos del discriminador, cada uno con batch distinto. [F4]"""
        loss_list                     = []
        score_real_last = score_fake_last = None

        for _ in range(self.n_critic):
            try:
                seq_in, _, real_traj = next(self._critic_iter)
            except StopIteration:
                self._critic_iter = iter(loader)
                seq_in, _, real_traj = next(self._critic_iter)

            seq_in    = seq_in.to(self.device)
            real_traj = real_traj.to(self.device)

            with torch.no_grad():
                fake_traj, _ = self._generate_sequence(seq_in)

            score_real = self.D(real_traj)
            score_fake = self.D(fake_traj)
            gp         = self.gp(self.D, real_traj, fake_traj)
            loss_D     = score_fake.mean() - score_real.mean() + gp

            self.opt_D.zero_grad()
            loss_D.backward()
            self.opt_D.step()
            loss_list.append(loss_D.item())
            score_real_last, score_fake_last = score_real, score_fake

        w_dist = (score_real_last.mean() - score_fake_last.mean()).item()
        return float(np.mean(loss_list)), w_dist

    def _step_G(self, seq_in, seq_out):
        """1 paso del generador."""
        seq_len            = seq_in.size(1)
        fake_traj, w_preds = self._generate_sequence(seq_in)
        loss_adv           = -self.D(fake_traj).mean()
        # [F12] MSE en loop — sin torch.cat, ahorra memoria en GPU ≤8GB
        loss_mse  = sum(F.mse_loss(wp, gt)
                        for wp, gt in zip(w_preds, seq_out.unbind(1))) / seq_len
        loss_phys = self._physics_loss(seq_in, w_preds)
        loss_G    = loss_adv + self.alpha_mse * loss_mse + self.beta_phys * loss_phys

        self.opt_G.zero_grad()
        loss_G.backward()
        nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)
        self.opt_G.step()
        return loss_G.item(), loss_mse.item(), loss_phys.item()

    # ── Validación ─────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, loader):
        """MSE + energía + enstrofía sobre el val set. [M4]"""
        self.G.eval()
        total_mse = total_energy = total_enstrophy = 0.0
        n = 0
        for seq_in, seq_out, _ in loader:
            seq_in  = seq_in.to(self.device)
            seq_out = seq_out.to(self.device)
            seq_len = seq_in.size(1)

            _, w_preds = self._generate_sequence(seq_in)

            total_mse += sum(
                F.mse_loss(wp, gt) for wp, gt in zip(w_preds, seq_out.unbind(1))
            ).item() / seq_len

            w_last = w_preds[-1].squeeze(1)   # (B, H, W)
            total_energy += 0.5 * (w_last ** 2).mean().item()

            # Enstrofía = ½ Σ |k|² |ŵ|²  (exacto en Fourier)
            wf = torch.fft.fft2(w_last)
            total_enstrophy += 0.5 * (self.ns_loss.K2 * wf.abs() ** 2).mean().item()
            n += 1

        self.G.train()
        return total_mse / n, total_energy / n, total_enstrophy / n

    # ── Espectro de energía cinética ───────────────────────

    @staticmethod
    def energy_spectrum(w: torch.Tensor):
        """
        E(k) por anillos en espacio de Fourier. [M5]

        E(k) = ½ Σ_{|k'|∈[k-0.5, k+0.5)} |ŵ(k')|²

        Para flujo de Kolmogorov turbulento se espera E(k) ~ k⁻³.

        Parámetros
        ----------
        w : (B, H, W) o (H, W)

        Retorna
        -------
        k_bins : (K,) ndarray — números de onda enteros 1..K
        E_k    : (K,) ndarray — energía por anillo (promedio sobre batch)
        """
        if w.ndim == 2:
            w = w.unsqueeze(0)
        B, H, W  = w.shape
        device   = w.device

        kx = torch.fft.fftfreq(W, d=1.0 / W).to(device)
        ky = torch.fft.fftfreq(H, d=1.0 / H).to(device)
        KY, KX = torch.meshgrid(ky, kx, indexing="ij")
        K_mag  = (KX ** 2 + KY ** 2).sqrt()   # (H, W)

        wf     = torch.fft.fft2(w)             # (B, H, W)
        E_2d   = 0.5 * wf.abs() ** 2          # (B, H, W)
        E_mean = E_2d.mean(dim=0)             # (H, W)

        k_max  = int(min(H, W) // 2)
        k_bins = np.arange(1, k_max + 1)
        E_k    = np.zeros(k_max)
        K_np   = K_mag.cpu().numpy()
        E_np   = E_mean.cpu().numpy()

        for i, k in enumerate(k_bins):
            mask   = (K_np >= k - 0.5) & (K_np < k + 0.5)
            E_k[i] = E_np[mask].sum() if mask.any() else 0.0

        return k_bins, E_k

    # ── Checkpointing ──────────────────────────────────────

    def save_checkpoint(self, epoch, val_mse):
        """[F2] Guarda estado completo: modelos, optimizadores, schedulers."""
        ckpt = {
            "epoch"  : epoch,
            "G"      : self.G.state_dict(),
            "D"      : self.D.state_dict(),
            "opt_G"  : self.opt_G.state_dict(),
            "opt_D"  : self.opt_D.state_dict(),
            "val_mse": val_mse,
        }
        if self.use_scheduler:
            ckpt["sched_G"] = self.sched_G.state_dict()
            ckpt["sched_D"] = self.sched_D.state_dict()
        torch.save(ckpt, self.log_dir / "best_checkpoint.pt")

    def load_checkpoint(self, path):
        """[F3] Restaura modelo, optimizadores y schedulers."""
        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["G"])
        self.D.load_state_dict(ckpt["D"])
        self.opt_G.load_state_dict(ckpt["opt_G"])
        self.opt_D.load_state_dict(ckpt["opt_D"])
        if self.use_scheduler and "sched_G" in ckpt:
            self.sched_G.load_state_dict(ckpt["sched_G"])
            self.sched_D.load_state_dict(ckpt["sched_D"])
        logger.info(
            f"Checkpoint cargado — época {ckpt['epoch']}, val MSE {ckpt['val_mse']:.6f}"
        )
        return ckpt["epoch"]

    # ── Visualizaciones ────────────────────────────────────

    def plot_fields(self, epoch):
        """
        Campos de vorticidad real vs generado para cada t. [M6]
        2 ejemplos × T pasos, 4 filas: real_0, fake_0, real_1, fake_1.
        Barra de color compartida. squeeze=False para robustez. [M10]
        Usa self.val_example fijo. [M9]
        """
        seq_in, _, real_traj = self.val_example
        seq_in    = seq_in[:2].to(self.device)
        real_traj = real_traj[:2]

        self.G.eval()
        with torch.no_grad():
            fake_traj, _ = self._generate_sequence(seq_in)
        self.G.train()

        real_np = real_traj.cpu().numpy()     # (2, T, H, W)
        fake_np = fake_traj.cpu().numpy()
        T       = real_np.shape[1]
        vmax    = max(np.abs(real_np).max(), np.abs(fake_np).max())

        fig, axes = plt.subplots(4, T, figsize=(3 * T, 10), squeeze=False)   # [M10]
        row_labels = ["Real (A)", "Generado (A)", "Real (B)", "Generado (B)"]

        for s, (r_row, f_row) in enumerate([(0, 1), (2, 3)]):
            for t in range(T):
                im = axes[r_row][t].imshow(
                    real_np[s, t], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    interpolation="nearest",
                )
                axes[r_row][t].set_title(f"t={t}", fontsize=8)
                axes[r_row][t].axis("off")
                axes[f_row][t].imshow(
                    fake_np[s, t], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    interpolation="nearest",
                )
                axes[f_row][t].axis("off")

        for row_idx, label in enumerate(row_labels):
            axes[row_idx][0].set_ylabel(label, fontsize=8, rotation=90, labelpad=4)

        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.35, label="vorticidad (norm.)")
        fig.suptitle(f"Campos de vorticidad — época {epoch}", fontsize=12)
        plt.tight_layout()
        path = self.log_dir / f"fields_epoch{epoch:04d}.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info(f"  Campos → {path}")

    def plot_losses(self):
        """
        Panel 2×3: losses adversariales, W-dist, LR,
        términos de L_G, val MSE, energía + enstrofía. [M7]
        Se sobreescribe en cada llamada — siempre refleja el estado actual.
        """
        h   = self.history
        eps = range(1, len(h["loss_D"]) + 1)

        fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)

        # (0,0) Losses adversariales
        axes[0][0].plot(eps, h["loss_D"], label="L_D", color="tab:red")
        axes[0][0].plot(eps, h["loss_G"], label="L_G", color="tab:blue")
        axes[0][0].set_title("Losses adversariales")
        axes[0][0].legend(); axes[0][0].set_xlabel("Época")

        # (0,1) Wasserstein distance
        axes[0][1].plot(eps, h["w_dist"], color="tab:green")
        axes[0][1].axhline(0, color="k", lw=0.8, ls="--")
        axes[0][1].set_title("Wasserstein distance"); axes[0][1].set_xlabel("Época")

        # (0,2) Learning rates (escala log)
        axes[0][2].plot(eps, h["lr_G"], label="lr_G", color="tab:purple")
        axes[0][2].plot(eps, h["lr_D"], label="lr_D", color="tab:orange")
        axes[0][2].set_title("Learning rates"); axes[0][2].legend()
        axes[0][2].set_xlabel("Época"); axes[0][2].set_yscale("log")

        # (1,0) Términos de L_G
        axes[1][0].plot(eps, h["loss_mse"],  label="MSE",  color="tab:cyan")
        axes[1][0].plot(eps, h["loss_phys"], label="Phys", color="tab:brown")
        axes[1][0].set_title("Términos de L_G")
        axes[1][0].legend(); axes[1][0].set_xlabel("Época")

        # (1,1) Val MSE
        axes[1][1].plot(eps, h["val_mse"], color="tab:red")
        axes[1][1].set_title("Val MSE"); axes[1][1].set_xlabel("Época")

        # (1,2) Métricas físicas — eje doble
        ax2 = axes[1][2].twinx()
        axes[1][2].plot(eps, h["val_energy"],    label="Energía",   color="tab:blue")
        ax2.plot(        eps, h["val_enstrophy"], label="Enstrofía", color="tab:orange", ls="--")
        axes[1][2].set_title("Métricas físicas (val)"); axes[1][2].set_xlabel("Época")
        axes[1][2].set_ylabel("Energía",   color="tab:blue")
        ax2.set_ylabel(       "Enstrofía", color="tab:orange")
        l1, lbl1 = axes[1][2].get_legend_handles_labels()
        l2, lbl2 = ax2.get_legend_handles_labels()
        axes[1][2].legend(l1 + l2, lbl1 + lbl2, fontsize=8)

        plt.suptitle("Historial de entrenamiento", fontsize=13)
        plt.tight_layout()
        path = self.log_dir / "losses.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info(f"  Curvas → {path}")

    def plot_spectrum(self, epoch):
        """
        E(k) real vs generado con línea k⁻³ de referencia. [M8]
        Figura esencial para paper — verifica cascada de energía de Kolmogorov.
        """
        seq_in, _, real_traj = self.val_example
        seq_in = seq_in.to(self.device)

        self.G.eval()
        with torch.no_grad():
            fake_traj, _ = self._generate_sequence(seq_in)
        self.G.train()

        w_real = real_traj[:, -1].float()   # (B, H, W) CPU
        w_fake = fake_traj[:, -1].cpu()

        k_real, E_real = self.energy_spectrum(w_real)
        k_fake, E_fake = self.energy_spectrum(w_fake)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(k_real, E_real, label="Real",     color="tab:blue", lw=2)
        ax.loglog(k_fake, E_fake, label="Generado", color="tab:red",  lw=2, ls="--")

        # Línea k⁻³ escalada al nivel real en k=2
        idx2  = np.where(k_real >= 2)[0][0]
        k_ref = k_real[idx2:]
        E_ref = E_real[idx2] * (k_ref / k_real[idx2]) ** (-3)
        ax.loglog(k_ref, E_ref, label=r"$k^{-3}$", color="k", lw=1, ls=":")

        ax.set_xlabel("Número de onda $k$", fontsize=12)
        ax.set_ylabel("$E(k)$",             fontsize=12)
        ax.set_title(f"Espectro de energía cinética — época {epoch}", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        path = self.log_dir / f"spectrum_epoch{epoch:04d}.png"
        plt.savefig(path, dpi=120)
        plt.close()
        logger.info(f"  Espectro → {path}")

    # ── Loop principal ─────────────────────────────────────

    def fit(self, train_loader, val_loader, n_epochs, log_every=10):
        best_val          = float("inf")
        self._critic_iter = iter(train_loader)   # [F14]

        # [M9] Fijar ejemplo de validación para visualizaciones reproducibles
        if self.val_example is None:
            self.val_example = next(iter(val_loader))

        for epoch in range(1, n_epochs + 1):
            self.G.train(); self.D.train()
            ep = {"ld": [], "wd": [], "lg": [], "mse": [], "phys": []}

            # [M3] barra de progreso
            pbar = tqdm(train_loader, desc=f"Epoch {epoch:4d}/{n_epochs}", leave=False)
            for seq_in, seq_out, _ in pbar:
                seq_in  = seq_in.to(self.device)
                seq_out = seq_out.to(self.device)
                ld, wd       = self._step_D(train_loader)
                lg, lmse, lp = self._step_G(seq_in, seq_out)
                ep["ld"].append(ld); ep["wd"].append(wd)
                ep["lg"].append(lg); ep["mse"].append(lmse); ep["phys"].append(lp)
                pbar.set_postfix({
                    "L_D": f"{ld:.3f}", "W": f"{wd:.3f}",
                    "L_G": f"{lg:.3f}", "mse": f"{lmse:.4f}",
                })

            val_mse, val_energy, val_enstrophy = self._validate(val_loader)

            # Historial
            self.history["loss_D"].append(float(np.mean(ep["ld"])))
            self.history["loss_G"].append(float(np.mean(ep["lg"])))
            self.history["w_dist"].append(float(np.mean(ep["wd"])))
            self.history["loss_mse"].append(float(np.mean(ep["mse"])))
            self.history["loss_phys"].append(float(np.mean(ep["phys"])))
            self.history["val_mse"].append(val_mse)
            self.history["val_energy"].append(val_energy)
            self.history["val_enstrophy"].append(val_enstrophy)
            self.history["lr_G"].append(self.opt_G.param_groups[0]["lr"])
            self.history["lr_D"].append(self.opt_D.param_groups[0]["lr"])

            # [M2] scheduler step
            if self.use_scheduler:
                self.sched_G.step(val_mse)
                self.sched_D.step(val_mse)

            # Checkpoint
            if val_mse < best_val:
                best_val = val_mse
                self.save_checkpoint(epoch, val_mse)
                logger.info(f"✅ Nuevo mejor — época {epoch}, val MSE={val_mse:.6f}")

            # [F13] liberar fragmentos CUDA cada 10 épocas
            if epoch % 10 == 0 and str(self.device) != "cpu":
                torch.cuda.empty_cache()

            # Visualizaciones periódicas
            if epoch % self.vis_freq == 0:
                self.plot_fields(epoch)
                self.plot_spectrum(epoch)
                self.plot_losses()

            if epoch % log_every == 0:
                logger.info(
                    f"Época {epoch:4d} | W {np.mean(ep['wd']):+.4f} | "
                    f"L_D {np.mean(ep['ld']):+.5f} | L_G {np.mean(ep['lg']):+.5f} | "
                    f"MSE {np.mean(ep['mse']):.5f} | Phys {np.mean(ep['phys']):.4f} | "
                    f"Val {val_mse:.5f} | E {val_energy:.4f} | Ω {val_enstrophy:.4f}"
                )

        # Plot final completo al terminar
        self.plot_losses()
        logger.info(
            f"\nMejor val MSE: {best_val:.6f} — {self.log_dir}/best_checkpoint.pt"
        )
        return self.history