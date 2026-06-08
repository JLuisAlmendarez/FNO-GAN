import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import logging
import json
import os
import random

logger = logging.getLogger(__name__)

# ── Configuración de logging ──────────────────────────────
def setup_logging(log_file="training.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

# ═══════════════════════════════════════════════════════════
# 1. DATASET
# ═══════════════════════════════════════════════════════════
class KolmogorovDataset(Dataset):
    # ... (sin cambios)
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
# 2. BLOQUE ESPECTRAL (versión original con complex)
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
    # ... (sin cambios)
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
    # ... (sin cambios)
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
# 5. RESIDUO NAVIER-STOKES
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
        return dwdt + spatial, nu, f

    def residuo_espacial(self, traj):
        B, T, H, W = traj.shape
        residuos = []
        nus = []
        fs = []
        for t in range(T - 1):
            r, nu, f = self._residuo_campo(traj[:, t], traj[:, t + 1])
            residuos.append(r)
            nus.append(nu)
            fs.append(f)
        return torch.stack(residuos, dim=1), nus, fs

    def energy_conservation_residual(self, traj, nus, fs):
        B, T, H, W = traj.shape
        res = []
        for t in range(T - 1):
            w_n    = traj[:, t]
            w_next = traj[:, t+1]
            wf_n    = torch.fft.fft2(w_n);    wf_n[:, 0, 0]    = 0
            wf_next = torch.fft.fft2(w_next); wf_next[:, 0, 0] = 0
            E_n    = 0.5 * (wf_n.abs()**2    * self.K2_inv).sum(dim=(1,2)) / (H*W)
            E_next = 0.5 * (wf_next.abs()**2 * self.K2_inv).sum(dim=(1,2)) / (H*W)
            dE_dt  = (E_next - E_n) / self.dt
            Z      = 0.5 * (w_n**2).mean(dim=(1,2))
            nu     = nus[t]
            f      = fs[t]
            f_omega = (f * w_n).mean(dim=(1,2))
            rhs    = -2 * nu * Z + f_omega
            res.append(dE_dt - rhs)
        return torch.stack(res, dim=1)

    def forward(self, traj):
        R, _, _ = self.residuo_espacial(traj)
        return (R ** 2).mean()

# ═══════════════════════════════════════════════════════════
# 6. ROLLOUT
# ═══════════════════════════════════════════════════════════
class Rollout:
    # ... (sin cambios)
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
# 7. FUNCIONES ESPECTRALES (sin cambios)
# ═══════════════════════════════════════════════════════════
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

def enstrophy_spectrum(w):
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

def palinstrophy_spectrum(w):
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

def transfer_spectrum(w, KX, KY, K2, K2_inv):
    if w.ndim == 2: w = w.unsqueeze(0)
    w  = w.to(KX.device)
    B, H, W = w.shape
    wf      = torch.fft.fft2(w); wf[:, 0, 0] = 0
    psi     = wf / K2_inv; psi[:, 0, 0] = 0
    u       =  torch.fft.ifft2( 1j * KY * psi).real
    v       =  torch.fft.ifft2(-1j * KX * psi).real
    adv     = u * torch.fft.ifft2(1j * KX * wf).real + v * torch.fft.ifft2(1j * KY * wf).real
    T_2d    = (wf.conj() * torch.fft.fft2(adv)).real.mean(dim=0)
    K_mag   = (KX**2 + KY**2).sqrt().cpu().numpy()
    T_np    = T_2d.cpu().numpy()
    k_max   = int(min(H, W) // 2)
    k_bins  = np.arange(1, k_max + 1)
    T_k     = np.array([T_np[(K_mag >= k-.5) & (K_mag < k+.5)].sum() for k in k_bins])
    return T_k, k_bins

def spectral_correlation(w_real, w_fake):
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

# ═══════════════════════════════════════════════════════════
# 8. UTILIDADES DE PERSISTENCIA ATÓMICA
# ═══════════════════════════════════════════════════════════
def save_atomic(obj, path):
    tmp_path = Path(str(path) + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)

def load_torch(path):
    return torch.load(path, map_location="cpu")

# ═══════════════════════════════════════════════════════════
# 9. BASE TRAINER — con early stopping (SIN AMP)
# ═══════════════════════════════════════════════════════════
class BaseTrainer:
    def __init__(self, log_dir="logs", resume=False, best_metric_name="val_loss",
                 patience=10):
        self.log_dir           = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.resume            = resume
        self.best_metric_name  = best_metric_name
        self.best_val          = float("inf")
        self.epoch_start       = 0
        self.history           = {}
        self.patience          = patience
        self.epochs_no_improve = 0
        self._setup_logger()
        if resume:
            self._resume_state()
        if not self.history:
            self._init_history_keys()

    def _setup_logger(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def _flush_logs(self):
        for handler in logging.getLogger().handlers:
            handler.flush()

    def _init_history_keys(self):
        pass

    def _resume_state(self):
        ckpt_path = self.log_dir / "latest_checkpoint.pt"
        hist_path = self.log_dir / "history.json"
        if ckpt_path.exists():
            self.logger.info(f"Reanudando desde {ckpt_path}")
            checkpoint = load_torch(ckpt_path)
            self._load_checkpoint_state(checkpoint)
            self.epoch_start       = checkpoint["epoch"] + 1
            self.best_val          = checkpoint.get("best_val", float("inf"))
            self.epochs_no_improve = checkpoint.get("epochs_no_improve", 0)
            self._restore_rng(checkpoint.get("rng_state"))
        else:
            self.logger.warning("No se encontró checkpoint; empezando desde cero.")
            self.epoch_start       = 0
            self.best_val          = float("inf")
            self.epochs_no_improve = 0
        if hist_path.exists():
            with open(hist_path, "r") as f:
                self.history = json.load(f)
            self.logger.info(f"Historial cargado ({len(next(iter(self.history.values())))} épocas previas).")

    def _restore_rng(self, rng_state):
        if rng_state is None: return
        torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and rng_state["cuda"] is not None:
            torch.cuda.set_rng_state_all(rng_state["cuda"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["python"])

    def _get_rng_state(self):
        return {
            "torch": torch.get_rng_state(),
            "cuda":  torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }

    def _load_checkpoint_state(self, checkpoint):
        raise NotImplementedError

    def _save_checkpoint_state(self, checkpoint, epoch, is_best):
        raise NotImplementedError

    def _get_best_model_state(self):
        raise NotImplementedError

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            "epoch":             epoch,
            "best_val":          self.best_val,
            "epochs_no_improve": self.epochs_no_improve,
            "rng_state":         self._get_rng_state(),
        }
        self._save_checkpoint_state(checkpoint, epoch, is_best)
        save_atomic(checkpoint, self.log_dir / "latest_checkpoint.pt")
        if is_best:
            save_atomic(self._get_best_model_state(), self.log_dir / "best_model.pt")
            self.logger.info(f"✅ Nuevo mejor modelo guardado (epoch {epoch}).")

    def _save_history(self):
        with open(self.log_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

    def _update_history(self, metrics_dict):
        for k, v in metrics_dict.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v)

    def _check_best(self, current_val):
        if current_val < self.best_val:
            self.best_val          = current_val
            self.epochs_no_improve = 0
            return True
        self.epochs_no_improve += 1
        return False

    @property
    def should_stop(self):
        return self.epochs_no_improve >= self.patience

    def log_epoch(self, epoch, metrics_dict):
        self._update_history(metrics_dict)
        self._save_history()
        current_val = metrics_dict.get(self.best_metric_name, None)
        is_best = False
        if current_val is not None:
            is_best = self._check_best(current_val)
        self.save_checkpoint(epoch, is_best=is_best)
        msg = f"Época {epoch:4d} | " + " | ".join(
            f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics_dict.items()
        )
        if is_best:
            msg += " ★"
        if self.epochs_no_improve > 0:
            msg += f" | no_improve: {self.epochs_no_improve}/{self.patience}"
        self.logger.info(msg)
        self._flush_logs()