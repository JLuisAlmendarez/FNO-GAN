"""
WGAFNOGP v2 — Rollout Autoregresivo, Flujo de Kolmogorov 2D
=============================================================
Dataset:   .npy shape (N, T, H, W)  — solo vorticidad
           e.g.  (1152, 100, 64, 64)

Generador: wₙ (1 canal) → wₙ₊₁  via FNO + residual connection
Discriminador: trayectoria (B, T+1, H, W) → escalar (WGAN)

Pérdidas:
  L_D = 𝔼[D(fake)] − 𝔼[D(real)] + λ·GP
  L_G = −𝔼[D(fake)] + α·MSE(pred, vort_{n+1}) + β·L_NS

  L_NS = ‖∂w/∂t + u·∂w/∂x + v·∂w/∂y − ν·∇²w‖²
         calculado en Fourier (exacto en malla periódica)
         u, v recuperados de w via función de corriente

Estructura:
  1. SpectralConv2d     — convolución en espacio de Fourier
  2. FNOBlock           — capa FNO completa
  3. FNOGenerator       — G_θ: wₙ → wₙ₊₁
  4. FNODiscriminator   — D_φ: trayectoria → escalar
  5. KolmogorovDataset  — carga .npy (N,T,H,W), ventanas deslizantes
  6. NavierStokesLoss   — residuo NS puro en Fourier (sin IBM)
  7. GradientPenalty    — GP en norma L² funcional sobre trayectorias (clase)
  8. Rollout            — generación autoregresiva con métricas
  9. WGAFNOGPTrainer    — loop adversarial completo
  10. main              — ejemplo de uso

FIXES aplicados (ronda 1 — bugs):
  - [FIX 1] w_true_cat reshape explícito en _step_G para evitar colapso silencioso de canal
  - [FIX 2] Checkpointing completo: guarda época, G, D, opt_G, opt_D, val_mse
  - [FIX 3] load_checkpoint() añadido a WGAFNOGPTrainer para reanudar entrenamiento
  - [FIX 4] _step_D itera sobre batches distintos del loader para los n_critic pasos
  - [FIX 5] assert shapes en GradientPenalty para detectar mismatches silenciosos

FIXES aplicados (ronda 2 — evaluación):
  - [FIX 6]  NavierStokesLoss: esquema advección consistente Euler explícito puro (todo en wₙ)
  - [FIX 7]  NavierStokesLoss: anular modo k=(0,0) de vorticidad antes de calcular ψ (conservación de masa)
  - [FIX 8]  NavierStokesLoss: detección de NaN/Inf con RuntimeError descriptivo
  - [FIX 9]  KolmogorovDataset: warning si w_std es degenerado + documentar .copy() obligatorio
  - [FIX 10] FNODiscriminator: .contiguous() antes de reshape tras permute
  - [FIX 11] Rollout.run: restaurar modo train tras eval + modo train explícito al salir
  - [FIX 12] _step_G: MSE acumulado en loop para evitar torch.cat duplicando memoria
  - [FIX 13] fit: torch.cuda.empty_cache() cada 10 épocas para GPU ≤8GB
  - [FIX 14] _critic_iter como atributo de instancia — elimina efecto secundario de _step_D
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path

# ═══════════════════════════════════════════════════════════
# 1. DATASET
# ═══════════════════════════════════════════════════════════
class KolmogorovDataset(Dataset):
    """
    Carga vorticidad Kolmogorov desde .npy shape (N, T, H, W).
    Construye ventanas sobre la marcha sin duplicar datos.
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
        self.seq_len = seq_len
        self.H, self.W = H, W
        self.N, self.T = N, T

        self.w_mean = float(w.mean())
        # [FIX 9] warning explícito si el dataset es casi constante
        raw_std = float(w.std())
        if raw_std < 1e-6:
            import warnings
            warnings.warn(
                f"w_std={raw_std:.2e} — dataset casi constante, "
                "normalización degenerada. Verifica el archivo .npy."
            )
        self.w_std = raw_std + 1e-8
        w = (w - self.w_mean) / self.w_std

        self.w = w                      # shape (N, T, H, W)
        self.n_windows = T - seq_len

        print(
            f"Dataset cargado: {N} trayectorias × {self.n_windows} ventanas "
            f"= {len(self):,} muestras  |  seq_len={seq_len}  |  H×W={H}×{W}"
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
        t0       = idx % self.n_windows

        traj = self.w[traj_idx, t0 : t0 + self.seq_len + 1]  # (seq_len+1, H, W)
        # [FIX 9] .copy() obligatorio: el slice en dim T produce un array
        # no contiguo en memoria; torch.from_numpy requiere array contiguo
        # y sin .copy() los workers del DataLoader pueden producir race conditions.
        traj_tensor = torch.from_numpy(traj.copy())
        seq_in      = traj_tensor[:-1].unsqueeze(1)   # (seq_len, 1, H, W)
        seq_out     = traj_tensor[1:].unsqueeze(1)    # (seq_len, 1, H, W)

        return seq_in, seq_out, traj_tensor


# ═══════════════════════════════════════════════════════════
# 2. BLOQUE ESPECTRAL
# ═══════════════════════════════════════════════════════════

class SpectralConv2d(nn.Module):
    """
    Convolución integral en espacio de Fourier.
    v' = F⁻¹( R · F(v) )
    R ∈ ℂ^{in_ch × out_ch × modes1 × modes2}  — parámetros aprendidos
    """

    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (in_ch * out_ch)
        self.W1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))
        self.W2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))

    def _mul(self, x, w):
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x):
        B, C, H, W = x.shape
        xf  = torch.fft.rfft2(x)
        out = torch.zeros(B, self.W1.shape[1], H, W // 2 + 1,
                          dtype=torch.cfloat, device=x.device)
        out[:, :,  :self.modes1, :self.modes2] = self._mul(xf[:, :,  :self.modes1, :self.modes2], self.W1)
        out[:, :, -self.modes1:, :self.modes2] = self._mul(xf[:, :, -self.modes1:, :self.modes2], self.W2)
        return torch.fft.irfft2(out, s=(H, W))


# ═══════════════════════════════════════════════════════════
# 3. CAPA FNO
# ═══════════════════════════════════════════════════════════

class FNOBlock(nn.Module):
    """
    v' = σ( F⁻¹(R·F(v)) + W·v )
    Término integral (global) + transformación local punto a punto.
    """

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
    G_θ: (wₙ, z) → wₙ₊₁   — generador estocástico

    Input:  (B, 1, H, W)   — vorticidad en t=n
            z ~ N(0,I)      — vector de ruido latente (B, z_dim, H, W)
                              broadcast espacial: el mismo z se repite en H×W
    Output: (B, 1, H, W)   — vorticidad en t=n+1

    La estocasticidad permite capturar la degeneración del flujo turbulento:
    múltiples wₙ₊₁ físicamente válidos desde la misma condición wₙ.

    Arquitectura:
      lift: (1 + z_dim) → hidden_ch   — canal de vorticidad + canales de ruido
      FNOBlocks × n_layers
      proj: hidden_ch → 1
      residual: wₙ + Δw

    z_dim=0 recupera el generador determinista original.
    """

    def __init__(self, hidden_ch=64, modes1=12, modes2=12, n_layers=4, z_dim=4):
        super().__init__()
        self.z_dim = z_dim
        # lift acepta 1 canal de vorticidad + z_dim canales de ruido
        self.lift = nn.Conv2d(1 + z_dim, hidden_ch, kernel_size=1)
        self.layers = nn.ModuleList([
            FNOBlock(hidden_ch, modes1, modes2) for _ in range(n_layers)
        ])
        self.proj = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_ch // 2, 1, kernel_size=1),
        )

    def forward(self, w_n, z=None):
        """
        w_n: (B, 1, H, W)
        z:   (B, z_dim, H, W) opcional — si None se muestrea N(0,I)

        Retorna: (B, 1, H, W)
        """
        B, _, H, W = w_n.shape

        if self.z_dim > 0:
            if z is None:
                # Muestreo en cada forward — estocasticidad durante entrenamiento
                z = torch.randn(B, self.z_dim, H, W, device=w_n.device)
            h = self.lift(torch.cat([w_n, z], dim=1))   # (B, 1+z_dim, H, W)
        else:
            # z_dim=0 → modo determinista, compatible con código antiguo
            h = self.lift(w_n)

        for layer in self.layers:
            h = layer(h)
        return w_n + self.proj(h)    # residual: wₙ + Δw


# ═══════════════════════════════════════════════════════════
# 5. DISCRIMINADOR
# ═══════════════════════════════════════════════════════════

class FNODiscriminator(nn.Module):
    """
    D_φ: trayectoria (B, T, H, W) → score escalar (B, 1)

    Juzga si una secuencia temporal completa es físicamente plausible.
    """

    def __init__(self, seq_len, hidden_ch=64, modes1=12, modes2=12, n_layers=4):
        super().__init__()
        self.seq_len = seq_len

        self.temporal_mix = nn.Sequential(
            nn.Conv1d(seq_len, hidden_ch // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_ch // 2, hidden_ch, kernel_size=1),
        )

        self.layers = nn.ModuleList([
            FNOBlock(hidden_ch, modes1, modes2) for _ in range(n_layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(hidden_ch, hidden_ch // 2),
            nn.GELU(),
            nn.Linear(hidden_ch // 2, 1),
        )

    def forward(self, traj):
        """
        traj: (B, T, H, W)
        returns: (B, 1)
        """
        B, T, H, W = traj.shape

        # [FIX 10] .contiguous() garantiza que el reshape no falle tras permute
        x = traj.permute(0, 2, 3, 1).contiguous().reshape(B * H * W, T, 1)
        x = self.temporal_mix(x)                          # (B·H·W, hidden_ch, 1)
        x = x.squeeze(-1)                                 # (B·H·W, hidden_ch)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, hidden_ch, H, W)

        for layer in self.layers:
            x = layer(x)

        x = x.mean(dim=(-2, -1))   # (B, hidden_ch)
        return self.head(x)        # (B, 1)


# ═══════════════════════════════════════════════════════════
# 6. PÉRDIDA NS PURA EN FOURIER
# ═══════════════════════════════════════════════════════════

class NavierStokesLoss(nn.Module):
    """
    Residuo de Navier-Stokes 2D en malla periódica:

      R = ∂w/∂t + u·∂w/∂x + v·∂w/∂y − ν·∇²w

    L_NS = ‖R‖²  promediado sobre batch y espacio.
    """

    def __init__(self, H, W, nu=1e-3, dt=0.01, device="cpu"):
        super().__init__()
        self.nu = nu
        self.dt = dt

        kx = torch.fft.fftfreq(W, d=1.0 / W)
        ky = torch.fft.fftfreq(H, d=1.0 / H)
        KY, KX = torch.meshgrid(ky, kx, indexing="ij")
        K2 = KX ** 2 + KY ** 2

        K2_inv = K2.clone()
        K2_inv[0, 0] = 1.0

        self.register_buffer("KX", KX)
        self.register_buffer("KY", KY)
        self.register_buffer("K2", K2)
        self.register_buffer("K2_inv", K2_inv)

    def _velocity_from_vorticity(self, w):
        """
        Recupera (u,v) de w via función de corriente ψ.
        [FIX 7] El modo k=(0,0) de w se anula antes de invertir
        para garantizar vorticidad media = 0 (conservación de masa).
        """
        wf = torch.fft.fft2(w)
        wf[:, 0, 0] = 0              # [FIX 7] anular componente de masa
        psi = wf / self.K2_inv
        psi[:, 0, 0] = 0
        u = torch.fft.ifft2( 1j * self.KY * psi).real
        v = torch.fft.ifft2(-1j * self.KX * psi).real
        return u, v

    def forward(self, w_n, w_pred):
        """
        w_n:    (B, 1, H, W)
        w_pred: (B, 1, H, W)
        Retorna: escalar — ‖R‖² promediado

        [FIX 6] Esquema Euler explícito puro: u, ∇w evaluados todos en wₙ.
                Esto hace el residuo NS matemáticamente consistente con un
                integrador de primer orden y justificable en un paper.
        [FIX 8] Detección de NaN/Inf antes de operar para evitar
                propagación silenciosa de gradientes inválidos.
        """
        # [FIX 8] Detección temprana de NaN/Inf
        if torch.isnan(w_pred).any() or torch.isinf(w_pred).any():
            raise RuntimeError(
                "NavierStokesLoss: w_pred contiene NaN/Inf — "
                "el generador divergió. Reduce lr_G o aumenta clip_grad_norm."
            )

        w_n    = w_n.squeeze(1)      # (B, H, W)
        w_pred = w_pred.squeeze(1)

        # ∂w/∂t — diferencia finita primer orden
        dwdt = (w_pred - w_n) / self.dt

        # [FIX 6] Todo evaluado en wₙ — Euler explícito puro
        u, v = self._velocity_from_vorticity(w_n)

        wn_f = torch.fft.fft2(w_n)
        dwdx = torch.fft.ifft2(1j * self.KX * wn_f).real   # gradiente de wₙ
        dwdy = torch.fft.ifft2(1j * self.KY * wn_f).real

        adv   = u * dwdx + v * dwdy
        lap_w = torch.fft.ifft2(-self.K2 * wn_f).real      # laplaciano de wₙ

        residual = dwdt + adv - self.nu * lap_w
        return (residual ** 2).mean()


# ═══════════════════════════════════════════════════════════
# 7. GRADIENT PENALTY
# ═══════════════════════════════════════════════════════════

class GradientPenalty(nn.Module):
    """
    GP = λ · 𝔼_û[ (‖∇_û D(û)‖_{L²} − 1)² ]
    û = interpolación convexa entre trayectoria real y falsa.
    """

    def __init__(self, lambda_gp=10.0):
        super().__init__()
        self.lambda_gp = lambda_gp

    def forward(self, discriminator, real_traj, fake_traj):
        # [FIX 5] assert para detectar mismatches silenciosos
        assert real_traj.shape == fake_traj.shape, (
            f"Shape mismatch en GP: real {real_traj.shape} vs fake {fake_traj.shape}"
        )

        B     = real_traj.size(0)
        alpha = torch.rand(B, 1, 1, 1, device=real_traj.device)
        interp = (alpha * real_traj + (1 - alpha) * fake_traj).requires_grad_(True)

        score = discriminator(interp)

        grad = torch.autograd.grad(
            outputs=score,
            inputs=interp,
            grad_outputs=torch.ones_like(score),
            create_graph=True,
            retain_graph=True,
        )[0]

        grad_norm = grad.flatten(1).norm(2, dim=1)   # (B,)
        return self.lambda_gp * ((grad_norm - 1) ** 2).mean()


# ═══════════════════════════════════════════════════════════
# 8. ROLLOUT
# ═══════════════════════════════════════════════════════════

class Rollout:
    """
    Genera trayectorias autoregresivas con G_θ entrenado.

    Métricas disponibles tras cada run():
      energy:    ½·‖w‖² promedio por paso  (n_steps+1,)
      rmse:      RMSE vs ground truth       (si w_true dado)
      rel_error: error relativo L²          (si w_true dado)
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
                   - None  → ruido fresco N(0,I) en cada paso (comportamiento por defecto)
                   - tensor → mismo z fijo en todos los pasos (trayectoria reproducible)
                   Útil en el paper para figuras comparativas con semilla fija.

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
        # [FIX 11] restaurar modo train para no afectar entrenamientos posteriores
        self.generator.train()
        return traj

    def _compute_metrics(self, traj, w_true):
        energy = 0.5 * (traj ** 2).mean(dim=(0, 2, 3))
        self.metrics["energy"] = energy.numpy()

        if w_true is not None:
            T    = min(traj.shape[1], w_true.shape[1])
            diff = traj[:, :T] - w_true[:, :T]
            self.metrics["rmse"] = diff.pow(2).mean(dim=(0, 2, 3)).sqrt().numpy()
            self.metrics["rel_error"] = (
                diff.norm(dim=(2, 3)) / (w_true[:, :T].norm(dim=(2, 3)) + 1e-8)
            ).mean(0).numpy()


# ═══════════════════════════════════════════════════════════
# 9. TRAINER ADVERSARIAL
# ═══════════════════════════════════════════════════════════

class WGAFNOGPTrainer:
    """
    Loop de entrenamiento WGAN-GP para rollout de vorticidad.

    Pérdidas:
      L_D = 𝔼[D(fake)] − 𝔼[D(real)] + λ·GP
      L_G = −𝔼[D(fake)] + α·MSE(pred, vort_{n+1}) + β·L_NS
    """

    def __init__(
        self,
        generator,
        discriminator,
        ns_loss,
        device,
        lr_G      = 1e-4,
        lr_D      = 1e-4,
        n_critic  = 5,
        lambda_gp = 10.0,
        alpha_mse = 1.0,
        beta_phys = 0.1,
    ):
        self.G         = generator.to(device)
        self.D         = discriminator.to(device)
        self.ns_loss   = ns_loss.to(device)
        self.device    = device
        self.n_critic  = n_critic
        self.lambda_gp = lambda_gp
        self.alpha_mse = alpha_mse
        self.beta_phys = beta_phys

        self.gp = GradientPenalty(lambda_gp=lambda_gp).to(device)

        self.opt_G = torch.optim.Adam(generator.parameters(),     lr=lr_G, betas=(0.0, 0.9))
        self.opt_D = torch.optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.0, 0.9))

        # [FIX 14] iterador del crítico como atributo — se inicializa en fit()
        # para evitar el efecto secundario de pasar loader_iter entre métodos
        self._critic_iter = None

        self.history = {
            "loss_D": [], "loss_G": [], "w_dist": [],
            "loss_mse": [], "loss_phys": [],
        }

    def _generate_sequence(self, seq_in, z=None):
        """
        seq_in: (B, seq_len, 1, H, W)
        z:      (B, z_dim, H, W) opcional — si None se muestrea N(0,I) en cada paso
                Pasar z fijo solo tiene sentido en rollout determinista para reproducibilidad.

        Retorna:
          fake_traj: (B, seq_len+1, H, W)
          w_preds:   lista de seq_len tensores (B, 1, H, W)
        """
        B, seq_len, _, H, W = seq_in.shape
        w_cur   = seq_in[:, 0]
        fake    = [w_cur.squeeze(1)]
        w_preds = []

        for t in range(seq_len):
            # z=None → ruido fresco en cada paso temporal (máxima diversidad)
            w_next = self.G(w_cur, z=z)
            fake.append(w_next.squeeze(1))
            w_preds.append(w_next)
            w_cur = w_next

        fake_traj = torch.stack(fake, dim=1)   # (B, seq_len+1, H, W)
        return fake_traj, w_preds

    def _physics_loss(self, seq_in, w_preds):
        total = 0.0
        for t, w_pred in enumerate(w_preds):
            w_n   = seq_in[:, t]
            total = total + self.ns_loss(w_n, w_pred)
        return total / len(w_preds)

    # [FIX 14] _step_D usa self._critic_iter — sin efectos secundarios hacia fit()
    def _step_D(self, loader):
        """
        n_critic pasos del discriminador, cada uno con un batch distinto.
        """
        loss_list = []
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

            gp     = self.gp(self.D, real_traj, fake_traj)
            loss_D = score_fake.mean() - score_real.mean() + gp

            self.opt_D.zero_grad()
            loss_D.backward()
            self.opt_D.step()
            loss_list.append(loss_D.item())

            score_real_last = score_real
            score_fake_last = score_fake

        w_dist = (score_real_last.mean() - score_fake_last.mean()).item()
        return float(np.mean(loss_list)), w_dist

    def _step_G(self, seq_in, seq_out):
        """1 paso del generador."""
        B, seq_len, _, H, W = seq_in.shape

        fake_traj, w_preds = self._generate_sequence(seq_in)

        loss_adv = -self.D(fake_traj).mean()

        # [FIX 12] MSE acumulado en loop — evita torch.cat que duplica memoria
        # especialmente crítico en GPU ≤8GB con seq_len largo
        loss_mse = sum(
            F.mse_loss(wp, gt)
            for wp, gt in zip(w_preds, seq_out.unbind(1))
        ) / seq_len

        loss_phys = self._physics_loss(seq_in, w_preds)

        loss_G = loss_adv + self.alpha_mse * loss_mse + self.beta_phys * loss_phys

        self.opt_G.zero_grad()
        loss_G.backward()
        nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)
        self.opt_G.step()

        return loss_G.item(), loss_mse.item(), loss_phys.item()

    @torch.no_grad()
    def _validate(self, loader):
        self.G.eval()
        total = 0.0
        for seq_in, seq_out, _ in loader:
            seq_in  = seq_in.to(self.device)
            seq_out = seq_out.to(self.device)
            _, seq_len, _, H, W = seq_in.shape

            _, w_preds = self._generate_sequence(seq_in)

            # [FIX 12] mismo patrón sin torch.cat
            batch_mse = sum(
                F.mse_loss(wp, gt)
                for wp, gt in zip(w_preds, seq_out.unbind(1))
            ) / seq_len
            total += batch_mse.item()

        self.G.train()
        return total / len(loader)

    # [FIX 3] Método para reanudar entrenamiento desde checkpoint
    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["G"])
        self.D.load_state_dict(ckpt["D"])
        self.opt_G.load_state_dict(ckpt["opt_G"])
        self.opt_D.load_state_dict(ckpt["opt_D"])
        print(f"Checkpoint cargado — época {ckpt['epoch']}, val MSE {ckpt['val_mse']:.6f}")
        return ckpt["epoch"]

    def fit(self, train_loader, val_loader, n_epochs, log_every=10):
        best_val = float("inf")
        # [FIX 14] inicializar iterador del crítico como atributo de instancia
        self._critic_iter = iter(train_loader)

        for epoch in range(1, n_epochs + 1):
            self.G.train(); self.D.train()
            ep = {"ld": [], "wd": [], "lg": [], "mse": [], "phys": []}

            for seq_in, seq_out, _ in train_loader:
                seq_in  = seq_in.to(self.device)
                seq_out = seq_out.to(self.device)

                # [FIX 14] _step_D ya no devuelve loader_iter
                ld, wd       = self._step_D(train_loader)
                lg, lmse, lp = self._step_G(seq_in, seq_out)

                ep["ld"].append(ld);   ep["wd"].append(wd)
                ep["lg"].append(lg);   ep["mse"].append(lmse)
                ep["phys"].append(lp)

            val_mse = self._validate(val_loader)

            for k, v in [("loss_D","ld"),("loss_G","lg"),("w_dist","wd"),
                          ("loss_mse","mse"),("loss_phys","phys")]:
                self.history[k].append(float(np.mean(ep[v])))

            # [FIX 2] Checkpoint completo con estado de optimizadores y época
            if val_mse < best_val:
                best_val = val_mse
                torch.save({
                    "epoch"  : epoch,
                    "G"      : self.G.state_dict(),
                    "D"      : self.D.state_dict(),
                    "opt_G"  : self.opt_G.state_dict(),
                    "opt_D"  : self.opt_D.state_dict(),
                    "val_mse": best_val,
                }, "best_checkpoint_wgafnogp.pt")

            # [FIX 13] liberar fragmentos de memoria CUDA cada 10 épocas
            if epoch % 10 == 0 and self.device != "cpu":
                torch.cuda.empty_cache()

            if epoch % log_every == 0:
                print(
                    f"Época {epoch:4d} | "
                    f"W-dist: {np.mean(ep['wd']):+.4f} | "
                    f"L_D: {np.mean(ep['ld']):+.5f} | "
                    f"L_G: {np.mean(ep['lg']):+.5f} | "
                    f"MSE: {np.mean(ep['mse']):.5f} | "
                    f"Phys: {np.mean(ep['phys']):.4f} | "
                    f"Val: {val_mse:.5f}"
                )

        print(f"\nMejor val MSE: {best_val:.6f}  — guardado en best_checkpoint_wgafnogp.pt")
        return self.history