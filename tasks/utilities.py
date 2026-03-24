"""
WGAFNOGP — Rollout Autoregresivo para Flujo de Kolmogorov 2D
=============================================================
Combina WGAN-GP con FNO para aprender el operador de avance wₙ → wₙ₊₁.

Arquitectura:
  - Generador G_θ  : FNOTimestepper  [wₙ, f, χ] → wₙ₊₁
  - Discriminador  : FNODiscriminator recibe una SECUENCIA completa
                     (B, T, H, W) y devuelve un score escalar
                     — juzga si la trayectoria ENTERA es plausible,
                     no solo un frame individual

Pérdida:
  L_G = −𝔼[D(traj_fake)] + α·L_data + β·L_physics
  L_D =  𝔼[D(traj_fake)] − 𝔼[D(traj_real)] + λ·GP

Estructura:
  1. SpectralConv2d          — convolución en Fourier
  2. FNOBlock                — capa FNO (integral + local)
  3. FNOGenerator            — G_θ: [wₙ, f, χ] → wₙ₊₁
  4. FNODiscriminator        — D_φ: trayectoria (B,T,H,W) → escalar
  5. KolmogorovDataset       — carga .npz, construye secuencias
  6. NavierStokesIBMLoss     — residuo NS-IBM en Fourier
  7. gradient_penalty        — GP en norma L² funcional
  8. Rollout                 — rollout autoregresivo con métricas
  9. WGAFNOGPTrainer         — loop adversarial completo
  10. main                   — ejemplo de uso
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path


# ═══════════════════════════════════════════════════════════
# 1. BLOQUE ESPECTRAL
# ═══════════════════════════════════════════════════════════

class SpectralConv2d(nn.Module):
    """v' = F⁻¹( R · F(v) ) — aprende R ∈ ℂ^{modes×modes×in×out}"""

    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        scale = 1 / (in_ch * out_ch)
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
# 2. CAPA FNO
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
# 3. GENERADOR — FNO Timestepper
# ═══════════════════════════════════════════════════════════

class FNOGenerator(nn.Module):
    """
    G_θ: [wₙ, f, χ] → wₙ₊₁

    Input:  (B, 3, H, W)
    Output: (B, 1, H, W)

    Usa conexión residual: predice Δw = wₙ₊₁ − wₙ
    """

    def __init__(self, in_channels=3, hidden_ch=64, modes1=12, modes2=12, n_layers=4):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, hidden_ch, kernel_size=1)
        self.layers = nn.ModuleList([
            FNOBlock(hidden_ch, modes1, modes2) for _ in range(n_layers)
        ])
        self.proj = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_ch // 2, 1, kernel_size=1),
        )

    def forward(self, x):
        """x: (B, 3, H, W) → (B, 1, H, W)"""
        w_n = x[:, :1]
        h   = self.lift(x)
        for layer in self.layers:
            h = layer(h)
        return w_n + self.proj(h)   # aprendizaje residual


# ═══════════════════════════════════════════════════════════
# 4. DISCRIMINADOR — juzga TRAYECTORIAS completas
# ═══════════════════════════════════════════════════════════

class FNODiscriminator(nn.Module):
    """
    D_φ: (B, T, H, W) → (B, 1)

    Recibe una secuencia temporal completa y devuelve un score escalar.
    Juzgar trayectorias (no frames individuales) es clave: un frame
    suelto puede parecer físicamente plausible pero la dinámica temporal
    puede ser incorrecta. El discriminador ve toda la secuencia y detecta
    inconsistencias temporales.

    Arquitectura:
      - Tratar T como canales → Conv temporal → FNO espacial → escalar
    """

    def __init__(self, seq_len=10, hidden_ch=64, modes1=12, modes2=12, n_layers=4):
        super().__init__()
        self.seq_len = seq_len

        # Mezcla temporal: Conv1d trata T como canales, L=1 por pixel
        # (B·H·W, T, 1) → Conv1d(in=T, out=hidden) → (B·H·W, hidden_ch, 1)
        self.temporal_mix = nn.Sequential(
            nn.Conv1d(seq_len, hidden_ch // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_ch // 2, hidden_ch, kernel_size=1),
        )

        # Capas FNO espaciales
        self.layers = nn.ModuleList([
            FNOBlock(hidden_ch, modes1, modes2) for _ in range(n_layers)
        ])

        # Cabeza escalar — sin sigmoid (WGAN usa scores sin acotar)
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

        # Mezcla temporal por pixel:
        # (B, T, H, W) → (B·H·W, T) como canales → Conv1d → (B·H·W, hidden_ch)
        # Conv1d espera (N, C_in, L): usamos C_in=T, L=1
        x = traj.permute(0, 2, 3, 1).reshape(B * H * W, T, 1)   # (B·H·W, T, 1)
        x = self.temporal_mix(x)                                   # (B·H·W, hidden_ch, 1)
        x = x.squeeze(-1)                                          # (B·H·W, hidden_ch)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)           # (B, hidden_ch, H, W)

        # Capas FNO espaciales
        for layer in self.layers:
            x = layer(x)

        # Global average pool → escalar
        x = x.mean(dim=(-2, -1))    # (B, hidden_ch)
        return self.head(x)         # (B, 1)


# ═══════════════════════════════════════════════════════════
# 5. DATASET
# ═══════════════════════════════════════════════════════════

class KolmogorovDataset(Dataset):
    """
    Carga trayectorias Kolmogorov 2D desde .npz.

    Formato esperado:
      vorticity: (N, T, H, W)   p.ej. (1152, 100, 64, 64)
      forcing:   (H, W)
      chi:       (H, W)         máscara IBM — si no existe usa zeros

    Cada muestra es una sub-secuencia de longitud seq_len+1:
      X_seq: (seq_len, H, W)    — frames w_0 ... w_{T-1}
      Y_seq: (seq_len, H, W)    — frames w_1 ... w_T  (targets)
    """

    def __init__(self, path, seq_len=10, vort_key="vorticity",
                 f_key="forcing", chi_key="chi"):
        path = Path(path)
        self.seq_len = seq_len

        # ── Cargar ─────────────────────────────────────────
        data = np.load(path)
        w    = data[vort_key].astype(np.float32)          # (N, T, H, W)
        f    = data[f_key].astype(np.float32)             # (H, W)
        chi  = data[chi_key].astype(np.float32) if chi_key in data \
               else np.zeros(f.shape, dtype=np.float32)

        N, T, H, W = w.shape
        assert T > seq_len, f"T={T} debe ser > seq_len={seq_len}"

        # ── Normalizar ─────────────────────────────────────
        self.w_mean = float(w.mean())
        self.w_std  = float(w.std()) + 1e-8
        w = (w - self.w_mean) / self.w_std

        f_max = float(np.abs(f).max()) + 1e-8
        f = f / f_max

        # ── Construir sub-secuencias deslizantes ───────────
        # Por cada trayectoria, extraer ventanas de longitud seq_len+1
        # Número de ventanas por trayectoria: T - seq_len
        seqs = []
        for t0 in range(T - seq_len):
            seqs.append(w[:, t0 : t0 + seq_len + 1])   # (N, seq_len+1, H, W)
        # seqs: lista de (N, seq_len+1, H, W) → concatenar en dim 0
        seqs = np.concatenate(seqs, axis=0)             # (N*(T-seq_len), seq_len+1, H, W)

        self.seqs = torch.from_numpy(seqs)              # (M, seq_len+1, H, W)

        # Campos estáticos
        self.f_tensor   = torch.from_numpy(f)[None, None]    # (1, 1, H, W)
        self.chi_tensor = torch.from_numpy(chi)[None, None]  # (1, 1, H, W)

        # f y chi expandidos para concatenar con cada frame
        self.f_field   = torch.from_numpy(f)    # (H, W)
        self.chi_field = torch.from_numpy(chi)  # (H, W)

        print(f"Dataset: {len(self):,} secuencias  |  seq_len={seq_len}  |  shape traj: {self.seqs.shape}")

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        """
        Retorna:
          seq_in:  (seq_len, 3, H, W) — [wₙ, f, χ] para cada paso
          seq_out: (seq_len, H, W)    — wₙ₊₁ para cada paso (targets)
          traj:    (seq_len+1, H, W)  — trayectoria completa (para D)
        """
        traj   = self.seqs[idx]                          # (seq_len+1, H, W)
        H, W   = traj.shape[-2], traj.shape[-1]

        f_exp   = self.f_field.expand(self.seq_len, H, W)    # (seq_len, H, W)
        chi_exp = self.chi_field.expand(self.seq_len, H, W)  # (seq_len, H, W)

        w_in  = traj[:-1].unsqueeze(1)   # (seq_len, 1, H, W)
        f_in  = f_exp.unsqueeze(1)       # (seq_len, 1, H, W)
        chi_in = chi_exp.unsqueeze(1)    # (seq_len, 1, H, W)

        seq_in  = torch.cat([w_in, f_in, chi_in], dim=1)  # (seq_len, 3, H, W)
        seq_out = traj[1:]                                  # (seq_len, H, W)

        return seq_in, seq_out, traj


# ═══════════════════════════════════════════════════════════
# 6. PÉRDIDA NS-IBM EN FOURIER
# ═══════════════════════════════════════════════════════════

class NavierStokesIBMLoss(nn.Module):
    """
    Residuo NS-IBM en Fourier:
      R = ∂w/∂t + u·∂w/∂x + v·∂w/∂y − ν·∇²w − f + (χ/η)·w + (1/η)·(v·∂χ/∂x − u·∂χ/∂y)
    """

    def __init__(self, H, W, nu=1e-3, dt=0.01, eta=1e-3, device="cpu"):
        super().__init__()
        self.nu, self.dt, self.eta = nu, dt, eta

        kx = torch.fft.fftfreq(W, d=1.0 / W).to(device)
        ky = torch.fft.fftfreq(H, d=1.0 / H).to(device)
        KY, KX = torch.meshgrid(ky, kx, indexing="ij")
        K2 = KX ** 2 + KY ** 2
        K2_inv = K2.clone(); K2_inv[0, 0] = 1.0

        self.register_buffer("KX", KX)
        self.register_buffer("KY", KY)
        self.register_buffer("K2", K2)
        self.register_buffer("K2_inv", K2_inv)

    def _velocity(self, wf):
        psi = wf / self.K2_inv
        psi[:, 0, 0] = 0
        u = torch.fft.ifft2( 1j * self.KY * psi).real
        v = torch.fft.ifft2(-1j * self.KX * psi).real
        return u, v

    def residual(self, w_n, w_pred, f, chi):
        wn_f    = torch.fft.fft2(w_n.squeeze(1))
        wpred_f = torch.fft.fft2(w_pred.squeeze(1))

        dwdt = (w_pred - w_n) / self.dt

        u, v      = self._velocity(wn_f)
        dwdx      = torch.fft.ifft2(1j * self.KX * wpred_f).real
        dwdy      = torch.fft.ifft2(1j * self.KY * wpred_f).real
        lap_w     = torch.fft.ifft2(-self.K2 * wpred_f).real.unsqueeze(1)
        adv       = (u * dwdx + v * dwdy).unsqueeze(1)

        chi_f    = torch.fft.fft2(chi.squeeze(1))
        dchi_dx  = torch.fft.ifft2(1j * self.KX * chi_f).real
        dchi_dy  = torch.fft.ifft2(1j * self.KY * chi_f).real
        ibm_vol  = (chi / self.eta) * w_pred
        ibm_surf = (1.0 / self.eta) * (v.unsqueeze(1) * dchi_dx - u.unsqueeze(1) * dchi_dy)

        return dwdt + adv - self.nu * lap_w - f + ibm_vol + ibm_surf

    def forward(self, w_n, w_pred, f, chi):
        res = self.residual(w_n, w_pred, f, chi)
        return (res ** 2).mean()


# ═══════════════════════════════════════════════════════════
# 7. GRADIENT PENALTY EN NORMA L² FUNCIONAL
# ═══════════════════════════════════════════════════════════

def gradient_penalty(discriminator, real_traj, fake_traj, device, lambda_gp=10.0):
    """
    GP sobre trayectorias completas:
      GP = λ · E_û[ (‖∇_û D(û)‖_{L²} − 1)² ]
    donde û es interpolación entre trayectoria real y falsa.
    """
    B = real_traj.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)
    interp = (alpha * real_traj + (1 - alpha) * fake_traj).requires_grad_(True)

    score = discriminator(interp)

    grad = torch.autograd.grad(
        outputs=score,
        inputs=interp,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Norma L² sobre toda la trayectoria (T, H, W)
    grad_norm = grad.flatten(1).norm(2, dim=1)
    return lambda_gp * ((grad_norm - 1) ** 2).mean()


# ═══════════════════════════════════════════════════════════
# 8. ROLLOUT
# ═══════════════════════════════════════════════════════════

class Rollout:
    """
    Genera trayectorias autoregresivas con G_θ entrenado.
    f y χ son estáticos → se guardan como atributos.
    Acumula métricas: energía, RMSE, error relativo L².
    """

    def __init__(self, generator, f, chi, device):
        self.generator = generator.to(device)
        self.f         = f.to(device)
        self.chi       = chi.to(device)
        self.device    = device
        self.metrics   = {}

    @torch.no_grad()
    def run(self, w0, n_steps, w_true=None):
        """
        w0:      (B, 1, H, W) — condición inicial
        n_steps: pasos a generar
        w_true:  (B, T, H, W) opcional — ground truth para métricas

        Retorna traj: (B, n_steps+1, H, W)
        """
        self.generator.eval()
        B, _, H, W = w0.shape
        f_rep   = self.f.expand(B, -1, H, W)
        chi_rep = self.chi.expand(B, -1, H, W)

        w_cur = w0.to(self.device)
        traj  = [w_cur.squeeze(1).cpu()]

        for _ in range(n_steps):
            x_in   = torch.cat([w_cur, f_rep, chi_rep], dim=1)
            w_next = self.generator(x_in)
            traj.append(w_next.squeeze(1).cpu())
            w_cur  = w_next

        traj = torch.stack(traj, dim=1)   # (B, n_steps+1, H, W)
        self._compute_metrics(traj, w_true)
        return traj

    def _compute_metrics(self, traj, w_true):
        energy = 0.5 * (traj ** 2).mean(dim=(0, 2, 3))
        self.metrics["energy"] = energy.numpy()

        if w_true is not None:
            T    = min(traj.shape[1], w_true.shape[1])
            diff = traj[:, :T] - w_true[:, :T]
            self.metrics["rmse"]      = diff.pow(2).mean(dim=(0,2,3)).sqrt().numpy()
            self.metrics["rel_error"] = (
                diff.norm(dim=(2,3)) / (w_true[:,:T].norm(dim=(2,3)) + 1e-8)
            ).mean(0).numpy()


# ═══════════════════════════════════════════════════════════
# 9. TRAINER ADVERSARIAL
# ═══════════════════════════════════════════════════════════

class WGAFNOGPTrainer:
    """
    Loop de entrenamiento WGAN-GP para rollout de fluidos.

    Cada batch contiene secuencias de longitud seq_len.
    El generador produce la secuencia paso a paso (rollout corto).
    El discriminador juzga la secuencia completa.

    Pérdidas:
      L_D = 𝔼[D(fake)] − 𝔼[D(real)] + λ·GP          (maximizar separación)
      L_G = −𝔼[D(fake)] + α·L_MSE + β·L_physics       (engañar D + física)
    """

    def __init__(
        self,
        generator,
        discriminator,
        ns_loss,
        device,
        lr_G       = 1e-4,
        lr_D       = 1e-4,
        n_critic   = 5,
        lambda_gp  = 10.0,
        alpha_mse  = 1.0,
        beta_phys  = 0.1,
    ):
        self.G          = generator.to(device)
        self.D          = discriminator.to(device)
        self.ns_loss    = ns_loss.to(device)
        self.device     = device
        self.n_critic   = n_critic
        self.lambda_gp  = lambda_gp
        self.alpha_mse  = alpha_mse
        self.beta_phys  = beta_phys

        self.opt_G = torch.optim.Adam(generator.parameters(),     lr=lr_G, betas=(0.0, 0.9))
        self.opt_D = torch.optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.0, 0.9))

        self.history = {
            "loss_D": [], "loss_G": [], "w_dist": [],
            "loss_mse": [], "loss_phys": [],
        }

    def _generate_trajectory(self, seq_in):
        """
        Aplica G paso a paso sobre la secuencia de entrada.

        seq_in: (B, seq_len, 3, H, W)
        returns fake_traj: (B, seq_len+1, H, W)
                           incluye w0 + seq_len pasos generados
        """
        B, seq_len, _, H, W = seq_in.shape

        # Condición inicial = primer frame de la secuencia
        w_cur   = seq_in[:, 0, :1]                   # (B, 1, H, W)
        fake    = [w_cur.squeeze(1)]                  # lista de (B, H, W)
        w_preds = []                                  # para la pérdida

        for t in range(seq_len):
            # Usar f y χ del input (canales 1 y 2)
            f_t   = seq_in[:, t, 1:2]                # (B, 1, H, W)
            chi_t = seq_in[:, t, 2:3]                # (B, 1, H, W)
            x_in  = torch.cat([w_cur, f_t, chi_t], dim=1)
            w_next = self.G(x_in)
            fake.append(w_next.squeeze(1))
            w_preds.append(w_next)
            w_cur = w_next

        fake_traj = torch.stack(fake, dim=1)          # (B, seq_len+1, H, W)
        return fake_traj, w_preds

    def _physics_loss(self, seq_in, w_preds):
        """
        Calcula pérdida física acumulada sobre la secuencia.
        w_preds: lista de seq_len tensores (B, 1, H, W)
        """
        total = 0.0
        for t, w_pred in enumerate(w_preds):
            w_n   = seq_in[:, t, :1]
            f     = seq_in[:, t, 1:2]
            chi   = seq_in[:, t, 2:3]
            total = total + self.ns_loss(w_n, w_pred, f, chi)
        return total / len(w_preds)

    def train_step_D(self, seq_in, real_traj):
        """n_critic pasos del discriminador"""
        B, seq_len, _, H, W = seq_in.shape
        losses = []

        for _ in range(self.n_critic):
            with torch.no_grad():
                fake_traj, _ = self._generate_trajectory(seq_in)

            score_real = self.D(real_traj)
            score_fake = self.D(fake_traj)

            gp     = gradient_penalty(self.D, real_traj, fake_traj,
                                      self.device, self.lambda_gp)
            loss_D = score_fake.mean() - score_real.mean() + gp

            self.opt_D.zero_grad()
            loss_D.backward()
            self.opt_D.step()
            losses.append(loss_D.item())

        w_dist = (score_real.mean() - score_fake.mean()).item()
        return np.mean(losses), w_dist

    def train_step_G(self, seq_in, seq_out):
        """1 paso del generador"""
        fake_traj, w_preds = self._generate_trajectory(seq_in)

        # Pérdida adversarial
        loss_adv = -self.D(fake_traj).mean()

        # Pérdida MSE (supervisión frame a frame)
        w_pred_stack = torch.cat(w_preds, dim=0)           # (B*seq_len, 1, H, W)
        w_true_stack = seq_out.reshape(-1, 1, *seq_out.shape[-2:])
        loss_mse = F.mse_loss(w_pred_stack, w_true_stack)

        # Pérdida física NS-IBM
        loss_phys = self._physics_loss(seq_in, w_preds)

        loss_G = loss_adv + self.alpha_mse * loss_mse + self.beta_phys * loss_phys

        self.opt_G.zero_grad()
        loss_G.backward()
        nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)
        self.opt_G.step()

        return loss_G.item(), loss_mse.item(), loss_phys.item()

    def fit(self, train_loader, val_loader, n_epochs, log_every=10):
        best_val_mse = float("inf")

        for epoch in range(1, n_epochs + 1):
            self.G.train(); self.D.train()

            ep_ld, ep_wdist, ep_lg, ep_mse, ep_phys = [], [], [], [], []

            for seq_in, seq_out, real_traj in train_loader:
                seq_in    = seq_in.to(self.device)
                seq_out   = seq_out.to(self.device)
                real_traj = real_traj.to(self.device)

                ld, wd = self.train_step_D(seq_in, real_traj)
                lg, lm, lp = self.train_step_G(seq_in, seq_out)

                ep_ld.append(ld); ep_wdist.append(wd)
                ep_lg.append(lg); ep_mse.append(lm); ep_phys.append(lp)

            # Validación
            val_mse = self._validate(val_loader)

            # Guardar métricas
            self.history["loss_D"].append(np.mean(ep_ld))
            self.history["loss_G"].append(np.mean(ep_lg))
            self.history["w_dist"].append(np.mean(ep_wdist))
            self.history["loss_mse"].append(np.mean(ep_mse))
            self.history["loss_phys"].append(np.mean(ep_phys))

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                torch.save(self.G.state_dict(), "best_generator_wgafnogp.pt")
                torch.save(self.D.state_dict(), "best_discriminator_wgafnogp.pt")

            if epoch % log_every == 0:
                print(
                    f"Época {epoch:4d} | "
                    f"W-dist: {np.mean(ep_wdist):+.4f} | "
                    f"L_D: {np.mean(ep_ld):+.4f} | "
                    f"L_G: {np.mean(ep_lg):+.4f} | "
                    f"MSE: {np.mean(ep_mse):.4f} | "
                    f"Phys: {np.mean(ep_phys):.4f} | "
                    f"Val MSE: {val_mse:.4f}"
                )

        print(f"\nMejor val MSE: {best_val_mse:.6f}")
        return self.history

    @torch.no_grad()
    def _validate(self, loader):
        self.G.eval()
        total_mse = 0.0
        for seq_in, seq_out, _ in loader:
            seq_in  = seq_in.to(self.device)
            seq_out = seq_out.to(self.device)
            _, w_preds = self._generate_trajectory(seq_in)
            w_pred_stack = torch.cat(w_preds, dim=0)
            w_true_stack = seq_out.reshape(-1, 1, *seq_out.shape[-2:])
            total_mse += F.mse_loss(w_pred_stack, w_true_stack).item()
        return total_mse / len(loader)


# ═══════════════════════════════════════════════════════════
# 10. MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Configuración ──────────────────────────────────────
    DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
    H, W      = 64, 64
    SEQ_LEN   = 10      # longitud de secuencia para D
    BATCH     = 8       # RTX 3070: puede subir a 16-32
    EPOCHS    = 200
    MODES     = 12
    HIDDEN    = 64
    N_LAYERS  = 4
    NU        = 1e-3
    DT        = 0.01
    ETA       = 1e-3
    N_CRITIC  = 5
    LAMBDA_GP = 10.0
    ALPHA_MSE = 1.0     # peso supervisión frame a frame
    BETA_PHYS = 0.1     # peso residuo NS-IBM

    print(f"Dispositivo: {DEVICE}")

    # ── Dataset ────────────────────────────────────────────
    # Con tu archivo real:
    #   dataset = KolmogorovDataset("tu_archivo.npz", seq_len=SEQ_LEN)
    #
    # Dataset sintético para prueba:
    print("Generando dataset sintético...")
    N_TRAJ, T = 20, 100
    w_fake  = np.random.randn(N_TRAJ, T, H, W).astype(np.float32)
    f_fake  = np.zeros((H, W), dtype=np.float32)
    y_grid  = np.linspace(0, 2 * np.pi, H, endpoint=False)
    f_fake += np.sin(4 * y_grid)[:, None]
    chi_fake = np.zeros((H, W), dtype=np.float32)
    np.savez("kolmogorov_synthetic.npz",
             vorticity=w_fake, forcing=f_fake, chi=chi_fake)

    dataset  = KolmogorovDataset("kolmogorov_synthetic.npz", seq_len=SEQ_LEN)
    n_val    = max(1, int(0.1 * len(dataset)))
    n_train  = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  drop_last=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, drop_last=False, num_workers=0)
    print(f"Train: {n_train:,}  |  Val: {n_val:,}  |  seq_len: {SEQ_LEN}")

    # ── Modelos ────────────────────────────────────────────
    generator = FNOGenerator(
        in_channels=3, hidden_ch=HIDDEN,
        modes1=MODES, modes2=MODES, n_layers=N_LAYERS,
    ).to(DEVICE)

    discriminator = FNODiscriminator(
        seq_len=SEQ_LEN + 1,   # D ve seq_len+1 frames (incluye w0)
        hidden_ch=HIDDEN,
        modes1=MODES, modes2=MODES, n_layers=N_LAYERS,
    ).to(DEVICE)

    ns_loss = NavierStokesIBMLoss(H=H, W=W, nu=NU, dt=DT, eta=ETA, device=DEVICE)

    n_G = sum(p.numel() for p in generator.parameters())
    n_D = sum(p.numel() for p in discriminator.parameters())
    print(f"Parámetros G: {n_G:,}  |  D: {n_D:,}")

    # ── Entrenamiento ──────────────────────────────────────
    trainer = WGAFNOGPTrainer(
        generator=generator,
        discriminator=discriminator,
        ns_loss=ns_loss,
        device=DEVICE,
        lr_G=1e-4, lr_D=1e-4,
        n_critic=N_CRITIC,
        lambda_gp=LAMBDA_GP,
        alpha_mse=ALPHA_MSE,
        beta_phys=BETA_PHYS,
    )

    history = trainer.fit(train_loader, val_loader, n_epochs=EPOCHS, log_every=10)

    # ── Rollout autoregresivo ──────────────────────────────
    print("\nGenerando rollout...")
    generator.load_state_dict(torch.load("best_generator_wgafnogp.pt", map_location=DEVICE))

    w0    = dataset.seqs[0, :1].unsqueeze(0)   # (1, 1, H, W)
    roller = Rollout(generator, dataset.f_tensor, dataset.chi_tensor, DEVICE)
    traj   = roller.run(w0, n_steps=99)

    print(f"Trayectoria: {traj.shape}")
    print(f"Energía t=0:  {roller.metrics['energy'][0]:.4f}")
    print(f"Energía t=99: {roller.metrics['energy'][-1]:.4f}")

    np.save("rollout_wgafnogp.npy", traj.numpy())

    # ── Guardar checkpoint completo ────────────────────────
    torch.save({
        "generator_state":     generator.state_dict(),
        "discriminator_state": discriminator.state_dict(),
        "config": {
            "hidden_ch": HIDDEN, "modes1": MODES, "modes2": MODES,
            "n_layers": N_LAYERS, "seq_len": SEQ_LEN,
        },
        "norm":    {"w_mean": dataset.w_mean, "w_std": dataset.w_std},
        "history": history,
    }, "wgafnogp_final.pt")
    print("Guardado en wgafnogp_final.pt")