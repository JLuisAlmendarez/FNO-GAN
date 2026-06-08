import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import logging

from lib.Common import (
    FNOBlock,
    BaseTrainer,
    spectral_correlation,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# DISCRIMINADORES
# ═══════════════════════════════════════════════════════════
class FNODiscriminatorStat(nn.Module):
    # ... (sin cambios)
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
        B, T, H, W = traj.shape
        x = traj.permute(0, 2, 3, 1).contiguous().reshape(B * H * W, T, 1)
        x = self.temporal_mix(x).squeeze(-1)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=(-2, -1))
        return self.head(x)


class FNODiscriminatorPhys(nn.Module):
    # ... (sin cambios)
    def __init__(self, seq_len, hidden_ch=64, modes1=12, modes2=12, n_layers=4):
        super().__init__()
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
        B, T, H, W = residuo.shape
        x = residuo.permute(0, 2, 3, 1).contiguous().reshape(B * H * W, T, 1)
        x = self.temporal_mix(x).squeeze(-1)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=(-2, -1))
        return self.head(x)


# ═══════════════════════════════════════════════════════════
# GRADIENT PENALTY
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
# TRAINER WGAN-GP (sin AMP, con early stopping)
# ═══════════════════════════════════════════════════════════
class WGAFNOGPTrainer(BaseTrainer):
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
        lambda_energy      = 0.1,
        use_scheduler      = True,
        scheduler_patience = 5,
        scheduler_factor   = 0.5,
        log_dir            = "logs_gan",
        resume             = False,
        patience           = 15,
    ):
        # --- Modelos ---
        self.G          = generator.to(device)
        self.D_stat     = d_stat.to(device)
        self.D_phys     = d_phys.to(device)
        self.ns_residuo = ns_residuo.to(device)
        self.device     = device

        # --- Optimizadores ---
        self.opt_G      = torch.optim.Adam(self.G.parameters(),      lr=lr_G, betas=(0.0, 0.9))
        self.opt_D_stat = torch.optim.Adam(self.D_stat.parameters(), lr=lr_D, betas=(0.0, 0.9))
        self.opt_D_phys = torch.optim.Adam(self.D_phys.parameters(), lr=lr_D, betas=(0.0, 0.9))

        # --- Schedulers ---
        self.use_scheduler = use_scheduler
        if use_scheduler:
            self.sched_G      = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.opt_G,      mode="min", factor=scheduler_factor, patience=scheduler_patience)
            self.sched_D_stat = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.opt_D_stat, mode="max", factor=scheduler_factor, patience=scheduler_patience)
            self.sched_D_phys = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.opt_D_phys, mode="max", factor=scheduler_factor, patience=scheduler_patience)

        # --- BaseTrainer ---
        super().__init__(log_dir=log_dir, resume=resume,
                         best_metric_name="val_mse",
                         patience=patience)

        # --- Resto de atributos ---
        self.n_critic      = n_critic
        self.lambda_energy = lambda_energy
        self.gp            = GradientPenalty(lambda_gp=lambda_gp).to(device)
        self._critic_iter  = None
        self.val_example   = None

    # ── BaseTrainer hooks ──────────────────────────────────
    def _init_history_keys(self):
        self.history = {
            "loss_D_stat": [], "loss_D_phys": [], "loss_G": [],
            "w_dist_stat": [], "w_dist_phys": [],
            "val_mse": [], "val_ns_residuo": [],
            "val_energy": [], "val_enstrophy": [],
            "val_spectral_correlation": [],
            "lr_G": [], "lr_D_stat": [], "lr_D_phys": [],
            "loss_energy": [], "val_loss_energy": [],
        }

    def _load_checkpoint_state(self, checkpoint):
        self.G.load_state_dict(checkpoint["G"])
        self.D_stat.load_state_dict(checkpoint["D_stat"])
        self.D_phys.load_state_dict(checkpoint["D_phys"])
        self.opt_G.load_state_dict(checkpoint["opt_G"])
        self.opt_D_stat.load_state_dict(checkpoint["opt_D_stat"])
        self.opt_D_phys.load_state_dict(checkpoint["opt_D_phys"])
        if self.use_scheduler and "sched_G" in checkpoint:
            self.sched_G.load_state_dict(checkpoint["sched_G"])
            self.sched_D_stat.load_state_dict(checkpoint["sched_D_stat"])
            self.sched_D_phys.load_state_dict(checkpoint["sched_D_phys"])

    def _save_checkpoint_state(self, checkpoint, epoch, is_best):
        checkpoint["G"]          = self.G.state_dict()
        checkpoint["D_stat"]     = self.D_stat.state_dict()
        checkpoint["D_phys"]     = self.D_phys.state_dict()
        checkpoint["opt_G"]      = self.opt_G.state_dict()
        checkpoint["opt_D_stat"] = self.opt_D_stat.state_dict()
        checkpoint["opt_D_phys"] = self.opt_D_phys.state_dict()
        if self.use_scheduler:
            checkpoint["sched_G"]      = self.sched_G.state_dict()
            checkpoint["sched_D_stat"] = self.sched_D_stat.state_dict()
            checkpoint["sched_D_phys"] = self.sched_D_phys.state_dict()

    def _get_best_model_state(self):
        return self.G.state_dict()

    # ── Generación ────────────────────────────────────────
    def _generate_sequence(self, seq_in, z=None):
        B, seq_len, _, H, W = seq_in.shape
        w_cur = seq_in[:, 0]
        fake  = [w_cur.squeeze(1)]
        for _ in range(seq_len):
            w_next = self.G(w_cur, z=z)
            fake.append(w_next.squeeze(1))
            w_cur  = w_next
        return torch.stack(fake, dim=1)

    # ── Paso discriminadores ───────────────────────────────
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

            R_real, _, _ = self.ns_residuo.residuo_espacial(real_traj)
            R_fake, _, _ = self.ns_residuo.residuo_espacial(fake_traj)

            # --- D_stat ---
            self.opt_D_stat.zero_grad()
            score_real_stat = self.D_stat(real_traj)
            score_fake_stat = self.D_stat(fake_traj.detach())
            gp_stat     = self.gp(self.D_stat, real_traj, fake_traj.detach())
            loss_D_stat = score_fake_stat.mean() - score_real_stat.mean() + gp_stat
            loss_D_stat.backward()
            self.opt_D_stat.step()

            # --- D_phys ---
            self.opt_D_phys.zero_grad()
            score_real_phys = self.D_phys(R_real)
            score_fake_phys = self.D_phys(R_fake.detach())
            gp_phys     = self.gp(self.D_phys, R_real, R_fake.detach())
            loss_D_phys = score_fake_phys.mean() - score_real_phys.mean() + gp_phys
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

    # ── Paso generador ────────────────────────────────────
    def _step_G(self, seq_in):
        self.opt_G.zero_grad()
        fake_traj       = self._generate_sequence(seq_in)
        R_fake, nus, fs = self.ns_residuo.residuo_espacial(fake_traj)
        loss_adv_stat   = -self.D_stat(fake_traj).mean()
        loss_adv_phys   = -self.D_phys(R_fake).mean()
        residuo_energia = self.ns_residuo.energy_conservation_residual(fake_traj, nus, fs)
        loss_energy     = (residuo_energia ** 2).mean()
        loss_G          = loss_adv_stat + loss_adv_phys + self.lambda_energy * loss_energy

        loss_G.backward()
        nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)
        self.opt_G.step()

        return loss_G.item(), loss_energy.item()

    # ── Validación ────────────────────────────────────────
    @torch.no_grad()
    def _validate(self, loader):
        self.G.eval()
        total_mse = total_energy = total_enstrophy = total_corr = total_ns = total_loss_energy = 0.0
        n = 0

        for seq_in, seq_out, _ in loader:
            seq_in  = seq_in.to(self.device)
            seq_out = seq_out.to(self.device)

            fake_traj       = self._generate_sequence(seq_in)
            seq_out_2d      = seq_out.squeeze(2)
            R_fake, nus, fs = self.ns_residuo.residuo_espacial(fake_traj)

            total_mse += F.mse_loss(fake_traj[:, 1:], seq_out_2d).item()
            total_ns  += (R_fake ** 2).mean().item()

            residuo_energia   = self.ns_residuo.energy_conservation_residual(fake_traj, nus, fs)
            total_loss_energy += (residuo_energia ** 2).mean().item()

            w_last_fake      = fake_traj[:, -1]
            w_last_real      = seq_out_2d[:, -1]
            total_energy    += 0.5 * (w_last_fake ** 2).mean().item()
            wf               = torch.fft.fft2(w_last_fake)
            total_enstrophy += 0.5 * (self.ns_residuo.K2 * wf.abs() ** 2).mean().item()

            _, C_k = spectral_correlation(w_last_real.cpu(), w_last_fake.cpu())
            total_corr += float(C_k.mean())
            n += 1

        self.G.train()
        return (
            total_mse       / n,
            total_ns        / n,
            total_energy    / n,
            total_enstrophy / n,
            total_corr      / n,
            total_loss_energy / n,
        )

    # ── Bucle principal ───────────────────────────────────
    def fit(self, train_loader, val_loader, n_epochs, log_every=5):
        self._critic_iter = iter(train_loader)
        if self.val_example is None:
            self.val_example = next(iter(val_loader))

        for epoch in range(self.epoch_start, n_epochs + 1):
            self.G.train(); self.D_stat.train(); self.D_phys.train()

            ep = {"ld_stat": [], "ld_phys": [], "lg": [], "le": [],
                  "wd_stat": [], "wd_phys": []}

            pbar = tqdm(train_loader, desc=f"Epoch {epoch:4d}/{n_epochs}", leave=False)
            for seq_in, seq_out, _ in pbar:
                seq_in = seq_in.to(self.device)

                ld_stat, ld_phys, wd_stat, wd_phys = self._step_D(train_loader)
                lg, le = self._step_G(seq_in)

                ep["ld_stat"].append(ld_stat); ep["ld_phys"].append(ld_phys)
                ep["lg"].append(lg);           ep["le"].append(le)
                ep["wd_stat"].append(wd_stat); ep["wd_phys"].append(wd_phys)

                pbar.set_postfix({
                    "Ws": f"{wd_stat:.2f}", "Wp": f"{wd_phys:.2f}",
                    "LG": f"{lg:.3f}",      "LE": f"{le:.4f}",
                })

            val_mse, val_ns, val_energy, val_enstrophy, val_corr, val_le = self._validate(val_loader)

            if self.use_scheduler:
                self.sched_G.step(val_mse)
                self.sched_D_stat.step(float(np.mean(ep["wd_stat"])))
                self.sched_D_phys.step(float(np.mean(ep["wd_phys"])))

            metrics = {
                "loss_D_stat":             float(np.mean(ep["ld_stat"])),
                "loss_D_phys":             float(np.mean(ep["ld_phys"])),
                "loss_G":                  float(np.mean(ep["lg"])),
                "loss_energy":             float(np.mean(ep["le"])),
                "w_dist_stat":             float(np.mean(ep["wd_stat"])),
                "w_dist_phys":             float(np.mean(ep["wd_phys"])),
                "val_mse":                 val_mse,
                "val_ns_residuo":          val_ns,
                "val_energy":              val_energy,
                "val_enstrophy":           val_enstrophy,
                "val_spectral_correlation": val_corr,
                "val_loss_energy":         val_le,
                "lr_G":                    self.opt_G.param_groups[0]["lr"],
                "lr_D_stat":               self.opt_D_stat.param_groups[0]["lr"],
                "lr_D_phys":               self.opt_D_phys.param_groups[0]["lr"],
            }

            self.log_epoch(epoch, metrics)

            if self.should_stop:
                self.logger.info(f"Early stopping en epoch {epoch} — "
                                 f"sin mejora en val_mse por {self.patience} épocas.")
                break

            if epoch % 10 == 0 and str(self.device) != "cpu":
                torch.cuda.empty_cache()

        self.logger.info(f"Entrenamiento finalizado. Mejor val_mse: {self.best_val:.6f}")
        return self.history