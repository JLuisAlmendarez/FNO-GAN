import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

from lib.Common import (
    FNOBlock,
    BaseTrainer,
    energy_spectrum,
    enstrophy_spectrum,
    palinstrophy_spectrum,
    transfer_spectrum,
    spectral_correlation,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# DISCRIMINADORES
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
# TRAINER WGAN-GP CON DOBLE DISCRIMINADOR (hereda de BaseTrainer)
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
        use_scheduler      = True,
        scheduler_patience = 5,
        scheduler_factor   = 0.5,
        log_dir            = "logs_gan",
        vis_freq           = 5,
        resume             = False,
    ):
        # Inicializar la infraestructura base
        super().__init__(log_dir=log_dir, resume=resume, best_metric_name="val_mse")

        # Modelos
        self.G          = generator.to(device)
        self.D_stat     = d_stat.to(device)
        self.D_phys     = d_phys.to(device)
        self.ns_residuo = ns_residuo.to(device)
        self.device     = device
        self.n_critic   = n_critic
        self.vis_freq   = vis_freq

        self.gp = GradientPenalty(lambda_gp=lambda_gp).to(device)

        # Optimizadores (se crean después de super() para que al cargar checkpoint se restaure bien)
        self.opt_G      = torch.optim.Adam(self.G.parameters(),      lr=lr_G, betas=(0.0, 0.9))
        self.opt_D_stat = torch.optim.Adam(self.D_stat.parameters(), lr=lr_D, betas=(0.0, 0.9))
        self.opt_D_phys = torch.optim.Adam(self.D_phys.parameters(), lr=lr_D, betas=(0.0, 0.9))

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

    # ── Métodos requeridos por BaseTrainer ─────────────────
    def _init_history_keys(self):
        self.history = {
            "loss_D_stat"  : [], "loss_D_phys"  : [], "loss_G": [],
            "w_dist_stat"  : [], "w_dist_phys"  : [],
            "val_mse"      : [], "val_ns_residuo": [],
            "val_energy"   : [], "val_enstrophy" : [],
            "val_spectral_correlation": [],
            "lr_G": [], "lr_D_stat": [], "lr_D_phys": [],
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
        checkpoint["G"]         = self.G.state_dict()
        checkpoint["D_stat"]    = self.D_stat.state_dict()
        checkpoint["D_phys"]    = self.D_phys.state_dict()
        checkpoint["opt_G"]     = self.opt_G.state_dict()
        checkpoint["opt_D_stat"] = self.opt_D_stat.state_dict()
        checkpoint["opt_D_phys"] = self.opt_D_phys.state_dict()
        if self.use_scheduler:
            checkpoint["sched_G"]      = self.sched_G.state_dict()
            checkpoint["sched_D_stat"] = self.sched_D_stat.state_dict()
            checkpoint["sched_D_phys"] = self.sched_D_phys.state_dict()

    def _get_best_model_state(self):
        return self.G.state_dict()

    # ── Generación de secuencia ────────────────────────────
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

            R_real = self.ns_residuo.residuo_espacial(real_traj)
            R_fake = self.ns_residuo.residuo_espacial(fake_traj)

            # D_stat
            score_real_stat = self.D_stat(real_traj)
            score_fake_stat = self.D_stat(fake_traj)
            gp_stat         = self.gp(self.D_stat, real_traj, fake_traj)
            loss_D_stat     = score_fake_stat.mean() - score_real_stat.mean() + gp_stat

            self.opt_D_stat.zero_grad()
            loss_D_stat.backward()
            self.opt_D_stat.step()

            # D_phys
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
        )

    # ── Espectros ──────────────────────────────────────────
    def _transfer_spectrum(self, w):
        return transfer_spectrum(
            w,
            self.ns_residuo.KX,
            self.ns_residuo.KY,
            self.ns_residuo.K2,
            self.ns_residuo.K2_inv,
        )

    # ── Visualizaciones (se mantienen igual, solo cambia la gestión de history) ──
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

        plt.suptitle("Historial de entrenamiento — WGAN-GP Dual Discriminator", fontsize=13)
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
        k_r, E_r = energy_spectrum(real_traj[:, -1].float())
        k_f, E_f = energy_spectrum(fake_traj[:, -1].cpu())
        self._plot_spectrum_generic(epoch, k_r, E_r, k_f, E_f,
            "$E(k)$", "Espectro de energía cinética",
            f"spectrum_epoch{epoch:04d}.png", ref_slope=-3, ref_label=r"$k^{-3}$")

    def plot_enstrophy_spectrum(self, epoch):
        real_traj, fake_traj = self._get_fake_for_plot()
        k_r, Z_r = enstrophy_spectrum(real_traj[:, -1].float())
        k_f, Z_f = enstrophy_spectrum(fake_traj[:, -1].cpu())
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
        k_r, P_r = palinstrophy_spectrum(real_traj[:, -1].float())
        k_f, P_f = palinstrophy_spectrum(fake_traj[:, -1].cpu())
        self._plot_spectrum_generic(epoch, k_r, P_r, k_f, P_f,
            "$P(k)$", "Espectro de palinstrofía",
            f"palinstrophy_spectrum_epoch{epoch:04d}.png", ref_slope=1, ref_label=r"$k^{+1}$")

    def plot_spectral_correlation(self, epoch):
        real_traj, fake_traj = self._get_fake_for_plot()
        k_bins, C_k = spectral_correlation(
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

    # ── Bucle de entrenamiento ─────────────────────────────
    def fit(self, train_loader, val_loader, n_epochs, log_every=5):
        self._critic_iter = iter(train_loader)
        if self.val_example is None:
            self.val_example = next(iter(val_loader))

        for epoch in range(self.epoch_start, n_epochs + 1):
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

            if self.use_scheduler:
                self.sched_G.step(val_mse)
                self.sched_D_stat.step(float(np.mean(ep["wd_stat"])))
                self.sched_D_phys.step(float(np.mean(ep["wd_phys"])))

            # Métricas de la época
            metrics = {
                "loss_D_stat": float(np.mean(ep["ld_stat"])),
                "loss_D_phys": float(np.mean(ep["ld_phys"])),
                "loss_G": float(np.mean(ep["lg"])),
                "w_dist_stat": float(np.mean(ep["wd_stat"])),
                "w_dist_phys": float(np.mean(ep["wd_phys"])),
                "val_mse": val_mse,
                "val_ns_residuo": val_ns,
                "val_energy": val_energy,
                "val_enstrophy": val_enstrophy,
                "val_spectral_correlation": val_corr,
                "lr_G": self.opt_G.param_groups[0]["lr"],
                "lr_D_stat": self.opt_D_stat.param_groups[0]["lr"],
                "lr_D_phys": self.opt_D_phys.param_groups[0]["lr"],
            }

            # Registrar, guardar checkpoint e historial
            self.log_epoch(epoch, metrics)

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

            # El log resumen ya se hace dentro de log_epoch, podemos omitir el bloque de log_every

        self.plot_losses()
        self.logger.info(f"Entrenamiento finalizado. Mejor val_mse: {self.best_val:.6f}")
        return self.history