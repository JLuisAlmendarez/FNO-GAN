import torch
import torch.nn as nn
from tqdm import tqdm
from lib.Common import BaseTrainer


class FNOPhysicsTrainer(BaseTrainer):
    def __init__(self, generator, ns_residuo, device, lr=1e-4, lambda_energy=0.1,
                 log_dir="logs_physics", resume=False, patience=10):
        self.G             = generator.to(device)
        self.ns_residuo    = ns_residuo.to(device)
        self.device        = device
        self.lambda_energy = lambda_energy
        self.opt           = torch.optim.Adam(self.G.parameters(), lr=lr)

        super().__init__(log_dir=log_dir, resume=resume,
                         best_metric_name="val_loss",
                         patience=patience)

    def _init_history_keys(self):
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_loss_energy": [], "val_loss_energy": [],
        }

    def _load_checkpoint_state(self, checkpoint):
        self.G.load_state_dict(checkpoint["model_state"])
        self.opt.load_state_dict(checkpoint["optimizer_state"])

    def _save_checkpoint_state(self, checkpoint, epoch, is_best):
        checkpoint["model_state"]     = self.G.state_dict()
        checkpoint["optimizer_state"] = self.opt.state_dict()

    def _get_best_model_state(self):
        return self.G.state_dict()

    def _generate_trajectory(self, w0, n_steps):
        traj  = [w0.squeeze(1)]
        w_cur = w0
        for _ in range(n_steps):
            w_next = self.G(w_cur)
            traj.append(w_next.squeeze(1))
            w_cur  = w_next
        return torch.stack(traj, dim=1)

    def train_epoch(self, loader):
        self.G.train()
        total_loss        = 0.0
        total_loss_energy = 0.0
        n    = 0
        pbar = tqdm(loader, desc="Training Physics")
        for seq_in, _, _ in pbar:
            seq_in = seq_in.to(self.device)
            w0     = seq_in[:, 0]

            traj_fake      = self._generate_trajectory(w0, seq_in.shape[1])
            R, nus, fs     = self.ns_residuo.residuo_espacial(traj_fake)
            loss_ns        = (R ** 2).mean()
            residuo_energia = self.ns_residuo.energy_conservation_residual(traj_fake, nus, fs)
            loss_energy    = (residuo_energia ** 2).mean()
            loss           = loss_ns + self.lambda_energy * loss_energy

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)
            self.opt.step()

            total_loss        += loss.item()
            total_loss_energy += loss_energy.item()
            n += 1
            pbar.set_postfix({"loss": f"{loss.item():.5f}", "E": f"{loss_energy.item():.5f}"})
        return total_loss / n, total_loss_energy / n

    @torch.no_grad()
    def validate(self, loader):
        self.G.eval()
        total_loss        = 0.0
        total_loss_energy = 0.0
        n = 0
        for seq_in, _, _ in loader:
            seq_in = seq_in.to(self.device)
            w0     = seq_in[:, 0]
            traj_fake       = self._generate_trajectory(w0, seq_in.shape[1])
            R, nus, fs      = self.ns_residuo.residuo_espacial(traj_fake)
            loss_ns         = (R ** 2).mean()
            residuo_energia = self.ns_residuo.energy_conservation_residual(traj_fake, nus, fs)
            loss_energy     = (residuo_energia ** 2).mean()
            total_loss        += loss_ns.item()
            total_loss_energy += loss_energy.item()
            n += 1
        self.G.train()
        return total_loss / n, total_loss_energy / n

    def fit(self, train_loader, val_loader, epochs=100):
        for epoch in range(self.epoch_start, epochs + 1):
            train_loss, train_le = self.train_epoch(train_loader)
            val_loss,   val_le   = self.validate(val_loader)

            metrics = {
                "train_loss":        train_loss,
                "val_loss":          val_loss,
                "train_loss_energy": train_le,
                "val_loss_energy":   val_le,
            }
            self.log_epoch(epoch, metrics)

            if self.should_stop:
                self.logger.info(f"Early stopping en epoch {epoch} — "
                                 f"sin mejora por {self.patience} épocas.")
                break

        self.logger.info(f"Entrenamiento finalizado. Mejor val_loss: {self.best_val:.6f}")
        return self.history