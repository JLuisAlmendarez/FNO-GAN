import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from lib.Common import BaseTrainer

class FNOPhysicsTrainer(BaseTrainer):
    def __init__(self, generator, ns_residuo, device, lr=1e-4, log_dir="logs_physics", resume=False):
        super().__init__(log_dir=log_dir, resume=resume, best_metric_name="val_loss")
        self.G = generator.to(device)
        self.ns_residuo = ns_residuo.to(device)
        self.device = device
        self.opt = torch.optim.Adam(self.G.parameters(), lr=lr)

    def _init_history_keys(self):
        self.history = {"train_loss": [], "val_loss": []}

    def _load_checkpoint_state(self, checkpoint):
        self.G.load_state_dict(checkpoint["model_state"])
        self.opt.load_state_dict(checkpoint["optimizer_state"])

    def _save_checkpoint_state(self, checkpoint, epoch, is_best):
        checkpoint["model_state"] = self.G.state_dict()
        checkpoint["optimizer_state"] = self.opt.state_dict()

    def _get_best_model_state(self):
        return self.G.state_dict()

    def _generate_trajectory(self, w0, n_steps):
        B, C, H, W = w0.shape
        traj = [w0.squeeze(1)]
        w_cur = w0
        for _ in range(n_steps):
            w_next = self.G(w_cur)
            traj.append(w_next.squeeze(1))
            w_cur = w_next
        return torch.stack(traj, dim=1)

    def train_epoch(self, loader):
        self.G.train()
        total_loss = 0.0
        n = 0
        pbar = tqdm(loader, desc="Training Physics")
        for seq_in, _, _ in pbar:
            seq_in = seq_in.to(self.device)
            w0 = seq_in[:, 0]
            traj_fake = self._generate_trajectory(w0, seq_in.shape[1])
            residuo = self.ns_residuo.residuo_espacial(traj_fake)
            loss = (residuo ** 2).mean()

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)
            self.opt.step()

            total_loss += loss.item()
            n += 1
            pbar.set_postfix({"loss": f"{loss.item():.5f}"})
        return total_loss / n

    @torch.no_grad()
    def validate(self, loader):
        self.G.eval()
        total_loss = 0.0
        n = 0
        for seq_in, _, _ in loader:
            seq_in = seq_in.to(self.device)
            w0 = seq_in[:, 0]
            traj_fake = self._generate_trajectory(w0, seq_in.shape[1])
            residuo = self.ns_residuo.residuo_espacial(traj_fake)
            loss = (residuo ** 2).mean()
            total_loss += loss.item()
            n += 1
        self.G.train()
        return total_loss / n

    def fit(self, train_loader, val_loader, epochs):
        for epoch in range(self.epoch_start, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            self.log_epoch(epoch, metrics)

        self.logger.info(f"Entrenamiento finalizado. Mejor val_loss: {self.best_val:.6f}")
        return self.history