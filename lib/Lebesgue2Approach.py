import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from lib.Common import BaseTrainer

class FNOSupervisedTrainer(BaseTrainer):
    def __init__(self, generator, device, lr=1e-4, log_dir="logs_mse", resume=False):
        super().__init__(log_dir=log_dir, resume=resume, best_metric_name="val_loss")
        self.G = generator.to(device)
        self.device = device
        self.opt = torch.optim.Adam(self.G.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

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

    def train_epoch(self, loader):
        self.G.train()
        total_loss = 0.0
        n = 0
        pbar = tqdm(loader, desc="Training MSE")
        for seq_in, seq_out, _ in pbar:
            seq_in = seq_in.to(self.device)
            seq_out = seq_out.to(self.device)
            B, T, C, H, W = seq_in.shape

            loss = 0.0
            for t in range(T):
                w_t = seq_in[:, t]
                w_next_pred = self.G(w_t)
                target = seq_out[:, t]
                loss += self.criterion(w_next_pred, target)
            loss /= T

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
        for seq_in, seq_out, _ in loader:
            seq_in = seq_in.to(self.device)
            seq_out = seq_out.to(self.device)
            B, T, C, H, W = seq_in.shape
            loss = 0.0
            for t in range(T):
                w_t = seq_in[:, t]
                w_next_pred = self.G(w_t)
                target = seq_out[:, t]
                loss += self.criterion(w_next_pred, target)
            loss /= T
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