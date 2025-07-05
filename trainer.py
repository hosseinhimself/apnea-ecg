## trainer.py

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """
    Trainer class to manage the training and validation procedure of the CNN-Transformer model.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
    ) -> None:
        """
        Initialize Trainer instance with model, dataloaders, and config.
        """
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader
        self.config: Dict = config
        torch.backends.cudnn.benchmark = True

        self.device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)

        training_cfg: dict = self.config.get("training", {})
        lr: float = float(training_cfg.get("learning_rate", 0.001))
        beta1: float = float(training_cfg.get("betas", {}).get("beta1", 0.9))
        beta2: float = float(training_cfg.get("betas", {}).get("beta2", 0.999))
        eps: float = float(training_cfg.get("epsilon", 1e-8))
        self.epochs: int = int(training_cfg.get("epochs", 70))

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps
        )
        self.criterion = nn.CrossEntropyLoss()

        self.best_val_loss: float = float("inf")
        self.best_epoch: int = -1

    def train(self) -> None:
        """
        Execute the training and validation loop.
        """
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_train_loss = 0.0
            correct_train = 0
            total_train = 0

            for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)

                if torch.isnan(loss):
                    warnings.warn(f"NaN loss detected at epoch {epoch}, batch {batch_idx}. Skipping step.")
                    continue

                loss.backward()
                
                # <--- FIX: Add gradient clipping for training stability --->
                # This helps prevent exploding gradients, which can be an issue in complex models.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()

                epoch_train_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct_train += (preds == batch_y).sum().item()
                total_train += batch_x.size(0)

            epoch_train_loss /= total_train if total_train > 0 else 1
            train_accuracy = correct_train / total_train if total_train > 0 else 0.0

            # Validation Phase
            self.model.eval()
            epoch_val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for batch_x, batch_y in self.val_loader:
                    batch_x = batch_x.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    epoch_val_loss += loss.item() * batch_x.size(0)
                    preds = torch.argmax(outputs, dim=1)
                    correct_val += (preds == batch_y).sum().item()
                    total_val += batch_x.size(0)

            epoch_val_loss /= total_val if total_val > 0 else 1
            val_accuracy = correct_val / total_val if total_val > 0 else 0.0

            print(
                f"Epoch [{epoch}/{self.epochs}] "
                f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {train_accuracy*100:.2f}% | "
                f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_accuracy*100:.2f}%"
            )

            if epoch_val_loss < self.best_val_loss:
                self.best_val_loss = epoch_val_loss
                self.best_epoch = epoch
                # No need to save checkpoint in this context, but the logic is fine.
                # self.save_checkpoint(f"best_model_epoch_{epoch}.pth")
                print(f"INFO: New best validation loss found: {self.best_val_loss:.4f}")

        print(f"Training complete. Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")

    def save_checkpoint(self, path: str) -> None:
        """Saves model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
        }
        torch.save(checkpoint, path)
        