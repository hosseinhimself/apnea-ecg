## trainer.py

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """
    Trainer class to manage the training and validation procedure of the CNN-Transformer model.

    Attributes:
        model (nn.Module): The deep learning model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        config (dict): Training configuration parameters.
        device (torch.device): Device to run training on (CPU or CUDA).
        optimizer (torch.optim.Optimizer): Optimizer (Adam) for the model parameters.
        criterion (nn.Module): Loss function (CrossEntropyLoss).
        epochs (int): Number of training epochs.
        best_val_loss (float): Best validation loss observed for checkpointing.
        best_epoch (int): Epoch number with the best validation loss.
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

        Args:
            model (nn.Module): Model instance from model.py.
            train_loader (DataLoader): DataLoader for training set.
            val_loader (DataLoader): DataLoader for validation set.
            config (dict): Configuration dictionary loaded from config.yaml.
        """
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader
        self.config: Dict = config

        # Device assignment
        self.device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)

        # Extract training hyperparameters from config with defaults
        training_cfg: dict = self.config.get("training", {})
        lr: float = float(training_cfg.get("learning_rate", 0.001))
        beta1: float = float(training_cfg.get("betas", {}).get("beta1", 0.9))
        beta2: float = float(training_cfg.get("betas", {}).get("beta2", 0.999))
        eps: float = float(training_cfg.get("epsilon", 1e-8))
        self.epochs: int = int(training_cfg.get("epochs", 70))

        # Optimizer setup: Adam with parameters from config
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps
        )
        # Loss function: CrossEntropyLoss appropriate for classification (binary)
        self.criterion = nn.CrossEntropyLoss()

        # Training state tracking
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = -1

    def train(self) -> None:
        """
        Execute the training and validation loop for the configured number of epochs.
        Prints progress and performance metrics per epoch.
        Saves best model checkpoint based on validation loss.
        """
        for epoch in range(1, self.epochs + 1):
            # --------- Training Phase ---------
            self.model.train()
            epoch_train_loss = 0.0
            correct_train = 0
            total_train = 0

            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)  # logits shape (batch_size, 2)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_train_loss += loss.item() * batch_x.size(0)

                # Compute training accuracy
                preds = torch.argmax(outputs, dim=1)
                correct_train += (preds == batch_y).sum().item()
                total_train += batch_x.size(0)

            epoch_train_loss /= total_train if total_train > 0 else 1
            train_accuracy = correct_train / total_train if total_train > 0 else 0.0

            # --------- Validation Phase ---------
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

            # Print epoch metrics
            print(
                f"Epoch [{epoch}/{self.epochs}] "
                f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {train_accuracy*100:.2f}% | "
                f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_accuracy*100:.2f}%"
            )

            # Save checkpoint if validation loss improved
            if epoch_val_loss < self.best_val_loss:
                self.best_val_loss = epoch_val_loss
                self.best_epoch = epoch
                checkpoint_path = f"best_model_epoch_{epoch}.pth"
                self.save_checkpoint(checkpoint_path)
                print(f"Saved best checkpoint to {checkpoint_path}")

        print(f"Training complete. Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")

    def save_checkpoint(self, path: str) -> None:
        """
        Save model checkpoint including optimizer state and training metadata.

        Args:
            path (str): File path (including filename) for saving the checkpoint.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
        }
        torch.save(checkpoint, path)
