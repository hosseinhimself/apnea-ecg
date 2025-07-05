## trainer.py

import warnings
from typing import Dict

import numpy as np # <--- FIX: Import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
# <--- FIX: Import Subset for type hinting
from torch.utils.data import Subset


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
        weight_decay: float = float(training_cfg.get("weight_decay", 1e-4))
        self.epochs: int = int(training_cfg.get("epochs", 70))
        self.early_stopping_patience: int = int(training_cfg.get("early_stopping_patience", 10))
        self.epochs_no_improve: int = 0

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        
        # <--- FIX: Calculate class weights to handle imbalance --->
        class_weights = self.calculate_class_weights(train_loader.dataset)
        class_weights = class_weights.to(self.device)
        print(f"Using calculated class weights: {class_weights.cpu().numpy()}")
        
        # <--- FIX: Pass the calculated weights to the loss function --->
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=5, verbose=True)

        self.best_val_loss: float = float("inf")
        self.best_epoch: int = -1

    # <--- FIX: New method to calculate weights --->
    def calculate_class_weights(self, train_dataset: Subset) -> torch.Tensor:
        """Calculates class weights inversely proportional to their frequency."""
        # We need to access the labels of the underlying dataset
        # In a Subset, the labels are in `dataset.dataset.tensors[1]` at `dataset.indices`
        full_dataset_labels = train_dataset.dataset.tensors[1]
        subset_labels = full_dataset_labels[train_dataset.indices]
        
        class_counts = np.bincount(subset_labels.cpu().numpy())
        
        # Handle case where a class might not be in the subset (unlikely but possible)
        if len(class_counts) < 2:
            return torch.ones(2)

        weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
        # Normalize weights
        weights = weights / weights.sum()
        
        return weights

    # The rest of the file remains the same...
    def train(self) -> None:
        """
        Execute the training and validation loop with early stopping.
        """
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_train_loss = 0.0
            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)

                if torch.isnan(loss):
                    warnings.warn("NaN loss detected. Skipping step.")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(self.train_loader)

            avg_val_loss, val_accuracy = self.validate()
            self.scheduler.step(avg_val_loss)

            print(
                f"Epoch [{epoch}/{self.epochs}] "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy*100:.2f}%"
            )

            if avg_val_loss < self.best_val_loss:
                print(f"Validation loss decreased ({self.best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model...")
                self.best_val_loss = avg_val_loss
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                self.save_checkpoint("best_model.pth")
            else:
                self.epochs_no_improve += 1
                # This print can be removed to reduce log clutter
                # print(f"Validation loss did not improve for {self.epochs_no_improve} epoch(s).")

            if self.epochs_no_improve >= self.early_stopping_patience:
                print(f"Early stopping triggered after {self.early_stopping_patience} epochs with no improvement.")
                break

        print(f"\nTraining complete. Best model from epoch {self.best_epoch} with validation loss: {self.best_val_loss:.4f}")

    def validate(self) -> tuple[float, float]:
        """Performs a validation pass and returns average loss and accuracy."""
        self.model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == batch_y).sum().item()
                total_val += batch_x.size(0)
        
        avg_loss = val_loss / len(self.val_loader)
        accuracy = correct_val / total_val if total_val > 0 else 0.0
        return avg_loss, accuracy

    def save_checkpoint(self, path: str) -> None:
        """Saves model checkpoint."""
        torch.save(self.model.state_dict(), path)