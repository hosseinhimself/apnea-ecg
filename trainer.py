import warnings
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Trainer:
    """
    This class manages the training and validation process of the model for accurate paper replication.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
    ) -> None:
        """
        Initializes the trainer with the model, data loaders, and configuration file.

        Args:
            model (nn.Module): The neural network model to be trained.
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            config (Dict): A dictionary containing training configuration parameters.
        """
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader
        self.config: Dict = config

        # Configure the device for training (GPU if available, otherwise CPU).
        self.device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device) # Move the model to the selected device.

        # Read specific training parameters directly from the configuration file.
        training_cfg: dict = self.config.get("training", {})
        lr: float = float(training_cfg.get("learning_rate", 0.001)) # Learning rate for the optimizer.
        beta1: float = float(training_cfg.get("betas", {}).get("beta1", 0.9)) # Beta1 parameter for Adam optimizer.
        beta2: float = float(training_cfg.get("betas", {}).get("beta2", 0.999)) # Beta2 parameter for Adam optimizer.
        eps: float = float(training_cfg.get("epsilon", 1e-8)) # Epsilon parameter for Adam optimizer.
        self.epochs: int = int(training_cfg.get("epochs", 70)) # Total number of training epochs.

        # Initialize the Adam optimizer with the model's parameters and specified hyperparameters from the paper.
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps
        )
        
        # Use CrossEntropyLoss as the loss function, as specified in the paper.
        # It's suitable for classification tasks and handles softmax internally for logits.
        self.criterion = nn.CrossEntropyLoss()

        # Variables for tracking the best model during training and implementing early stopping.
        self.best_val_loss: float = float("inf") # Initialize with infinity to ensure first validation loss is lower.
        self.best_epoch: int = -1 # Stores the epoch number when the best model was saved.
        self.early_stopping_patience: int = 10  # Number of epochs to wait for improvement before stopping.
        self.epochs_no_improve: int = 0 # Counter for consecutive epochs without validation loss improvement.

    def train(self) -> None:
        """
        Executes the main training and validation loop for the specified number of epochs.
        It manages the forward and backward passes, optimization, and model saving.
        """
        print("[Training] Starting training process based on the paper's methodology...")

        for epoch in range(1, self.epochs + 1):
            self.model.train() # Set the model to training mode (enables dropout, BatchNorm updates).
            epoch_train_loss = 0.0
            
            # --- Training Loop for current epoch ---
            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device, non_blocking=True) # Move input data to the specified device.
                batch_y = batch_y.to(self.device, non_blocking=True) # Move target labels to the specified device.

                self.optimizer.zero_grad() # Clear previous gradients to prevent accumulation.
                outputs = self.model(batch_x) # Perform forward pass to get model outputs (logits).
                loss = self.criterion(outputs, batch_y) # Calculate the loss between outputs and true labels.

                if torch.isnan(loss): # Check for NaN loss, which can indicate training instability.
                    warnings.warn(f"NaN loss detected at epoch {epoch}. Skipping step.")
                    continue

                loss.backward() # Perform backward pass to compute gradients.
                # Clip gradients to a maximum norm to prevent exploding gradients.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step() # Update model parameters using the optimizer.

                epoch_train_loss += loss.item() # Accumulate batch loss for epoch's average.

            avg_train_loss = epoch_train_loss / len(self.train_loader) # Calculate average training loss for the epoch.

            # --- Validation Loop ---
            avg_val_loss, val_accuracy = self.validate() # Run validation to evaluate model performance.

            # Print epoch summary.
            print(
                f"Epoch [{epoch:02d}/{self.epochs}] "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy*100:.2f}%"
            )

            # --- Best Model Saving and Early Stopping Logic ---
            if avg_val_loss < self.best_val_loss: # Check if validation loss has improved.
                print(f"INFO: Validation loss decreased ({self.best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model to 'best_model.pth'")
                self.best_val_loss = avg_val_loss # Update best validation loss.
                self.best_epoch = epoch # Record the epoch of the best model.
                self.epochs_no_improve = 0 # Reset counter for epochs without improvement.
                torch.save(self.model.state_dict(), "best_model.pth") # Save the model's state dictionary.
            else:
                self.epochs_no_improve += 1 # Increment counter if no improvement.

            if self.epochs_no_improve >= self.early_stopping_patience: # Check if early stopping criteria are met.
                print(f"\nEarly stopping triggered after {self.early_stopping_patience} epochs with no improvement.")
                break # Exit the training loop.
        
        # Final message after training completion or early stopping.
        print(f"\nTraining complete. Best model saved from epoch {self.best_epoch} with validation loss: {self.best_val_loss:.4f}")

    def validate(self) -> tuple[float, float]:
        """
        Performs a full validation pass over the validation dataset.
        Returns the average validation loss and accuracy.
        """
        self.model.eval() # Set the model to evaluation mode (disables dropout, BatchNorm statistics freezing).
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad(): # Disable gradient calculations during validation for efficiency.
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                outputs = self.model(batch_x) # Get model predictions.
                loss = self.criterion(outputs, batch_y) # Calculate loss.
                val_loss += loss.item() # Accumulate loss.
                
                preds = torch.argmax(outputs, dim=1) # Get predicted class labels (index of max logit).
                correct_val += (preds == batch_y).sum().item() # Count correctly predicted samples.
                total_val += batch_x.size(0) # Accumulate total samples.
        
        # Calculate average loss and accuracy. Handle cases where loader might be empty to avoid division by zero.
        avg_loss = val_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        accuracy = correct_val / total_val if total_val > 0 else 0.0
        
        return avg_loss, accuracy # Return calculated metrics.