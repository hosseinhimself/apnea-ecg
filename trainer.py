## trainer.py

import warnings
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Trainer:
    """
    این کلاس فرآیند آموزش و اعتبارسنجی مدل را برای بازتولید دقیق مقاله مدیریت می‌کند.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
    ) -> None:
        """
        مقادیر اولیه ترینر را بر اساس مدل، دیتا لودرها و فایل کانفیگ تنظیم می‌کند.
        """
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader
        self.config: Dict = config

        # تنظیمات دستگاه (GPU یا CPU)
        self.device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)

        # خواندن پارامترهای آموزش دقیقاً از فایل کانفیگ
        training_cfg: dict = self.config.get("training", {})
        lr: float = float(training_cfg.get("learning_rate", 0.001))
        beta1: float = float(training_cfg.get("betas", {}).get("beta1", 0.9))
        beta2: float = float(training_cfg.get("betas", {}).get("beta2", 0.999))
        eps: float = float(training_cfg.get("epsilon", 1e-8))
        self.epochs: int = int(training_cfg.get("epochs", 70))

        # استفاده از اپتیمایزر Adam با پارامترهای مقاله
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps
        )
        
        # استفاده از تابع Loss استاندارد CrossEntropyLoss بدون وزن‌دهی
        self.criterion = nn.CrossEntropyLoss()

        # متغیرها برای ذخیره بهترین مدل و توقف زودهنگام (Early Stopping)
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = -1
        self.early_stopping_patience: int = 10  # می‌توان این مقدار را نیز در کانفیگ تعریف کرد
        self.epochs_no_improve: int = 0

    def train(self) -> None:
        """
        حلقه اصلی آموزش و اعتبارسنجی را برای تعداد اپاک‌های مشخص شده اجرا می‌کند.
        """
        print("[Training] Starting training process based on the paper's methodology...")

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_train_loss = 0.0
            
            # --- حلقه آموزش ---
            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)

                if torch.isnan(loss):
                    warnings.warn(f"NaN loss detected at epoch {epoch}. Skipping step.")
                    continue

                loss.backward()
                # Gradient Clipping برای جلوگیری از انفجار گرادیان‌ها
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(self.train_loader)

            # --- حلقه اعتبارسنجی ---
            avg_val_loss, val_accuracy = self.validate()

            print(
                f"Epoch [{epoch:02d}/{self.epochs}] "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy*100:.2f}%"
            )

            # --- منطق ذخیره بهترین مدل و توقف زودهنگام ---
            if avg_val_loss < self.best_val_loss:
                print(f"INFO: Validation loss decreased ({self.best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model to 'best_model.pth'")
                self.best_val_loss = avg_val_loss
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                # ذخیره بهترین مدل
                torch.save(self.model.state_dict(), "best_model.pth")
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {self.early_stopping_patience} epochs with no improvement.")
                break
        
        print(f"\nTraining complete. Best model saved from epoch {self.best_epoch} with validation loss: {self.best_val_loss:.4f}")

    def validate(self) -> tuple[float, float]:
        """
        یک دور اعتبارسنجی کامل را روی مجموعه داده اعتبارسنجی انجام می‌دهد.
        میانگین Loss و دقت را برمی‌گرداند.
        """
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
        
        avg_loss = val_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        accuracy = correct_val / total_val if total_val > 0 else 0.0
        
        return avg_loss, accuracy