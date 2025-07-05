import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Optimizer
        training_cfg = config.get("training", {})
        self.optimizer = AdamW(
            model.parameters(),
            lr=training_cfg.get("learning_rate", 0.0005),
            weight_decay=training_cfg.get("weight_decay", 0.001)
        )
        
        # Scheduler
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        # Loss function with class weights
        class_weights = torch.tensor([1.0, 1.5], device=self.device)  # Higher weight for apnea class
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        self.best_val_loss = float('inf')
        self.epochs = training_cfg.get("epochs", 50)

    def train(self):
        for epoch in range(1, self.epochs + 1):
            # Training
            self.model.train()
            train_loss, correct, total = 0, 0, 0
            
            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            train_loss /= len(self.train_loader.dataset)
            train_acc = correct / total
            
            # Validation
            val_loss, val_acc = self.validate()
            
            # Step scheduler
            self.scheduler.step()
            
            # Print metrics
            print(f"Epoch {epoch}/{self.epochs}: "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pth")
                print("Saved best model!")

    def validate(self):
        self.model.eval()
        val_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        val_loss /= len(self.val_loader.dataset)
        val_acc = correct / total
        
        return val_loss, val_acc