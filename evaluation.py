import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, recall_score, 
    precision_score, f1_score, roc_auc_score
)
from collections import defaultdict

class Evaluation:
    def __init__(self, model, test_loader, threshold=0.5):
        self.model = model
        self.test_loader = test_loader
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
    def evaluate(self):
        all_true, all_pred, all_probs = [], [], []
        
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                logits = self.model(batch_x)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = (probs >= self.threshold).long()
                
                all_true.extend(batch_y.cpu().numpy())
                all_pred.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        metrics = {
            "accuracy": accuracy_score(all_true, all_pred),
            "sensitivity": recall_score(all_true, all_pred, pos_label=1),
            "specificity": recall_score(all_true, all_pred, pos_label=0),
            "precision": precision_score(all_true, all_pred),
            "f1_score": f1_score(all_true, all_pred),
            "auc": roc_auc_score(all_true, all_probs)
        }
        
        return metrics
    
    def aggregate_recording_results(self):
        rec_true, rec_pred = defaultdict(list), defaultdict(list)
        
        with torch.no_grad():
            for (batch_x, batch_y), record_ids in zip(self.test_loader, self.test_loader.dataset.record_ids):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                logits = self.model(batch_x)
                preds = torch.argmax(logits, dim=1)
                
                for rid, true, pred in zip(record_ids, batch_y.cpu(), preds.cpu()):
                    rec_true[rid].append(true.item())
                    rec_pred[rid].append(pred.item())
        
        # Recording-wise metrics
        recording_acc = []
        ahi_errors = []
        
        for rid in rec_true:
            true_labels = np.array(rec_true[rid])
            pred_labels = np.array(rec_pred[rid])
            
            # Majority voting
            true_rec_label = 1 if np.mean(true_labels) > 0.5 else 0
            pred_rec_label = 1 if np.mean(pred_labels) > 0.5 else 0
            recording_acc.append(true_rec_label == pred_rec_label)
            
            # AHI calculation (events per hour)
            seg_len = 60  # seconds
            duration_h = (len(true_labels) * seg_len) / 3600
            true_ahi = np.sum(true_labels) / duration_h
            pred_ahi = np.sum(pred_labels) / duration_h
            ahi_errors.append(abs(true_ahi - pred_ahi))
        
        return {
            "recording_classification_accuracy": np.mean(recording_acc),
            "ahi_mae": np.mean(ahi_errors)
        }