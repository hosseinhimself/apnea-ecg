import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import wfdb

class DatasetLoader:
    def __init__(self, config):
        self.config = config
        self.dataset_root = config["dataset"].get("root_dir", "apnea-ecg")
        self.fs = config["dataset"].get("sampling_rate", 100)
        self.segment_lengths = config["dataset"].get("segment_lengths", [60])
        self.batch_size = config["training"].get("batch_size", 32)
        
        # تغییر این بخش برای مدیریت مواردی که preprocessing وجود ندارد
        pp_cfg = config.get("preprocessing", {})
        bp_cfg = pp_cfg.get("bandpass_filter", {})
        
        self.lowcut = bp_cfg.get("lowcut", 0.5)
        self.highcut = bp_cfg.get("highcut", 40.0)
        self.order = bp_cfg.get("order", 4)
        self.normalization_method = pp_cfg.get("normalization", "z-score")

    def _find_records_path(self):
        if not os.path.isdir(self.dataset_root):
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_root}")
        
        # Check root directory
        if any(f.endswith(".dat") for f in os.listdir(self.dataset_root)):
            return self.dataset_root
            
        # Check records subdirectory
        records_dir = os.path.join(self.dataset_root, "records")
        if os.path.isdir(records_dir) and any(f.endswith(".dat") for f in os.listdir(records_dir)):
            return records_dir
            
        raise FileNotFoundError("No .dat files found in dataset directory")

    def load_raw_data(self):
        record_files = [f[:-4] for f in os.listdir(self.records_path) if f.endswith(".dat")]
        if not record_files:
            raise FileNotFoundError("No record files found")
            
        for record_id in record_files:
            try:
                record = wfdb.rdrecord(os.path.join(self.records_path, record_id))
                signal = record.p_signal[:, 0] if record.p_signal.ndim > 1 else record.p_signal
                self.raw_signals[record_id] = signal.astype(np.float32)
            except Exception as e:
                print(f"Error loading {record_id}: {str(e)}")
                continue

    def preprocess_and_segment(self):
        if not self.raw_signals:
            raise RuntimeError("No raw signals loaded")
            
        for seg_len in self.segment_lengths:
            seg_samples = seg_len * self.fs
            segments, labels, record_ids = [], [], []
            
            for record_id, signal in self.raw_signals.items():
                # Filter signal
                filtered = butter_bandpass_filter(
                    signal, self.lowcut, self.highcut, self.fs, self.order
                )
                
                # Load annotations
                try:
                    ann = wfdb.rdann(os.path.join(self.records_path, record_id), "apn")
                except:
                    continue
                
                for sample, symbol in zip(ann.sample, ann.symbol):
                    if symbol not in ('A', 'N'):
                        continue
                    
                    label = 1 if symbol == 'A' else 0
                    start = sample
                    end = start + seg_samples
                    
                    if end > len(filtered):
                        continue
                        
                    segment = filtered[start:end]
                    segment = normalize(segment)
                    
                    segments.append(segment)
                    labels.append(label)
                    record_ids.append(record_id)
            
            # Convert to tensors
            segments_tensor = torch.tensor(np.array(segments), dtype=torch.float32).unsqueeze(-1)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            
            self.segment_data[seg_len] = {
                "segments": segments_tensor,
                "labels": labels_tensor,
                "record_ids": record_ids
            }
            
        return self.segment_data[self.segment_lengths[0]]["segments"], \
               self.segment_data[self.segment_lengths[0]]["labels"]

    def get_dataloaders(self, segment_length=None, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        seg_len = segment_length or self.segment_lengths[0]
        data = self.segment_data.get(seg_len)
        if not data:
            raise ValueError(f"No data for segment length {seg_len}")
            
        # Split by record IDs
        unique_records = list(set(data["record_ids"]))
        random.shuffle(unique_records)
        
        n_train = int(len(unique_records) * train_ratio)
        n_val = int(len(unique_records) * val_ratio)
        
        train_records = set(unique_records[:n_train])
        val_records = set(unique_records[n_train:n_train+n_val])
        test_records = set(unique_records[n_train+n_val:])
        
        # Create indices for each split
        train_indices = [i for i, rid in enumerate(data["record_ids"]) if rid in train_records]
        val_indices = [i for i, rid in enumerate(data["record_ids"]) if rid in val_records]
        test_indices = [i for i, rid in enumerate(data["record_ids"]) if rid in test_records]
        
        # Create datasets
        dataset = TensorDataset(data["segments"], data["labels"])
        dataset.record_ids = data["record_ids"]  # Attach record IDs
        
        train_loader = DataLoader(
            Subset(dataset, train_indices),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            Subset(dataset, val_indices),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        test_loader = DataLoader(
            Subset(dataset, test_indices),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader, test_loader