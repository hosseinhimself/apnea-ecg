## dataset_loader.py

import os
import random
import warnings
from typing import Tuple, Dict, List, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

from preprocessing import butter_bandpass_filter, normalize

try:
    import wfdb
except ImportError:
    raise ImportError("The wfdb package is required. Install it via `pip install wfdb`.")

class DatasetLoader:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        ds_cfg = config.get("dataset", {})
        self.dataset_root: str = ds_cfg.get("root_dir", os.path.join(os.getcwd(), "apnea-ecg"))
        self.fs: int = ds_cfg.get("sampling_rate", 100)
        # <<< FIX: Focus on the segment length specified in config
        self.segment_length_sec: int = ds_cfg.get("segment_lengths", [180])[0]

        preprocessing_cfg = config.get("preprocessing", {})
        bandpass_cfg = preprocessing_cfg.get("bandpass_filter", {})
        self.lowcut: float = bandpass_cfg.get("lowcut", 0.5)
        self.highcut: float = bandpass_cfg.get("highcut", 48.0)
        self.filter_order: int = bandpass_cfg.get("order", 4)
        
        training_cfg = config.get("training", {})
        self.batch_size: int = training_cfg.get("batch_size", 32)
        
        self.records_path = self._find_records_path()
        self.random_seed: int = 42

        # <<< FIX: Define released and withheld sets as per PhysioNet specification
        # Released set: a01-a20, b01-b05, c01-c10
        self.released_records_names = [f'a{i:02d}' for i in range(1, 21)] + \
                                      [f'b{i:02d}' for i in range(1, 6)] + \
                                      [f'c{i:02d}' for i in range(1, 11)]
        # Withheld set: x01-x35
        self.withheld_records_names = [f'x{i:02d}' for i in range(1, 36)]

    def _find_records_path(self) -> str:
        if not os.path.isdir(self.dataset_root):
            raise FileNotFoundError(f"Dataset root directory not found: {self.dataset_root}")
        if any(f.endswith(".dat") for f in os.listdir(self.dataset_root)):
            return self.dataset_root
        records_subdir = os.path.join(self.dataset_root, "records")
        if os.path.isdir(records_subdir) and any(f.endswith(".dat") for f in os.listdir(records_subdir)):
            return records_subdir
        raise FileNotFoundError(f"Could not find any .dat record files in '{self.dataset_root}' or its 'records' subdirectory.")

    def _load_and_segment_set(self, record_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        seg_len_samples = int(self.segment_length_sec * self.fs)
        all_segments, all_labels = [], []

        for record_id in record_names:
            try:
                record_path = os.path.join(self.records_path, record_id)
                record = wfdb.rdrecord(record_path)
                signal = record.p_signal
            except Exception as e:
                warnings.warn(f"Could not read record {record_id}, skipping. Error: {e}")
                continue

            if np.isnan(signal).any():
                signal = np.nan_to_num(signal)

            if signal.ndim > 1:
                signal = signal[:, 0]

            filtered_signal = butter_bandpass_filter(signal.flatten().astype(np.float32), self.lowcut, self.highcut, self.fs, self.filter_order)
            
            try:
                annotations = wfdb.rdann(record_path, extension="apn")
            except Exception:
                warnings.warn(f"No annotations for {record_id}, skipping.")
                continue

            for sample_idx, symbol in zip(annotations.sample, annotations.symbol):
                if symbol not in ('A', 'N'):
                    continue
                label = 1 if symbol == 'A' else 0

                # <<< FIX: Logic for 3-minute window as per paper
                # The annotation is for the middle minute.
                if self.segment_length_sec == 180:
                    start_idx = sample_idx - (60 * self.fs)
                    end_idx = start_idx + seg_len_samples
                else: # Logic for 1-minute window
                    start_idx = sample_idx
                    end_idx = start_idx + (60 * self.fs)

                if start_idx < 0 or end_idx > len(filtered_signal):
                    continue
                
                segment = filtered_signal[start_idx:end_idx]
                
                if len(segment) != seg_len_samples:
                    continue

                normalized_segment = normalize(segment)
                all_segments.append(torch.from_numpy(normalized_segment.copy()).float())
                all_labels.append(label)

        if not all_segments:
            return torch.empty(0), torch.empty(0)

        segments_tensor = torch.stack(all_segments, dim=0).unsqueeze(-1)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long)
        return segments_tensor, labels_tensor

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        # --- Load and process the released set for training and validation ---
        print("Loading and processing released set for training/validation...")
        train_val_segments, train_val_labels = self._load_and_segment_set(self.released_records_names)
        
        if train_val_segments.size(0) == 0:
            raise RuntimeError("No data loaded from the released set.")

        train_val_dataset = TensorDataset(train_val_segments, train_val_labels)
        
        # Split released set into 80% train, 20% validation
        n_samples = len(train_val_dataset)
        indices = list(range(n_samples))
        split = int(np.floor(0.2 * n_samples))
        
        # Reproducible shuffle
        np.random.seed(self.random_seed)
        np.random.shuffle(indices)
        
        train_indices, val_indices = indices[split:], indices[:split]
        
        train_dataset = Subset(train_val_dataset, train_indices)
        val_dataset = Subset(train_val_dataset, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # --- Load and process the withheld set for testing ---
        print("Loading and processing withheld set for testing...")
        test_segments, test_labels = self._load_and_segment_set(self.withheld_records_names)
        
        if test_segments.size(0) == 0:
            raise RuntimeError("No data loaded from the withheld set.")

        test_dataset = TensorDataset(test_segments, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
        return train_loader, val_loader, test_loader