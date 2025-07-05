## dataset_loader.py

import os
import random
import warnings
from typing import Tuple, Dict, List, Optional, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

from preprocessing import butter_bandpass_filter, normalize

# To read WFDB format files and annotations from PhysioNet
try:
    import wfdb
except ImportError:
    raise ImportError(
        "The wfdb package is required for ECG signal and annotation reading. "
        "Install it via `pip install wfdb`."
    )


class DatasetLoader:
    """
    DatasetLoader is responsible for loading the Apnea-ECG dataset from PhysioNet,
    applying preprocessing, segmenting, labeling apnea events, and providing
    PyTorch DataLoaders for train, validation, and test splits.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes DatasetLoader with configuration parameters.

        Args:
            config (dict): Configuration dictionary parsed from config.yaml.
        """
        self.config = config

        # Dataset parameters from config
        ds_cfg = config.get("dataset", {})
        self.dataset_name: str = ds_cfg.get("name", "Apnea-ECG")
        self.dataset_root: str = ds_cfg.get("root_dir", os.path.join(os.getcwd(), "apnea-ecg"))
        self.fs: int = ds_cfg.get("sampling_rate", 100)
        self.segment_lengths_sec: List[int] = ds_cfg.get("segment_lengths", [60])

        # Preprocessing parameters from config
        preprocessing_cfg = config.get("preprocessing", {})
        bandpass_cfg = preprocessing_cfg.get("bandpass_filter", {})
        self.lowcut: float = bandpass_cfg.get("lowcut", 0.5)
        self.highcut: float = bandpass_cfg.get("highcut", 48.0)
        self.filter_order: int = bandpass_cfg.get("order", 4)
        self.normalization_method: str = preprocessing_cfg.get("normalization", "z-score")

        # Training parameters from config
        training_cfg = config.get("training", {})
        self.batch_size: int = training_cfg.get("batch_size", 32)
        
        self.records_path = self._find_records_path()

        # Internal data storage
        self.raw_signals: Dict[str, np.ndarray] = {}
        self.segment_data: Dict[int, Dict[str, Any]] = {}

        # Random seed for reproducibility
        self.random_seed: int = 42

    def _find_records_path(self) -> str:
        """Finds the correct path to the .dat files, which could be in the root or a 'records' subdir."""
        if not os.path.isdir(self.dataset_root):
            raise FileNotFoundError(f"Dataset root directory not found: {self.dataset_root}")
        
        if any(f.endswith(".dat") for f in os.listdir(self.dataset_root)):
            return self.dataset_root
            
        records_subdir = os.path.join(self.dataset_root, "records")
        if os.path.isdir(records_subdir) and any(f.endswith(".dat") for f in os.listdir(records_subdir)):
            return records_subdir
            
        raise FileNotFoundError(f"Could not find any .dat record files in '{self.dataset_root}' or its 'records' subdirectory.")


    def load_raw_data(self) -> None:
        """
        Loads all raw ECG signals from the dataset directory.
        """
        record_files = sorted([f[:-4] for f in os.listdir(self.records_path) if f.endswith(".dat")])
        if not record_files:
            raise FileNotFoundError(f"No record files (.dat) found in '{self.records_path}'.")

        self.raw_signals.clear()

        for record_id in record_files:
            record_path = os.path.join(self.records_path, record_id)
            try:
                record = wfdb.rdrecord(record_path)
                ecg_signal = record.p_signal

                # <--- FIX: Handle potential NaNs from the source file right after loading --->
                # This is the primary cause of the learning failure.
                if np.isnan(ecg_signal).any():
                    warnings.warn(f"NaNs found in record {record_id}, converting them to 0.0.", UserWarning)
                    ecg_signal = np.nan_to_num(ecg_signal)

                if ecg_signal.ndim > 1:
                    if ecg_signal.shape[1] > 1:
                        warnings.warn(
                            f"Record {record_id} has {ecg_signal.shape[1]} channels; using channel 0.",
                            UserWarning
                        )
                    ecg_signal = ecg_signal[:, 0]
                
                self.raw_signals[record_id] = ecg_signal.flatten().astype(np.float32)

            except Exception as e:
                warnings.warn(f"Could not read record {record_id}. Error: {e}")
                continue
        
        # DEBUG print is fine, keeping it as is.
        print("\n[Debug] Sample of loaded apnea annotations:")
        sample_records_to_debug = [r for r in record_files if r in self.raw_signals][:3]
        for rid in sample_records_to_debug:
            try:
                ann = wfdb.rdann(os.path.join(self.records_path, rid), extension="apn")
                print(f"  Record '{rid}' Symbols: {ann.symbol[:5]}")
            except Exception:
                print(f"  Record '{rid}': No annotations found or error reading.")
        print("-" * 20)


    def preprocess_and_segment(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies preprocessing and segments signals based on per-minute annotations.
        """
        if not self.raw_signals:
            raise RuntimeError("Raw signals not loaded. Call load_raw_data() first.")
        self.segment_data.clear()

        for seg_len_sec in self.segment_lengths_sec:
            seg_len_samples = int(seg_len_sec * self.fs)

            all_segments = []
            all_labels = []
            all_record_ids_for_segments = []

            for record_id, signal in self.raw_signals.items():
                signal_len = len(signal)
                
                filtered_signal = butter_bandpass_filter(
                    signal=signal,
                    lowcut=self.lowcut,
                    highcut=self.highcut,
                    fs=self.fs,
                    order=self.filter_order,
                )

                try:
                    record_path = os.path.join(self.records_path, record_id)
                    annotations = wfdb.rdann(record_path, extension="apn")
                except Exception:
                    continue

                for sample_idx, symbol in zip(annotations.sample, annotations.symbol):
                    if symbol not in ('A', 'N'):
                        continue

                    label = 1 if symbol == 'A' else 0
                    
                    if seg_len_sec == 180:
                        start_idx = sample_idx - (60 * self.fs)
                        end_idx = start_idx + seg_len_samples
                    else:
                        start_idx = sample_idx
                        end_idx = start_idx + seg_len_samples

                    if start_idx < 0 or end_idx > signal_len:
                        continue
                        
                    segment = filtered_signal[start_idx:end_idx]
                    
                    if len(segment) != seg_len_samples:
                        continue

                    # <--- NOTE: With NaN problem solved, this function is now safe to use.
                    normalized_segment = normalize(segment)

                    all_segments.append(torch.from_numpy(normalized_segment.copy()).float()) # Use .copy() to be safe
                    all_labels.append(label)
                    all_record_ids_for_segments.append(record_id)

            if not all_segments:
                self.segment_data[seg_len_sec] = {"segments": torch.empty(0), "labels": torch.empty(0), "record_ids": []}
                continue

            segments_tensor = torch.stack(all_segments, dim=0).unsqueeze(-1)
            labels_tensor = torch.tensor(all_labels, dtype=torch.long)
            
            print(f"📊 Label distribution for {seg_len_sec}s: {torch.bincount(labels_tensor)}")

            self.segment_data[seg_len_sec] = {
                "segments": segments_tensor,
                "labels": labels_tensor,
                "record_ids": all_record_ids_for_segments,
            }

        first_seg_len = self.segment_lengths_sec[0]
        data = self.segment_data.get(first_seg_len)
        if data is None or data["segments"].numel() == 0:
            raise RuntimeError(f"No data was processed for segment length {first_seg_len}s.")
            
        return data["segments"], data["labels"]

    def get_dataloaders(
        self,
        segment_length: Optional[int] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        shuffle_seed: Optional[int] = None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Splits the dataset by patient (record ID) and creates PyTorch DataLoaders.
        """
        if segment_length is None:
            segment_length = self.segment_lengths_sec[0]

        data = self.segment_data.get(segment_length)
        if data is None:
            raise RuntimeError(f"Data for segment length {segment_length}s not found. Run preprocess_and_segment() first.")

        segments_tensor, labels_tensor, record_ids = data["segments"], data["labels"], data["record_ids"]

        if segments_tensor.size(0) == 0:
            raise RuntimeError(f"No segments available for segment length {segment_length}s.")

        unique_records = sorted(list(set(record_ids)))
        
        random.seed(shuffle_seed if shuffle_seed is not None else self.random_seed)
        random.shuffle(unique_records)

        n_records = len(unique_records)
        n_train = int(n_records * train_ratio)
        n_val = int(n_records * val_ratio)
        
        train_records = set(unique_records[:n_train])
        val_records = set(unique_records[n_train : n_train + n_val])
        test_records = set(unique_records[n_train + n_val :])

        def get_indices_for_split(target_records: set) -> List[int]:
            return [i for i, rec_id in enumerate(record_ids) if rec_id in target_records]

        train_indices = get_indices_for_split(train_records)
        val_indices = get_indices_for_split(val_records)
        test_indices = get_indices_for_split(test_records)

        full_dataset = TensorDataset(segments_tensor, labels_tensor)
        
        full_dataset.record_ids = record_ids

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        test_dataset = Subset(full_dataset, test_indices)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader