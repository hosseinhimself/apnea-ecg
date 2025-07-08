# dataset_loader.py

import os
import random
import warnings
from typing import Tuple, Dict, List, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

# Import custom preprocessing functions
from preprocessing import butter_bandpass_filter, normalize

try:
    # Attempt to import wfdb, a library for reading WFDB physiological data
    import wfdb
except ImportError:
    # If wfdb is not installed, raise an error guiding the user to install it
    raise ImportError("The wfdb package is required. Install it via `pip install wfdb`.")

class DatasetLoader:
    """
    A class to load, preprocess, and prepare ECG datasets for Obstructive Sleep Apnea (OSA) detection.
    It handles data loading from PhysioNet's Apnea-ECG database, preprocessing steps like filtering
    and normalization, and segmenting the data into appropriate time windows as described in the paper.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the DatasetLoader with configuration parameters.

        Args:
            config (Dict[str, Any]): A dictionary containing various configuration settings
                                     for dataset loading, preprocessing, and training.
        """
        self.config = config
        ds_cfg = config.get("dataset", {})
        # Determine the root directory for the dataset, defaulting to 'apnea-ecg' in the current working directory
        self.dataset_root: str = ds_cfg.get("root_dir", os.path.join(os.getcwd(), "apnea-ecg"))
        # Set the sampling rate of the ECG signals, default is 100 Hz
        self.fs: int = ds_cfg.get("sampling_rate", 100)
        # The segment length for input to the model, obtained from config.
        # [cite_start]The paper specifies using 3-minute (180 seconds) epochs[cite: 33].
        self.segment_length_sec: int = ds_cfg.get("segment_lengths", [180])[0] # FIX: Focus on the segment length specified in config

        preprocessing_cfg = config.get("preprocessing", {})
        bandpass_cfg = preprocessing_cfg.get("bandpass_filter", {})
        # [cite_start]Low cutoff frequency for the Butterworth bandpass filter [cite: 225]
        self.lowcut: float = bandpass_cfg.get("lowcut", 0.5)
        # [cite_start]High cutoff frequency for the Butterworth bandpass filter [cite: 225]
        self.highcut: float = bandpass_cfg.get("highcut", 48.0)
        # [cite_start]Order of the Butterworth bandpass filter [cite: 225]
        self.filter_order: int = bandpass_cfg.get("order", 4)
        
        training_cfg = config.get("training", {})
        # [cite_start]Batch size for DataLoader [cite: 240]
        self.batch_size: int = training_cfg.get("batch_size", 32)
        
        # Discover the actual path where the record files (.dat) are located
        self.records_path = self._find_records_path()
        # [cite_start]Random seed for reproducibility, as mentioned in the paper for data splitting[cite: 231].
        self.random_seed: int = 42

        # [cite_start]Define the record names for the released and withheld sets as per PhysioNet Apnea-ECG database specifications[cite: 218].
        # [cite_start]These sets are used for training/validation and testing, respectively[cite: 237].
        self.released_records_names = [f'a{i:02d}' for i in range(1, 21)] + \
                                      [f'b{i:02d}' for i in range(1, 6)] + \
                                      [f'c{i:02d}' for i in range(1, 11)]
        self.withheld_records_names = [f'x{i:02d}' for i in range(1, 36)]

    def _find_records_path(self) -> str:
        """
        Identifies the directory containing the WFDB record files (.dat).
        It checks the specified dataset root and a common 'records' subdirectory.

        Returns:
            str: The path to the directory containing the record files.

        Raises:
            FileNotFoundError: If no .dat record files are found in expected locations.
        """
        # Check if the dataset root itself contains .dat files
        if not os.path.isdir(self.dataset_root):
            raise FileNotFoundError(f"Dataset root directory not found: {self.dataset_root}")
        if any(f.endswith(".dat") for f in os.listdir(self.dataset_root)):
            return self.dataset_root
        
        # If not, check a 'records' subdirectory within the root
        records_subdir = os.path.join(self.dataset_root, "records")
        if os.path.isdir(records_subdir) and any(f.endswith(".dat") for f in os.listdir(records_subdir)):
            return records_subdir
        
        # If neither contains .dat files, raise an error
        raise FileNotFoundError(f"Could not find any .dat record files in '{self.dataset_root}' or its 'records' subdirectory.")

    def _load_and_segment_set(self, record_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads ECG signals and their annotations for a given list of record names,
        then segments them into fixed-length epochs with corresponding labels.

        Args:
            record_names (List[str]): A list of record IDs (e.g., 'a01', 'x05') to process.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
                                               - A tensor of ECG segments (shape: [num_segments, segment_length_samples, 1])
                                               - A tensor of corresponding labels (0 for normal, 1 for apnea)
        """
        # Calculate the length of a segment in samples based on segment_length_sec and sampling rate
        seg_len_samples = int(self.segment_length_sec * self.fs)
        all_segments, all_labels = [], []

        for record_id in record_names:
            try:
                record_path = os.path.join(self.records_path, record_id)
                # Read the ECG record using wfdb
                record = wfdb.rdrecord(record_path)
                signal = record.p_signal
            except Exception as e:
                warnings.warn(f"Could not read record {record_id}, skipping. Error: {e}")
                continue

            # Handle potential NaN values in the signal by converting them to zeros
            if np.isnan(signal).any():
                signal = np.nan_to_num(signal)

            # [cite_start]If the signal has multiple channels, select the first one (single-channel ECG as per paper) [cite: 28, 108]
            if signal.ndim > 1:
                signal = signal[:, 0]

            # [cite_start]Apply bandpass filtering to the raw ECG signal [cite: 224, 225]
            filtered_signal = butter_bandpass_filter(signal.flatten().astype(np.float32), self.lowcut, self.highcut, self.fs, self.filter_order)
            
            try:
                # [cite_start]Read annotations (apnea events) for the current record [cite: 222]
                annotations = wfdb.rdann(record_path, extension="apn")
            except Exception:
                warnings.warn(f"No annotations for {record_id}, skipping.")
                continue

            # Iterate through each annotated event
            for sample_idx, symbol in zip(annotations.sample, annotations.symbol):
                # [cite_start]Only consider 'A' (apnea) or 'N' (normal) symbols [cite: 222]
                if symbol not in ('A', 'N'):
                    continue
                label = 1 if symbol == 'A' else 0 # 1 for apnea, 0 for normal

                # [cite_start]FIX: Implement the 3-minute window logic as described in the paper [cite: 33, 111, 279, 281]
                # The paper states that the annotation refers to the middle minute of a 3-minute epoch.
                if self.segment_length_sec == 180: # If using 3-min input window
                    # The `sample_idx` from annotations typically refers to the start of the 1-minute epoch.
                    # For a 3-minute epoch centered around this 1-minute epoch (X_n),
                    # the start index for the 3-minute window (X_{n-1}, X_n, X_{n+1}) would be 1 minute before X_n.
                    start_idx = sample_idx - (60 * self.fs) # Go back one minute from the annotated minute's start
                    end_idx = start_idx + seg_len_samples # End of the 3-minute window
                else: # Logic for 1-minute window, where sample_idx is the start of the 1-min segment
                    start_idx = sample_idx
                    end_idx = start_idx + (60 * self.fs) # 1-minute duration

                # Ensure the segment falls within the bounds of the signal
                if start_idx < 0 or end_idx > len(filtered_signal):
                    continue
                
                # Extract the segment from the filtered signal
                segment = filtered_signal[start_idx:end_idx]
                
                # Ensure the extracted segment has the correct length
                if len(segment) != seg_len_samples:
                    continue

                # [cite_start]Normalize the extracted segment [cite: 227]
                normalized_segment = normalize(segment)
                # Append the processed segment and its label
                all_segments.append(torch.from_numpy(normalized_segment.copy()).float())
                all_labels.append(label)

        # If no segments were loaded, return empty tensors
        if not all_segments:
            return torch.empty(0), torch.empty(0)

        # Stack all segments into a single tensor and add an unsqueeze dimension for CNN input (channels)
        segments_tensor = torch.stack(all_segments, dim=0).unsqueeze(-1)
        # Convert labels to a LongTensor
        labels_tensor = torch.tensor(all_labels, dtype=torch.long)
        return segments_tensor, labels_tensor

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepares and returns PyTorch DataLoaders for the training, validation, and test sets.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: A tuple containing the DataLoader for:
                                                       - Training set
                                                       - Validation set
                                                       - Test set
        """
        # --- Load and process the released set for training and validation ---
        print("Loading and processing released set for training/validation...")
        # [cite_start]The released set is used for training and validation [cite: 237]
        train_val_segments, train_val_labels = self._load_and_segment_set(self.released_records_names)
        
        if train_val_segments.size(0) == 0:
            raise RuntimeError("No data loaded from the released set.")

        # Create a TensorDataset from the loaded segments and labels
        train_val_dataset = TensorDataset(train_val_segments, train_val_labels)
        
        # [cite_start]Split the released set into 80% training and 20% validation as described [cite: 238]
        n_samples = len(train_val_dataset)
        indices = list(range(n_samples))
        split = int(np.floor(0.2 * n_samples)) # 20% for validation
        
        # [cite_start]Shuffle indices for reproducibility using the predefined random seed [cite: 231]
        np.random.seed(self.random_seed)
        np.random.shuffle(indices)
        
        # Determine train and validation indices
        train_indices, val_indices = indices[split:], indices[:split]
        
        # Create Subset datasets for training and validation
        train_dataset = Subset(train_val_dataset, train_indices)
        val_dataset = Subset(train_val_dataset, val_indices)

        # Create DataLoaders for training and validation sets
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # --- Load and process the withheld set for testing ---
        print("Loading and processing withheld set for testing...")
        # [cite_start]The withheld set is used exclusively for testing [cite: 237]
        test_segments, test_labels = self._load_and_segment_set(self.withheld_records_names)
        
        if test_segments.size(0) == 0:
            raise RuntimeError("No data loaded from the withheld set.")

        # Create a TensorDataset for the test set
        test_dataset = TensorDataset(test_segments, test_labels)
        # Create a DataLoader for the test set
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Print the number of samples in each dataset split
        print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
        return train_loader, val_loader, test_loader