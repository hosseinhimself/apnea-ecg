import os
import random
import warnings
from typing import Tuple, Dict, List, Optional, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

from preprocessing import butter_bandpass_filter, normalize

try:
    import wfdb
except ImportError:
    raise ImportError(
        "The wfdb package is required for ECG signal and annotation reading. "
        "Install it via `pip install wfdb`."
    )


class DatasetLoader:
    """
    Loads and preprocesses the PhysioNet Apnea-ECG dataset when files may live flat
    or under a "records/" subfolder. Automatically detects and handles both layouts.

    - Point `config["dataset"]["root_dir"]` at the parent folder.
    - Flat layout: all .dat/.hea/.apn in root_dir.
    - Nested layout: .dat/.hea/.apn under root_dir/records.
    - Annotation extension is 'apn'.
    - If multi-channel, selects channel 0 with a warning.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

        # Dataset parameters
        ds_cfg = config.get("dataset", {})
        self.dataset_name: str = ds_cfg.get("name", "Apnea-ECG")
        self.dataset_root: str = ds_cfg.get(
            "root_dir", os.path.join(os.getcwd(), "apnea-ecg")
        )
        self.fs: int = ds_cfg.get("sampling_rate", 100)
        self.segment_lengths_sec: List[int] = ds_cfg.get("segment_lengths", [60])

        # Preprocessing parameters
        prep = config.get("preprocessing", {})
        bp = prep.get("bandpass_filter", {})
        self.lowcut: float = bp.get("lowcut", 0.5)
        self.highcut: float = bp.get("highcut", 48.0)
        self.filter_order: int = bp.get("order", 4)

        # Training parameters
        train_cfg = config.get("training", {})
        self.batch_size: int = train_cfg.get("batch_size", 32)

        # Internal storage
        self.raw_signals: Dict[str, np.ndarray] = {}
        self.apnea_annotations: Dict[str, List[Tuple[int, int]]] = {}
        self.segment_data: Dict[int, Dict[str, Any]] = {}
        self.random_seed: int = 42

    def load_raw_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Detects layout, reads every `<record_id>.dat/.hea` pair and its `<record_id>.apn`
        annotation. Converts annotation points → continuous intervals.
        """
        base_dir = self.dataset_root
        if not os.path.isdir(base_dir):
            raise FileNotFoundError(f"Dataset folder not found: {base_dir}")

        # Look for .dat files in root; if none, check for 'records' subfolder
        files = os.listdir(base_dir)
        dat_files = [f for f in files if f.lower().endswith(".dat")]
        if not dat_files:
            nested = os.path.join(base_dir, "records")
            if os.path.isdir(nested):
                base_dir = nested
                files = os.listdir(base_dir)
                dat_files = [f for f in files if f.lower().endswith(".dat")]
        if not dat_files:
            raise FileNotFoundError(
                f"No `.dat` files found in {self.dataset_root} or its 'records/' subfolder."
            )

        record_ids = sorted({f[:-4] for f in dat_files})
        self.raw_signals.clear()
        self.apnea_annotations.clear()

        for rid in record_ids:
            rec_path = os.path.join(base_dir, rid)
            try:
                rec = wfdb.rdrecord(rec_path)
            except Exception as e:
                raise RuntimeError(f"Failed to read record {rid}: {e}")

            # Select first channel if multi-signal
            if rec.n_sig != 1:
                warnings.warn(
                    f"Record {rid} has {rec.n_sig} channels; using channel 0."
                )
            sig = rec.p_signal[:, 0].astype(np.float32).flatten()
            self.raw_signals[rid] = sig

            # Read annotations (.apn)
            try:
                ann = wfdb.rdann(rec_path, extension="apnea")
            except Exception as e:
                raise RuntimeError(f"Failed to read .apn for {rid}: {e}")
            positions = [int(s) for s, sym in zip(ann.sample, ann.symbol) if sym.upper() == "A"]
            self.apnea_annotations[rid] = self._positions_to_intervals(sorted(positions))

        return {"signals": self.raw_signals, "annotations": self.apnea_annotations}

    def _positions_to_intervals(self, positions: List[int]) -> List[Tuple[int, int]]:
        if not positions:
            return []
        intervals = []
        start = prev = positions[0]
        for pos in positions[1:]:
            if pos == prev + 1:
                prev = pos
            else:
                intervals.append((start, prev))
                start = prev = pos
        intervals.append((start, prev))
        return intervals

    def preprocess_and_segment(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.raw_signals:
            raise RuntimeError("Call load_raw_data() before preprocessing!")

        self.segment_data.clear()
        for seg_sec in self.segment_lengths_sec:
            seg_samps = int(seg_sec * self.fs)
            segs, labs, rec_ids = [], [], []

            for rid, sig in self.raw_signals.items():
                filtered = butter_bandpass_filter(
                    signal=sig,
                    lowcut=self.lowcut,
                    highcut=self.highcut,
                    fs=self.fs,
                    order=self.filter_order,
                )
                n_seg = len(filtered) // seg_samps
                if n_seg == 0:
                    continue
                truncated = filtered[: n_seg * seg_samps].reshape(n_seg, seg_samps)

                labels = np.zeros(n_seg, dtype=np.int64)
                for i in range(n_seg):
                    start = i * seg_samps
                    end = start + seg_samps - 1
                    covered = 0
                    for a0, a1 in self.apnea_annotations[rid]:
                        ov0 = max(start, a0)
                        ov1 = min(end, a1)
                        if ov0 <= ov1:
                            covered += (ov1 - ov0 + 1)
                    if covered / seg_samps > 0.5:
                        labels[i] = 1

                normed = np.vstack([normalize(tr) for tr in truncated]).astype(np.float32)
                segs.append(torch.from_numpy(normed).unsqueeze(-1))
                labs.append(torch.from_numpy(labels))
                rec_ids += [rid] * n_seg

            if segs:
                self.segment_data[seg_sec] = {
                    "segments": torch.cat(segs, dim=0),
                    "labels": torch.cat(labs, dim=0),
                    "record_ids": rec_ids,
                }
            else:
                self.segment_data[seg_sec] = {"segments": torch.empty(0), "labels": torch.empty(0), "record_ids": []}

        first = self.segment_lengths_sec[0]
        data = self.segment_data[first]
        return data["segments"], data["labels"]

    def get_dataloaders(
        self,
        segment_length: Optional[int] = None,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        shuffle_seed: Optional[int] = None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        if segment_length is None:
            segment_length = self.segment_lengths_sec[0]
        data = self.segment_data.get(segment_length)
        if data is None:
            raise RuntimeError("No preprocessed data—run preprocess_and_segment() first.")

        X, y, rec_ids = data["segments"], data["labels"], data["record_ids"]
        if X.size(0) == 0:
            raise RuntimeError(f"No segments for {segment_length}s")

        ids = sorted(set(rec_ids))
        random.seed(shuffle_seed or self.random_seed)
        random.shuffle(ids)
        n = len(ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_ids = set(ids[:n_train])
        val_ids = set(ids[n_train : n_train + n_val])
        test_ids = set(ids[n_train + n_val :])

        def inds_for(split_ids):
            return [i for i, rid in enumerate(rec_ids) if rid in split_ids]

        ds = TensorDataset(X, y)
        train_ds = Subset(ds, inds_for(train_ids))
        val_ds   = Subset(ds, inds_for(val_ids))
        test_ds  = Subset(ds, inds_for(test_ids))

        return (
            DataLoader(train_ds, batch_size=self.batch_size, shuffle=True),
            DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False),
            DataLoader(test_ds,  batch_size=self.batch_size, shuffle=False),
        )