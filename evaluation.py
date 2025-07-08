## evaluation.py

from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
)

from model import Model 


class Evaluation:
    """
    Performs evaluation of the trained CNN-Transformer model on test data.
    
    Supports:
    - Segment-wise evaluation (accuracy, sensitivity, specificity, f1_score, auc).
    - Recording-wise aggregation for classification accuracy and AHI MAE.
    """

    def __init__(self, model: Model, test_loader: DataLoader, threshold: float = 0.5) -> None:
        """
        Initialize with trained model and test DataLoader.
        Sets model to eval mode and configures device.

        Args:
            model (Model): Trained model instance for inference.
            test_loader (DataLoader): Test dataset loader yielding batches (inputs, labels)
                Note: inputs shape (batch_size, seq_len, 1), labels shape (batch_size,)
                Assumes dataset_loader provides recording IDs for segments as an attribute or by unpacking.
                Since design does not explicitly say test_loader batches contain record_ids,
                we require _test_loader_ to have an attribute 'dataset' with 'record_ids' list aligned to dataset indices.
            threshold (float, optional): Threshold for binary apnea classification from positive class probability.
                Defaults to 0.5.
        """
        self.model: Model = model
        self.test_loader: DataLoader = test_loader
        self.threshold: float = threshold

        self.device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)
        self.model.eval()

        # After evaluate() run, these will be populated for use in aggregate_recording_results()
        self._true_labels: Optional[np.ndarray] = None  # shape: (num_samples,)
        self._pred_labels: Optional[np.ndarray] = None  # shape: (num_samples,)
        self._pred_probs: Optional[np.ndarray] = None  # shape: (num_samples,) positive class prob
        self._segment_record_ids: Optional[List[str]] = None  # record IDs per segment, len == num_samples

    def evaluate(self) -> Dict[str, float]:
        """
        Perform segment-wise model evaluation on the test set.

        Returns:
            dict: Dictionary with keys:
                'accuracy', 'sensitivity', 'specificity', 'f1_score', 'auc'
                Values are floats representing metric scores (fractions, e.g. 0.882 for 88.2%).
        """
        all_true: List[int] = []
        all_pred: List[int] = []
        all_probs: List[float] = []
        all_record_ids: List[str] = []

        # Attempt to extract segment record_ids from dataset attribute:
        # We expect that test_loader.dataset has attribute 'dataset' or 'record_ids' to access record_ids
        # For Subset, dataset_loader.store 'record_ids' in segment_data[segment_len]['record_ids']
        # so test_loader.dataset.dataset.record_ids or test_loader.dataset.record_ids
        # We will try to fallback gracefully.

        # Build index to record_id map for dataset indices in test_loader dataset
        record_ids_idx_map: Dict[int, str] = {}

        dataset = getattr(self.test_loader, "dataset", None)
        base_dataset = None

        # Handling PyTorch Subset datasets or TensorDataset wrapped into Subset
        if dataset is not None:
            if hasattr(dataset, "dataset"):
                base_dataset = dataset.dataset
            else:
                base_dataset = dataset

        # Try to get record_ids list aligned to base_dataset indices
        record_ids_list: Optional[List[str]] = None
        if base_dataset is not None:
            # Try attribute 'record_ids' expected in base_dataset as per DatasetLoader.segment_data
            record_ids_list = getattr(base_dataset, "record_ids", None)

            # If not found, try __dict__ keys or others
            if record_ids_list is None:
                record_ids_list = getattr(base_dataset, "_record_ids", None)

        # If record_ids_list found, map test indices accordingly
        # For Subset, test_loader.dataset.indices provides mapping
        if record_ids_list is not None:
            if hasattr(dataset, "indices"):
                for local_idx, base_idx in enumerate(dataset.indices):
                    if base_idx < len(record_ids_list):
                        record_ids_idx_map[local_idx] = record_ids_list[base_idx]
            else:
                # If no indices mapping, assume direct
                for idx in range(len(dataset)):
                    if idx < len(record_ids_list):
                        record_ids_idx_map[idx] = record_ids_list[idx]

        # If no record_ids mapping available, fallback all unknown
        default_record_id = "unknown"

        # Accumulate predictions and labels
        # Will be appended in order of batches processing
        segment_counter = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                # batch may be tuple: (inputs, labels) or (inputs, labels, record_ids)
                # But DataLoader typically yields tuples matching dataset __getitem__,
                # DatasetLoader uses TensorDataset with 2 elements only, so no record_id in batch
                # We'll assign record_ids if possible from record_ids_idx_map

                if isinstance(batch, (list, tuple)):
                    if len(batch) >= 2:
                        batch_x = batch[0]
                        batch_y = batch[1]
                    else:
                        raise ValueError(
                            f"Unexpected batch length {len(batch)} from test_loader, expected at least 2 elements."
                        )
                else:
                    raise ValueError("Expected batch to be tuple or list.")

                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                logits = self.model(batch_x)  # (batch_size, 2)
                probs = F.softmax(logits, dim=1)[:, 1]  # positive class probabilities, shape (batch_size,)

                preds = (probs >= self.threshold).to(torch.int64)

                all_true.extend(batch_y.cpu().tolist())
                all_pred.extend(preds.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())

                # Determine record ids for this batch indices
                batch_size = batch_y.size(0)
                batch_record_ids: List[str] = []
                for i in range(batch_size):
                    if segment_counter in record_ids_idx_map:
                        batch_record_ids.append(record_ids_idx_map[segment_counter])
                    else:
                        batch_record_ids.append(default_record_id)
                    segment_counter += 1
                all_record_ids.extend(batch_record_ids)

        # Convert all to numpy arrays for metrics
        true_np = np.array(all_true, dtype=np.int32)
        pred_np = np.array(all_pred, dtype=np.int32)
        probs_np = np.array(all_probs, dtype=np.float32)
        record_ids_np = all_record_ids  # list of str, same length as above arrays

        # Store for later use in recording aggregation
        self._true_labels = true_np
        self._pred_labels = pred_np
        self._pred_probs = probs_np
        self._segment_record_ids = record_ids_np

        # Compute metrics
        # Accuracy
        accuracy_val = accuracy_score(true_np, pred_np)

        # Sensitivity: recall for positive class (apnea=1)
        sensitivity_val = recall_score(true_np, pred_np, pos_label=1, zero_division=0)

        # Specificity: recall for negative class (non-apnea=0)
        specificity_val = recall_score(true_np, pred_np, pos_label=0, zero_division=0)

        # F1 score (harmonic mean precision-recall)
        f1_val = f1_score(true_np, pred_np, zero_division=0)

        # AUC (Area Under ROC Curve)
        # If only one class in ground truth, roc_auc_score fails; handle gracefully
        try:
            auc_val = roc_auc_score(true_np, probs_np)
        except ValueError:
            auc_val = float('nan')

        return {
            "accuracy": accuracy_val,
            "sensitivity": sensitivity_val,
            "specificity": specificity_val,
            "f1_score": f1_val,
            "auc": auc_val,
        }

    def aggregate_recording_results(self) -> Dict[str, float]:
        """
        Aggregate segment-wise predictions to derive recording-level classification and
        estimate AHI mean absolute error (MAE).
        
        Recording-level classification:
        - Majority vote of segment labels (>50% apnea segments -> apnea positive)

        AHI estimation assumptions:
        - Predicted AHI = (number of predicted apnea-positive segments * segment duration (mins)) / total recording duration (hours)
        - True AHI calculated similarly from true labels
        - Then compute MAE over recordings.

        Returns:
            dict: With keys
                'recording_classification_accuracy': fraction accuracy over recordings,
                'ahi_mae': mean absolute error of AHI estimation.
        """
        if (
            self._true_labels is None
            or self._pred_labels is None
            or self._segment_record_ids is None
        ):
            raise RuntimeError(
                "Segment-level predictions not available. Run evaluate() before aggregate_recording_results()."
            )

        # Group segment predictions and true labels by recording ID
        # Using defaultdict to accumulate data per recording
        rec_true_segments: Dict[str, List[int]] = defaultdict(list)
        rec_pred_segments: Dict[str, List[int]] = defaultdict(list)

        # Also collect count of segments per recording
        rec_segment_counts: Dict[str, int] = defaultdict(int)

        # We will also store segment count to estimate recording duration
        # Segment duration in seconds can be deduced by segment length from one batch — not stored here,
        # so we attempt to deduce from the test_loader dataset segment length:
        # (dataset_loader.py stores segments of shape (N, segment_length_samples, 1) - segment_length_samples = segment_length_sec * fs)
        # We'll attempt to get from dataset stored in test_loader dataset:
        segment_length_sec: int = -1  # unknown default
        sampling_rate: int = 100  # as per config and paper

        # Extract segment_length_sec from test_loader dataset example
        # We try to find the segment length from the first input batch shape if possible
        # If test_loader is empty, fallback to config sampling_rate and no segment length (we cannot estimate AHI)
        try:
            first_batch = next(iter(self.test_loader))
            if isinstance(first_batch, (list, tuple)) and len(first_batch) >= 1:
                batch_x = first_batch[0]
                # batch_x shape (batch_size, segment_len, 1)
                if isinstance(batch_x, torch.Tensor) and batch_x.ndim == 3:
                    seg_len_samples = batch_x.shape[1]
                    segment_length_sec = int(seg_len_samples // sampling_rate)
        except Exception:
            # fallback values if something fails
            segment_length_sec = -1

        if segment_length_sec <= 0:
            # Default to 60 seconds since paper config uses 60 and 180
            segment_length_sec = 60

        # Collect segments grouped by recording
        for rec_id, t_label, p_label in zip(
            self._segment_record_ids, self._true_labels, self._pred_labels
        ):
            rec_true_segments[rec_id].append(t_label)
            rec_pred_segments[rec_id].append(p_label)
            rec_segment_counts[rec_id] += 1

        # Compute recording-wise ground truth and predicted labels (majority voting)
        rec_gt_labels: Dict[str, int] = {}
        rec_pred_labels: Dict[str, int] = {}
        rec_true_ahi_list: List[float] = []
        rec_pred_ahi_list: List[float] = []

        # AHI = (# apnea events per hour of sleep)
        # Approximate apnea events as number of apnea-positive segments
        # and total recording length = #segments * segment_length_sec (seconds) -> divide by 3600 for hours
        for rec_id in rec_true_segments.keys():
            true_segs = rec_true_segments[rec_id]
            pred_segs = rec_pred_segments[rec_id]
            n_segs = len(true_segs)

            if n_segs == 0:
                # Skip empty recording just in case
                continue

            # Majority apnea segments => label 1, else 0
            true_label = 1 if sum(true_segs) > (n_segs / 2) else 0
            pred_label = 1 if sum(pred_segs) > (n_segs / 2) else 0

            rec_gt_labels[rec_id] = true_label
            rec_pred_labels[rec_id] = pred_label

            # Estimate AHI for recording
            duration_hours = (n_segs * segment_length_sec) / 3600.0
            # If duration_hours is zero (unlikely), fallback small eps to avoid div by zero
            if duration_hours < 1e-8:
                duration_hours = 1e-8

            # Number of apnea events approximated as total apnea-positive segments # (paper states up to segment-level)
            true_ahi = sum(true_segs) / duration_hours
            pred_ahi = sum(pred_segs) / duration_hours

            rec_true_ahi_list.append(true_ahi)
            rec_pred_ahi_list.append(pred_ahi)

        # Compute recording-wise classification accuracy
        rec_true_labels_np = np.array(list(rec_gt_labels.values()), dtype=np.int32)
        rec_pred_labels_np = np.array(list(rec_pred_labels.values()), dtype=np.int32)

        if len(rec_true_labels_np) == 0:
            # No recordings to evaluate
            recording_acc = float('nan')
        else:
            recording_acc = accuracy_score(rec_true_labels_np, rec_pred_labels_np)

        # Compute AHI MAE over recordings
        if len(rec_true_ahi_list) == 0:
            ahi_mae: float = float('nan')
        else:
            true_ahi_np = np.array(rec_true_ahi_list, dtype=np.float32)
            pred_ahi_np = np.array(rec_pred_ahi_list, dtype=np.float32)
            ahi_mae = float(np.mean(np.abs(true_ahi_np - pred_ahi_np)))

        return {
            "recording_classification_accuracy": recording_acc,
            "ahi_mae": ahi_mae,
        }
