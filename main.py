## main.py

import os
import sys
import yaml
import argparse
from typing import Dict, Any

import torch

from dataset_loader import DatasetLoader
from model import Model
from trainer import Trainer
from evaluation import Evaluation


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path (str): Path to YAML config file.

    Returns:
        Dict[str, Any]: Parsed config dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If YAML file is malformed.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def run_experiment_for_segment_length(
    config: Dict[str, Any],
    segment_length_sec: int,
) -> None:
    """
    Run full experiment pipeline (data prep, model training, evaluation) for a given segment length.

    Args:
        config (Dict[str, Any]): Full configuration dictionary.
        segment_length_sec (int): Segment length in seconds (e.g. 60 or 180).
    """
    print(f"\n=== Starting experiment for segment length = {segment_length_sec} seconds ===")

    # --- STEP 1: Prepare DatasetLoader for this segment length ---

    # Deep copy config to avoid side effects
    import copy
    config_copy = copy.deepcopy(config)

    # Override dataset segment_lengths to a single length for this run
    if "dataset" not in config_copy:
        raise RuntimeError("Missing 'dataset' section in config.")
    config_copy["dataset"]["segment_lengths"] = [segment_length_sec]

    # Instantiate DatasetLoader
    print("[Data] Initializing DatasetLoader...")
    dataset_loader = DatasetLoader(config_copy)

    # Load raw data and annotations
    print("[Data] Loading raw ECG signals and apnea annotations...")
    try:
        dataset_loader.load_raw_data()
    except Exception as e:
        print(f"ERROR: Failed to load raw data - {str(e)}")
        sys.exit(1)

    # Preprocess, filter, normalize and segment signals
    print("[Data] Preprocessing and segmenting ECG signals...")
    try:
        segments, labels = dataset_loader.preprocess_and_segment()
        print("📊 Label distribution:", torch.bincount(labels))
    except Exception as e:
        print(f"ERROR: Preprocessing and segmentation failed - {str(e)}")
        sys.exit(1)

    if segments.size(0) == 0 or labels.size(0) == 0:
        print(f"WARNING: No segments or labels available for segment length {segment_length_sec}s, skipping experiment.")
        return

    # Obtain dataloaders (train/val/test splits) from the segmented data
    print("[Data] Creating train, validation, and test DataLoaders...")
    try:
        train_loader, val_loader, test_loader = dataset_loader.get_dataloaders(
            segment_length=segment_length_sec,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            shuffle_seed=42,
        )
    except Exception as e:
        print(f"ERROR: Failed to create DataLoaders - {str(e)}")
        sys.exit(1)

    if len(train_loader.dataset) == 0:
        print("ERROR: Training dataset is empty after split - aborting.")
        sys.exit(1)
    if len(val_loader.dataset) == 0:
        print("ERROR: Validation dataset is empty after split - aborting.")
        sys.exit(1)
    if len(test_loader.dataset) == 0:
        print("ERROR: Test dataset is empty after split - aborting.")
        sys.exit(1)

    print(f"[Data] Train set size: {len(train_loader.dataset)} samples")
    print(f"[Data] Validation set size: {len(val_loader.dataset)} samples")
    print(f"[Data] Test set size: {len(test_loader.dataset)} samples")

    # DEBUG: Inspect the first batch of the training loader
    print("\n[Debug] Inspecting first batch of train_loader...")
    try:
        first_batch_data, first_batch_labels = next(iter(train_loader))
        print(f"  Data shape: {first_batch_data.shape}")
        print(f"  Labels in first batch: {first_batch_labels}")
        if len(first_batch_labels) > 0:
            print(f"  Label distribution in first batch: {torch.bincount(first_batch_labels.long())}")
    except StopIteration:
        print("  Could not get a batch from train_loader, it might be empty.")
    print("-" * 20)

    # --- STEP 2: Instantiate Model with config parameters ---

    print("[Model] Building CNN-Transformer model...")
    model_params = config_copy.get("model", {})
    if not model_params:
        print("WARNING: No 'model' config section found, defaulting.")
        model_params = {}

    model = Model(params={"model": model_params})

    # --- STEP 3: Set up Trainer and train model ---

    print("[Training] Starting training process...")
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, config=config_copy)
    trainer.train()
    print("[Training] Training finished.")

    # --- STEP 4: Evaluate model on test set ---

    print("[Evaluation] Evaluating model on test set...")
    evaluator = Evaluation(model=model, test_loader=test_loader)
    segment_metrics = evaluator.evaluate()
    recording_metrics = evaluator.aggregate_recording_results()

    # --- STEP 5: Report results ---

    print(f"\n[Results] Segment-wise metrics for segment length = {segment_length_sec} seconds:")
    def fmt_float(v: float) -> str:
        if v is None or (isinstance(v,float) and (v != v)):  # Check NaN
            return "N/A"
        return f"{v*100:.2f}%"

    print(f"  Accuracy:           {fmt_float(segment_metrics.get('accuracy', float('nan')))}")
    print(f"  Sensitivity:        {fmt_float(segment_metrics.get('sensitivity', float('nan')))}")
    print(f"  Specificity:        {fmt_float(segment_metrics.get('specificity', float('nan')))}")
    print(f"  F1 Score:           {fmt_float(segment_metrics.get('f1_score', float('nan')))}")
    auc_val = segment_metrics.get("auc", float('nan'))
    if isinstance(auc_val, float) and (auc_val != auc_val):
        print(f"  AUC:                N/A")
    else:
        print(f"  AUC:                {auc_val:.4f}")

    print(f"\n[Results] Recording-wise metrics for segment length = {segment_length_sec} seconds:")
    rec_acc = recording_metrics.get("recording_classification_accuracy", float('nan'))
    ahi_mae = recording_metrics.get("ahi_mae", float('nan'))
    if rec_acc != rec_acc:
        print(f"  Recording classification accuracy: N/A")
    else:
        print(f"  Recording classification accuracy: {rec_acc*100:.2f}%")
    if ahi_mae != ahi_mae:
        print(f"  AHI MAE:                           N/A")
    else:
        print(f"  AHI Mean Absolute Error (MAE):    {ahi_mae:.4f} events/hour")

    print(f"\n=== Experiment for segment length = {segment_length_sec} seconds completed ===\n")


def main() -> None:
    """
    Main function for the OSA Detection experiment pipeline.
    Parses config.yaml, runs experiments for specified segment lengths.
    """
    parser = argparse.ArgumentParser(description="Run OSA detection experiments with CNN-Transformer model.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"ERROR: Failed to load config file '{args.config}': {e}")
        sys.exit(1)

    # Extract segment lengths from config, fallback to [60] if missing or empty
    segment_lengths = []
    try:
        segment_lengths = config.get("dataset", {}).get("segment_lengths", [])
        if not isinstance(segment_lengths, list) or not segment_lengths:
            print("WARNING: segment_lengths missing or empty in config; defaulting to [60].")
            segment_lengths = [60]
        else:
            segment_lengths = [int(x) for x in segment_lengths]
    except Exception:
        print("WARNING: Unable to parse segment_lengths from config; defaulting to [60].")
        segment_lengths = [60]

    print("==========================================")
    print("Obstructive Sleep Apnea Detection - CNN-Transformer")
    print("Reproduction based on config file:", args.config)
    print("Segment lengths to experiment:", segment_lengths)
    print("==========================================")

    for seg_len in segment_lengths:
        try:
            run_experiment_for_segment_length(config, seg_len)

        except Exception as e:
            print(f"ERROR: Exception during experiment with segment length {seg_len}s: {str(e)}")
            print("Skipping to next segment length if available.")
            continue

    print("All experiments completed.")


if __name__ == "__main__":
    main()
