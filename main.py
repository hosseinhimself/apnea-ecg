# main.py

import os
import sys
import yaml
import argparse
from typing import Dict, Any

import torch

# Import custom modules for data loading, model definition, training, and evaluation
from dataset_loader import DatasetLoader
from model import Model
from trainer import Trainer
from evaluation import Evaluation


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads the YAML configuration file.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration parameters.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
    """
    # Check if the configuration file exists
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    # Open and load the YAML file
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def main() -> None:
    """
    Main function to execute the complete pipeline for Obstructive Sleep Apnea (OSA) detection.
    This includes data preparation, model building, training, and evaluation,
    following the methodology described in the paper.
    """
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Run OSA detection experiments based on the paper.")
    # Add argument for specifying the configuration file path
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml", # Default config file name
        help="Path to configuration YAML file (default: config.yaml)",
    )
    args = parser.parse_args()

    try:
        # Load the configuration from the specified YAML file
        config = load_config(args.config)
    except Exception as e:
        # If config loading fails, print an error and exit
        print(f"ERROR: Failed to load config file '{args.config}': {e}")
        sys.exit(1)

    # Print a header for the experiment run
    print("======================================================")
    print("OSA Detection - Exact Replication of the Paper")
    print(f"Using config file: {args.config}")
    print("======================================================")

    try:
        # --- Step 1: Dataset Preparation ---
        # Initialize the DatasetLoader, which handles loading, preprocessing,
        # and splitting of the Apnea-ECG dataset into training, validation, and test sets.
        print("[Data] Initializing DatasetLoader and creating dataloaders...")
        dataset_loader = DatasetLoader(config)
        train_loader, val_loader, test_loader = dataset_loader.get_dataloaders()
        print("[Data] Dataloaders created successfully.")

        # --- Step 2: Model Construction ---
        # Build the CNN-Transformer model based on the architecture described in the paper.
        # Model parameters are extracted from the loaded configuration.
        print("[Model] Building CNN-Transformer model as per the paper's architecture...")
        model = Model(params=config)

        # --- Step 3: Model Training ---
        # Initialize the Trainer with the model, data loaders, and configuration.
        # Start the training process. The trainer saves the best performing model.
        print("[Training] Initializing trainer and starting training process...")
        trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, config=config)
        trainer.train()
        print("[Training] Training finished.")

        # --- Step 4: Model Evaluation ---
        # Load the state dictionary of the best model saved during training.
        print("[Evaluation] Loading best model and evaluating on the test set...")
        model.load_state_dict(torch.load("best_model.pth"))
        
        # Initialize the Evaluation module with the loaded model and test data loader.
        # Perform segment-wise evaluation.
        evaluator = Evaluation(model=model, test_loader=test_loader)
        segment_metrics = evaluator.evaluate()
        
        # This section can be activated if per-recording aggregation results are needed,
        # which are typically used for the final AHI-based classification.
        # recording_metrics = evaluator.aggregate_recording_results()

        # --- Step 5: Report Results ---
        # Print the segment-wise classification metrics obtained on the test set.
        print("\n[Results] Segment-wise metrics on the test set:")
        # Helper function to format float values as percentages or "N/A" if invalid
        def fmt_float(v: float) -> str:
            if v is None or (isinstance(v,float) and (v != v)): # Check for None or NaN
                return "N/A"
            return f"{v*100:.2f}%"

        # Display key performance metrics
        print(f"  Accuracy:           {fmt_float(segment_metrics.get('accuracy'))}")
        print(f"  Sensitivity:        {fmt_float(segment_metrics.get('sensitivity'))}")
        print(f"  Specificity:        {fmt_float(segment_metrics.get('specificity'))}")
        print(f"  F1 Score:           {fmt_float(segment_metrics.get('f1_score'))}")
        # AUC is reported as a raw float, not percentage
        auc_val = segment_metrics.get("auc")
        print(f"  AUC:                {auc_val:.4f}" if isinstance(auc_val, float) else "N/A")
        
    except Exception as e:
        # Catch any unexpected errors during the experiment, print traceback, and exit.
        print(f"\nFATAL ERROR during experiment: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        sys.exit(1)

    print("\nAll experiments completed.")


if __name__ == "__main__":
    # Entry point of the script. Calls the main function.
    main()