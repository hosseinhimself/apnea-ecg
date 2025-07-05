import yaml
import argparse
from dataset_loader import DatasetLoader
from model import Model
from trainer import Trainer
from evaluation import Evaluation

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

def run_experiment(config, segment_length):
    print(f"\n=== Running experiment for segment length: {segment_length}s ===")
    
    # Prepare dataset
    dataset_loader = DatasetLoader(config)
    dataset_loader.load_raw_data()
    segments, labels = dataset_loader.preprocess_and_segment()
    
    # Get data loaders
    train_loader, val_loader, test_loader = dataset_loader.get_dataloaders(
        segment_length=segment_length,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )
    
    # Initialize model
    model = Model(config)
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()
    
    # Evaluate
    evaluator = Evaluation(model, test_loader)
    segment_metrics = evaluator.evaluate()
    recording_metrics = evaluator.aggregate_recording_results()
    
    # Print results
    print("\n=== Results ===")
    print("Segment-wise Metrics:")
    for k, v in segment_metrics.items():
        print(f"{k:15}: {v:.4f}")
    
    print("\nRecording-wise Metrics:")
    for k, v in recording_metrics.items():
        print(f"{k:15}: {v:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    segment_lengths = config["dataset"].get("segment_lengths", [60])
    
    for seg_len in segment_lengths:
        run_experiment(config, seg_len)

if __name__ == "__main__":
    main()