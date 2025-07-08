## predict_single.py

import torch
import yaml
import numpy as np
import random

# Import necessary classes from other project files
from model import Model
from dataset_loader import DatasetLoader

def load_config(config_path: str = "config.yaml"):
    """Loads the config file for model settings."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def predict_single_sample(model_path: str = "best_model.pth"):
    """
    Loads a random sample from the test dataset and displays the model's
    prediction alongside the actual label.
    """
    print("--- Starting prediction process for a single sample ---")

    # 1. Load config
    config = load_config()
    print("Config file loaded successfully.")

    # 2. Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(params=config)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'. Please train the model first.")
        return
        
    model.to(device)
    model.eval()  # Set the model to evaluation mode (very important)
    print(f"Model from '{model_path}' loaded successfully and moved to device '{device}'.")

    # 3. Prepare the test dataset
    # We only need the test data loader to get a sample
    print("Preparing the test dataset...")
    dataset_loader = DatasetLoader(config)
    _, _, test_loader = dataset_loader.get_dataloaders()
    print("Test dataset prepared.")

    # 4. Select a random sample from the test dataset
    # Convert the dataset to a list to be able to pick a random sample
    test_dataset_list = list(test_loader.dataset)
    if not test_dataset_list:
        print("Error: Test dataset is empty.")
        return
        
    random_index = random.randint(0, len(test_dataset_list) - 1)
    sample_ecg, actual_label_idx = test_dataset_list[random_index]
    
    print(f"\nA random sample with index {random_index} was selected from the test set.")

    # 5. Perform prediction
    # The sample needs a batch dimension, so we add one
    # Model input shape: (batch_size, sequence_length, channels) -> (1, 18000, 1)
    sample_ecg_batch = sample_ecg.unsqueeze(0).to(device)

    with torch.no_grad():  # Disable gradient calculation for faster inference
        logits = model(sample_ecg_batch)
        # Convert logits to probabilities using softmax
        probabilities = torch.softmax(logits, dim=1)
        # Choose the class with the highest probability
        predicted_label_idx = torch.argmax(probabilities, dim=1).item()

    # 6. Display the results
    # Define class names for better display
    class_names = {0: "Normal", 1: "Apnea"}

    actual_label_name = class_names.get(actual_label_idx.item(), "Unknown")
    predicted_label_name = class_names.get(predicted_label_idx, "Unknown")

    print("\n" + "="*40)
    print("Final Prediction Result:")
    print(f"    - Actual Label:      {actual_label_idx.item()} -> {actual_label_name}")
    print(f"    - Predicted Label:   {predicted_label_idx} -> {predicted_label_name}")
    print("-"*40)
    
    # Display more details (probabilities)
    prob_normal = probabilities[0, 0].item() * 100
    prob_apnea = probabilities[0, 1].item() * 100
    print("Details of probabilities calculated by the model:")
    print(f"    - Probability of being Normal: {prob_normal:.2f}%")
    print(f"    - Probability of being Apnea:  {prob_apnea:.2f}%")
    print("="*40)

    if actual_label_idx.item() == predicted_label_idx:
        print("\n✅ Prediction was correct!")
    else:
        print("\n❌ Prediction was incorrect.")


if __name__ == "__main__":
    predict_single_sample()
