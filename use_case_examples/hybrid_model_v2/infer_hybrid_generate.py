#!/usr/bin/env python
"""
Showcase for the hybrid model converter for MNIST_CNN_Hybrid.

This script loads the configuration dumped during compilation, instantiates the MNIST_CNN_Hybrid model,
and configures it to use a remote FHE server for its remote submodules. It then loads the MNIST test dataset
and interactively lets you choose a sample (by index) to run inference on, printing the predicted digit,
the raw model output, and the inference time.

Usage example:
    python showcase_mnist.py --snapshot "snapshots/pure.pth"
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torchvision import datasets

from concrete.ml.torch.hybrid_model import HybridFHEMode, HybridFHEModel
from mnist_model import MNIST_CNN

# Environment configuration
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Showcase for the hybrid model converter for MNIST_CNN_Hybrid."
    )
    parser.add_argument(
        "--snapshot",
        "-s",
        type=str,
        required=True,
        help="Path to snapshot weight file"
    )
    args = parser.parse_args()
    
    # Load configuration dumped during compilation
    with open("configuration.json", "r") as file:
        configuration = json.load(file)

    module_names = configuration["module_names"]
    model_name = configuration["model_name"]
    model_name_no_special_char = configuration["model_name_no_special_char"]

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"Using device: {device}")

    # Instantiate the MNIST_CNN_Hybrid model
    model = MNIST_CNN()
    model.load_state_dict(torch.load(f"snapshots/{args.snapshot}"))
    model.to(device)

    # Create the hybrid model wrapper to use the remote FHE server
    hybrid_model = HybridFHEModel(
        model,
        module_names,
        server_remote_address="http://0.0.0.0:8000",
        model_name=model_name_no_special_char,
        verbose=True,
    )
    # Initialize client connections using the "clients" folder
    path_to_clients = Path(__file__).parent / "clients"
    hybrid_model.init_client(path_to_clients=path_to_clients)
    hybrid_model.set_fhe_mode(HybridFHEMode.REMOTE)

    # Load MNIST test dataset for interactive inference demonstration
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    print("MNIST test dataset loaded.")

    # For demonstration, we use sample index 0 (this can be made interactive if needed)
    sample_index = 0

    sample_image, true_label = test_dataset[sample_index]

    # Add batch dimension and move to device
    input_tensor = sample_image.unsqueeze(0).to(device)

    print(f"Processing sample index {sample_index} (true label: {true_label}).")
    print("*" * 30)

    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    end_time = time.time()
    inference_time = end_time - start_time

    # Determine the predicted digit
    predicted_digit = output.argmax(dim=1).item()

    print(f"Inference time: {inference_time:.4f} seconds")
    print(f"Predicted digit: {predicted_digit}")
    print(f"Raw model output: {output}")
    print("*" * 30)
