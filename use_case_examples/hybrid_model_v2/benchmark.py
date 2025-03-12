#!/usr/bin/env python
"""
Benchmark script for MNIST model inference.

Usage:
    python benchmark.py -b <benchmark_name> -n <num_images> -m <model_name> -M <module_name1,module_module2,...> -s <snapshot_weight_path>

This script runs inference on N images from the MNIST test set,
measures inference times, computes classification metrics (accuracy,
precision, recall, F1, and confusion matrix) and saves the results in a JSON file.
It also saves the confusion matrix as a PNG image.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torchvision import datasets

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Import the hybrid model and the MNIST model.
from concrete.ml.torch.hybrid_model import HybridFHEMode, HybridFHEModel
from mnist_model import MNIST_CNN


def generate_report_dict(all_labels, all_preds, test_data):
    """
    Computes classification metrics and returns a dictionary.
    """
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cr = classification_report(all_labels, all_preds, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    return {
         "accuracy": accuracy,
         "precision": precision,
         "recall": recall,
         "f1": f1,
         "classification_report": cr,
         "confusion_matrix": cm.tolist()
    }


def save_confusion_matrix(cm, classes, output_filename):
    """
    Saves a heatmap of the confusion matrix as a PNG file.
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark MNIST model inference.")
    parser.add_argument("--benchmark-name", "-b", type=str, required=True, help="Name of the benchmark")
    parser.add_argument("--num-images", "-n", type=int, required=True, help="Number of images to run inference on")
    parser.add_argument("--model-name", "-m", type=str, help="Name of the model (used for the hybrid model)", default="mnist_cnn")
    parser.add_argument("--module-names", "-M", type=lambda s: [] if s.strip() == "" else s.split(','), required=True, help="Comma-separated list of module names")
    parser.add_argument("--snapshot", "-s", type=str, required=True, help="Path to snapshot weight file")
    args = parser.parse_args()

    # Set device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"Using device: {device}")

    # Instantiate the MNIST_CNN model and move to the appropriate device
    model = MNIST_CNN()
    model.load_state_dict(torch.load(f"snapshots/{args.snapshot}"))
    model.to(device)

    # Create the hybrid model wrapper to use the remote FHE server
    hybrid_model = HybridFHEModel(
        model,
        args.module_names,
        server_remote_address="http://0.0.0.0:8000",
        model_name=args.model_name,
        verbose=True,
    )
    # Initialize client connections (assumes a "clients" folder in the same directory)
    path_to_clients = Path(__file__).parent / "clients"
    hybrid_model.init_client(path_to_clients=path_to_clients)
    hybrid_model.set_fhe_mode(HybridFHEMode.REMOTE)

    # Load the MNIST test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    print("MNIST test dataset loaded.")

    num_images = args.num_images
    if num_images > len(test_dataset):
        print(f"Requested {num_images} images, but the test dataset only has {len(test_dataset)} images. Running on all available images.")
        num_images = len(test_dataset)

    inference_times = []
    all_true_labels = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        for i in range(num_images):
            sample_image, true_label = test_dataset[i]
            # Add a batch dimension and move to device
            input_tensor = sample_image.unsqueeze(0).to(device)
            start_time = time.time()
            # Run inference (using the original model; adjust to hybrid_model(input_tensor) if needed)
            output = model(input_tensor)
            end_time = time.time()
            inference_time = end_time - start_time
            inference_times.append(inference_time)

            predicted_label = output.argmax(dim=1).item()
            all_true_labels.append(true_label)
            all_preds.append(predicted_label)
            print(f"Image {i+1}/{num_images}: True label = {true_label}, Predicted = {predicted_label}, Time = {inference_time:.4f} sec")

    # Calculate time metrics
    total_time = sum(inference_times)
    min_time = min(inference_times)
    max_time = max(inference_times)
    avg_time = total_time / num_images

    # Generate classification report
    report_dict = generate_report_dict(all_true_labels, all_preds, test_dataset)

    # Create directory for benchmark outputs based on benchmark name
    benchmark_dir = os.path.join("benchmark", args.benchmark_name)
    os.makedirs(benchmark_dir, exist_ok=True)
    
    # Save the confusion matrix as PNG
    # Use test_dataset.classes if available; otherwise, assume 10 classes labeled 0-9.
    if hasattr(test_dataset, 'classes'):
        classes = test_dataset.classes
    else:
        classes = [str(i) for i in range(10)]
    cm_output_file = os.path.join(benchmark_dir, "confusion_matrix.png")
    save_confusion_matrix(report_dict["confusion_matrix"], classes, cm_output_file)
    print(f"Confusion matrix saved to {cm_output_file}")
    
    # Prepare the benchmark output
    benchmark_output = {
        "benchmark_name": args.benchmark_name,
        "num_images": num_images,
        "execution_time": {
            "min": min_time,
            "max": max_time,
            "average": avg_time,
            "total": total_time
        },
        "classification_report": report_dict,
        "confusion_matrix_png": cm_output_file
    }
    
    # Write benchmark results to a JSON file
    output_filename = os.path.join(benchmark_dir, "report.json")
    with open(output_filename, "w") as outfile:
        json.dump(benchmark_output, outfile, indent=4)
    print(f"Benchmark results saved to {output_filename}")


if __name__ == "__main__":
    main()
