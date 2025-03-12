#!/usr/bin/env python
"""
Compile and convert the MNIST_CNN model to FHE.

This script loads a sample batch from the MNIST training set, instantiates the hybrid CNN model,
and then compiles the remote (FHE) parts using the concrete-ml HybridFHEModel converter.
The remote submodules (by default) are:
    - remote1: first convolution
    - pool1: pooling after conv1
    - remote2: second convolution
    - pool2: pooling after conv2
    - flatten: flattening operation
    - remote3: first fully-connected layer
    - remote4: final fully-connected layer

Usage example:
    python compile_mnist.py --model-name "mnist_cnn" --module-names "remote1,pool1,remote2,pool2,flatten,remote3,remote4" --snapshot "snapshots/pure.pth" --batch-size 32 --num-samples 32
"""

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import List, Union

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from concrete.ml.torch.hybrid_model import HybridFHEModel
from mnist_model import MNIST_CNN
from utils.compression import prune_model


def compile_model(
    model_name: str,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    module_names: Union[str, List],
    models_dir: Path,
):
    """
    Create a HybridFHEModel from the provided model, compile it with given inputs,
    and save the compiled model to the specified directory.
    """
    # Create a hybrid model wrapper
    hybrid_model = HybridFHEModel(model, module_names)
    # Compile the model (using 8-bit quantization for FHE execution)
    hybrid_model.compile_model(inputs, n_bits=8)

    from torchsummary import summary

    # Print the summary of the hybrid model
    print("Hybrid Model Summary:")
    summary(hybrid_model.model, input_size=inputs.shape[1:])

    # Ensure the directory exists and prepare model-specific subdirectory
    models_dir.mkdir(exist_ok=True)
    model_dir = models_dir / model_name
    print(f"Saving compiled model to {model_dir}")
    via_mlir = bool(int(os.environ.get("VIA_MLIR", 1)))
    hybrid_model.save_and_clear_private_info(model_dir, via_mlir=via_mlir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile a hybrid model.")
    parser.add_argument("--model-name", "-m", type=str, required=True, help="Name of the model")
    parser.add_argument(
        "--module-names",
        "-M",
        type=lambda s: [] if s.strip() == "" else s.split(','),
        required=True,
        help="Comma-separated list of module names"
    )
    parser.add_argument(
        "--snapshot",
        "-s",
        type=str,
        required=True,
        help="Path to snapshot weight file"
    )
    args = parser.parse_args()

    model_name = args.model_name
    module_names = args.module_names
    snapshot_path = args.snapshot
    print(f"Module names: {module_names}")
    print(f"Snapshot weight file: {snapshot_path}")

    num_samples = 32
    data_root = "../data"
    models_dir = Path(__file__).parent / os.environ.get("MODELS_DIR_NAME", "compiled_models")
    models_dir.mkdir(exist_ok=True)

    # Use CPU for compilation
    device = "cpu"
    print(f"Using device: {device}")

    # Load a sample of MNIST data with the same transforms as in training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    sample_batch = next(iter(loader))
    sample_images, _ = sample_batch  # We only need the images for the forward pass
    sample_images = sample_images.to(device)

    # Instantiate the MNIST CNN hybrid model and move to CPU
    model = MNIST_CNN()
    model.load_state_dict(torch.load(f"snapshots/{args.snapshot}"))
    model.to(device)

    # model = prune_model(model) # comment if no pruning

    # Compile the model with the provided sample input
    # Using deepcopy to avoid in-place modifications during multiple compilations
    compile_model(
        model_name,
        deepcopy(model),
        sample_images,
        module_names,
        models_dir
    )
