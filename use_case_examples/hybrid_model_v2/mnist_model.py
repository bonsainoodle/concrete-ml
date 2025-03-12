import torch.nn as nn
import torch.nn.functional as F


class MNIST_CNN_Hybrid(nn.Module):
    """
    A hybrid MNIST CNN where all linear and pooling operations are executed
    remotely (and can be FHE-compiled), while the activation function is run
    on the client in clear.
    
    The remote submodules are:
      - remote1: first convolution
      - pool1: pooling after conv1
      - remote2: second convolution
      - pool2: pooling after conv2
      - flatten: flattening operation
      - remote3: first fully-connected layer
      - remote4: final fully-connected layer
    """
    def __init__(self, model_size=1, activation=lambda x: x**2):
        super(MNIST_CNN_Hybrid, self).__init__()
        assert 1 <= model_size <= 10, "model_size should be between 1 and 10"
        self.activation = activation  # client-side activation

        base_channels = 8
        scale = 0.25  # scaling factor for intermediate layers
        conv1_out = max(1, int(base_channels * model_size * scale))
        conv2_out = max(1, int(base_channels * model_size * 2 * scale))
        fc1_out   = max(1, int(128 * model_size * scale))

        # Remote submodules: these layers will be compiled for FHE execution.
        self.remote1 = nn.Conv2d(1, conv1_out, kernel_size=3, padding=1)
        self.pool1   = nn.AvgPool2d(kernel_size=2, stride=2)
        self.remote2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=3, padding=1)
        self.pool2   = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.remote3 = nn.Linear(conv2_out * 7 * 7, fc1_out)
        self.remote4 = nn.Linear(fc1_out, 10)

    def forward(self, x):
        # --- Round 1 ---
        # Remote: conv1
        x = self.remote1(x)
        # Client: activation (in clear)
        x = self.activation(x)
        # Remote: pooling
        x = self.pool1(x)
        
        # --- Round 2 ---
        # Remote: conv2
        x = self.remote2(x)
        # Client: activation
        x = self.activation(x)
        # Remote: pooling
        x = self.pool2(x)
        
        # --- Round 3 ---
        # Remote: flatten and first FC layer
        x = self.flatten(x)
        x = self.remote3(x)
        # Client: activation
        x = self.activation(x)
        # Remote: final FC layer
        x = self.remote4(x)
        return x