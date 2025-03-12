import torch.nn as nn


class MNIST_CNN(nn.Module):
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
    def __init__(self):
        super(MNIST_CNN, self).__init__()

        self.activation = nn.ReLU()  # client-side activation
        self.flatten = nn.Flatten() #

       # Base number of channels without scaling
        base_channels = 8

        # Set sizes directly without scaling
        conv1_out = base_channels
        conv2_out = base_channels * 2
        fc1_out = 128

        # Remote submodules: these layers will be compiled for FHE execution.
        self.remote1 = nn.Conv2d(1, conv1_out, kernel_size=3, padding=1)
        self.pool1   = nn.AvgPool2d(kernel_size=2, stride=2)
        self.remote2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=3, padding=1)
        self.pool2   = nn.AvgPool2d(kernel_size=2, stride=2)
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