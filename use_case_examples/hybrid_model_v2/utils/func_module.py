import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

class FunctionWrapper(nn.Module):
    """
    A generic module that wraps a function.
    
    Args:
        func (Callable): A callable function (e.g. torch.nn.functional.relu)
                         that will be applied in the forward pass.
    """
    def __init__(self, func: Callable[[torch.Tensor], torch.Tensor]):
        super(FunctionWrapper, self).__init__()
        if not callable(func):
            raise ValueError("The provided function must be callable.")
        self.func = func

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Simply apply the wrapped function to the input tensor.
        print("On est al frerot")
        coucou = self.func(input)
        print(type(self.func))
        return coucou