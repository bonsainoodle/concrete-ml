import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


# Function to prune a model copy with a given pruning amount
def prune_model(model, amount, seed = None):
    pruned_model = copy.deepcopy(model)
    if seed is not None:
        torch.manual_seed(seed)
    # Apply random unstructured pruning to each Linear layer
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Linear):
            prune.random_unstructured(module, name='weight', amount=amount)
            # Optionally, remove the pruning reparameterization so the model is ready for inference
            prune.remove(module, 'weight')
    return pruned_model