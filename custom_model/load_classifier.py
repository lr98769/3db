## From https://3db.github.io/3db/usage/custom_inference.html#updating-the-configuration-file
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoadClassifier(nn.Module):
    def __init__(self, output_dim, path_to_model):
        super().__init__()

        # model must be saved with a structure
        self.model = torch.load(path_to_model)
        self.model.eval()

    def forward(self, x):
        return self.model.forward(x)