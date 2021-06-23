## From https://3db.github.io/3db/usage/custom_inference.html#updating-the-configuration-file
import torch
import torch.nn as nn
import torch.nn.functional as F
from robustness.model_utils import make_and_restore_model
from robustness.datasets import ImageNet

class LoadResNet50(nn.Module):
    def __init__(self, output_dim, path_to_model):
        super().__init__()

        ds = ImageNet('/tmp')
        self.model, _ = make_and_restore_model(arch='resnet50', resume_path=path_to_model, dataset=ds)
        self.model.eval()

    def forward(self, x):
        return self.model.forward(x)[0]