import torch
import torch.nn as nn
import torch.nn.functional as F

class GeLU(nn.Module):
    def forward(self, x):
        return F.gelu(x)