import torch
import torch.nn as nn


class IdiotPredictor(nn.Module):
    def __init__(self, img_size: int, num_classes: int = 1):
        super(IdiotPredictor, self).__init__()
        self.fc = nn.Sequential(nn.Linear(img_size * img_size * 3, num_classes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class AlwaysBenign(nn.Module):
    """simple pl"""

    def __init__(self):
        super(AlwaysBenign, self).__init__()
