import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

# from torchvision.models import SwinTransformer


class BaseCNNPredictor(nn.Module):
    def __init__(self, num_classes: int = 1, img_size: int = 64):
        super(BaseCNNPredictor, self).__init__()

        self.img_size = img_size

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.Flatten(),
        )

        self.flatten_size = self._determine_flatten_size()

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        self.prediction_head = nn.Sequential(nn.Linear(32, num_classes), nn.Sigmoid())

    def _determine_flatten_size(self) -> int:
        x = torch.randn(1, 3, self.img_size, self.img_size)
        x = self.cnn(x)
        return x.view(1, -1).size(1)

    def forward(self, x) -> torch.Tensor:
        x = self.cnn(x)
        x = self.fc(x)
        x = self.prediction_head(x)
        return x


class SelfCompressingCNN(nn.Module):
    """A simple implementation based on the paper: https://arxiv.org/pdf/2301.13142"""


def QFunc(x, e, b):
    return 2**e * round(min(max(2 ** (-e) * x, -(2 ** (b - 1))), 2 ** (b - 1) - 1))
