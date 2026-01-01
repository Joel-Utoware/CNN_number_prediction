import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from PIL import Image
import mlflow
from mlflow.models import infer_signature


class CNN(nn.Module):
    def __init__(self, input_size=(28, 28)):
        super().__init__()


        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),#layers to extract features
            nn.BatchNorm2d(32),#accelerates training & improves stability
            nn.ReLU(),# add non-linearity
            nn.MaxPool2d(kernel_size=2),#reduce dimensionality

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )


        self._to_linear = self._get_flattened_size(input_size)

        # fully connected (fc) layers for classification
        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 256),
            nn.ReLU(),
            nn.Dropout(0.3),# prevents overfitting
            nn.Linear(256, 10)
        )

    # compute flattened feature size for FC layer instead of manually calculating it
    def _get_flattened_size(self, input_size):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_size)  # batch size = 1
            out = self.features(dummy)
            return out.view(1, -1).shape[1]  # return flattened feature size

    #forward pass of model (how it runs)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
    
    
def get_best_model():
    best_model = CNN()
    best_model.load_state_dict(torch.load('best_model.pth'))
    best_model.eval()  # set to evaluation mode not training
    return best_model
