# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import torch.nn as nn
# import torch.optim as optim
# from torch.cuda.amp import autocast, GradScaler
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import matplotlib.pyplot as plt
# from PIL import Image
# import mlflow
# from mlflow.models import infer_signature
# from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path

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
    
    
def get_best_model(device="cpu"):
    model = CNN().to(device)

    model_path = Path(__file__).parent / "best_model.pth"
    state = torch.load(model_path, map_location=device)

    model.load_state_dict(state)
    model.eval() # set to evaluation mode not training
    return model

def transform_image(image):
    #new transform so new data matches format of train/test data
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.functional.invert,
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform(image).unsqueeze(0)  # add batch dimension

def predict_digit(image_path, model):
    # load and preprocess image
    if not isinstance(image_path, (str, Path)):
        raise TypeError("image_path must be a path")

    if not Path(image_path).exists():
        raise FileNotFoundError(f"{image_path} does not exist")
    image = Image.open(image_path)
    image = transform_image(image)

    # prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        probabilities = torch.nn.functional.softmax(output, dim=1)

    return predicted.item(), probabilities.squeeze().tolist()

def show_transformed_image(image_path):
    # load and preprocess image
    image = Image.open(image_path)
    image = transform_image(image)
    plt.imshow(image.squeeze(), cmap='gray')#show new preprocessed image
    plt.show()