import torch
import torchvision


class ResNet(torch.nn.Module):
    """
    Model.
    """
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50(weights=None)

    def forward(self, x):
        return self.model(x)
