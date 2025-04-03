import torch
import torchvision


class ResNet(torch.nn.Module):
    """
    Model. Choose required size of ResNet via attributes.
    """
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50(weights=None)

    def forward(self, x):
        return self.model(x)
