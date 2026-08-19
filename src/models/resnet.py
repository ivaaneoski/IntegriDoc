import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Binary(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        if pretrained:
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = resnet18(weights=None)
            
        # Replace the final fully connected layer for binary classification
        num_ftrs = self.backbone.fc.in_features
        
        # Following the spec: Dropout + Linear
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, 2)
        )

    def forward(self, x):
        return self.backbone(x)
