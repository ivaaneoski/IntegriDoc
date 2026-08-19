import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class SmallCNN(nn.Module):
    def __init__(self, in_channels: int, out_features: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, out_features)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ForensicFusionNet(nn.Module):
    def __init__(self, use_ela=True, use_residual=True):
        super().__init__()
        self.use_ela = use_ela
        self.use_residual = use_residual
        
        # Branch 1: RGB Backbone
        self.rgb_backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        rgb_out_features = self.rgb_backbone.fc.in_features
        self.rgb_backbone.fc = nn.Identity() # Remove the final layer
        
        # Branch 2: Forensic Backbone
        forensic_channels = 0
        if use_ela: forensic_channels += 1
        if use_residual: forensic_channels += 1
            
        forensic_out_features = 128
        
        if forensic_channels > 0:
            self.forensic_backbone = SmallCNN(in_channels=forensic_channels, out_features=forensic_out_features)
        else:
            self.forensic_backbone = None
            forensic_out_features = 0
            
        # Fusion Head
        total_features = rgb_out_features + forensic_out_features
        self.fusion_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(total_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2) # Binary classification
        )

    def forward(self, rgb, forensics=None):
        f_rgb = self.rgb_backbone(rgb)
        
        if self.forensic_backbone is not None and forensics is not None:
            f_for = self.forensic_backbone(forensics)
            fused = torch.cat((f_rgb, f_for), dim=1)
        else:
            fused = f_rgb
            
        return self.fusion_head(fused)
