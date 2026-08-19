import torch.nn as nn
import segmentation_models_pytorch as smp

class DocumentSegmentationModel(nn.Module):
    def __init__(self, encoder_name: str = "resnet18", encoder_weights: str = "imagenet"):
        """
        Initializes a U-Net for binary segmentation of tampered regions.
        Output is a 1-channel spatial mask logit.
        """
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1, # 1 class (binary mask: tampered or not)
            activation=None # We will use BCEWithLogitsLoss
        )
        
    def forward(self, x):
        return self.model(x)
