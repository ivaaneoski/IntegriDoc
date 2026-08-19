import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

class DocumentDataset(Dataset):
    def __init__(self, manifest_path: str, transform=None):
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
            
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        record = self.manifest[idx]
        img_path = record["image_path"]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # Label: 0 for REAL, 1 for TAMPERED
        label = 1 if record["label"] == "TAMPERED" else 0
        
        return image, torch.tensor(label, dtype=torch.long)
