import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
from src.forensics.ela import calculate_ela
from src.forensics.residuals import calculate_gaussian_residual

class DocumentDataset(Dataset):
    def __init__(self, manifest_path: str, transform=None, return_forensics: bool = False):
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
            
        self.return_forensics = return_forensics
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
            image_tensor = self.transform(image)
            
        # Label: 0 for REAL, 1 for TAMPERED
        label = 1 if record["label"] == "TAMPERED" else 0
        
        if not self.return_forensics:
            return image_tensor, torch.tensor(label, dtype=torch.long)
            
        # Generate Forensics on the fly
        ela_res = calculate_ela(image)
        ela_img = ela_res["ela_image"].convert("L")
        
        res_res = calculate_gaussian_residual(image)
        res_img = res_res["residual_image"].convert("L")
        
        forensic_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        ela_tensor = forensic_transform(ela_img)
        res_tensor = forensic_transform(res_img)
        
        forensics_tensor = torch.cat((ela_tensor, res_tensor), dim=0)
        
        return image_tensor, forensics_tensor, torch.tensor(label, dtype=torch.long)
