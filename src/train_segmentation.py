import os
import sys
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data.dataset import DocumentDataset
from src.models.segmentation import DocumentSegmentationModel
from src.localization.metrics import calculate_iou, calculate_dice

def train(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Setup DataLoader
    train_dataset = DocumentDataset("data/manifests/train.json", return_mask=True)
    val_dataset = DocumentDataset("data/manifests/val.json", return_mask=True)
    
    batch_size = config.get("batch_size", 16)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Setup Model
    model = DocumentSegmentationModel(encoder_name="resnet18", encoder_weights="imagenet").to(device)
    
    # 3. Setup Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    lr = float(config.get("lr", 1e-3))
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    epochs = config.get("epochs", 5)
    
    # Setup Checkpoint Dir
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"results/segmentation/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting segmentation training for {epochs} epochs. Output dir: {out_dir}")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images, masks, _ = batch
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images, masks, _ = batch
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                
                # Metrics
                preds = torch.sigmoid(outputs).cpu().numpy()
                true_masks = masks.cpu().numpy()
                
                batch_iou = calculate_iou(preds, true_masks)
                batch_dice = calculate_dice(preds, true_masks)
                
                val_iou += batch_iou
                val_dice += batch_dice
                num_batches += 1
                
        val_loss /= len(val_loader.dataset)
        val_iou /= num_batches
        val_dice /= num_batches
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val IoU: {val_iou:.4f} - Val Dice: {val_dice:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / "best_seg.pt")
            
            metrics = {
                "val_loss": val_loss,
                "val_iou": val_iou,
                "val_dice": val_dice
            }
            with open(out_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
                
    print(f"Training complete. Best model saved to {out_dir / 'best_seg.pt'}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_segmentation.yaml")
    args = parser.parse_args()
    train(args.config)
