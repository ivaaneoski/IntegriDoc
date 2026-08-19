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
from src.models.resnet import ResNet18Binary
from src.evaluation.metrics import calculate_metrics, plot_confusion_matrix

def train(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Setup DataLoader
    train_dataset = DocumentDataset("data/manifests/train.json")
    val_dataset = DocumentDataset("data/manifests/val.json")
    
    batch_size = config.get("batch_size", 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Setup Model
    model = ResNet18Binary(pretrained=True).to(device)
    
    # 3. Setup Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    lr_head = float(config.get("lr_head", 1e-3))
    optimizer = optim.AdamW(model.parameters(), lr=lr_head)
    epochs = config.get("epochs", 5)
    
    # Setup Checkpoint Dir
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"results/runs/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting training for {epochs} epochs. Output dir: {out_dir}")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        
        metrics = calculate_metrics(all_labels, all_preds, all_probs)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {metrics['accuracy']:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / "best.pt")
            
            # Save metrics
            with open(out_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
                
            plot_confusion_matrix(all_labels, all_preds, str(out_dir / "confusion_matrix.png"))
            
    print(f"Training complete. Best model saved to {out_dir / 'best.pt'}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_resnet18.yaml")
    args = parser.parse_args()
    train(args.config)
