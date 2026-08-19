import os
import sys
import json
import random
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data.generator import SyntheticDocumentGenerator
from src.data.tamper_ops import apply_text_replacement, apply_copy_move
from src.data.split import group_based_split, filter_manifest_by_source

def generate_dataset(num_sources: int = 100, output_dir: str = "data/synthetic"):
    out_dir = Path(output_dir)
    img_dir = out_dir / "images"
    mask_dir = out_dir / "masks"
    manifest_dir = Path("data/manifests")
    
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    generator = SyntheticDocumentGenerator()
    manifest = []
    source_ids = []
    
    print(f"Generating {num_sources} source documents and variants...")
    
    for i in range(num_sources):
        source_id = f"idcard_{i:05d}"
        source_ids.append(source_id)
        
        # Authentic
        auth_doc = generator.generate_id_card(source_id)
        auth_filename = f"{source_id}_auth.png"
        auth_doc["image"].save(img_dir / auth_filename)
        
        manifest.append({
            "source_id": source_id,
            "image_id": f"{source_id}_auth",
            "image_path": str(img_dir / auth_filename),
            "label": "REAL",
            "tamper_type": None
        })
        
        # Tampered 1 (Text Replacement)
        field_to_tamper = random.choice(["name", "dob", "id_number", "issue_date", "expiry_date"])
        tamp1 = apply_text_replacement(auth_doc, field_to_tamper, "TAMPERED")
        tamp1_filename = f"{source_id}_tamp1.png"
        tamp1_mask_filename = f"{source_id}_tamp1_mask.png"
        
        tamp1["image"].save(img_dir / tamp1_filename)
        tamp1["mask"].save(mask_dir / tamp1_mask_filename)
        
        manifest.append({
            "source_id": source_id,
            "image_id": f"{source_id}_tamp1",
            "image_path": str(img_dir / tamp1_filename),
            "mask_path": str(mask_dir / tamp1_mask_filename),
            "label": "TAMPERED",
            "tamper_type": "text_replacement"
        })
        
        # Tampered 2 (Copy Move)
        tamp2 = apply_copy_move(auth_doc)
        tamp2_filename = f"{source_id}_tamp2.png"
        tamp2_mask_filename = f"{source_id}_tamp2_mask.png"
        
        tamp2["image"].save(img_dir / tamp2_filename)
        tamp2["mask"].save(mask_dir / tamp2_mask_filename)
        
        manifest.append({
            "source_id": source_id,
            "image_id": f"{source_id}_tamp2",
            "image_path": str(img_dir / tamp2_filename),
            "mask_path": str(mask_dir / tamp2_mask_filename),
            "label": "TAMPERED",
            "tamper_type": "copy_move"
        })
        
    print("Splitting dataset...")
    train_ids, val_ids, test_ids = group_based_split(source_ids)
    
    train_manifest = filter_manifest_by_source(manifest, train_ids)
    val_manifest = filter_manifest_by_source(manifest, val_ids)
    test_manifest = filter_manifest_by_source(manifest, test_ids)
    
    with open(manifest_dir / "train.json", "w") as f:
        json.dump(train_manifest, f, indent=2)
    with open(manifest_dir / "val.json", "w") as f:
        json.dump(val_manifest, f, indent=2)
    with open(manifest_dir / "test.json", "w") as f:
        json.dump(test_manifest, f, indent=2)
        
    print(f"Generated Total: {len(manifest)} images")
    print(f"Train: {len(train_manifest)} | Val: {len(val_manifest)} | Test: {len(test_manifest)}")
    
if __name__ == "__main__":
    generate_dataset(num_sources=2000)
