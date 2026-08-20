import os
import sys
import json
import argparse
from pathlib import Path

def setup_external_dataset_manifest(
    dataset_name: str,
    raw_images_dir: str,
    output_manifest: str
):
    """
    Parses a downloaded external dataset directory (e.g. CASIA v2, Defacto, SROIE)
    and formats it into standard IntegriDoc manifest format.
    """
    raw_path = Path(raw_images_dir)
    if not raw_path.exists():
        print(f"Error: Directory {raw_images_dir} does not exist.")
        return
        
    images = list(raw_path.glob("**/*.jpg")) + list(raw_path.glob("**/*.png"))
    print(f"Found {len(images)} images in {raw_images_dir}")
    
    manifest = []
    for idx, img_p in enumerate(images):
        fname = img_p.name.lower()
        # Heuristic for common dataset naming conventions
        is_tampered = ("tampered" in fname or "forged" in fname or "tp" in fname or "tamp" in fname)
        label = "TAMPERED" if is_tampered else "REAL"
        
        manifest.append({
            "source_id": f"{dataset_name}_{idx:06d}",
            "image_id": img_p.stem,
            "image_path": str(img_p.resolve()),
            "label": label,
            "tamper_type": "external_forgery" if is_tampered else None,
            "dataset_origin": dataset_name
        })
        
    out_file = Path(output_manifest)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_file, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Successfully generated manifest with {len(manifest)} entries at {output_manifest}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format open external dataset for IntegriDoc")
    parser.add_argument("--dataset_name", type=str, default="casia_v2")
    parser.add_argument("--raw_images_dir", type=str, required=True)
    parser.add_argument("--output_manifest", type=str, default="data/manifests/external_dataset.json")
    args = parser.parse_args()
    
    setup_external_dataset_manifest(args.dataset_name, args.raw_images_dir, args.output_manifest)
