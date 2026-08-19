import os
import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.generator import SyntheticDocumentGenerator
from src.data.tamper_ops import apply_text_replacement, apply_copy_move
from src.forensics.ela import calculate_ela
from src.forensics.residuals import calculate_gaussian_residual, calculate_laplacian_residual

def main():
    print("=== IntegriDoc Phase 0 Smoke Test ===")
    
    # 1. Setup output directory
    out_dir = Path("results/smoke_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    try:
        with open("configs/default.yaml", "r") as f:
            config = yaml.safe_load(f)
            ela_quality = config.get("data", {}).get("jpeg_quality_ela", 90)
    except Exception as e:
        print(f"Failed to load config: {e}")
        ela_quality = 90
        
    print(f"1. Generating authentic document...")
    generator = SyntheticDocumentGenerator(seed=12345)
    auth_doc = generator.generate_id_card(source_id="smoke_test_001")
    
    auth_path = out_dir / "01_authentic.png"
    auth_doc["image"].save(auth_path)
    print(f"   -> Saved authentic document to {auth_path}")
    
    print(f"2. Applying tampering (text replacement)...")
    tampered_doc = apply_text_replacement(auth_doc, field_name="dob", new_value="01/01/1990")
    
    tamp_path = out_dir / "02_tampered.png"
    tampered_doc["image"].save(tamp_path)
    
    mask_path = out_dir / "03_tamper_mask.png"
    tampered_doc["mask"].save(mask_path)
    print(f"   -> Saved tampered image and mask to {out_dir}")
    
    print(f"3. Running ELA analysis...")
    ela_res = calculate_ela(tampered_doc["image"], quality=ela_quality)
    ela_path = out_dir / "04_ela_heatmap.png"
    ela_res["ela_image"].save(ela_path)
    print(f"   -> ELA max difference: {ela_res['max_diff']}")
    print(f"   -> Saved ELA map to {ela_path}")
    
    print(f"4. Running Residual analysis...")
    gauss_res = calculate_gaussian_residual(tampered_doc["image"])
    gauss_path = out_dir / "05_gaussian_residual.png"
    gauss_res["residual_image"].save(gauss_path)
    
    lap_res = calculate_laplacian_residual(tampered_doc["image"])
    lap_path = out_dir / "06_laplacian_residual.png"
    lap_res["residual_image"].save(lap_path)
    
    print(f"   -> Gaussian residual score: {gauss_res['score']:.4f}")
    print(f"   -> Laplacian residual score: {lap_res['score']:.4f}")
    print(f"   -> Saved residual maps to {out_dir}")
    
    print("\n=== Smoke Test Complete! ===")
    print(f"Review artifacts in {out_dir}")

if __name__ == "__main__":
    main()
