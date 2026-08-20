import os
import sys
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data.diverse_tamper_ops import (
    apply_digital_text_overlay,
    apply_gray_patch_splicing,
    apply_stamp_copy_move,
    get_fallback_font
)
from src.data.real_world_augmentations import apply_camera_capture_pipeline
from src.data.split import group_based_split, filter_manifest_by_source

def generate_medical_prescription(source_id: str) -> Image.Image:
    w, h = 600, 800
    img = Image.new('RGB', (w, h), color=(248, 248, 246)) # Off-white paper
    draw = ImageDraw.Draw(img)
    
    font_title = get_fallback_font(22)
    font_sub = get_fallback_font(13)
    font_body = get_fallback_font(14)
    
    # Clinic Header
    clinic_names = ["ASTHA CLINIC & DIAGNOSTICS", "SANJIVINI MULTI SPECIALITY HOSPITAL", "METRO HEALTHCARE & CLINIC", "APEX WELLNESS CLINIC"]
    doctors = ["Dr. Arvind Kumar Sharma (MBBS, MD)", "Dr. Vikram Chaudhary (MBBS, MS)", "Dr. Radhika Nair (MD, D.ORTHO)", "Dr. Amit Kulkarni (MBBS, DNB)"]
    
    c_name = random.choice(clinic_names)
    doc_name = random.choice(doctors)
    
    # Header bar
    draw.rectangle([0, 0, w, 70], fill=(230, 240, 245))
    draw.text((20, 15), c_name, fill=(20, 60, 120), font=font_title)
    draw.text((20, 45), doc_name, fill=(80, 80, 80), font=font_sub)
    draw.line([(0, 70), (w, 70)], fill=(180, 190, 200), width=2)
    
    # Patient meta
    draw.text((20, 85), f"Patient: Patient_{source_id[:6]} | Age: {random.randint(18, 70)} | Gender: {random.choice(['M', 'F'])}", fill=(40, 40, 40), font=font_sub)
    draw.text((w - 150, 85), f"Date: 02/08/2026", fill=(40, 40, 40), font=font_sub)
    draw.line([(20, 110), (w - 20, 110)], fill=(210, 210, 210), width=1)
    
    # Clinical Notes (simulating doctor writing / report)
    draw.text((30, 130), "Rx / Clinical Evaluation:", fill=(20, 50, 100), font=font_body)
    
    diagnoses = ["Acute Bronchitis & Viral Fever", "Synovitis Both Knees - Bilateral", "Lumbar Spine Strain & Muscle Spasm", "Gastritis & Reflux Disorder"]
    draw.text((40, 165), f"Diagnosis: {random.choice(diagnoses)}", fill=(60, 60, 60), font=font_body)
    
    # Medicine table
    draw.rectangle([30, 210, w - 30, 460], outline=(180, 180, 180), width=1)
    draw.line([(30, 245), (w - 30, 245)], fill=(180, 180, 180), width=1)
    draw.text((40, 220), "Medicine / Formulation", fill=(30, 30, 30), font=font_body)
    draw.text((320, 220), "Dosage", fill=(30, 30, 30), font=font_body)
    draw.text((450, 220), "Duration", fill=(30, 30, 30), font=font_body)
    
    meds = [
        ("Tab. Pantocid 40mg", "1-0-0 (Empty Stomach)", "15 Days"),
        ("Cap. Becosules Z", "0-1-0 (After Meals)", "30 Days"),
        ("Tab. Dolo 650mg", "1-0-1 (SOS / Fever)", "5 Days"),
        ("Syp. Grilinctus 10ml", "1-1-1 (Twice Daily)", "7 Days"),
        ("Tab. Shelcal 500mg", "0-0-1 (Post Dinner)", "30 Days")
    ]
    
    y = 260
    for med, dose, dur in meds[:4]:
        draw.text((40, y), med, fill=(50, 50, 50), font=font_sub)
        draw.text((320, y), dose, fill=(50, 50, 50), font=font_sub)
        draw.text((450, y), dur, fill=(50, 50, 50), font=font_sub)
        y += 45
        
    # Signature & Stamp
    draw.text((w - 180, h - 100), "Doctor's Signature", fill=(100, 100, 100), font=font_sub)
    draw.ellipse([w - 170, h - 80, w - 40, h - 30], outline=(50, 80, 160), width=2)
    draw.text((w - 145, h - 60), "VERIFIED", fill=(50, 80, 160), font=font_sub)
    
    return img

def generate_invoice_receipt(source_id: str) -> Image.Image:
    w, h = 600, 750
    img = Image.new('RGB', (w, h), color=(252, 252, 250))
    draw = ImageDraw.Draw(img)
    
    font_title = get_fallback_font(20)
    font_sub = get_fallback_font(13)
    font_body = get_fallback_font(14)
    
    # Store Header
    stores = ["METROPOLIS PHARMA & DIAGNOSTICS", "GLOBAL MEDICAL SUPPLIES LTD", "HEALTHPLUS RETAIL CORP"]
    draw.text((20, 20), random.choice(stores), fill=(20, 20, 20), font=font_title)
    draw.text((20, 50), "Tax Invoice / Bill of Supply", fill=(90, 90, 90), font=font_sub)
    draw.text((w - 180, 20), f"Invoice #: INV-{source_id[:6].upper()}", fill=(60, 60, 60), font=font_sub)
    draw.text((w - 180, 45), "Date: 15/08/2026", fill=(60, 60, 60), font=font_sub)
    
    draw.line([(20, 80), (w - 20, 80)], fill=(160, 160, 160), width=2)
    
    # Items
    draw.text((30, 95), "Description", fill=(30, 30, 30), font=font_body)
    draw.text((380, 95), "Qty", fill=(30, 30, 30), font=font_body)
    draw.text((480, 95), "Amount ($)", fill=(30, 30, 30), font=font_body)
    draw.line([(20, 120), (w - 20, 120)], fill=(200, 200, 200), width=1)
    
    items = [
        ("Diagnostic Lab Blood Test", "1", "120.00"),
        ("X-Ray Imaging (Chest / Lumbar)", "2", "250.00"),
        ("Prescription Medication Bundle", "3", "85.50"),
        ("Physiotherapy Consultation", "1", "90.00")
    ]
    
    y = 135
    total = 0.0
    for desc, qty, amt in items:
        draw.text((30, y), desc, fill=(60, 60, 60), font=font_sub)
        draw.text((390, y), qty, fill=(60, 60, 60), font=font_sub)
        draw.text((490, y), amt, fill=(60, 60, 60), font=font_sub)
        total += float(amt)
        y += 40
        
    draw.line([(20, y + 10), (w - 20, y + 10)], fill=(160, 160, 160), width=2)
    draw.text((350, y + 25), "TOTAL AMOUNT:", fill=(20, 20, 20), font=font_body)
    draw.text((490, y + 25), f"${total:.2f}", fill=(20, 20, 20), font=font_body)
    
    return img

def generate_generalized_dataset(
    num_sources: int = 500,
    output_dir: str = "data/generalized_dataset",
    manifest_dir: str = "data/manifests"
):
    out_path = Path(output_dir)
    img_dir = out_path / "images"
    mask_dir = out_path / "masks"
    man_path = Path(manifest_dir)
    
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    man_path.mkdir(parents=True, exist_ok=True)
    
    manifest = []
    source_ids = []
    
    print(f"Generating {num_sources} multi-template source documents with realistic tampers & camera physics...")
    
    tamper_phrases = [
        "3 Days rest", "7 Days Medical Leave", "Vitamin D - 19.8 (Abnormal)",
        "Diagnosis: Unfit for Duty", "$1,450.00 Reimbursement Approved",
        "Tab. Morphine Sulfate 15mg", "14 Days Home Quarantine"
    ]
    
    for i in range(num_sources):
        source_id = f"doc_{i:05d}"
        source_ids.append(source_id)
        
        # 1. Pick document template
        doc_type = "medical" if random.random() < 0.6 else "invoice"
        if doc_type == "medical":
            base_img = generate_medical_prescription(source_id)
        else:
            base_img = generate_invoice_receipt(source_id)
            
        # --- A. Authentic Variant (With Camera Pipeline) ---
        auth_cam_img = apply_camera_capture_pipeline(base_img)
        auth_filename = f"{source_id}_auth.jpg"
        auth_cam_img.save(img_dir / auth_filename, "JPEG", quality=85)
        
        manifest.append({
            "source_id": source_id,
            "image_id": f"{source_id}_auth",
            "image_path": str(img_dir / auth_filename),
            "label": "REAL",
            "tamper_type": None,
            "doc_type": doc_type
        })
        
        # --- B. Tampered Variant 1: Digital Text Overlay / Splicing ---
        phrase = random.choice(tamper_phrases)
        tamp1_res = apply_digital_text_overlay(base_img, text=phrase, font_size=random.randint(15, 20))
        tamp1_cam_img = apply_camera_capture_pipeline(tamp1_res["image"])
        
        t1_filename = f"{source_id}_tamp1_text.jpg"
        t1_mask_filename = f"{source_id}_tamp1_mask.png"
        
        tamp1_cam_img.save(img_dir / t1_filename, "JPEG", quality=85)
        tamp1_res["mask"].save(mask_dir / t1_mask_filename)
        
        manifest.append({
            "source_id": source_id,
            "image_id": f"{source_id}_tamp1",
            "image_path": str(img_dir / t1_filename),
            "mask_path": str(mask_dir / t1_mask_filename),
            "label": "TAMPERED",
            "tamper_type": "digital_text_overlay",
            "doc_type": doc_type
        })
        
        # --- C. Tampered Variant 2: Gray Patch / Whiteout Splicing OR Copy-Move ---
        if random.random() < 0.5:
            patch_phrase = random.choice(["Tab. Calcitron 500mg", "30 Days Complete Bedrest", "$890.00 Subtotal"])
            tamp2_res = apply_gray_patch_splicing(base_img, patch_text=patch_phrase)
            tamp_type = "gray_patch_splicing"
        else:
            tamp2_res = apply_stamp_copy_move(base_img)
            tamp_type = "stamp_copy_move"
            
        tamp2_cam_img = apply_camera_capture_pipeline(tamp2_res["image"])
        t2_filename = f"{source_id}_tamp2_{tamp_type}.jpg"
        t2_mask_filename = f"{source_id}_tamp2_mask.png"
        
        tamp2_cam_img.save(img_dir / t2_filename, "JPEG", quality=85)
        tamp2_res["mask"].save(mask_dir / t2_mask_filename)
        
        manifest.append({
            "source_id": source_id,
            "image_id": f"{source_id}_tamp2",
            "image_path": str(img_dir / t2_filename),
            "mask_path": str(mask_dir / t2_mask_filename),
            "label": "TAMPERED",
            "tamper_type": tamp_type,
            "doc_type": doc_type
        })
        
    print(f"Total Images Generated: {len(manifest)}")
    
    # Split manifest
    train_ids, val_ids, test_ids = group_based_split(source_ids)
    train_man = filter_manifest_by_source(manifest, train_ids)
    val_man = filter_manifest_by_source(manifest, val_ids)
    test_man = filter_manifest_by_source(manifest, test_ids)
    
    with open(man_path / "train.json", "w") as f:
        json.dump(train_man, f, indent=2)
    with open(man_path / "val.json", "w") as f:
        json.dump(val_man, f, indent=2)
    with open(man_path / "test.json", "w") as f:
        json.dump(test_man, f, indent=2)
        
    print(f"Saved Manifests: Train ({len(train_man)}), Val ({len(val_man)}), Test ({len(test_man)})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_sources", type=int, default=300)
    parser.add_argument("--output_dir", type=str, default="data/generalized_dataset")
    args = parser.parse_args()
    
    generate_generalized_dataset(num_sources=args.num_sources, output_dir=args.output_dir)
