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

def generate_id_card(source_id: str) -> Image.Image:
    w, h = 800, 500
    img = Image.new('RGB', (w, h), color=(240, 245, 255)) # Light blue tint
    draw = ImageDraw.Draw(img)
    
    font_title = get_fallback_font(24)
    font_sub = get_fallback_font(16)
    font_body = get_fallback_font(18)
    
    # Header
    draw.rectangle([0, 0, w, 80], fill=(20, 60, 140))
    draw.text((w // 2 - 150, 20), "NATIONAL IDENTITY CARD", fill=(255, 255, 255), font=font_title)
    draw.text((w // 2 - 100, 50), "REPUBLIC OF GENERIC", fill=(200, 220, 255), font=font_sub)
    
    # Photo placeholder
    draw.rectangle([30, 100, 200, 320], fill=(200, 200, 200), outline=(100, 100, 100), width=2)
    draw.text((85, 200), "PHOTO", fill=(100, 100, 100), font=font_body)
    
    # Details
    draw.text((230, 110), "ID Number:", fill=(100, 100, 100), font=font_sub)
    draw.text((350, 110), f"{random.randint(1000,9999)} {random.randint(1000,9999)} {random.randint(1000,9999)}", fill=(20, 20, 20), font=font_title)
    
    draw.text((230, 160), "Full Name:", fill=(100, 100, 100), font=font_sub)
    first_names = ["John", "Michael", "Sarah", "Emily", "David"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones"]
    draw.text((350, 160), f"{random.choice(first_names)} {random.choice(last_names)}", fill=(20, 20, 20), font=font_body)
    
    draw.text((230, 210), "DOB:", fill=(100, 100, 100), font=font_sub)
    draw.text((350, 210), f"{random.randint(1, 28)}/0{random.randint(1,9)}/19{random.randint(70,99)}", fill=(20, 20, 20), font=font_body)
    
    draw.text((230, 260), "Gender:", fill=(100, 100, 100), font=font_sub)
    draw.text((350, 260), random.choice(["MALE", "FEMALE"]), fill=(20, 20, 20), font=font_body)
    
    draw.text((230, 310), "Address:", fill=(100, 100, 100), font=font_sub)
    draw.text((350, 310), f"{random.randint(10, 999)} Main Street, City", fill=(20, 20, 20), font=font_body)
    
    # Bottom Bar / Machine Readable Zone
    mrz = f"I<GEN{random.randint(100000,999999)}<<<<<<<<<<<<<<<\n{random.choice(last_names).upper()}<<{random.choice(first_names).upper()}<<<<<<<<<<<<<<<<<<<<"
    draw.rectangle([0, h - 80, w, h], fill=(255, 255, 255))
    draw.text((20, h - 70), mrz, fill=(0, 0, 0), font=get_fallback_font(22))
    
    return img

def generate_screenshot(source_id: str) -> Image.Image:
    w, h = 1080, 1920 # Mobile screen ratio
    img = Image.new('RGB', (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    font_large = get_fallback_font(48)
    font_med = get_fallback_font(36)
    font_small = get_fallback_font(28)
    
    # Status bar
    draw.rectangle([0, 0, w, 80], fill=(240, 240, 240))
    draw.text((40, 20), "12:00", fill=(0, 0, 0), font=font_small)
    draw.text((w - 150, 20), "5G | 100%", fill=(0, 0, 0), font=font_small)
    
    # App Header
    apps = ["BankingApp", "ChatApp", "CryptoWallet"]
    app_choice = random.choice(apps)
    draw.rectangle([0, 80, w, 220], fill=(40, 120, 200))
    draw.text((40, 120), app_choice, fill=(255, 255, 255), font=font_large)
    
    if app_choice == "BankingApp":
        draw.text((40, 300), "Current Balance", fill=(100, 100, 100), font=font_med)
        draw.text((40, 360), f"${random.randint(100, 9999):,}.{random.randint(10, 99)}", fill=(0, 0, 0), font=font_large)
        
        # Transactions
        y = 500
        for i in range(5):
            draw.text((40, y), f"Payment to {random.choice(['Amazon', 'Uber', 'Walmart'])}", fill=(0, 0, 0), font=font_med)
            draw.text((w - 200, y), f"-${random.randint(10, 200)}.00", fill=(200, 40, 40), font=font_med)
            draw.line([(40, y + 60), (w - 40, y + 60)], fill=(220, 220, 220), width=2)
            y += 100
    else:
        # Chat App
        y = 300
        for i in range(6):
            is_me = random.choice([True, False])
            msg = f"Message {i+1} content goes here..."
            x = w - 400 if is_me else 40
            bg = (200, 240, 200) if is_me else (240, 240, 240)
            draw.rectangle([x, y, x + 360, y + 100], fill=bg, outline=(200, 200, 200))
            draw.text((x + 20, y + 30), msg, fill=(0, 0, 0), font=font_med)
            y += 150
            
    return img

def integrate_real_world_documents(manifest: list, img_dir: Path, mask_dir: Path, num_real: int = 500, source_ids_pool: list = None):
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets library not found. Skipping RVL-CDIP real-world integration.")
        return manifest, source_ids_pool
        
    print(f"Streaming {num_real} real-world documents from RVL-CDIP...")
    
    try:
        ds = load_dataset("rvl_cdip", split="train", streaming=True)
    except Exception as e:
        print(f"Failed to load rvl_cdip: {e}")
        return manifest, source_ids_pool
        
    count = 0
    for item in ds:
        if count >= num_real:
            break
            
        source_id = f"real_rvl_{count:05d}"
        if source_ids_pool is not None:
            source_ids_pool.append(source_id)
            
        img = item['image'].convert('RGB')
        
        # Tamper 50% of the real documents
        if random.random() < 0.5:
            # Keep authentic
            auth_filename = f"{source_id}_auth.jpg"
            img.save(img_dir / auth_filename, "JPEG", quality=85)
            manifest.append({
                "source_id": source_id,
                "image_id": f"{source_id}_auth",
                "image_path": str(img_dir / auth_filename),
                "label": "REAL",
                "tamper_type": None,
                "doc_type": "real_scanned"
            })
        else:
            # Tamper it
            if random.random() < 0.5:
                tamp_res = apply_digital_text_overlay(img, text="APPROVED", font_size=24)
                tamp_type = "digital_text_overlay"
            else:
                tamp_res = apply_stamp_copy_move(img)
                tamp_type = "stamp_copy_move"
                
            tamp_filename = f"{source_id}_tamp_{tamp_type}.jpg"
            mask_filename = f"{source_id}_tamp_mask.png"
            
            tamp_res["image"].save(img_dir / tamp_filename, "JPEG", quality=85)
            tamp_res["mask"].save(mask_dir / mask_filename)
            
            manifest.append({
                "source_id": source_id,
                "image_id": f"{source_id}_tamp",
                "image_path": str(img_dir / tamp_filename),
                "mask_path": str(mask_dir / mask_filename),
                "label": "TAMPERED",
                "tamper_type": tamp_type,
                "doc_type": "real_scanned"
            })
            
        count += 1
        
    return manifest, source_ids_pool

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
        doc_type_choice = random.choice(["medical", "invoice", "id_card", "screenshot"])
        if doc_type_choice == "medical":
            base_img = generate_medical_prescription(source_id)
        elif doc_type_choice == "invoice":
            base_img = generate_invoice_receipt(source_id)
        elif doc_type_choice == "id_card":
            base_img = generate_id_card(source_id)
        else:
            base_img = generate_screenshot(source_id)
            
        doc_type = doc_type_choice
            
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
        
    print(f"Total Synthetic Images Generated: {len(manifest)}")
    
    # 4. Integrate Real-World Scans
    manifest, source_ids = integrate_real_world_documents(manifest, img_dir, mask_dir, num_real=1000, source_ids_pool=source_ids)
    
    print(f"Total Combined Images Generated: {len(manifest)}")
    
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
