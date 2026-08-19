import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def apply_text_replacement(auth_doc: dict, field_name: str, new_value: str = "TAMPERED") -> dict:
    """
    Replaces a specific text field in the authentic document.
    """
    img = auth_doc["image"].copy()
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    
    # Get bounding box of the original field
    if field_name not in auth_doc["bboxes"]:
        raise ValueError(f"Field {field_name} not found in bboxes")
        
    norm_bbox = auth_doc["bboxes"][field_name]
    width, height = img.size
    
    x1, y1 = int(norm_bbox[0] * width), int(norm_bbox[1] * height)
    x2, y2 = int(norm_bbox[2] * width), int(norm_bbox[3] * height)
    
    # Create mask before tampering
    mask = Image.new('L', img.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    
    # Erase old text (assuming white background for simplicity in this smoke test)
    # Give a little padding
    pad = 2
    draw.rectangle([x1-pad, y1-pad, x2+pad, y2+pad], fill="white")
    
    # Draw new text
    draw.text((x1, y1), new_value, fill="black", font=font)
    
    # Update mask
    new_bbox = draw.textbbox((x1, y1), new_value, font=font)
    mask_draw.rectangle([
        min(x1-pad, new_bbox[0]), 
        min(y1-pad, new_bbox[1]), 
        max(x2+pad, new_bbox[2]), 
        max(y2+pad, new_bbox[3])
    ], fill=255)
    
    # Store normalized updated bbox
    tampered_bbox = [
        min(x1-pad, new_bbox[0]) / width,
        min(y1-pad, new_bbox[1]) / height,
        max(x2+pad, new_bbox[2]) / width,
        max(y2+pad, new_bbox[3]) / height
    ]
    
    return {
        "image": img,
        "mask": mask,
        "tamper_type": "text_replacement",
        "bbox": tampered_bbox
    }

def apply_copy_move(auth_doc: dict) -> dict:
    """
    Copies a random patch from the image and pastes it elsewhere.
    """
    img = auth_doc["image"].copy()
    width, height = img.size
    
    # Source patch (e.g. from the portrait area)
    patch_size = 50
    sx1 = random.randint(400, 500)
    sy1 = random.randint(100, 200)
    sx2, sy2 = sx1 + patch_size, sy1 + patch_size
    
    patch = img.crop((sx1, sy1, sx2, sy2))
    
    # Target location (e.g. over the ID number)
    tx1 = random.randint(50, 300)
    ty1 = random.randint(200, 350)
    
    # Paste
    img.paste(patch, (tx1, ty1))
    
    # Mask
    mask = Image.new('L', img.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle([tx1, ty1, tx1 + patch_size, ty1 + patch_size], fill=255)
    
    tampered_bbox = [
        tx1 / width,
        ty1 / height,
        (tx1 + patch_size) / width,
        (ty1 + patch_size) / height
    ]
    
    return {
        "image": img,
        "mask": mask,
        "tamper_type": "copy_move",
        "bbox": tampered_bbox
    }
