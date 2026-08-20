import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import Dict, Tuple, List, Optional

def get_fallback_font(size: int = 16):
    try:
        # Try common system fonts on Windows/Linux
        for font_name in ["arial.ttf", "calibri.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]:
            try:
                return ImageFont.truetype(font_name, size)
            except IOError:
                continue
    except Exception:
        pass
    return ImageFont.load_default()

def apply_digital_text_overlay(
    image: Image.Image,
    text: str,
    location: Optional[Tuple[int, int]] = None,
    font_size: int = 18,
    text_color: Optional[Tuple[int, int, int]] = None
) -> Dict:
    """
    Overlays a crisp, digital font string onto the document image.
    This mimics edits like adding '3 Days rest' or 'Vitamin D - 19.8' onto photographed/printed paper.
    """
    img = image.copy()
    w, h = img.size
    mask = Image.new('L', (w, h), 0)
    
    draw = ImageDraw.Draw(img)
    mask_draw = ImageDraw.Draw(mask)
    
    font = get_fallback_font(font_size)
    
    if text_color is None:
        # Occasionally bright red, blue, or crisp pitch-black
        colors = [(20, 20, 20), (180, 20, 20), (20, 40, 180), (10, 10, 10)]
        text_color = random.choice(colors)
        
    if location is None:
        # Pick a random safe location
        x = random.randint(int(w * 0.15), int(w * 0.7))
        y = random.randint(int(h * 0.2), int(h * 0.8))
    else:
        x, y = location
        
    bbox = draw.textbbox((x, y), text, font=font)
    draw.text((x, y), text, fill=text_color, font=font)
    
    # Fill mask in tampered zone with slight padding
    pad = 3
    tampered_zone = [
        max(0, bbox[0] - pad),
        max(0, bbox[1] - pad),
        min(w, bbox[2] + pad),
        min(h, bbox[3] + pad)
    ]
    mask_draw.rectangle(tampered_zone, fill=255)
    
    norm_bbox = [
        tampered_zone[0] / w,
        tampered_zone[1] / h,
        tampered_zone[2] / w,
        tampered_zone[3] / h
    ]
    
    return {
        "image": img,
        "mask": mask,
        "tamper_type": "digital_text_overlay",
        "bbox": norm_bbox
    }

def apply_gray_patch_splicing(
    image: Image.Image,
    patch_text: str,
    location: Optional[Tuple[int, int]] = None,
    font_size: int = 16
) -> Dict:
    """
    Pastes a rectangular patch with a slight background luminance mismatch
    and writes text inside it. Mimics whiteout/erasure patches (e.g. Astha Clinic medicine list).
    """
    img = image.copy()
    w, h = img.size
    mask = Image.new('L', (w, h), 0)
    
    draw = ImageDraw.Draw(img)
    mask_draw = ImageDraw.Draw(mask)
    font = get_fallback_font(font_size)
    
    if location is None:
        x = random.randint(int(w * 0.2), int(w * 0.6))
        y = random.randint(int(h * 0.3), int(h * 0.7))
    else:
        x, y = location
        
    # Measure text box
    temp_img = Image.new('RGB', (10, 10))
    temp_draw = ImageDraw.Draw(temp_img)
    t_bbox = temp_draw.textbbox((0, 0), patch_text, font=font)
    pw = (t_bbox[2] - t_bbox[0]) + random.randint(20, 40)
    ph = (t_bbox[3] - t_bbox[1]) + random.randint(10, 20)
    
    # Slight gray/off-white patch tint (luminance mismatch)
    patch_brightness = random.randint(205, 235)
    patch_color = (patch_brightness, patch_brightness + random.randint(-3, 3), patch_brightness + random.randint(-3, 3))
    
    patch_rect = [x, y, min(w, x + pw), min(h, y + ph)]
    draw.rectangle(patch_rect, fill=patch_color)
    
    # Draw text inside patch
    tx = x + 8
    ty = y + (ph - (t_bbox[3] - t_bbox[1])) // 2
    draw.text((tx, ty), patch_text, fill=(30, 30, 30), font=font)
    
    mask_draw.rectangle(patch_rect, fill=255)
    
    norm_bbox = [
        patch_rect[0] / w,
        patch_rect[1] / h,
        patch_rect[2] / w,
        patch_rect[3] / h
    ]
    
    return {
        "image": img,
        "mask": mask,
        "tamper_type": "gray_patch_splicing",
        "bbox": norm_bbox
    }

def apply_stamp_copy_move(
    image: Image.Image,
    source_box: Optional[Tuple[int, int, int, int]] = None
) -> Dict:
    """
    Copies a region (e.g. stamp, signature, logo, or text) from one part of the document
    and pastes it elsewhere.
    """
    img = image.copy()
    w, h = img.size
    mask = Image.new('L', (w, h), 0)
    mask_draw = ImageDraw.Draw(mask)
    
    if source_box is None:
        pw = random.randint(50, 120)
        ph = random.randint(40, 80)
        sx = random.randint(int(w * 0.1), int(w * 0.8) - pw)
        sy = random.randint(int(h * 0.1), int(h * 0.8) - ph)
    else:
        sx, sy, sx2, sy2 = source_box
        pw = sx2 - sx
        ph = sy2 - sy
        
    crop = img.crop((sx, sy, sx + pw, sy + ph))
    
    # Target location
    tx = random.randint(int(w * 0.1), int(w * 0.8) - pw)
    ty = random.randint(int(h * 0.1), int(h * 0.8) - ph)
    
    img.paste(crop, (tx, ty))
    target_rect = [tx, ty, tx + pw, ty + ph]
    mask_draw.rectangle(target_rect, fill=255)
    
    norm_bbox = [
        target_rect[0] / w,
        target_rect[1] / h,
        target_rect[2] / w,
        target_rect[3] / h
    ]
    
    return {
        "image": img,
        "mask": mask,
        "tamper_type": "stamp_copy_move",
        "bbox": norm_bbox
    }
