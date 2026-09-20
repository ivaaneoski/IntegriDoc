import pytesseract
from PIL import Image
from typing import List, Dict, Any, Union

def extract_text_regions(image_input: Union[str, Image.Image]) -> List[Dict[str, Any]]:
    """
    Extracts text regions from an image using pytesseract.
    Accepts either an image path (str) or a PIL Image instance.
    Returns a list of dictionaries containing text, bounding box (normalized [x1, y1, x2, y2]), and confidence.
    """
    try:
        if isinstance(image_input, str):
            img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            img = image_input.convert('RGB')
        else:
            return []
            
        width, height = img.size
        if width <= 0 or height <= 0:
            return []
        
        # Run Tesseract OCR and get detailed data
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
        regions = []
        n_boxes = len(data.get('level', []))
        
        for i in range(n_boxes):
            text = str(data['text'][i]).strip()
            try:
                conf = float(data['conf'][i])
            except (ValueError, TypeError):
                conf = 0.0
            
            # Filter out empty text and low confidence predictions (e.g. below 40)
            if text and conf > 40:
                x = max(0, data['left'][i])
                y = max(0, data['top'][i])
                w = max(1, data['width'][i])
                h = max(1, data['height'][i])
                
                # Normalize coordinates to 0.0 - 1.0 range [x1, y1, x2, y2]
                norm_bbox = [
                    round(x / width, 4), 
                    round(y / height, 4), 
                    round(min(width, x + w) / width, 4), 
                    round(min(height, y + h) / height, 4)
                ]
                
                regions.append({
                    "text": text,
                    "bbox": norm_bbox,
                    "confidence": round(conf / 100.0, 3)
                })
                
        return regions
    except Exception as e:
        # Gracefully handle when tesseract-ocr binary is not installed on system
        return []

def match_tamper_regions_with_text(
    ocr_regions: List[Dict[str, Any]], 
    tamper_boxes: List[List[float]],
    iou_thresh: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Identifies which OCR text blocks intersect with detected tamper bounding boxes.
    Boxes are in normalized coordinates [x1, y1, x2, y2] (0.0 to 1.0).
    """
    flagged_text = []
    
    for ocr in ocr_regions:
        ox1, oy1, ox2, oy2 = ocr["bbox"]
        for t_idx, t_box in enumerate(tamper_boxes):
            tx1, ty1, tx2, ty2 = t_box
            
            # Compute intersection
            ix1 = max(ox1, tx1)
            iy1 = max(oy1, ty1)
            ix2 = min(ox2, tx2)
            iy2 = min(oy2, ty2)
            
            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                ocr_area = max(1e-6, (ox2 - ox1) * (oy2 - oy1))
                overlap_ratio = inter_area / ocr_area
                
                if overlap_ratio >= iou_thresh:
                    flagged_text.append({
                        "text": ocr["text"],
                        "confidence": ocr["confidence"],
                        "ocr_bbox": ocr["bbox"],
                        "tamper_region_index": t_idx,
                        "overlap_ratio": round(overlap_ratio, 3)
                    })
                    break
                    
    return flagged_text

if __name__ == "__main__":
    import sys
    import json
    if len(sys.argv) > 1:
        regions = extract_text_regions(sys.argv[1])
        print(json.dumps(regions, indent=2))
