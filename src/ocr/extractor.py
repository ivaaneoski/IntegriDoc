import pytesseract
from PIL import Image
import numpy as np

def extract_text_regions(image_path: str):
    """
    Extracts text regions from an image using pytesseract.
    Returns a list of dictionaries containing text, bounding box (normalized), and confidence.
    """
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        width, height = img.size
        
        # Run Tesseract OCR and get detailed data
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
        regions = []
        n_boxes = len(data['level'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            # Filter out empty text and low confidence predictions (e.g. below 50)
            if text and conf > 50:
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                
                # Normalize coordinates to 0.0 - 1.0 range
                norm_bbox = [
                    round(x / width, 4), 
                    round(y / height, 4), 
                    round((x + w) / width, 4), 
                    round((y + h) / height, 4)
                ]
                
                regions.append({
                    "text": text,
                    "bbox": norm_bbox,
                    "confidence": conf / 100.0
                })
                
        return regions
    except Exception as e:
        print(f"OCR Extraction Failed. Is tesseract-ocr installed on the system? Error: {e}")
        return []

if __name__ == "__main__":
    # Smoke test if run directly
    import sys
    if len(sys.argv) > 1:
        regions = extract_text_regions(sys.argv[1])
        import json
        print(json.dumps(regions, indent=2))
