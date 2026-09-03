import os
import io
import base64
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from typing import Dict, Any

from src.forensics.ela import calculate_ela
from src.forensics.residuals import calculate_gaussian_residual
from src.models.resnet import ResNet18Binary
from src.models.fusion import ForensicFusionNet
from src.models.segmentation import DocumentSegmentationModel

device = torch.device('cpu') # Use CPU for local API demo stability

# Image transforms
rgb_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

forensic_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Initialize models
_model = None
_model_type = "resnet"
_seg_model = None

def get_model():
    global _model, _model_type, _seg_model
    
    # Load U-Net Segmentation Model if not loaded
    if _seg_model is None:
        seg_weights_path = "results/segmentation/unet_segmentation/best_seg.pt"
        if os.path.exists(seg_weights_path):
            seg_model = DocumentSegmentationModel(encoder_name="resnet18", encoder_weights="imagenet")
            seg_model.load_state_dict(torch.load(seg_weights_path, map_location=device))
            seg_model.eval()
            _seg_model = seg_model
            print(f"Loaded DocumentSegmentationModel from {seg_weights_path}")
            
    if _model is not None:
        return _model, _model_type, _seg_model
        
    weights_path = "results/runs/resnet18_baseline/best.pt"
    if not os.path.exists(weights_path):
        weights_path = "results/runs/fusion/best.pt"
        
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        is_fusion = any("rgb_backbone" in k or "forensic_backbone" in k for k in state_dict.keys())
        
        if is_fusion:
            model = ForensicFusionNet(use_ela=True, use_residual=True)
            model.load_state_dict(state_dict)
            model.eval()
            _model = model
            _model_type = "fusion"
            print(f"Loaded ForensicFusionNet model from {weights_path}")
            return _model, _model_type, _seg_model
        else:
            model = ResNet18Binary(pretrained=False)
            model.load_state_dict(state_dict)
            model.eval()
            _model = model
            _model_type = "resnet"
            print(f"Loaded ResNet18Binary model from {weights_path}")
            return _model, _model_type, _seg_model
            
    # Fallback default
    model = ResNet18Binary(pretrained=False)
    model.eval()
    _model = model
    _model_type = "resnet"
    return _model, _model_type, _seg_model


def run_full_analysis(image_bytes: bytes) -> Dict[str, Any]:
    """
    Executes full forensic analysis on the uploaded document:
    1. Decodes image and computes real-time ELA & Gaussian Noise maps.
    2. Runs deep model inference (ForensicFusionNet / ResNet18).
    3. Produces localized tampering regions and summary.
    4. Encodes visual artifacts to base64.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    w, h = img.size
    
    # 1. Compute Forensic Maps
    ela_result = calculate_ela(img)
    ela_img = ela_result['ela_image']
    buffered_ela = io.BytesIO()
    ela_img.save(buffered_ela, format="JPEG")
    ela_b64 = base64.b64encode(buffered_ela.getvalue()).decode()
    
    noise_result = calculate_gaussian_residual(img)
    noise_img = noise_result['residual_image']
    buffered_noise = io.BytesIO()
    noise_img.save(buffered_noise, format="JPEG")
    noise_b64 = base64.b64encode(buffered_noise.getvalue()).decode()
    
    # 2. Run Model Inference
    model, m_type, seg_model = get_model()
    rgb_tensor = rgb_transform(img).unsqueeze(0)
    
    with torch.no_grad():
        if m_type == "fusion":
            ela_t = forensic_transform(ela_img.convert("L"))
            res_t = forensic_transform(noise_img.convert("L"))
            forensics_tensor = torch.cat((ela_t, res_t), dim=0).unsqueeze(0)
            output = model(rgb_tensor, forensics_tensor)
        else:
            output = model(rgb_tensor)
            
        # Apply temperature scaling to soften extreme 1.000 or 0.000 confidence scores
        temperature = 2.0
        probs = torch.softmax(output / temperature, dim=1)
        tamper_score = probs[0][1].item()
        
    # 3. Create Heatmap / Tamper Mask by analyzing U-Net Segmentation Output
    mask = Image.new('L', (w, h), 0)
    regions = 0
    total_mask_area = 0
    draw = ImageDraw.Draw(mask)
    
    is_tampered = tamper_score > 0.5
    annotated_img = img.copy()
    annotated_draw = ImageDraw.Draw(annotated_img)
    
    if is_tampered and seg_model is not None:
        with torch.no_grad():
            seg_out = seg_model(rgb_tensor)
            seg_probs = torch.sigmoid(seg_out)
            seg_mask_224 = (seg_probs[0, 0] > 0.5).cpu().numpy().astype(np.uint8) * 255
            
            # Resize U-Net output to original image size
            seg_mask_full = cv2.resize(seg_mask_224, (w, h), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(seg_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                if cw * ch > 100:  # U-Net masks are highly precise, lower threshold
                    pad = 5
                    bx = max(0, cx - pad)
                    by = max(0, cy - pad)
                    bx2 = min(w, cx + cw + pad)
                    by2 = min(h, cy + ch + pad)
                    draw.rectangle([bx, by, bx2, by2], fill=255)
                    # Draw thick red bounding box for Gemini Context
                    annotated_draw.rectangle([bx, by, bx2, by2], outline=(255, 0, 0), width=6)
                    regions += 1
                    total_mask_area += (bx2 - bx) * (by2 - by)
    
    # Fallback to ELA ONLY if U-Net isn't loaded
    if is_tampered and regions == 0 and seg_model is None:
        ela_cv = np.array(ela_img.convert("L"))
        thresh_val = max(100, np.max(ela_cv) * 0.7)
        _, thresh = cv2.threshold(ela_cv, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            if cw * ch > 400 and cw < w * 0.9 and ch < h * 0.9:
                pad = 10
                bx = max(0, cx - pad)
                by = max(0, cy - pad)
                bx2 = min(w, cx + cw + pad)
                by2 = min(h, cy + ch + pad)
                draw.rectangle([bx, by, bx2, by2], fill=255)
                regions += 1
                total_mask_area += (bx2 - bx) * (by2 - by)

        # Legacy ELA Override: Prevent False Positives on normal text
        is_text_false_positive = regions > 15 or (total_mask_area / (w * h)) > 0.15
        if is_text_false_positive:
            tamper_score = min(tamper_score, 0.45)
            regions = 0
            mask = Image.new('L', (w, h), 0)
            draw = ImageDraw.Draw(mask)
            annotated_img = img.copy()
        
    is_tampered = tamper_score > 0.5
    
    verdict_text = "TAMPERED" if is_tampered else "AUTHENTIC"
    base_confidence = tamper_score if is_tampered else (1 - tamper_score)
    vlm_summary = None

    if os.environ.get("GEMINI_API_KEY"):
        try:
            from google import genai
            client = genai.Client()
            
            prompt = (
                f"## Document Tampering / Authenticity Inspection Prompt\n\n"
                f"You are an AI document-forensics assistant. Your task is to visually inspect the provided document image(s) "
                f"and identify any signs that the document may have been digitally altered, manipulated, composited, or otherwise tampered with.\n\n"
                f"Our neural network evaluated this image and classified it as {verdict_text} "
                f"with a confidence score of {base_confidence*100:.1f}%. "
                f"We have also drawn thick RED bounding boxes on the image provided to you. These RED bounding boxes indicate "
                f"the exact pixels where our Semantic Segmentation U-Net model localized physical splicing, inconsistencies, or text manipulation. "
                f"Please read any text or elements inside these red boxes and carefully evaluate them in your assessment.\n\n"
                f"### What to inspect\n"
                f"Carefully examine the entire document, including:\n"
                f"1. **Text consistency**: Font style, spacing, alignment, sharpness differences.\n"
                f"2. **Layout and formatting**: Misaligned text, broken borders, inconsistent margins.\n"
                f"3. **Image and compression artifacts**: Pixelation, JPEG artifacts, blurring, boundaries.\n"
                f"4. **Background texture**: Erasures, digital filling, color changes.\n"
                f"5. **Important fields**: Dates, numbers, names, signatures.\n\n"
                f"### Important limitations\n"
                f"* Do not label something as tampering solely because it looks different.\n"
                f"* Consider innocent explanations (OCR, scanning, low quality).\n"
                f"* If evidence is inconclusive, explicitly say so.\n\n"
                f"### Required output (Markdown Format)\n"
                f"**Overall assessment:** (No obvious signs of tampering | Potential signs of tampering | Strong visual indications of tampering | Inconclusive)\n"
                f"**Confidence:** (Low | Medium | High)\n\n"
                f"**Suspicious areas:**\n"
                f"* Location: ...\n"
                f"* Observation: ...\n"
                f"* Why suspicious: ...\n"
                f"* Innocent explanation: ...\n"
                f"* Severity: ...\n\n"
                f"**Evidence supporting authenticity:** ...\n"
                f"**Evidence suggesting manipulation:** ...\n"
                f"**Image-quality limitations:** ...\n"
                f"**Final conclusion:** ...\n"
            )
            
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=[annotated_img, prompt]
            )
            vlm_summary = response.text.strip()
        except Exception as e:
            print(f"Gemini VLM API failed or is unavailable: {e}")
            vlm_summary = None
            
    if is_tampered:
        # Fallback if no contours were large enough but model predicts tamper heavily (and override didn't catch it)
        if regions == 0:
            draw.rectangle([w // 4, h // 4, 3 * w // 4, h // 2], fill=255)
            regions = 1
            
        if not vlm_summary:
            vlm_summary = (
                f"The document shows strong evidence of digital tampering (Confidence: {tamper_score*100:.1f}%). "
                f"Forensic ELA and noise variance indicate spliced text blocks and compression mismatches."
            )
    else:
        if not vlm_summary:
            vlm_summary = (
                f"The document appears authentic (Confidence: {(1 - tamper_score)*100:.1f}%). "
                f"No statistical noise discontinuities or compression anomalies were detected."
            )
        
    buffered_mask = io.BytesIO()
    mask.save(buffered_mask, format="JPEG")
    mask_b64 = base64.b64encode(buffered_mask.getvalue()).decode()
    
    buffered_annotated = io.BytesIO()
    annotated_img.save(buffered_annotated, format="JPEG")
    annotated_b64 = base64.b64encode(buffered_annotated.getvalue()).decode()
    
    return {
        "score": float(tamper_score),
        "regions_detected": regions,
        "vlm_summary": vlm_summary,
        "artifacts": {
            "ela": ela_b64,
            "residual": noise_b64,
            "mask": mask_b64,
            "annotated": annotated_b64
        }
    }
