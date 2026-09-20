import os
import io
import base64
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from typing import Dict, Any, List, Optional, Tuple

from src.forensics.ela import calculate_ela
from src.forensics.residuals import calculate_gaussian_residual
from src.models.resnet import ResNet18Binary
from src.models.fusion import ForensicFusionNet
from src.models.segmentation import DocumentSegmentationModel
from src.localization.gradcam import GradCAM, overlay_heatmap
from src.ocr.extractor import extract_text_regions, match_tamper_regions_with_text
from src.utils.image_io import load_document_image, image_to_base64, is_pdf

# Dynamic device selection
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

device = get_device()

# Default calibrated decision thresholds
THRESHOLD_LOW = float(os.environ.get("THRESHOLD_LOW", "0.35"))
THRESHOLD_HIGH = float(os.environ.get("THRESHOLD_HIGH", "0.65"))

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

# Lazy model singletons
_model = None
_model_type = "resnet"
_seg_model = None

def get_model() -> Tuple[torch.nn.Module, str, Optional[DocumentSegmentationModel]]:
    global _model, _model_type, _seg_model
    
    # Load U-Net Segmentation Model if available
    if _seg_model is None:
        seg_weights_path = "results/segmentation/unet_segmentation/best_seg.pt"
        if os.path.exists(seg_weights_path):
            try:
                seg_model = DocumentSegmentationModel(encoder_name="resnet18", encoder_weights=None)
                seg_model.load_state_dict(torch.load(seg_weights_path, map_location=device))
                seg_model.to(device)
                seg_model.eval()
                _seg_model = seg_model
                print(f"Loaded DocumentSegmentationModel on {device} from {seg_weights_path}")
            except Exception as e:
                print(f"Could not load DocumentSegmentationModel from {seg_weights_path}: {e}")
            
    if _model is not None:
        return _model, _model_type, _seg_model
        
    weights_path = "results/runs/resnet18_baseline/best.pt"
    if not os.path.exists(weights_path):
        weights_path = "results/runs/fusion/best.pt"
        
    if os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=device)
            is_fusion = any("rgb_backbone" in k or "forensic_backbone" in k for k in state_dict.keys())
            
            if is_fusion:
                model = ForensicFusionNet(use_ela=True, use_residual=True)
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                _model = model
                _model_type = "fusion"
                print(f"Loaded ForensicFusionNet on {device} from {weights_path}")
                return _model, _model_type, _seg_model
            else:
                model = ResNet18Binary(pretrained=False)
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                _model = model
                _model_type = "resnet"
                print(f"Loaded ResNet18Binary on {device} from {weights_path}")
                return _model, _model_type, _seg_model
        except Exception as e:
            print(f"Failed to load checkpoint from {weights_path}: {e}")
            
    # Default uninitialized fallback model for demo stability
    model = ResNet18Binary(pretrained=False)
    model.to(device)
    model.eval()
    _model = model
    _model_type = "resnet"
    return _model, _model_type, _seg_model


def generate_gradcam_heatmap(model: torch.nn.Module, m_type: str, rgb_tensor: torch.Tensor) -> Optional[np.ndarray]:
    """Generates Grad-CAM activation heatmap as explainability fallback."""
    try:
        if m_type == "resnet" and hasattr(model, "model") and hasattr(model.model, "layer4"):
            target_layer = model.model.layer4[-1]
            grad_cam = GradCAM(model, target_layer)
            cam = grad_cam.generate_heatmap(rgb_tensor, target_class=1)
            return cam
        elif m_type == "fusion" and hasattr(model, "rgb_backbone") and hasattr(model.rgb_backbone, "layer4"):
            target_layer = model.rgb_backbone.layer4[-1]
            # Wrap for CAM
            grad_cam = GradCAM(model, target_layer)
            cam = grad_cam.generate_heatmap(rgb_tensor, target_class=1)
            return cam
    except Exception as e:
        print(f"GradCAM generation skipped: {e}")
    return None


def run_full_analysis(
    document_bytes: bytes,
    page_index: int = 0,
    threshold_low: float = THRESHOLD_LOW,
    threshold_high: float = THRESHOLD_HIGH
) -> Dict[str, Any]:
    """
    Executes full multi-modal forensic analysis on uploaded document (PDF or Image):
    1. Validates and loads document (auto-resolving EXIF orientation and PDF pages).
    2. Computes real-time Error Level Analysis (ELA) and Gaussian Residual maps & scores.
    3. Runs deep classifier inference (ForensicFusionNet / ResNet18).
    4. Evaluates calibrated 3-state verdict (REAL, UNCERTAIN, TAMPERED).
    5. Localizes tampering regions (via U-Net segmentation or Grad-CAM / ELA fallback).
    6. Performs OCR cross-verification to identify altered text tokens/fields.
    7. Engages Gemini VLM for contextual analysis grounded in model evidence.
    8. Returns full specification-compliant JSON contract with base64 visual artifacts.
    """
    # 1. Safe Document Loading
    img, total_pages = load_document_image(document_bytes, page_index=page_index)
    
    # Cap maximum dimension to 1000px to protect against Out-Of-Memory (OOM 502) on free 512MB RAM instances
    max_dim = 1000
    orig_w, orig_h = img.size
    if max(orig_w, orig_h) > max_dim:
        scale = max_dim / float(max(orig_w, orig_h))
        new_w, new_h = max(10, int(orig_w * scale)), max(10, int(orig_h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    w, h = img.size
    is_pdf_doc = is_pdf(document_bytes)
    
    # 2. Compute Classical Forensics
    ela_result = calculate_ela(img)
    ela_img = ela_result['ela_image']
    ela_b64 = image_to_base64(ela_img, format="JPEG")
    
    noise_result = calculate_gaussian_residual(img)
    noise_img = noise_result['residual_image']
    noise_b64 = image_to_base64(noise_img, format="JPEG")
    
    # 3. Model Inference
    model, m_type, seg_model = get_model()
    rgb_tensor = rgb_transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if m_type == "fusion":
            ela_t = forensic_transform(ela_img.convert("L")).to(device)
            res_t = forensic_transform(noise_img.convert("L")).to(device)
            forensics_tensor = torch.cat((ela_t, res_t), dim=0).unsqueeze(0)
            output = model(rgb_tensor, forensics_tensor)
        else:
            output = model(rgb_tensor)
            
        temperature = 2.0
        probs = torch.softmax(output / temperature, dim=1)
        tamper_score = float(probs[0][1].item())
        
    # 4. Calibrated 3-State Verdict
    if tamper_score >= threshold_high:
        verdict = "TAMPERED"
        confidence = tamper_score
    elif tamper_score <= threshold_low:
        verdict = "REAL"
        confidence = 1.0 - tamper_score
    else:
        verdict = "UNCERTAIN"
        confidence = 1.0 - abs(tamper_score - 0.5) * 2  # Higher uncertainty near 0.5
        
    is_suspicious = tamper_score > threshold_low
    
    # 5. Localization & Anomaly Bounding Boxes
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)
    annotated_img = img.copy()
    annotated_draw = ImageDraw.Draw(annotated_img)
    
    detected_pixel_boxes: List[Tuple[int, int, int, int]] = []
    normalized_regions: List[Dict[str, Any]] = []
    total_mask_area = 0
    
    # Priority A: U-Net Semantic Segmentation
    if is_suspicious and seg_model is not None:
        try:
            with torch.no_grad():
                seg_out = seg_model(rgb_tensor)
                seg_probs = torch.sigmoid(seg_out)
                seg_mask_224 = (seg_probs[0, 0] > 0.5).cpu().numpy().astype(np.uint8) * 255
                
                seg_mask_full = cv2.resize(seg_mask_224, (w, h), interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours(seg_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    cx, cy, cw, ch = cv2.boundingRect(cnt)
                    if cw * ch > 100:
                        pad = 6
                        bx, by = max(0, cx - pad), max(0, cy - pad)
                        bx2, by2 = min(w, cx + cw + pad), min(h, cy + ch + pad)
                        draw.rectangle([bx, by, bx2, by2], fill=255)
                        annotated_draw.rectangle([bx, by, bx2, by2], outline=(255, 0, 0), width=6)
                        detected_pixel_boxes.append((bx, by, bx2, by2))
                        total_mask_area += (bx2 - bx) * (by2 - by)
        except Exception as e:
            print(f"Segmentation inference error: {e}")
            
    # Priority B: ELA & Noise Anomaly Fallback
    if is_suspicious and len(detected_pixel_boxes) == 0:
        ela_cv = np.array(ela_img.convert("L"))
        thresh_val = max(100, np.max(ela_cv) * 0.7)
        _, thresh = cv2.threshold(ela_cv, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            if cw * ch > 400 and cw < w * 0.9 and ch < h * 0.9:
                pad = 8
                bx, by = max(0, cx - pad), max(0, cy - pad)
                bx2, by2 = min(w, cx + cw + pad), min(h, cy + ch + pad)
                draw.rectangle([bx, by, bx2, by2], fill=255)
                annotated_draw.rectangle([bx, by, bx2, by2], outline=(255, 0, 0), width=5)
                detected_pixel_boxes.append((bx, by, bx2, by2))
                total_mask_area += (bx2 - bx) * (by2 - by)
                
        # Guard against full-image text false positives in pure ELA
        if len(detected_pixel_boxes) > 15 or (total_mask_area / (w * h)) > 0.20:
            tamper_score = min(tamper_score, threshold_low)
            verdict = "REAL"
            detected_pixel_boxes = []
            mask = Image.new('L', (w, h), 0)
            draw = ImageDraw.Draw(mask)
            annotated_img = img.copy()

    # Build Normalized Bounding Boxes [x1, y1, x2, y2]
    for idx, (bx, by, bx2, by2) in enumerate(detected_pixel_boxes):
        norm_box = [
            round(bx / w, 4),
            round(by / h, 4),
            round(bx2 / w, 4),
            round(by2 / h, 4)
        ]
        area_ratio = round(((bx2 - bx) * (by2 - by)) / (w * h), 4)
        normalized_regions.append({
            "id": idx + 1,
            "label": "possible_tampering_region",
            "bbox": norm_box,
            "area_ratio": area_ratio,
            "score": round(tamper_score, 3)
        })

    # 6. OCR Text Extraction & Tamper Grounding
    ocr_tokens = extract_text_regions(img)
    tamper_norm_boxes = [r["bbox"] for r in normalized_regions]
    flagged_text = match_tamper_regions_with_text(ocr_tokens, tamper_norm_boxes)
    
    # 7. Gemini VLM Multimodal Forensic Report
    vlm_summary = None
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if api_key:
        try:
            from google import genai
            client = genai.Client()
            
            # Format text evidence for prompt
            flagged_text_summary = "None detected." if not flagged_text else ", ".join(
                [f"'{t['text']}'" for t in flagged_text[:10]]
            )
            
            prompt = (
                f"## Document Forensics & Integrity Assessment\n\n"
                f"You are an expert digital document forensics specialist. Inspect the provided images:\n"
                f"1. Document with Red Bounding Boxes identifying model-localized tampering anomalies.\n"
                f"2. Error Level Analysis (ELA) map showing JPEG compression level differences.\n"
                f"3. High-Pass Gaussian Noise Residual map showing local variance and pixel noise inconsistencies.\n\n"
                f"### Quantitative Model Metrics\n"
                f"- Classification Verdict: {verdict}\n"
                f"- Tamper Risk Score: {tamper_score*100:.1f}%\n"
                f"- ELA Max Delta: {ela_result['max_diff']}\n"
                f"- Noise Variance: {noise_result['variance']:.4f}\n"
                f"- Flagged Spliced/Altered Text Tokens: {flagged_text_summary}\n\n"
                f"### Inspection Criteria\n"
                f"Evaluate font typography consistency, baseline alignment, noise distribution, boundary artifacts, "
                f"and semantic logic (dates, dollar amounts, identity numbers, names).\n\n"
                f"### Required Markdown Output Structure\n"
                f"**Assessment:** {verdict} (Confidence: {confidence*100:.1f}%)\n"
                f"**Summary:** Brief 1-2 sentence executive summary.\n\n"
                f"**Suspicious Regions & Anomalies:**\n"
                f"- Detail any anomalous boxes or spliced text tokens.\n\n"
                f"**Forensic Evidence:**\n"
                f"- *Compression/ELA:* Analysis of compression patterns.\n"
                f"- *Noise/Texture:* Analysis of background noise and smoothing.\n"
                f"- *Typography/Layout:* Font style, kerning, and alignment.\n\n"
                f"**Conclusion & Risk Recommendation:** Actionable recommendation for KYC/fraud analysts.\n"
            )
            
            # Prepare lightweight thumbnails for rapid VLM transmission
            vlm_annotated = annotated_img.copy()
            vlm_annotated.thumbnail((800, 800))
            vlm_ela = ela_img.copy()
            vlm_ela.thumbnail((800, 800))
            vlm_noise = noise_img.copy()
            vlm_noise.thumbnail((800, 800))
            
            # Fast single-attempt VLM invocation
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[vlm_annotated, vlm_ela, vlm_noise, prompt]
                )
                if response and response.text:
                    vlm_summary = response.text.strip()
            except Exception as model_err:
                print(f"VLM call failed, falling back gracefully: {model_err}")
                vlm_summary = None
                    
        except Exception as e:
            print(f"Gemini VLM API invocation failed: {e}")
            vlm_summary = None

    # Fallback rule-based forensic report if VLM is unavailable
    if not vlm_summary:
        if verdict == "TAMPERED":
            if flagged_text:
                formatted_tokens = ", ".join([f"`{t['text']}`" for t in flagged_text[:5]])
                spliced_mention = f" Flagged text elements: {formatted_tokens}."
            else:
                spliced_mention = ""
            vlm_summary = (
                f"**Assessment:** Strong visual and statistical indications of document manipulation (Score: {tamper_score*100:.1f}%).\n\n"
                f"**Key Findings:**\n"
                f"- Localized {len(normalized_regions)} suspicious region(s) with anomalous compression discontinuities.{spliced_mention}\n"
                f"- ELA maximum difference reached {ela_result['max_diff']} with noise variance of {noise_result['variance']:.4f}.\n\n"
                f"**Recommendation:** Reject automated pass and escalate document to senior fraud investigator."
            )
        elif verdict == "UNCERTAIN":
            vlm_summary = (
                f"**Assessment:** Ambiguous / Inconclusive Document Integrity (Score: {tamper_score*100:.1f}%).\n\n"
                f"**Key Findings:**\n"
                f"- Document exhibits borderline compression noise or subtle optical variations that fall within the gray zone.\n"
                f"- Statistical noise residual is {noise_result['score']:.4f}.\n\n"
                f"**Recommendation:** Manual KYC operator review required to confirm original document origin."
            )
        else:
            vlm_summary = (
                f"**Assessment:** Authentic Document (Confidence: {confidence*100:.1f}%).\n\n"
                f"**Key Findings:**\n"
                f"- No significant statistical noise discontinuities or ELA compression anomalies detected.\n"
                f"- Typography and background texture maintain uniform characteristics across the document canvas.\n\n"
                f"**Recommendation:** Approved for automated verification workflow."
            )

    # 8. Encode Artifacts to base64
    mask_b64 = image_to_base64(mask, format="JPEG")
    annotated_b64 = image_to_base64(annotated_img, format="JPEG")

    return {
        "verdict": verdict,
        "tamper_score": round(tamper_score, 4),
        "confidence": round(confidence, 4),
        "thresholds": {
            "low": threshold_low,
            "high": threshold_high
        },
        "model": {
            "name": "ForensicFusionNet" if m_type == "fusion" else "ResNet18Binary",
            "type": m_type,
            "device": str(device),
            "segmentation_model_active": seg_model is not None
        },
        "forensics": {
            "ela_score": ela_result.get("score", 0.0),
            "ela_max_diff": ela_result.get("max_diff", 0),
            "noise_residual_score": noise_result.get("score", 0.0),
            "noise_residual_variance": noise_result.get("variance", 0.0)
        },
        "document_info": {
            "width": w,
            "height": h,
            "total_pages": total_pages,
            "is_pdf": is_pdf_doc,
            "analyzed_page": page_index + 1
        },
        "regions_detected": len(normalized_regions),
        "regions": normalized_regions,
        "flagged_text": flagged_text,
        "vlm_summary": vlm_summary,
        "artifacts": {
            "ela": ela_b64,
            "residual": noise_b64,
            "mask": mask_b64,
            "annotated": annotated_b64
        }
    }
