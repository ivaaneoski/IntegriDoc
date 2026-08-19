import json
from typing import List, Dict

def build_forensic_prompt(ocr_data: List[Dict], classifier_score: float, has_localization_heatmap: bool = True) -> str:
    """
    Constructs the structured prompt for the VLM (Qwen2.5-VL), bundling the extracted 
    forensic evidence into a strict system prompt.
    """
    
    ocr_context = json.dumps(ocr_data, indent=2) if ocr_data else "No text detected."
    
    heatmap_context = (
        "A Grad-CAM / U-Net localization mask is provided in the image bundle, highlighting suspected altered pixels." 
        if has_localization_heatmap else 
        "No localization mask provided."
    )
    
    prompt = f"""You are a document-forensics assistant.
You are not allowed to infer authenticity from personal identity, nationality, or visual stereotypes.
Base every conclusion only on the supplied image and forensic evidence.

Inputs:
- Document image (attached)
- OCR regions:
{ocr_context}
- Detector score: {classifier_score:.4f} (0.0 = REAL, 1.0 = TAMPERED)
- Localization: {heatmap_context}

Task:
1. Identify suspicious visual regions in the document (e.g., inconsistent fonts, misaligned text, background noise anomalies).
2. Describe which supplied evidence supports each region.
3. Distinguish your observed evidence from mere inference.
4. Recommend one of the following actions: "REAL", "MANUAL_REVIEW", or "TAMPERED".
5. OUTPUT JSON ONLY.

Follow this exact JSON schema:
{{
  "summary": "<brief overall assessment>",
  "suspicious_regions": [
    {{
      "field": "<name of the field or region>",
      "reason": "<why it is suspicious based on evidence>",
      "evidence_strength": "<low/medium/high>"
    }}
  ],
  "recommended_action": "<REAL|MANUAL_REVIEW|TAMPERED>",
  "limitations": [
    "<any limitations or assumptions made>"
  ]
}}
"""
    return prompt
