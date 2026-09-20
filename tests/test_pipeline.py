import io
import pytest
import numpy as np
from PIL import Image, ImageDraw

from apps.api.services.pipeline import run_full_analysis, get_model
from src.forensics.ela import calculate_ela
from src.forensics.residuals import calculate_gaussian_residual, calculate_laplacian_residual
from src.ocr.extractor import extract_text_regions, match_tamper_regions_with_text
from src.utils.image_io import load_document_image, is_pdf, image_to_base64

def create_synthetic_image(tampered: bool = False) -> bytes:
    """Creates an in-memory synthetic document image for testing."""
    img = Image.new('RGB', (300, 300), color=(250, 248, 240))
    draw = ImageDraw.Draw(img)
    
    # Draw simulated authentic document elements
    draw.text((30, 30), "INVOICE #1042", fill=(20, 20, 20))
    draw.text((30, 70), "Date: 2026-04-15", fill=(40, 40, 40))
    draw.text((30, 110), "Amount Due: $1,250.00", fill=(40, 40, 40))
    draw.line([(30, 150), (270, 150)], fill=(100, 100, 100), width=2)
    
    if tampered:
        # Simulate text splicing with harsh boundary discontinuity
        splice = Image.new('RGB', (100, 30), color=(255, 255, 255))
        s_draw = ImageDraw.Draw(splice)
        s_draw.text((5, 5), "$9,999.00", fill=(0, 0, 0))
        img.paste(splice, (130, 105))
        
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=95)
    return buf.getvalue()


def test_calculate_ela():
    raw_bytes = create_synthetic_image(tampered=False)
    img = Image.open(io.BytesIO(raw_bytes))
    result = calculate_ela(img)
    
    assert "ela_image" in result
    assert "ela_array" in result
    assert "max_diff" in result
    assert "score" in result
    assert isinstance(result["score"], float)
    assert result["score"] >= 0.0


def test_calculate_residuals():
    raw_bytes = create_synthetic_image(tampered=False)
    img = Image.open(io.BytesIO(raw_bytes))
    
    gaussian_res = calculate_gaussian_residual(img)
    assert "residual_image" in gaussian_res
    assert "score" in gaussian_res
    assert "variance" in gaussian_res
    assert isinstance(gaussian_res["variance"], float)
    
    laplacian_res = calculate_laplacian_residual(img)
    assert "residual_image" in laplacian_res
    assert "score" in laplacian_res
    assert "variance" in laplacian_res


def test_ocr_extraction_and_matching():
    raw_bytes = create_synthetic_image(tampered=True)
    img = Image.open(io.BytesIO(raw_bytes))
    
    # Extract OCR tokens
    tokens = extract_text_regions(img)
    assert isinstance(tokens, list)
    
    # Test bounding box matching with simulated tamper box
    simulated_tamper_boxes = [[0.4, 0.3, 0.8, 0.5]]
    flagged = match_tamper_regions_with_text(tokens, simulated_tamper_boxes)
    assert isinstance(flagged, list)


def test_pipeline_execution_authentic():
    raw_bytes = create_synthetic_image(tampered=False)
    result = run_full_analysis(raw_bytes)
    
    assert "verdict" in result
    assert result["verdict"] in ["REAL", "UNCERTAIN", "TAMPERED"]
    assert "tamper_score" in result
    assert "confidence" in result
    assert "thresholds" in result
    assert "model" in result
    assert "forensics" in result
    assert "document_info" in result
    assert "artifacts" in result
    assert "ela" in result["artifacts"]
    assert "residual" in result["artifacts"]
    assert "mask" in result["artifacts"]
    assert "annotated" in result["artifacts"]


def test_pipeline_execution_tampered():
    raw_bytes = create_synthetic_image(tampered=True)
    result = run_full_analysis(raw_bytes)
    
    assert "verdict" in result
    assert "regions" in result
    assert "flagged_text" in result
    assert isinstance(result["regions"], list)


def test_invalid_image_payload():
    with pytest.raises(ValueError):
        load_document_image(b"not_an_image_corrupted_data")

    with pytest.raises(ValueError):
        load_document_image(b"")
