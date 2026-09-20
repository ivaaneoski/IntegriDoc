import io
import pytest
from PIL import Image, ImageDraw
import pypdfium2 as pdfium
from fastapi.testclient import TestClient

from apps.api.main import app
from src.utils.image_io import load_document_image, is_pdf

client = TestClient(app)

def create_synthetic_pdf() -> bytes:
    """Creates a synthetic multi-page PDF document in memory."""
    # Create page 1 image
    img1 = Image.new('RGB', (400, 500), color=(255, 255, 255))
    d1 = ImageDraw.Draw(img1)
    d1.text((50, 50), "PDF Document - Page 1", fill=(0, 0, 0))
    d1.text((50, 100), "Official Statement", fill=(50, 50, 50))
    
    # Create page 2 image
    img2 = Image.new('RGB', (400, 500), color=(240, 245, 255))
    d2 = ImageDraw.Draw(img2)
    d2.text((50, 50), "PDF Document - Page 2", fill=(0, 0, 0))
    d2.text((50, 100), "Transaction Log: $4,500.00", fill=(50, 50, 50))
    
    pdf_bytes = io.BytesIO()
    img1.save(pdf_bytes, format="PDF", save_all=True, append_images=[img2])
    return pdf_bytes.getvalue()

def test_pdf_detection_and_rendering():
    pdf_data = create_synthetic_pdf()
    assert is_pdf(pdf_data) is True
    
    # Render page 1
    page1_img, total_pages = load_document_image(pdf_data, page_index=0)
    assert total_pages == 2
    assert isinstance(page1_img, Image.Image)
    assert page1_img.mode == 'RGB'
    assert page1_img.size[0] > 100 and page1_img.size[1] > 100
    
    # Render page 2
    page2_img, _ = load_document_image(pdf_data, page_index=1)
    assert isinstance(page2_img, Image.Image)

def test_analyze_pdf_endpoint():
    pdf_data = create_synthetic_pdf()
    response = client.post(
        "/v1/analyze?page=0",
        files={"file": ("statement.pdf", pdf_data, "application/pdf")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["document_info"]["is_pdf"] is True
    assert data["document_info"]["total_pages"] == 2
    assert data["document_info"]["analyzed_page"] == 1
    assert "verdict" in data
    assert "tamper_score" in data
