import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from apps.api.main import app

client = TestClient(app)

def make_test_image_bytes() -> bytes:
    img = Image.new('RGB', (200, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "IntegriDoc Test", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "device" in data
    assert "model_type" in data
    assert "thresholds" in data

def test_ui_endpoints():
    response = client.get("/")
    assert response.status_code == 200
    assert "INTEGRIDOC" in response.text

    docs_resp = client.get("/documentation")
    assert docs_resp.status_code == 200

def test_analyze_endpoint_valid_image():
    img_bytes = make_test_image_bytes()
    response = client.post(
        "/v1/analyze",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "verdict" in data
    assert "tamper_score" in data
    assert "artifacts" in data
    assert "ela" in data["artifacts"]

def test_analyze_endpoint_empty_file():
    response = client.post(
        "/v1/analyze",
        files={"file": ("empty.jpg", b"", "image/jpeg")}
    )
    assert response.status_code == 400

def test_analyze_endpoint_corrupt_file():
    response = client.post(
        "/v1/analyze",
        files={"file": ("corrupt.jpg", b"corrupted_garbage_bytes", "image/jpeg")}
    )
    assert response.status_code == 400
