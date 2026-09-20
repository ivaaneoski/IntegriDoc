# IntegriDoc

Explainable document forgery detection, tamper localization, and multimodal forensic reporting.

IntegriDoc is an end-to-end document forensics workstation and API designed to detect, localize, and explain digital document manipulation. By combining classical signal forensics (Error Level Analysis and Gaussian Noise Residuals), deep dual-stream neural architectures, semantic segmentation, OCR cross-verification, and Vision-Language Models (VLMs), IntegriDoc provides transparent, verifiable evidence for identity verification, KYC, and document compliance workflows.

---

## Table of Contents

- [Key Capabilities](#key-capabilities)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Environment Configuration](#environment-configuration)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [Output Contract](#output-contract)
- [Testing & Quality Assurance](#testing--quality-assurance)
- [Research & Model Training](#research--model-training)
- [License](#license)

---

## Key Capabilities

- **Multi-Format Document Ingestion**: Seamlessly ingests single and multi-page PDF documents, PNG, JPEG, WEBP, and TIFF files with automatic EXIF orientation normalization and decompression bomb protection.
- **Classical Forensics Extraction**:
  - *Error Level Analysis (ELA)*: Computes compression delta maps and statistical error scores to expose re-saved text blocks and spliced elements.
  - *High-Pass Noise Residuals*: Analyzes Gaussian and Laplacian spatial residuals to detect local sensor noise inconsistencies and digital smoothing.
- **Deep Tamper Classification**:
  - *ResNet-18 Baseline*: Classifies document authenticity using visual feature extractors.
  - *ForensicFusionNet*: Dual-stream architecture fusing RGB pixel data with classical forensic feature maps.
- **Calibrated Three-State Decision Policy**: Implements an enterprise triage policy (`AUTHENTIC`, `UNCERTAIN` for human-in-the-loop analyst review, `TAMPERED`) rather than an arbitrary binary threshold.
- **Precise Anomaly Localization**: Generates pixel-level tamper masks and normalized bounding boxes `[x1, y1, x2, y2]` using U-Net Semantic Segmentation with Grad-CAM activation fallback.
- **OCR Text Cross-Verification**: Intersects localized tamper masks with Tesseract OCR tokens to identify exactly which text fields (e.g., invoice totals, dates, names, account numbers) were altered.
- **Multimodal VLM Forensic Reporting**: Cascades evidence (annotated document, ELA map, noise residual map, and spliced text tokens) to Gemini models for structured, human-readable audit reports.
- **Interactive Forensic Workstation**: A modern, high-contrast web interface for drag-and-drop document audits, real-time telemetry, and visual layer inspection.

---

## System Architecture

```text
                           +-------------------------------+
                           | Document Upload (PDF / Image) |
                           +---------------+---------------+
                                           |
                                           v
                           +-------------------------------+
                           |  Ingestion & Preprocessing    |
                           | (EXIF Transpose / PDF Render) |
                           +---------------+---------------+
                                           |
                    +----------------------+----------------------+
                    v                                             v
       +-------------------------+                   +-------------------------+
       |   Error Level Analysis  |                   | Gaussian Noise Residual |
       |     (Compression)       |                   |   (Variance / Texture)  |
       +------------+------------+                   +------------+------------+
                    |                                             |
                    +----------------------+----------------------+
                                           v
                       +---------------------------------------+
                       |   Neural Classification & Fusion     |
                       |  (ResNet-18 / ForensicFusionNet)      |
                       +-------------------+-------------------+
                                           |
                    +----------------------+----------------------+
                    v                                             v
       +-------------------------+                   +-------------------------+
       |   U-Net Segmentation    |                   |   Grad-CAM Attention    |
       |  (Pixel-Level Heatmap)  |                   |       (Fallback)        |
       +------------+------------+                   +------------+------------+
                    |                                             |
                    +----------------------+----------------------+
                                           v
                       +---------------------------------------+
                       |      OCR Cross-Verification           |
                       |   (Spliced Text Token Matching)       |
                       +-------------------+-------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |  Multimodal VLM Forensic Evaluation   |
                       |         (Gemini / Qwen-VL)            |
                       +-------------------+-------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |   FastAPI Response + Web Workstation  |
                       +---------------------------------------+
```

---

## Repository Structure

```text
IntegriDoc/
|-- apps/
|   |-- api/
|   |   |-- main.py                  # FastAPI application and routing
|   |   `-- services/
|   |       `-- pipeline.py          # Core multi-modal forensic analysis pipeline
|   `-- frontend/
|       |-- app.js                   # Client logic, API integration, and rendering
|       |-- documentation.html       # Forensic methodology documentation
|       |-- index.html               # Forensic Workstation user interface
|       |-- privacy.html             # Privacy policy page
|       `-- support.html             # Technical support page
|-- configs/
|   |-- data.yaml                    # Dataset generation configuration
|   |-- localization.yaml            # Segmentation and localization hyperparams
|   |-- train_resnet18.yaml          # ResNet-18 training configuration
|   `-- vlm.yaml                     # Vision-Language Model configuration
|-- src/
|   |-- data/                        # Synthetic dataset generation & transforms
|   |-- forensics/
|   |   |-- ela.py                   # Error Level Analysis implementation
|   |   `-- residuals.py             # Gaussian and Laplacian noise residuals
|   |-- localization/
|   |   |-- gradcam.py               # Grad-CAM explainability hooks
|   |   `-- metrics.py               # IoU and localization metrics
|   |-- models/
|   |   |-- fusion.py                # ForensicFusionNet dual-stream architecture
|   |   |-- resnet.py                # ResNet-18 classifier architecture
|   |   `-- segmentation.py          # U-Net semantic segmentation network
|   |-- ocr/
|   |   `-- extractor.py             # OCR extraction and box-intersection matching
|   |-- utils/
|   |   `-- image_io.py              # Safe image and PDF loading utilities
|   `-- vlm/
|       |-- prompt_builder.py        # Structured forensic prompt construction
|       `-- qwen_interpreter.py      # Local VLM inference engine
|-- tests/
|   |-- test_api.py                  # FastAPI integration and endpoint tests
|   |-- test_pdf.py                  # PDF document ingestion and rendering tests
|   `-- test_pipeline.py             # Forensics and pipeline unit tests
|-- pyproject.toml                   # Project metadata and dependency configuration
`-- README.md                        # Project documentation
```

---

## Installation & Setup

### Prerequisites

- Python 3.11 or higher
- Tesseract OCR (optional, for text token extraction)
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
  - macOS: `brew install tesseract`
  - Windows: Download installer from official Tesseract GitHub releases.

### 1. Clone the Repository

```bash
git clone https://github.com/ivaaneoski/IntegriDoc.git
cd IntegriDoc
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv

# On Linux / macOS:
source venv/bin/activate

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -e .[dev]
```

---

## Environment Configuration

Create a `.env` file in the root directory:

```env
# Google Gemini VLM API Key (Optional, enables multimodal reasoning)
GEMINI_API_KEY=your_gemini_api_key_here

# Calibrated Decision Thresholds (Optional, defaults to 0.35 and 0.65)
THRESHOLD_LOW=0.35
THRESHOLD_HIGH=0.65
```

---

## Running the Application

### 1. Start the API & Forensic Workstation

Launch the server with live reload:

```bash
python apps/api/main.py
```

Or using Uvicorn directly:

```bash
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Access the Application

- **Forensic Workstation Web UI**: Open `http://localhost:8000` in your web browser.
- **Interactive Swagger Documentation**: Open `http://localhost:8000/docs`.
- **ReDoc Specification**: Open `http://localhost:8000/redoc`.

---

## API Reference

### Health Check

`GET /health`

Returns engine status, active device (`cuda`, `mps`, or `cpu`), loaded models, and threshold configurations.

**Response:**
```json
{
  "status": "ok",
  "version": "1.1.0",
  "device": "cpu",
  "model_type": "resnet",
  "segmentation_loaded": false,
  "vlm_configured": true,
  "thresholds": {
    "low": 0.35,
    "high": 0.65
  }
}
```

---

### Analyze Document

`POST /v1/analyze`

Executes the full forensic analysis pipeline on an uploaded image or PDF document.

**Parameters:**
- `file` (multipart/form-data, required): The document image or PDF file.
- `page` (query integer, optional, default: `0`): Target page index for multi-page PDF files.
- `threshold_low` (query float, optional): Custom lower threshold for `AUTHENTIC` classification.
- `threshold_high` (query float, optional): Custom upper threshold for `TAMPERED` classification.

---

## Output Contract

The API responds with a structured, machine-readable JSON schema:

```json
{
  "verdict": "TAMPERED",
  "tamper_score": 0.892,
  "confidence": 0.892,
  "thresholds": {
    "low": 0.35,
    "high": 0.65
  },
  "model": {
    "name": "ForensicFusionNet",
    "type": "fusion",
    "device": "cpu",
    "segmentation_model_active": true
  },
  "forensics": {
    "ela_score": 0.084,
    "ela_max_diff": 142,
    "noise_residual_score": 8.12,
    "noise_residual_variance": 42.18
  },
  "document_info": {
    "width": 1200,
    "height": 1600,
    "total_pages": 1,
    "is_pdf": false,
    "analyzed_page": 1
  },
  "regions_detected": 1,
  "regions": [
    {
      "id": 1,
      "label": "possible_tampering_region",
      "bbox": [0.35, 0.42, 0.68, 0.49],
      "area_ratio": 0.0231,
      "score": 0.892
    }
  ],
  "flagged_text": [
    {
      "text": "$9,500.00",
      "confidence": 0.92,
      "ocr_bbox": [0.36, 0.43, 0.65, 0.48],
      "tamper_region_index": 0,
      "overlap_ratio": 0.94
    }
  ],
  "vlm_summary": "**Assessment:** Strong visual and statistical indications of document manipulation...",
  "artifacts": {
    "ela": "<base64_jpeg>",
    "residual": "<base64_jpeg>",
    "mask": "<base64_jpeg>",
    "annotated": "<base64_jpeg>"
  }
}
```

*Note: Bounding boxes are normalized to `[0.0, 1.0]` as `[x1, y1, x2, y2]`.*

---

## Testing & Quality Assurance

IntegriDoc includes a comprehensive automated test suite covering unit forensics, PDF extraction, model fallbacks, and API integration.

Run the test suite with verbose output:

```bash
pytest tests/ -v
```

### Test Coverage

- **`tests/test_api.py`**: API health diagnostics, UI serving, valid image analysis, empty file rejection, and corrupt payload validation.
- **`tests/test_pdf.py`**: Multi-page PDF generation, memory rendering via `pypdfium2`, and end-to-end PDF analysis via `/v1/analyze`.
- **`tests/test_pipeline.py`**: ELA computation, Gaussian and Laplacian residual calculation, OCR token extraction, box intersection matching, and synthetic authentic vs. tampered document analysis.

---

## Research & Model Training

### Generating Synthetic Forgery Data

IntegriDoc provides synthetic document generators that apply controlled digital tampering (text splicing, copy-move, digital whiteout, font mismatching) to benchmark models:

```bash
python src/data/generator.py --config configs/data.yaml
```

### Training Deep Classifiers

To train baseline ResNet-18 or ForensicFusionNet models:

```bash
python src/train.py --config configs/train_resnet18.yaml
```

### Training U-Net Segmentation

To train the pixel-level semantic localization network:

```bash
python src/train_segmentation.py --config configs/localization.yaml
```

Trained checkpoints are saved to `results/runs/` and `results/segmentation/`.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
