# IntegriDoc: Explainable Document Forgery Detection & Forensic Localization

IntegriDoc is a multimodal document forensics platform and API designed to detect, localize, and explain digital document manipulations. Built for identity-verification (KYC) systems, financial audit pipelines, and forensic researchers, IntegriDoc bridges the gap between opaque deep-learning classifiers and verifiable forensic science by unifying classical signal analysis, dual-stream neural architectures, pixel-level semantic segmentation, OCR text grounding, and Vision-Language Models (VLMs).

---

## Table of Contents

- [1. Executive Summary & Problem Context](#1-executive-summary--problem-context)
- [2. Multi-Layered Forensic Architecture](#2-multi-layered-forensic-architecture)
  - [Layer 1: Error Level Analysis (ELA)](#layer-1-error-level-analysis-ela)
  - [Layer 2: High-Pass Noise Residuals](#layer-2-high-pass-noise-residuals)
  - [Layer 3: Neural Classification & Dual-Stream Fusion](#layer-3-neural-classification--dual-stream-fusion)
  - [Layer 4: Pixel-Level Localization & Grad-CAM](#layer-4-pixel-level-localization--grad-cam)
  - [Layer 5: OCR Text Grounding & Splicing Correlation](#layer-5-ocr-text-grounding--splicing-correlation)
  - [Layer 6: Multimodal Vision-Language Reasoning](#layer-6-multimodal-vision-language-reasoning)
- [3. Calibrated Three-State Decision Policy](#3-calibrated-three-state-decision-policy)
- [4. Synthetic Data Generation Engine](#4-synthetic-data-generation-engine)
- [5. System Architecture Flow](#5-system-architecture-flow)
- [6. Repository File Layout](#6-repository-file-layout)
- [7. Installation & Environment Setup](#7-installation--environment-setup)
- [8. Running the Application](#8-running-the-application)
- [9. REST API Reference](#9-rest-api-reference)
- [10. JSON Response Contract & Field Definitions](#10-json-response-contract--field-definitions)
- [11. Interactive Forensic Workstation](#11-interactive-forensic-workstation)
- [12. Testing & Verification Suite](#12-testing--verification-suite)
- [13. Model Training & Fine-Tuning Guide](#13-model-training--fine-tuning-guide)
- [14. License & Research Citation](#14-license--research-citation)

---

## 1. Executive Summary & Problem Context

Digital document forgery—ranging from altered dates and amounts on bank statements to fabricated credentials on identity cards—poses a major vulnerability in automated onboarding and compliance systems. Traditional machine-learning approaches suffer from key shortcomings:

1. **Opaque Decisions**: Standard classifiers output a single probability score without explaining which visual regions caused the prediction.
2. **Adversarial Fragility**: Pure RGB models can be tricked by image compression, scanning artifacts, or uniform background textures.
3. **Lack of Text Grounding**: Existing tools fail to correlate optical anomalies with the semantic meaning of the manipulated text fields.

IntegriDoc addresses these challenges by processing documents across multiple orthogonal inspection modalities:

- **Signal Level**: Exposing compression boundaries and noise variance anomalies.
- **Visual Representation Level**: Extracting deep spatial embeddings across RGB and forensic channels.
- **Semantic Level**: Localizing exact pixel coordinates, segmenting altered text tokens, and generating structured natural-language inspection reports.

---

## 2. Multi-Layered Forensic Architecture

IntegriDoc evaluates documents through a six-stage forensic pipeline:

```text
+------------------------------------------------------------------------------------+
|                                Document Input                                      |
|                       (PDF Multi-Page / PNG / JPEG / TIFF)                         |
+-----------------------------------------+------------------------------------------+
                                          |
                                          v
+------------------------------------------------------------------------------------+
|                         Pre-Processing & Validation                                |
|        - EXIF Orientation Transposition (ImageOps.exif_transpose)                  |
|        - PDF Rasterization at 200 DPI (pypdfium2)                                  |
|        - Decompression Bomb Defense (Max 60M Pixels)                               |
+-------------------+--------------------------------------------+-------------------+
                    |                                            |
                    v                                            v
+---------------------------------------+    +---------------------------------------+
|  Layer 1: Error Level Analysis (ELA)  |    |  Layer 2: Gaussian Noise Residuals    |
|  - JPEG Recompression delta map       |    |  - High-pass spatial filtering        |
|  - Dynamic 95th percentile scaling    |    |  - Local variance & texture analysis  |
|  - Max delta & mean error metrics     |    |  - Discontinuity detection            |
+-------------------+-------------------+    +-------------------+-------------------+
                    |                                            |
                    +--------------------+-----------------------+
                                         |
                                         v
+------------------------------------------------------------------------------------+
|  Layer 3: Neural Classification & Dual-Stream Feature Fusion                       |
|  - Stream A: RGB Backbone (ResNet-18 spatial features)                             |
|  - Stream B: Forensic Backbone (Concatenated ELA + Residual 2-channel tensor)      |
|  - Late Fusion: Concatenated linear projection with temperature-scaled softmax     |
+----------------------------------------+-------------------------------------------+
                                         |
                                         v
+------------------------------------------------------------------------------------+
|  Layer 4: Semantic Localization & Anomaly Mapping                                  |
|  - Primary: U-Net Semantic Segmentation (Pixel-level binary tamper mask)           |
|  - Secondary Fallback: Grad-CAM Activation Heatmaps & ELA Contour Analysis         |
|  - Normalized Bounding Box Extraction: [x1, y1, x2, y2] in [0.0, 1.0]              |
+----------------------------------------+-------------------------------------------+
                                         |
                                         v
+------------------------------------------------------------------------------------+
|  Layer 5: OCR Text Grounding & Splicing Correlation                                |
|  - Text token extraction with normalized bounding boxes (pytesseract)             |
|  - Spatial Intersection-over-Area (IoA) matching with tamper masks                 |
|  - Flags specific altered fields: dates, dollar amounts, identity numbers          |
+----------------------------------------+-------------------------------------------+
                                         |
                                         v
+------------------------------------------------------------------------------------+
|  Layer 6: Multimodal Vision-Language Reasoning (Gemini VLM)                        |
|  - Multi-image prompt ingestion (Annotated Document + ELA Map + Noise Map)         |
|  - Contextual validation of typography, layout, numbers, and optical evidence      |
|  - Multi-tier fallback cascade with structured audit report output                 |
+------------------------------------------------------------------------------------+
```

---

### Layer 1: Error Level Analysis (ELA)

When an image is saved as a JPEG, each 8x8 pixel block is transformed into the frequency domain via Discrete Cosine Transform (DCT) and quantized according to a quality matrix. If an external element (such as a spliced number or altered name) is pasted into a pre-existing JPEG and re-saved, the pasted region undergoes its first compression cycle while the surrounding image undergoes an additional compression cycle.

IntegriDoc computes ELA by recompressing the image at a uniform quality level $Q=90$:

$$\Delta(x, y) = |I(x, y) - \mathcal{J}_{Q=90}(I)(x, y)|$$

To make subtle compression differences visually distinct and computationally accessible, dynamic percentile scaling is applied:

$$S = \frac{255}{\max(1.0, P_{95}(\Delta))}$$

$$I_{\text{ELA}}(x, y) = \text{clip}\left(\Delta(x, y) \cdot S, 0, 255\right)$$

The module calculates:
- `ela_max_diff`: Peak difference magnitude between original and recompressed frames.
- `ela_score`: Mean normalized error scalar across the entire canvas.

---

### Layer 2: High-Pass Noise Residuals

Camera sensors and document scanners leave intrinsic high-frequency noise patterns (Photo-Response Non-Uniformity or PRNU) across a document page. Digital tampering (such as erasing text with an eye-dropper brush, digital whiteout, or pasting synthetic text) alters or smooths this micro-texture.

IntegriDoc isolates high-frequency spatial components by subtracting a Gaussian-smoothed version of the image:

$$R_G(x, y) = I(x, y) - (G_\sigma * I)(x, y)$$

Where $G_\sigma$ is a Gaussian kernel with standard deviation $\sigma = 2.0$. The module computes:
- `noise_residual_score`: Mean high-frequency magnitude across the image.
- `noise_residual_variance`: Spatial variance of the high-frequency residual, exposing sudden drops in noise density where digital erasure occurred.

In addition, Laplacian second-order spatial derivatives ($\nabla^2 I$) are available to evaluate edge-transition sharpness across text boundaries.

---

### Layer 3: Neural Classification & Dual-Stream Fusion

IntegriDoc implements two classification backbones:

1. **ResNet-18 Binary Classifier**: Processes standard 3-channel RGB inputs normalized with ImageNet statistics:

$$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

2. **ForensicFusionNet (Dual-Stream Architecture)**:
   - **Stream 1 (RGB Branch)**: ResNet-18 backbone extracting 512-dimensional semantic spatial representations.
   - **Stream 2 (Forensic Branch)**: ResNet-18 backbone processing a 2-channel tensor consisting of the normalized ELA grayscale map and the Gaussian residual magnitude map.
   - **Fusion Head**: Concatenates both 512-dimensional feature vectors into a 1024-dimensional joint representation, followed by a multi-layer perceptron with dropout ($p=0.3$) and linear projection to 2 output logits ($[\text{logits}_{\text{authentic}}, \text{logits}_{\text{tampered}}]$).

Inference applies temperature scaling ($T=2.0$) to calibrate softmax probabilities and prevent overconfident extreme predictions:

$$P(\text{Tampered}) = \frac{e^{\text{logit}_{\text{tampered}} / T}}{e^{\text{logit}_{\text{authentic}} / T} + e^{\text{logit}_{\text{tampered}} / T}}$$

---

### Layer 4: Pixel-Level Localization & Grad-CAM

To isolate exact physical forgery locations, IntegriDoc employs a hierarchical localization mechanism:

1. **U-Net Semantic Segmentation**:
   - Encoder: ResNet-18 pre-trained backbone.
   - Decoder: Progressive upsampling with skip-connections from intermediate encoder stages.
   - Output: 1-channel continuous sigmoid probability mask $M \in [0, 1]^{H \times W}$.
   - Binary thresholding at $\tau = 0.5$ followed by morphological contour extraction (`cv2.findContours`) produces tight bounding boxes around spliced regions.

2. **Grad-CAM Attention Fallback**:
   - When segmentation checkpoints are uninitialized, IntegriDoc computes Gradient-weighted Class Activation Mapping (Grad-CAM) on the final convolutional layer (`layer4`):

$$\alpha_k = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial Y^{\text{tampered}}}{\partial A_{i, j}^k}$$

$$L_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k \alpha_k A^k\right)$$

3. **Coordinate Normalization**: All bounding box coordinates are converted into standard relative coordinates $[x_1, y_1, x_2, y_2] \in [0.0, 1.0]^4$, ensuring client compatibility regardless of native document resolution.

---

### Layer 5: OCR Text Grounding & Splicing Correlation

Detecting that a document contains an anomaly is only half the battle; investigators must know **which text field was changed**.

1. IntegriDoc runs OCR text extraction via Tesseract, obtaining word tokens, bounding boxes, and recognition confidence scores.
2. For every detected tamper bounding box $B_{\text{tamper}} = [tx_1, ty_1, tx_2, ty_2]$ and OCR token box $B_{\text{ocr}} = [ox_1, oy_1, ox_2, oy_2]$, the intersection ratio is calculated:

$$\text{IntersectionArea} = \max(0, \min(ox_2, tx_2) - \max(ox_1, tx_1)) \times \max(0, \min(oy_2, ty_2) - \max(oy_1, ty_1))$$

$$\text{OverlapRatio} = \frac{\text{IntersectionArea}}{\text{Area}(B_{\text{ocr}})}$$

3. If $\text{OverlapRatio} \ge 0.05$, the text token is flagged (e.g., `"$9,500.00"`, `"2026-11-30"`, `"John Doe"`), directly tying visual and signal anomalies to document semantic fields.

---

### Layer 6: Multimodal Vision-Language Reasoning

Flagged anomalies and visual evidence are synthesized into a structured natural-language audit report using a Vision-Language Model cascade:

1. **Multi-Image Prompt Composition**:
   - **Image 1**: The original document overlaid with high-visibility red bounding boxes around localized tamper zones.
   - **Image 2**: The Error Level Analysis (ELA) compression difference map.
   - **Image 3**: The Gaussian high-pass noise residual map.
2. **Contextual Metadata**: Injects quantitative classifier risk scores, ELA delta values, noise variance scalars, and the list of flagged OCR text tokens into the system prompt.
3. **Model Cascade**: Sequentially attempts execution across Gemini models (`gemini-2.5-flash` $\rightarrow$ `gemini-1.5-flash` $\rightarrow$ `gemini-3.7-flash`).
4. **Offline Rule-Based Fallback**: If no API key is configured or the system is operating in an air-gapped environment, a deterministic rule-based forensic report is generated directly from the statistical metrics.

---

## 3. Calibrated Three-State Decision Policy

Standard binary classification ($P > 0.5$) creates unacceptable operational risks in KYC and fraud detection: borderline documents near $0.51$ are labeled definitively fraudulent, while documents at $0.49$ pass undetected.

IntegriDoc establishes an explicit three-tier triage policy:

```text
Score:  0.0 ---------------- [0.35] ---------------- [0.65] ---------------- 1.0
             AUTHENTIC              UNCERTAIN               TAMPERED
         (Automated Pass)       (Human KYC Review)      (Automated Reject)
```

- **`AUTHENTIC`** ($\text{Score} < \tau_{\text{low}}$, Default: $< 0.35$):
  - Document displays consistent compression history, uniform noise distribution, and no typography discontinuities.
  - Action: Approved for automated processing.
- **`UNCERTAIN`** ($\tau_{\text{low}} \le \text{Score} \le \tau_{\text{high}}$, Default: $0.35 - 0.65$):
  - Document shows minor anomalies that may result from benign causes (multiple scan cycles, heavy recompression, or mobile camera sensor artifacts).
  - Action: Escalated to a human fraud analyst for secondary verification.
- **`TAMPERED`** ($\text{Score} > \tau_{\text{high}}$, Default: $> 0.65$):
  - Strong visual, structural, and signal evidence of digital splicing, font alteration, or digital erasure.
  - Action: Rejected and logged for fraud investigation.

Thresholds are configurable at runtime via environment variables or per-request API parameters.

---

## 4. Synthetic Data Generation Engine

To train deep models without relying on scarce manual forgery datasets, IntegriDoc includes a parameterized synthetic document generator in `src/data/generator.py`:

- **Text Splicing**: Replaces numeric fields, dates, and names with alternate typography rendered at differing font weights and kerning.
- **Copy-Move Forgery**: Duplicates genuine stamps, signatures, or characters across different coordinates on the document canvas.
- **Digital Inpainting / Whiteout**: Blurs and fills text regions to simulate erasure before new text insertion.
- **Noise & Compression Degradation**: Applies localized Gaussian noise mismatches, bilateral smoothing, and multi-generation JPEG recompression.
- **Ground-Truth Mask Export**: Automatically generates pixel-accurate binary masks for segmentation training.

---

## 5. System Architecture Flow

```text
+-------------------------------------------------------------------------------+
|                             Client Request Layer                              |
|           - Browser Upload (Forensic Workstation Web Interface)               |
|           - Direct REST API Calls (cURL / Python SDK / Microservices)          |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                             FastAPI Gateway Layer                             |
|           - CORS Middleware & Static File Serving                             |
|           - Health & Engine Telemetry Routing (/health)                       |
|           - Asynchronous File Ingestion & Validation (/v1/analyze)            |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                       Pipeline Orchestration Layer                            |
|                            (pipeline.py)                                      |
|   +-----------------------------------------------------------------------+   |
|   | 1. Safe Load (PDF / Image) -> EXIF Transpose -> RGB Normalization     |   |
|   | 2. Compute ELA Map & Noise Residual Map                               |   |
|   | 3. Deep Classifier Inference -> Calibrated Tamper Score               |   |
|   | 4. Semantic Localization -> Extract Normalized [x1, y1, x2, y2]       |   |
|   | 5. Run OCR -> Cross-Reference Overlapping Text Tokens                 |   |
|   | 6. Invoke Multimodal VLM -> Generate Audit Narrative                  |   |
|   | 7. Serialize Artifacts (Base64 JPEG) -> Return JSON Response          |   |
|   +-----------------------------------------------------------------------+   |
+---------------------------------------+---------------------------------------+
                                        |
                    +-------------------+-------------------+
                    v                                       v
+---------------------------------------+   +-----------------------------------+
|      FastAPI JSON API Response        |   |   Forensic Workstation UI Render  |
|  - Verdict & Calibrated Risk Score    |   |   - Interactive Verdict Badge     |
|  - Normalized Bounding Box Regions    |   |   - Forensic Quad Metrics HUD     |
|  - Flagged Spliced Text Tokens        |   |   - Spliced Text Chip Indicators  |
|  - Formatted Markdown Audit Report    |   |   - Expandable VLM Report Reader  |
|  - Base64 ELA, Noise & Mask Artifacts |   |   - 4-Tile Visual Evidence Gallery|
+---------------------------------------+   +-----------------------------------+
```

---

## 6. Repository File Layout

```text
IntegriDoc/
|-- apps/
|   |-- api/
|   |   |-- __init__.py
|   |   |-- main.py                      # FastAPI server routes, health checks & error handlers
|   |   `-- services/
|   |       |-- __init__.py
|   |       `-- pipeline.py              # End-to-end multi-modal forensic analysis pipeline
|   `-- frontend/
|       |-- app.js                       # Frontend state machine, API client, and dynamic DOM renderer
|       |-- documentation.html           # In-app forensic methodology and architectural reference
|       |-- index.html                   # High-contrast Forensic Workstation user interface
|       |-- privacy.html                 # Data privacy & retention statement
|       `-- support.html                 # Enterprise support and contact directory
|-- checkpoints/                         # Directory for persistent model weight files
|-- configs/
|   |-- data.yaml                        # Configuration for synthetic dataset generator
|   |-- localization.yaml                # Hyperparameters for segmentation model training
|   |-- train_efficientnet.yaml          # EfficientNet classifier training configuration
|   |-- train_resnet18.yaml              # ResNet-18 classifier training configuration
|   `-- vlm.yaml                         # VLM prompt templates and inference parameters
|-- data/
|   |-- manifests/                       # Dataset split manifests (train, validation, test)
|   |-- processed/                       # Pre-processed and normalized document tensors
|   |-- raw/                             # Raw source document images (uncommitted)
|   `-- synthetic/                       # Locally generated synthetic forgery benchmarks
|-- notebooks/                           # Jupyter notebooks for forensic exploration & evaluation
|-- reports/                             # Generated benchmark evaluation reports & confusion matrices
|-- results/
|   |-- runs/                            # Classifier checkpoints (best.pt) and training logs
|   `-- segmentation/                    # U-Net segmentation weights (best_seg.pt)
|-- scripts/                             # Utility scripts for smoke testing and batch inference
|-- src/
|   |-- __init__.py
|   |-- data/
|   |   |-- __init__.py
|   |   |-- dataset.py                   # PyTorch Dataset classes for document forensics
|   |   `-- generator.py                 # Synthetic document manipulation and forgery generator
|   |-- evaluation/
|   |   |-- __init__.py
|   |   `-- metrics.py                   # Accuracy, F1-score, precision, recall, and ROC-AUC metrics
|   |-- forensics/
|   |   |-- __init__.py
|   |   |-- ela.py                       # Error Level Analysis computation & dynamic scaling
|   |   `-- residuals.py                 # Gaussian and Laplacian high-pass spatial residuals
|   |-- inference/
|   |   `-- __init__.py
|   |-- localization/
|   |   |-- __init__.py
|   |   |-- gradcam.py                   # PyTorch Grad-CAM hooks and heatmap overlay generator
|   |   `-- metrics.py                   # Intersection over Union (IoU) localization metrics
|   |-- models/
|   |   |-- __init__.py
|   |   |-- fusion.py                    # Dual-stream ForensicFusionNet architecture
|   |   |-- resnet.py                    # ResNet-18 binary classification network
|   |   `-- segmentation.py              # ResNet18-UNet document segmentation architecture
|   |-- ocr/
|   |   |-- __init__.py
|   |   `-- extractor.py                 # Tesseract OCR extraction and tamper-box overlap matching
|   |-- utils/
|   |   |-- __init__.py
|   |   `-- image_io.py                  # Safe multi-format image and PDF document loading
|   `-- vlm/
|       |-- __init__.py
|       |-- prompt_builder.py            # Multimodal prompt formatting for forensic reasoning
|       `-- qwen_interpreter.py          # Local open-weight VLM interface (Qwen2-VL)
|-- tests/
|   |-- test_api.py                      # Integration tests for FastAPI endpoints and error handling
|   |-- test_pdf.py                      # Tests for multi-page PDF ingestion and rasterization
|   `-- test_pipeline.py                 # Unit tests for forensics, OCR matching, and pipeline execution
|-- pyproject.toml                       # Python package configuration, dependencies, and test settings
`-- README.md                            # Comprehensive project documentation
```

---

## 7. Installation & Environment Setup

### System Prerequisites

- Python 3.11 or higher
- Tesseract OCR (recommended for text token extraction)
  - Ubuntu / Debian: `sudo apt-get update && sudo apt-get install -y tesseract-ocr`
  - macOS (Homebrew): `brew install tesseract`
  - Windows: Install via official Windows binaries and ensure `tesseract.exe` is in your system `PATH`.

### Step-by-Step Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ivaaneoski/IntegriDoc.git
   cd IntegriDoc
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv venv

   # Linux / macOS:
   source venv/bin/activate

   # Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Package Dependencies**:
   ```bash
   pip install -e .[dev]
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   # Google Gemini VLM API Key (Optional; enables multimodal forensic reasoning)
   GEMINI_API_KEY=your_gemini_api_key_here

   # Calibrated Risk Thresholds (Optional; defaults to 0.35 and 0.65)
   THRESHOLD_LOW=0.35
   THRESHOLD_HIGH=0.65
   ```

---

## 8. Running the Application

### Start the Server

Run the application through Python:

```bash
python apps/api/main.py
```

Or using Uvicorn with multi-worker support:

```bash
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Access Points

- **Interactive Forensic Workstation**: `http://localhost:8000/`
- **Swagger Interactive API Documentation**: `http://localhost:8000/docs`
- **ReDoc API Reference**: `http://localhost:8000/redoc`
- **In-App Forensic Documentation**: `http://localhost:8000/documentation`

---

## 9. REST API Reference

### 1. Health & Telemetry Check

`GET /health`

Returns server health, active computing hardware, loaded model types, and threshold configurations.

**cURL Example:**
```bash
curl -X GET http://localhost:8000/health
```

**Response (HTTP 200 OK):**
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

### 2. Document Analysis

`POST /v1/analyze`

Executes the complete forensic analysis pipeline on an uploaded image or PDF document.

**Query Parameters:**
- `page` (integer, optional, default: `0`): Target page index for multi-page PDF documents (0-indexed).
- `threshold_low` (float, optional, default: `0.35`): Custom lower threshold for `AUTHENTIC` classification.
- `threshold_high` (float, optional, default: `0.65`): Custom upper threshold for `TAMPERED` classification.

**Request Body (multipart/form-data):**
- `file` (binary, required): The document image (`PNG`, `JPEG`, `WEBP`, `TIFF`) or PDF document.

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/v1/analyze?page=0" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/invoice_document.pdf"
```

**Python SDK Example:**
```python
import requests

url = "http://localhost:8000/v1/analyze"
params = {"page": 0, "threshold_low": 0.35, "threshold_high": 0.65}

with open("statement.pdf", "rb") as doc_file:
    files = {"file": ("statement.pdf", doc_file, "application/pdf")}
    response = requests.post(url, params=params, files=files)
    
result = response.json()
print(f"Verdict: {result['verdict']} (Score: {result['tamper_score']})")
```

---

## 10. JSON Response Contract & Field Definitions

```json
{
  "verdict": "TAMPERED",
  "tamper_score": 0.8924,
  "confidence": 0.8924,
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
    "ela_score": 0.0841,
    "ela_max_diff": 142,
    "noise_residual_score": 8.125,
    "noise_residual_variance": 42.1804
  },
  "document_info": {
    "width": 1200,
    "height": 1600,
    "total_pages": 3,
    "is_pdf": true,
    "analyzed_page": 1
  },
  "regions_detected": 1,
  "regions": [
    {
      "id": 1,
      "label": "possible_tampering_region",
      "bbox": [0.352, 0.421, 0.684, 0.495],
      "area_ratio": 0.0245,
      "score": 0.892
    }
  ],
  "flagged_text": [
    {
      "text": "$9,500.00",
      "confidence": 0.94,
      "ocr_bbox": [0.358, 0.428, 0.672, 0.489],
      "tamper_region_index": 0,
      "overlap_ratio": 0.965
    }
  ],
  "vlm_summary": "**Assessment:** Strong visual and statistical indications of document manipulation...",
  "artifacts": {
    "ela": "<base64_encoded_jpeg_string>",
    "residual": "<base64_encoded_jpeg_string>",
    "mask": "<base64_encoded_jpeg_string>",
    "annotated": "<base64_encoded_jpeg_string>"
  }
}
```

### Response Field Descriptions

| Field | Type | Description |
| :--- | :--- | :--- |
| `verdict` | String | Tri-state classification decision: `AUTHENTIC`, `UNCERTAIN`, or `TAMPERED`. |
| `tamper_score` | Float | Calibrated probability of document manipulation ($0.0 - 1.0$). |
| `confidence` | Float | Degree of certainty associated with the assigned verdict. |
| `thresholds` | Object | Active decision boundaries applied during evaluation (`low` and `high`). |
| `model` | Object | Metadata regarding the active inference model, architecture, and hardware device. |
| `forensics` | Object | Statistical metrics from classical Error Level Analysis and Noise Residual extraction. |
| `document_info` | Object | Document dimensions, page count, format type, and analyzed page index. |
| `regions_detected`| Integer | Total count of localized suspicious physical regions. |
| `regions` | Array | List of localized anomaly bounding boxes formatted as $[x_1, y_1, x_2, y_2]$ normalized to $[0.0, 1.0]$. |
| `flagged_text` | Array | OCR text tokens spatially intersecting with detected tamper bounding boxes. |
| `vlm_summary` | String | Structured multimodal forensic report in Markdown format. |
| `artifacts` | Object | Base64-encoded JPEG strings for ELA, Noise Residual, Mask, and Annotated Overlay layers. |

---

## 11. Interactive Forensic Workstation

The built-in web interface (`apps/frontend/`) provides an analyst-first workstation designed for inspection:

1. **Document Ingestion Zone**: Drag-and-drop support for PDF documents and image files with real-time file metadata preview.
2. **Dynamic Verdict Card**: Color-coded verdict badge displaying `AUTHENTIC` (Acid Green), `UNCERTAIN` (Amber), or `TAMPERED` (Crimson) alongside calibrated risk scores.
3. **Forensic Quad Metrics HUD**: Real-time display of Model Confidence %, Anomaly Region Count, ELA Max Delta, and Noise Residual Variance.
4. **Spliced Text Indicator**: Highlights OCR tokens located directly inside tamper zones for rapid verification of altered figures.
5. **Multimodal Report Reader**: Smooth markdown viewer rendering structured VLM forensic analysis.
6. **4-Tile Visual Evidence Gallery**: High-resolution inspectable tiles for ELA compression maps, Gaussian noise maps, binary tamper masks, and annotated bounding box overlays.

---

## 12. Testing & Verification Suite

IntegriDoc maintains an automated test suite covering unit forensics, PDF handling, model fallbacks, and API integration.

### Running All Tests

```bash
pytest tests/ -v
```

### Test Suite Structure

```text
tests/
|-- test_api.py         # Tests /health, /v1/analyze, empty file rejection, and corrupt byte handling
|-- test_pdf.py         # Tests in-memory multi-page PDF generation, rasterization, and API processing
`-- test_pipeline.py    # Tests ELA extraction, noise variance, OCR matching, and synthetic document classification
```

---

## 13. Model Training & Fine-Tuning Guide

### 1. Generating Synthetic Training Data

Generate benchmark datasets of authentic and manipulated documents:

```bash
python src/data/generator.py --config configs/data.yaml
```

### 2. Training the ResNet-18 Classifier

Train the baseline binary classifier:

```bash
python src/train.py --config configs/train_resnet18.yaml
```

Checkpoints will be saved to `results/runs/resnet18_baseline/best.pt`.

### 3. Training the U-Net Segmentation Network

Train the pixel-level semantic tamper localization model:

```bash
python src/train_segmentation.py --config configs/localization.yaml
```

Checkpoints will be saved to `results/segmentation/unet_segmentation/best_seg.pt`.

---

## 14. License & Research Citation

This project is licensed under the MIT License. See the `LICENSE` file for details.

If you use IntegriDoc in your research or production systems, please cite:

```bibtex
@software{integridoc2026,
  author = {IntegriDoc Research Team},
  title = {IntegriDoc: Explainable Document Forgery Detection and Multimodal Forensic Localization},
  year = {2026},
  url = {https://github.com/ivaaneoski/IntegriDoc}
}
```
