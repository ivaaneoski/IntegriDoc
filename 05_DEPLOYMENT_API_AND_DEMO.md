# Stage 5 — Deployment, API & Demo

## 1. Deployment stack

Recommended:

- **FastAPI** — inference API;
- **Pydantic** — request/response validation;
- **PyTorch** — model execution;
- **OpenCV/Pillow** — image handling;
- **React/Next.js or simple HTML frontend** — UI;
- **Docker** — reproducible runtime.

A Streamlit/Gradio prototype can be used first, but FastAPI is preferred for the portfolio-grade backend because it demonstrates an actual service boundary.

## 2. API endpoints

### `GET /health`
Returns runtime health and loaded model version.

### `GET /models`
Returns loaded detector/localizer/VLM versions.

### `POST /v1/analyze`
Accepts multipart image upload and optional inference flags.

Request flags:

```json
{
  "run_ocr": true,
  "run_vlm": true,
  "return_artifacts": true
}
```

Response follows the Stage 1 JSON contract.

### `POST /v1/forensics/ela`
Returns ELA artifact metadata and image.

### `POST /v1/forensics/residual`
Returns residual artifact metadata and image.

## 3. Input validation

Reject:

- unsupported MIME types;
- oversized files;
- corrupted images;
- images with zero/invalid dimensions.

Recommended starting limit: 10 MB upload.

Convert to RGB for model inference while retaining the original upload separately for forensic comparison in memory only.

## 4. Inference pipeline

```text
Upload
 ↓
Validate
 ↓
Decode
 ↓
Normalize / preserve original
 ↓
ELA + residuals
 ↓
Classifier
 ↓
Localization
 ↓
Grad-CAM
 ↓
OCR (optional)
 ↓
VLM (optional)
 ↓
Evidence aggregation
 ↓
JSON response
```

The inference service must be deterministic for the same model/checkpoint/config where possible.

## 5. Evidence aggregation

Start with a transparent rule-based aggregator.

Example conceptual score:

```text
final_score =
    0.70 * deep_model_score
  + 0.15 * ela_anomaly_score
  + 0.15 * residual_anomaly_score
```

Do **not** claim this weighting is statistically optimal. Treat it as a baseline and compare against:

- deep model alone;
- learned meta-classifier;
- calibrated ensemble.

Better P1 implementation: train a small logistic-regression/meta-model on validation features.

## 6. UI requirements

The demo should look like an AI forensic workstation, not a generic image classifier.

### Main layout

**Left:** uploaded document.

**Right:**
- verdict badge;
- tamper score;
- model name/version;
- suspicious regions;
- forensic evidence summary.

### Evidence tabs

1. Original
2. ELA
3. Noise residual
4. Grad-CAM
5. Localization mask
6. OCR regions
7. VLM explanation

### Final verdict area

Example:

```text
TAMPERED
Tamper score: 0.92

Most suspicious area:
DOB / ID number region

Evidence:
• localized activation in the edited field
• elevated local ELA difference
• residual inconsistency at the same region

Recommended action:
MANUAL REVIEW
```

## 7. UI warnings

Always display:

> “This is a research prototype. A model score is not legal or identity-proof evidence.”

Do not expose sensitive model internals to imply certainty.

## 8. Artifact storage

For local demo mode:

```text
artifacts/
├── uploads/
├── ela/
├── residuals/
├── heatmaps/
├── masks/
└── reports/
```

Use random IDs rather than original file names.

Do not persist uploaded documents by default. If persistence is implemented later, state retention policy clearly.

## 9. Docker

Create:

- `Dockerfile.api`
- `Dockerfile.web`
- `docker-compose.yml`

The API container should support CPU inference for smoke testing.

GPU deployment must be optional and separately documented.

## 10. Observability

Log:

- request ID;
- model version;
- preprocessing success/failure;
- inference latency by module;
- output verdict;
- error code.

Never log raw uploaded document content or OCR text by default.

## 11. API tests

Minimum integration tests:

- health endpoint;
- valid image upload;
- invalid MIME type;
- oversized payload;
- corrupted image;
- deterministic response schema;
- missing model checkpoint;
- optional module disabled/enabled.

## 12. Demo dataset

Commit only a tiny set of synthetic demonstration examples.

Include:

- authentic document;
- text replacement;
- copy-move;
- photo/signature replacement;
- post-processed tampered sample;
- hard-negative authentic sample.

Make it impossible for a reviewer to confuse the examples with real private documents.
