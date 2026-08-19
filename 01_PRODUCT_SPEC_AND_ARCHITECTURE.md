# Stage 1 — Product Specification & System Architecture

## 1. Product identity

**Product:** IntegriDoc

**Tagline:** Explainable document forgery detection and localization.

**Target users:**
- KYC/fraud analysts;
- identity-verification systems;
- document-processing engineers;
- ML researchers evaluating image-forensics models.

**Project positioning:** research prototype, not a production identity-verification product.

## 2. Core user journey

1. User uploads a document image.
2. System validates file type, dimensions, size, and readability.
3. Preprocessing normalizes the image without destroying forensic evidence.
4. Classical forensic modules generate ELA and noise-residual maps.
5. Deep classifier predicts tampered vs real.
6. Localization module generates a tamper heatmap/mask.
7. Optional OCR extracts visible text and bounding boxes.
8. Optional VLM reviews the document plus model evidence and produces a structured explanation.
9. Aggregator combines detector outputs into a final verdict.
10. API returns JSON and UI renders the evidence.

## 3. Required output contract

```json
{
  "verdict": "TAMPERED",
  "tamper_score": 0.923,
  "threshold": 0.50,
  "model": {
    "name": "resnet18_ela_fusion",
    "version": "1.0.0"
  },
  "forensics": {
    "ela_score": 0.81,
    "noise_residual_score": 0.74
  },
  "localization": {
    "available": true,
    "mask_uri": "...",
    "heatmap_uri": "...",
    "iou": null
  },
  "regions": [
    {
      "label": "possible_text_tampering",
      "bbox": [0.42, 0.60, 0.71, 0.73],
      "score": 0.88
    }
  ],
  "explanation": "The model found an anomalous region around the lower-right text field; ELA and residual maps also show stronger local inconsistency there.",
  "warnings": []
}
```

Coordinates in API output should be normalized to `[0, 1]` and documented as `[x1, y1, x2, y2]`.

## 4. Verdict policy
Use a three-state interface:

- `REAL`: score below lower threshold.
- `UNCERTAIN`: score in an explicit gray zone.
- `TAMPERED`: score above upper threshold.

Do not hard-code `0.5` as the final production threshold. Select thresholds from validation data, ideally using a target operating point such as high recall at an acceptable false-positive rate.

## 5. System architecture

```text
                         +-------------------+
                         |   Upload Image    |
                         +---------+---------+
                                   |
                                   v
                         +-------------------+
                         | Validation /      |
                         | Preprocessing      |
                         +----+---------+-----+
                              |         |
                    +---------+         +---------+
                    v                             v
             +-------------+               +-------------+
             | ELA         |               | Noise       |
             | module      |               | residual    |
             +------+------+               +------+------+ 
                    |                             |
                    +-------------+---------------+
                                  v
                       +-------------------------+
                       | Deep Tamper Classifier  |
                       | ResNet18 -> EfficientNet|
                       +------------+------------+
                                    |
                      +-------------+-------------+
                      |                           |
                      v                           v
             +----------------+        +-------------------+
             | Grad-CAM       |        | Localization Head |
             | explanation    |        | / segmentation    |
             +--------+-------+        +---------+---------+
                      |                           |
                      +-------------+-------------+
                                    v
                         +-----------------------+
                         | Evidence Aggregator   |
                         +-----------+-----------+
                                     |
                      +--------------+--------------+
                      |                             |
                      v                             v
              +---------------+            +----------------+
              | OCR / layout |            | Optional VLM    |
              | evidence      |            | forensic report |
              +-------+-------+            +-------+--------+
                      |                            |
                      +-------------+--------------+
                                    v
                         +-----------------------+
                         | JSON API + Web Demo   |
                         +-----------------------+
```

## 6. Repository architecture

```text
integridoc/
├── apps/
│   ├── api/
│   └── web/
├── configs/
│   ├── data.yaml
│   ├── train_resnet18.yaml
│   ├── train_efficientnet.yaml
│   ├── localization.yaml
│   └── vlm.yaml
├── data/
│   ├── raw/            # never committed
│   ├── processed/      # never committed
│   ├── synthetic/      # can be locally generated; large files ignored
│   └── manifests/
├── notebooks/
├── src/
│   ├── data/
│   ├── forensics/
│   ├── models/
│   ├── localization/
│   ├── ocr/
│   ├── vlm/
│   ├── inference/
│   ├── evaluation/
│   └── utils/
├── tests/
├── scripts/
├── checkpoints/
├── reports/
├── results/
├── docs/
├── README.md
├── pyproject.toml
├── Makefile
└── .gitignore
```

## 7. Engineering standards

- Python 3.11 recommended.
- PyTorch + torchvision for core CV training.
- Hugging Face Transformers/PEFT only for VLM components.
- FastAPI for serving.
- Pydantic for API schemas.
- OpenCV + Pillow for image processing.
- Albumentations or torchvision transforms for augmentation.
- pytest for unit/integration tests.
- Ruff + mypy where practical.
- YAML configuration instead of hidden constants.
- Every experiment receives an ID and writes a machine-readable result file.
- Random seeds must be configurable.
- Dataset manifests must record source document IDs so group-based splits are enforceable.

## 8. Model registry requirements
Every trained artifact must have:

- model name;
- architecture;
- git commit SHA;
- data version/hash;
- training config;
- input resolution;
- threshold;
- metrics;
- date;
- checkpoint path;
- known limitations.

## 9. Non-functional requirements

### Performance
The first goal is correctness and reproducibility, not maximum throughput.

### Reproducibility
A fresh Colab session should be able to reproduce the main reported experiment from a documented command/notebook.

### Interpretability
The demo must expose the original image, forensic maps, model heatmap, final verdict, and confidence/score.

### Failure visibility
Never silently return a verdict when preprocessing fails. Return a structured `UNCERTAIN` or validation error.

## 10. Definition of success

The first release is successful when a reviewer can:

1. clone the repository;
2. generate synthetic data;
3. train or download a documented checkpoint;
4. run evaluation;
5. upload an example image;
6. see verdict + localization + forensic evidence;
7. reproduce the key metrics.
