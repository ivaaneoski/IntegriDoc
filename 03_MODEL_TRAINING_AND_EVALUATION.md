# Stage 3 — Model Training, Localization & Evaluation

## 1. Baseline ladder

Do not begin with a large transformer.

Run experiments in this order:

1. ELA-only heuristic baseline;
2. residual-only heuristic baseline;
3. ResNet18 RGB classifier;
4. ResNet18 + forensic-channel fusion;
5. EfficientNet-B0 comparison;
6. localization head;
7. optional ViT experiment.

The point is to measure whether every added component improves the system.

## 2. ResNet18 classifier

Use torchvision’s pretrained ResNet18 as the first deep classifier. The current torchvision documentation exposes ImageNet pretrained weights and the standard 224×224 preprocessing. 

### Architecture

Input: RGB document image, default 224×224.

Backbone: pretrained ResNet18.

Head:

```text
2048-ish final feature representation
        ↓
Dropout(0.2)
        ↓
Linear(feature_dim, 2)
```

Use actual backbone feature dimensions from the implementation rather than hard-coding an incorrect number.

### Training phases

**Phase A:** freeze backbone, train classification head.

**Phase B:** unfreeze final residual block(s).

**Phase C:** optional full fine-tune at lower learning rate.

## 3. Loss function

Start with weighted cross-entropy if the dataset is imbalanced.

Optional experiments:

- focal loss;
- label smoothing;
- class-balanced loss.

Record every loss choice in the experiment manifest.

## 4. Recommended starting hyperparameters

These are starting points, not claimed optimal values:

```yaml
image_size: 224
batch_size: 32
optimizer: adamw
lr_head: 1e-3
lr_backbone: 1e-4
weight_decay: 1e-4
epochs: 15
warmup_epochs: 1
scheduler: cosine
mixed_precision: true
early_stopping_patience: 4
seed: 42
```

The AI agent must expose all values through YAML config or CLI flags.

## 5. Forensic-channel fusion experiment

Create a second experiment in which the model receives RGB + forensic representations.

Option A: early fusion

- RGB: 3 channels
- ELA: 1 channel
- residual: 1 channel
- total: 5 channels

Option B: dual-branch fusion

```text
RGB → CNN branch ───────┐
                        ├→ fusion MLP → classifier
ELA → small CNN branch ─┤
Residual → small CNN ───┘
```

Prefer Option B for the research experiment because it lets the model learn separate representations and is easier to ablate.

## 6. EfficientNet-B0

After ResNet18 is stable, train EfficientNet-B0 under the same split and evaluation protocol.

The comparison must report:

- accuracy;
- precision;
- recall;
- F1;
- ROC-AUC;
- parameter count;
- checkpoint size;
- approximate inference latency;
- memory usage where measurable.

Never invent results.

## 7. Localization model

Start with the simplest useful architecture: U-Net-like encoder-decoder using a pretrained CNN encoder if available.

Input:

- RGB image;
- optionally ELA/residual channels.

Output:

- one-channel tamper mask logits.

### Loss
Start with:

`BCEWithLogitsLoss + DiceLoss`

Optionally test focal/Tversky-style variants later.

### Metrics

- IoU;
- Dice/F1;
- pixel precision;
- pixel recall;
- boundary-aware metric if later needed.

## 8. Weak localization baseline

Before training a segmentation model, implement:

1. Grad-CAM heatmap;
2. normalize heatmap;
3. threshold at validation-selected operating point;
4. compare with ground-truth mask.

This is explicitly a weak explanation, not a true segmentation prediction.

## 9. Grad-CAM requirements

Implement a reusable API:

```python
heatmap = explain(model, image, target_class="tampered")
```

Generate side-by-side artifacts:

- original image;
- grayscale/colored heatmap;
- overlay;
- binary thresholded map.

For reports, annotate the predicted class and score.

## 10. Calibration

If the UI calls a number a “confidence,” calibration must be considered.

Evaluate:

- raw score;
- reliability curve;
- Expected Calibration Error (ECE);
- optional temperature scaling on validation data.

Until calibration is implemented, label the UI value `tamper score` or `model score`.

## 11. Evaluation metrics

Classification:

- accuracy;
- precision;
- recall;
- F1;
- ROC-AUC;
- PR-AUC;
- confusion matrix;
- false-positive rate;
- false-negative rate.

Localization:

- IoU;
- Dice;
- pixel precision;
- pixel recall.

Operational:

- latency;
- throughput;
- model size;
- peak memory.

## 12. Threshold selection

Use validation data to choose a threshold based on the intended operating objective.

Recommended reporting:

- threshold for best F1;
- threshold achieving target recall;
- threshold at an explicit false-positive operating point.

Never tune the threshold on the final test set.

## 13. Required experiment matrix

Create a CSV/JSON result table with at least:

| ID | Model | Input | ELA | Residual | Split | F1 | ROC-AUC | IoU | Latency |
|---|---|---|---|---|---|---:|---:|---:|---:|
| E01 | ELA heuristic | ELA | ✓ | — | test | — | — | — | — |
| E02 | Residual heuristic | residual | — | ✓ | test | — | — | — | — |
| E03 | ResNet18 | RGB | — | — | test | — | — | — | — |
| E04 | ResNet18-Fusion | RGB+ELA+residual | ✓ | ✓ | test | — | — | — | — |
| E05 | EfficientNet-B0 | RGB | — | — | test | — | — | — | — |
| E06 | Localization | RGB | ✓ | ✓ | test | — | — | — | — |

Replace `—` only when the metric is logically applicable.

## 14. Ablation study

Required ablations:

### Ablation A — forensic inputs
- RGB only;
- RGB + ELA;
- RGB + residual;
- RGB + ELA + residual.

### Ablation B — tamper diversity
- text only;
- copy-move only;
- mixed tampering.

### Ablation C — post-processing robustness
Evaluate the same tampered examples before and after whole-image JPEG compression, resize, blur, and brightness changes.

### Ablation D — cross-template generalization
Train on some template families and test on an unseen family.

This is especially valuable because it tests whether the system learned forensics rather than template memorization.

## 15. Failure analysis

For the top 50 false positives and false negatives, store:

- original image;
- predicted score;
- ground truth;
- tamper type;
- template family;
- ELA map;
- residual map;
- Grad-CAM;
- optional OCR evidence.

Create a notebook/report grouping failures into categories.

## 16. Reproducibility

Every run must write:

```text
results/runs/<experiment_id>/
├── config.yaml
├── metrics.json
├── predictions.csv
├── confusion_matrix.png
├── curves.png
├── sample_predictions/
├── model_summary.txt
└── git_commit.txt
```

Use deterministic seeds where practical and explicitly document any nondeterminism from GPU operations.
