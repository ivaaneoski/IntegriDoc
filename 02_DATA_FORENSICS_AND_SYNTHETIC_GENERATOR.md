# Stage 2 — Data, Image Forensics & Synthetic Tampering

## 1. Dataset strategy

Use three layers of data:

### Layer A — synthetic documents (primary)
Create fictional document templates and generate both authentic and tampered variants. This provides exact ground-truth masks.

### Layer B — public document-tampering research data
Use a permitted research dataset such as DocTamper for domain evaluation. The official repository states that its dataset is available via BaiduDrive/Kaggle, is restricted to non-commercial use, and requires access procedures; the build must verify the current repository terms before downloading. Do not redistribute the dataset. 

### Layer C — generic image-forensics data (optional)
A dataset such as CASIA can be used as a secondary robustness check, but generic image-forensics data must not be presented as equivalent to identity-document data.

## 2. Synthetic document generator

### Template families
Implement at least five fictional template families:

1. identity card;
2. employee badge;
3. university card;
4. certificate;
5. invoice/salary-slip style document.

Every template should be a deterministic drawing function or editable SVG/HTML/CSS template so the source geometry is known.

### Synthetic data fields
Use fictional values generated from seeded dictionaries:

- full name;
- date of birth;
- issue date;
- expiry date;
- document number;
- organization;
- address-like fictional text;
- signature glyph generated specifically for the dataset;
- portrait/avatar generated from an approved synthetic source.

Never use real identity information.

## 3. Data manifest

Every generated image gets a JSON record:

```json
{
  "source_id": "idcard_00421",
  "image_id": "idcard_00421_tamper_03",
  "split": "train",
  "label": "tampered",
  "template_family": "identity_card",
  "tamper_type": "text_replacement",
  "tamper_params": {
    "field": "dob",
    "font_changed": true,
    "jpeg_quality": 82
  },
  "mask_path": "masks/idcard_00421_tamper_03.png",
  "bbox": [0.54, 0.31, 0.77, 0.39],
  "seed": 183920,
  "generator_version": "0.3.0"
}
```

## 4. Group-based splitting

Split by `source_id`, not by rendered image.

Recommended starting split:

- 70% source documents → train
- 15% → validation
- 15% → test

All authentic and forged variants derived from the same source must remain in one split. This is required to avoid near-duplicate leakage.

## 5. Tamper operators

Implement each operator as a pure, testable function that returns:

`tampered_image, binary_mask, metadata`

### P0 operators

1. **Text replacement**
   - replace text within an existing field;
   - allow font size, spacing, alignment, anti-aliasing variation;
   - compute exact changed mask from rendered region.

2. **Number replacement**
   - replace dates, IDs, amounts, years;
   - vary string length and character shapes.

3. **Copy-move**
   - crop a source patch from inside the same image;
   - transform with translation/rotation/mild scaling;
   - paste to target location.

4. **Splicing**
   - insert a region from another synthetic document/template;
   - introduce boundary blending or imperfect blending.

5. **Signature replacement**
   - use synthetic signature assets only;
   - vary scale, rotation, opacity, and blending.

6. **Photo replacement**
   - replace a synthetic portrait/avatar region;
   - preserve overall layout.

### P1 operators

7. region deletion/inpainting;
8. font-family replacement;
9. character-spacing changes;
10. localized blur/sharpening;
11. inconsistent JPEG recompression;
12. local color/contrast shift;
13. local noise mismatch;
14. subtle perspective distortion.

## 6. Hard-negative generation

A strong detector must not solve the task by learning trivial editing signatures.

Generate authentic-but-transformed images with:

- global JPEG recompression;
- global resize;
- global brightness/contrast shift;
- whole-image mild noise;
- whole-image blur;
- scan-like artifacts.

These are **negative controls**, not tampered labels.

Also generate tampered images followed by full-image transformations, so tampering and post-processing are decoupled.

## 7. Mask rules

The mask should represent the smallest defensible changed region.

Store:

- binary mask PNG;
- optional soft mask for blended edits;
- bounding box;
- tamper type;
- source region if copy-move.

Do not use Grad-CAM itself as the ground-truth mask.

## 8. Error Level Analysis (ELA)

Implementation requirements:

1. load image in RGB;
2. save a recompressed JPEG at configurable quality `Q` (default 90);
3. read the recompressed image;
4. calculate absolute per-pixel difference;
5. convert to luminance or retain RGB channels;
6. normalize using a configurable percentile policy rather than hard-coded min/max only;
7. create a heatmap visualization;
8. optionally compute a scalar ELA anomaly score.

Important interpretation:
ELA is a forensic signal, not proof of manipulation. It is sensitive to image history and compression pipeline.

## 9. Noise residual analysis

Implement at least two residual methods:

### Method A: Gaussian high-pass residual
`residual = image - gaussian_blur(image, sigma)`

### Method B: Laplacian residual
Use a Laplacian/high-frequency filter and absolute magnitude.

Produce:

- residual map;
- normalized visualization;
- local anomaly statistics;
- optional scalar score.

Evaluate whether residual features actually add value through ablation rather than assuming they do.

## 10. Forensic feature interface

Create a common interface:

```python
class ForensicAnalyzer(Protocol):
    def analyze(self, image: Image.Image) -> ForensicResult: ...
```

`ForensicResult` should contain:

- name;
- image/map;
- scalar metrics;
- optional region proposals;
- warnings;
- debug metadata.

## 11. Data quality checks

Build a validation script that detects:

- unreadable images;
- empty masks;
- masks outside image bounds;
- tampered images identical to authentic source;
- incorrect label distribution;
- duplicate perceptual hashes across splits;
- suspiciously similar images across splits;
- source IDs appearing in multiple splits.

Fail CI/data preparation if any hard violation occurs.

## 12. Dataset sizes

Start small and scale:

### Smoke dataset
- 500 source documents;
- 1–2 tampered variants each;
- all tamper types represented.

### Main dataset
- 5,000–20,000 source documents/templates rendered with varied content.
- 2–4 tampered variants per source.

### Large experiment
Only create this if model learning saturates and compute allows it.

Do not optimize for raw image count; optimize for diversity of transformations and clean source separation.

## 13. Data augmentation

Recommended classification augmentation:

- resize/crop;
- tiny rotation;
- mild brightness/contrast;
- small Gaussian noise;
- mild blur;
- JPEG recompression.

Do not over-augment because the forensic evidence can be destroyed.

For localization, ensure geometric transforms are applied identically to the image and mask.

## 14. External evaluation rules

When evaluating an external dataset:

- never train on its test split;
- preserve dataset-native test definitions where possible;
- report domain shift;
- report per-manipulation results;
- do not compare numbers from different preprocessing/evaluation protocols as if they were directly equivalent.

## 15. Data deliverables

The AI coding agent must create:

- `scripts/generate_synthetic_dataset.py`
- `scripts/validate_dataset.py`
- `src/data/generator.py`
- `src/data/tamper_ops.py`
- `src/data/masks.py`
- `src/data/manifests.py`
- `src/forensics/ela.py`
- `src/forensics/residuals.py`
- `src/forensics/visualize.py`
- tests for every tamper operator and forensic module.
