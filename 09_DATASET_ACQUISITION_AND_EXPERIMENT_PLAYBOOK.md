# Stage 9 — Dataset Acquisition & Experiment Playbook

This is the practical data/research playbook for obtaining data, deciding what to train on, and creating credible experiments.

## 1. Dataset hierarchy

Use the following hierarchy rather than treating all datasets as interchangeable.

| Layer | Dataset/data | Purpose | Status |
|---|---|---|---|
| A | IntegriDoc synthetic dataset | Primary training + exact masks | Required |
| B | DocTamper | External document-tampering evaluation | Strongly recommended if access permits |
| C | Generic image-forensics data | Secondary robustness test | Optional |
| D | Small hand-designed synthetic hard negatives | Failure analysis/demo | Required |

The original project notes correctly identify the synthetic dataset as the centerpiece because it gives exact control over the tampering process and ground-truth masks. fileciteturn0file0L203-L214

---

## 2. Synthetic data: what you should actually generate

Do not make every sample a cartoon-perfect ID card.

Use at least five template families:

1. identity card;
2. employee badge;
3. university card;
4. certificate;
5. invoice/salary-slip-style document.

Within each family, randomize:

- layout spacing;
- font family from permitted local fonts;
- font size;
- line spacing;
- text length;
- color;
- background texture;
- logo-like geometric marks;
- portrait/avatar position;
- signature position;
- border style;
- paper/scan appearance.

All identities and values must be fictional.

---

## 3. Synthetic tampering distribution

A good first main dataset should not contain only text replacement.

Example target distribution:

```text
25% text/number replacement
15% copy-move
15% splicing
10% signature replacement
10% photo replacement
10% deletion/inpainting
05% font/spacing manipulation
10% mixed multi-tamper
```

Also create authentic hard negatives with global image transformations.

These percentages are a starting experiment design, not a required universal distribution.

---

## 4. Dataset sizes to use

### Phase A — smoke

100 source documents
+
1–2 tampered variants per source

Purpose: confirm code correctness.

### Phase B — development

500–1,000 source documents
+
2–3 variants per source

Purpose: debug training behavior and evaluate pipeline design.

### Phase C — main research dataset

5,000–20,000 source documents/templates
+
2–4 variants per source

Purpose: final training and ablation studies.

### Phase D — additional diversity

Add new template families and transformation combinations, not merely more copies of existing layouts.

---

## 5. How to acquire DocTamper

The official DocTamper GitHub repository says the dataset is available through BaiduDrive and Kaggle and is restricted to non-commercial use with an access/request procedure. The repository also distinguishes public inference/data-synthesis materials from restricted training materials. citeturn932630search0

Official starting point:

https://github.com/qcf-568/DocTamper

### Your exact workflow

1. Open the official repository.
2. Read its current README/license/access instructions.
3. Decide whether your intended academic/portfolio use complies.
4. Request access using the current procedure if required.
5. Download from the permitted source.
6. Store raw files outside Git.
7. Convert only as permitted by the dataset terms.
8. Create a manifest entry with:
   - dataset name;
   - version/date;
   - source URL;
   - access method;
   - license/use restriction;
   - local path;
   - checksum if available;
   - whether files can be redistributed (normally no for restricted datasets).
9. Keep the dataset-specific preprocessing code separate from your synthetic-data code.

### Do not

- download a random mirror and assume it is authorized;
- upload the dataset to your GitHub repository;
- place raw images in a public demo;
- describe restricted data as your own dataset;
- train on restricted data and then publish the raw samples.

---

## 6. Why DocTamper should be evaluation, not your only dataset

DocTamper is specifically about tampered text in document images, making it highly relevant to this project. Its official repository also provides visualization and research code. citeturn932630search0

However, the strongest personal research story is still:

```text
I designed a controllable synthetic benchmark
        ↓
trained models under known ground truth
        ↓
then tested domain transfer on an external document-tampering dataset
```

That lets you discuss both controlled experiments and generalization.

---

## 7. External evaluation protocol

For every external dataset, create a separate evaluation configuration.

Do not mix its test data into your synthetic training set unless the experiment explicitly studies domain adaptation and documents that fact.

Recommended table:

| Evaluation | Training data | Test data | Purpose |
|---|---|---|---|
| S1 | Synthetic | Synthetic test | In-domain benchmark |
| S2 | Synthetic | Unseen synthetic template family | Generalization |
| S3 | Synthetic | Post-processed synthetic | Robustness |
| S4 | Synthetic | External document dataset | Domain transfer |
| S5 | External + synthetic | External held-out test | Optional adaptation |

---

## 8. Data leakage checks

Before every final experiment, run:

```bash
python -m src.data.validate --check-split-leakage
```

Also verify:

- source IDs do not cross splits;
- perceptual duplicates do not cross splits;
- near-identical generated variants stay together;
- augmentation is not accidentally applied using test information;
- threshold selection uses validation only;
- test results are calculated once after the experiment protocol is frozen.

---

## 9. Research experiment ladder

### E01 — classical ELA

Question:

> Does recompression difference provide useful localization/anomaly signal on our data?

Output:

- ELA maps;
- heuristic score;
- qualitative examples.

### E02 — residuals

Question:

> Do local high-frequency inconsistencies differ between authentic and tampered documents?

Output:

- residual maps;
- heuristic statistics.

### E03 — RGB ResNet18

Question:

> How strong is a standard transfer-learning image classifier without explicit forensic inputs?

### E04 — forensic fusion

Question:

> Does adding ELA/residual evidence improve generalization over RGB alone?

This should be the centerpiece ablation.

### E05 — EfficientNet-B0

Question:

> Does a different efficient CNN architecture produce a better accuracy/latency tradeoff?

### E06 — Grad-CAM

Question:

> Is the classifier actually attending to plausible suspicious regions?

### E07 — segmentation/localization

Question:

> Can a model predict pixel-level tamper regions rather than only a document-level label?

### E08 — robustness

Question:

> Does the detector survive common post-processing?

### E09 — unseen template families

Question:

> Is the model learning forensics or memorizing layouts?

### E10 — external document dataset

Question:

> Does performance transfer beyond the synthetic domain?

### V1 — VLM image-only

Question:

> Can a VLM produce a useful forensic assessment from the raw document image?

### V2 — VLM + OCR

Question:

> Does structured text/layout evidence improve reasoning?

### V3 — VLM + ELA/residual/localization

Question:

> Does explicit forensic evidence improve grounding and explanation quality?

### V4 — LoRA fine-tuning

Question:

> Can domain-specific synthetic evidence examples improve structured forensic reasoning?

---

## 10. Experiment log template

For every run, record:

```yaml
experiment_id: E04
name: resnet18_fusion
purpose: compare RGB-only vs RGB+ELA+residual
git_sha: "..."
dataset_version: synthetic_v1
split_protocol: source_group_70_15_15
seed: 42
model_checkpoint: "..."
input_resolution: 224
batch_size: 32
optimizer: AdamW
learning_rate: 0.0001
epochs: 15
threshold_selection: validation_best_f1
metrics:
  accuracy: null
  precision: null
  recall: null
  f1: null
  roc_auc: null
  pr_auc: null
notes: ""
```

Fill numeric values only after running the experiment.

---

## 11. What counts as a strong result

A strong result is **not necessarily the highest accuracy**.

A strong project can instead show:

```text
RGB baseline       → strong classifier
Forensic fusion    → better recall / robustness
Localization       → useful masks
Robustness test    → known degradation pattern
VLM               → better evidence grounding
Failure analysis   → identified limitations
```

The research value comes from understanding why each change helped or did not help.

---

## 12. Failure cases you should intentionally create

Create tests for:

1. very small text replacement;
2. same-font text replacement;
3. copy-move with rotation;
4. copy-move after JPEG recompression;
5. blended splice;
6. photo replacement with similar-looking image;
7. authentic document with aggressive JPEG recompression;
8. authentic document with scan noise;
9. authentic document with resize/blur;
10. multiple edits in one image.

Then inspect whether the model fails systematically.

---

## 13. Hard negative philosophy

The model should not simply learn:

```text
JPEG artifact → fake
sharp image → real
synthetic template → fake
```

It should be exposed to:

- globally recompressed authentic documents;
- globally noisy authentic documents;
- resized authentic documents;
- blurred authentic documents;
- tampered documents with full-image recompression;
- tampered documents with realistic scan artifacts.

This forces the learning problem toward local inconsistency rather than trivial global shortcuts.

---

## 14. Optional 2026 research extension: multi-scale ELA

Recent 2026 work has explored multi-scale ELA and related forensic features for CPU-friendly image-level forgery screening, reinforcing the idea that ELA should be treated as a measurable feature family rather than a single visual trick. citeturn932630academia38

For IntegriDoc, an optional experiment can compare:

```text
single-quality ELA
vs
multi-quality ELA
```

For example:

```text
Q = 60, 70, 80, 85, 90, 95
```

Extract per-quality maps and compare whether the multi-quality representation improves classification or localization.

Do not add this until the basic ELA experiment works.

---

## 15. VLM dataset creation

The VLM should not be trained on arbitrary prose.

Generate structured examples from your own known ground truth.

Each example should contain:

```text
raw document image
+
OCR boxes
+
ELA map
+
residual map
+
classifier score
+
localization mask/bbox
+
ground-truth tamper type
```

The target response should distinguish:

- observed evidence;
- supported inference;
- uncertainty;
- recommended action.

This creates a defensible research question rather than a generic “ask a VLM whether the image is fake” demo.

---

## 16. VLM model selection

Qwen2.5-VL is available in 3B, 7B, and 72B variants in the current Hugging Face documentation. citeturn452132search1

For your project:

### Start with 3B

Use for:

- prompt experiments;
- structured-output testing;
- early Colab trials.

### Move to 7B

Use for:

- stronger multimodal reasoning;
- the final VLM baseline;
- LoRA experiment if your GPU budget allows.

The current Qwen2.5-VL model cards provide Transformers usage and Colab-oriented examples. citeturn452132search0turn452132search8

Do not start with 72B for this project.

---

## 17. How to decide whether LoRA is worth keeping

Run:

```text
V1 raw VLM
V2 VLM + forensic evidence
V3 LoRA-tuned VLM + forensic evidence
```

Keep LoRA only if it improves a measurable target such as:

- structured output validity;
- evidence-grounding score;
- tamper-type classification F1;
- hallucination rate;
- consistency across repeated evaluations.

If LoRA does not improve these metrics, document the negative result.

A negative result is still a valid research result when the experiment is controlled.

---

## 18. Dataset folder layout

Recommended final local layout:

```text
data/
├── synthetic/
│   ├── images/
│   ├── masks/
│   ├── manifests/
│   └── previews/
├── external/
│   ├── doctamper/        # never committed
│   └── generic_forensics/ # never committed
└── manifests/
    ├── synthetic.yaml
    ├── external_sources.yaml
    └── versions.yaml
```

---

## 19. Dataset manifest entry example

```yaml
- name: DocTamper
  role: external_document_tampering_evaluation
  source_url: https://github.com/qcf-568/DocTamper
  access_date: YYYY-MM-DD
  license_or_terms: "record current terms from official source"
  redistributable: false
  local_path: data/external/doctamper
  checksum: "if provided"
  notes: "Do not commit or redistribute raw dataset"
```

The AI builder should never silently fill a license field with a guessed value.

---

## 20. Publication-quality result tables

Create three final tables.

### Table A — classifier benchmark

| Model | Input | Precision | Recall | F1 | ROC-AUC | Latency |
|---|---|---:|---:|---:|---:|---:|
| ELA heuristic | ELA | — | — | — | — | — |
| ResNet18 | RGB | — | — | — | — | — |
| ResNet18 Fusion | RGB+ELA+Residual | — | — | — | — | — |
| EfficientNet-B0 | RGB | — | — | — | — | — |

### Table B — localization

| Method | Dice | IoU | Pixel Precision | Pixel Recall |
|---|---:|---:|---:|---:|
| Grad-CAM threshold | — | — | — | — |
| Segmentation model | — | — | — | — |
| SAM2-assisted | — | — | — | — |

### Table C — generalization

| Train → Test | F1 | Recall | FPR | Notes |
|---|---:|---:|---:|---|
| Synthetic → Synthetic | — | — | — | in-domain |
| Synthetic → Unseen Template | — | — | — | cross-template |
| Synthetic → External | — | — | — | domain transfer |

Only populate with measured results.

---

## 21. The experiment narrative you should aim for

Your final project should tell this story:

> I started with interpretable forensic signals because I wanted to understand what visual evidence changed under tampering. I then built a controlled synthetic benchmark so every manipulation had known pixel-level ground truth. A standard RGB CNN gave me a strong baseline, after which I tested whether explicit ELA/residual evidence improved the detector. I added localization and robustness experiments to determine whether the system was actually learning tampering signals rather than memorizing document templates. Finally, I tested whether OCR + forensic evidence improved a multimodal VLM's ability to explain suspicious regions, and whether LoRA fine-tuning improved that evidence-grounded reasoning.

That is much stronger than saying:

> “I fine-tuned a model on fake documents.”
