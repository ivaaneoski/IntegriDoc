# Stage 10 — Final Ship, Portfolio & HyperVerge Interview Checklist

## 1. Final repository structure

The GitHub repository should contain:

```text
integridoc/
├── apps/
├── configs/
├── docs/
├── notebooks/
├── scripts/
├── src/
├── tests/
├── examples/
├── reports/
├── results/
├── README.md
├── pyproject.toml
├── Makefile
├── LICENSE
└── .gitignore
```

Do not commit:

```text
.venv/
__pycache__/
data/raw/
data/external/
checkpoints/*.pt
*.safetensors
.env
private documents
```

Use GitHub Releases, Hugging Face model repositories, Google Drive, or another appropriate artifact store for large models when the licensing and platform terms permit it.

---

## 2. README must answer these questions immediately

Within the first screen of the README:

1. What is IntegriDoc?
2. Why was it built?
3. What does it output?
4. What is the core architecture?
5. How do I run it?
6. What data does it use?
7. What were the measured results?
8. What are its limitations?

Include a visual architecture diagram and at least one end-to-end example.

---

## 3. Suggested README opening

```text
# IntegriDoc

Explainable document forgery detection and tamper localization.

IntegriDoc combines classical image forensics, deep learning, synthetic tampering,
localization, OCR, and optional multimodal reasoning to determine whether a
fictional/public document image contains signs of manipulation.

The project is a research prototype and is not a substitute for legal, identity,
or fraud-investigation decisions.
```

---

## 4. Portfolio demo sequence

When showing the project, use this order:

### Screen 1 — Upload

Upload a synthetic tampered document.

### Screen 2 — Verdict

Show:

```text
TAMPERED
Tamper score: <actual measured output>
```

### Screen 3 — Forensic evidence

Show:

- original;
- ELA;
- residual;
- Grad-CAM;
- localization.

### Screen 4 — Field-level analysis

Show OCR boxes and suspicious region alignment.

### Screen 5 — VLM explanation

Show evidence-grounded JSON or human-readable explanation.

### Screen 6 — Research results

Show baseline vs fusion vs EfficientNet.

### Screen 7 — Failure analysis

Show one false positive and one false negative.

This final screen makes the project feel like research rather than a polished classifier demo.

---

## 5. Resume bullet strategy

Do not write bullets until the final experiment table is frozen.

Template:

```text
Built IntegriDoc, an explainable document-forensics pipeline combining ELA/noise-residual analysis, synthetic tamper generation with pixel-level ground truth, and transfer-learning CNNs for document-level forgery detection and localization; evaluated [N] source documents across [M] tamper classes and improved [metric] from [baseline] to [result].
```

Second bullet:

```text
Designed an evidence-grounded multimodal analysis pipeline combining OCR, forensic maps, localization outputs, and [VLM] with [LoRA/PEFT if actually implemented], evaluating [grounding/validity metric] across [N] synthetic cases.
```

Only replace placeholders with actual numbers.

---

## 6. HyperVerge interview positioning

The project directly maps to the job categories supplied in the original JD:

```text
HyperVerge need                  → IntegriDoc evidence
--------------------------------------------------------------
Image forgery detection          → ELA + residuals + CNN detector
Document understanding           → OCR + document templates
Identity verification adjacency  → fictional ID/document workflows
VLM                               → Qwen2.5-VL experiment
SFT / PEFT / LoRA                 → evidence-grounded VLM fine-tuning
Synthetic data generation         → controlled tamper generator
Model evaluation                  → metrics + ablations
Failure analysis                  → false-positive/false-negative reports
Deployment                        → FastAPI + UI + Docker
End-to-end ownership              → data → training → evaluation → deployment
```

The strongest message is that every system component exists because it answers a research or engineering question.

---

## 7. Interview questions you should be ready for

### Why did you use synthetic data?

Answer around controllability and masks:

> I needed exact pixel-level ground truth for tamper localization and wanted to control the manipulation type, location, and post-processing independently.

### Why ELA?

> It gives an interpretable signal related to local recompression differences, but I treated it as one forensic feature rather than a definitive detector.

### What is wrong with random train/test splitting?

> Multiple forged variants can be near duplicates of the same source document, so random splitting leaks source-specific information and inflates performance.

### Why ResNet18 first?

> It is a strong, lightweight transfer-learning baseline that lets me establish whether the problem and data pipeline work before increasing model complexity.

### Why add ELA/residual channels?

> I wanted to test whether explicit forensic evidence contributes information beyond the RGB representation. I evaluated this through ablations rather than assuming it would help.

### Why Grad-CAM?

> It provides a model-attention explanation, but I explicitly distinguish it from a ground-truth segmentation mask.

### Why a VLM?

> The VLM stage asks a different question: not only whether the image looks suspicious, but whether a multimodal model can interpret OCR and forensic evidence together and produce a grounded explanation.

### Why LoRA?

> It lets me adapt a multimodal model without full-model fine-tuning, which is more feasible under limited compute and is directly relevant to PEFT workflows.

### What did not work?

This is where your actual failure analysis should be used.

Never invent a failure story.

---

## 8. Final demo dataset

Create a tiny public-safe demo set under:

```text
examples/demo/
```

Include:

```text
authentic_idcard.png
tampered_text.png
tampered_copy_move.png
tampered_signature.png
tampered_photo.png
authentic_compressed.png
authentic_noisy.png
multiple_tamper.png
```

All examples should be fictional/synthetic or otherwise clearly permitted.

---

## 9. Model card

For the final detector include:

- model architecture;
- training dataset;
- split protocol;
- input resolution;
- training config;
- metrics;
- threshold selection;
- intended use;
- out-of-scope use;
- known failure modes;
- calibration status;
- robustness limitations;
- compute used;
- license dependencies.

---

## 10. Dataset card

For the synthetic dataset include:

- generation method;
- template families;
- tamper operators;
- number of source documents;
- number of variants;
- split policy;
- mask definition;
- known biases;
- synthetic-vs-real domain gap;
- intended use;
- prohibited use.

For external datasets, link to the official source rather than republishing the dataset.

---

## 11. What makes the project “amazing”

The quality ladder is:

```text
Level 1
Classifier

Level 2
Classifier + ELA

Level 3
Synthetic data + masks + proper evaluation

Level 4
Localization + Grad-CAM + robustness

Level 5
External evaluation + ablations + failure analysis

Level 6
OCR + VLM evidence grounding

Level 7
LoRA fine-tuning + measurable VLM improvements

Level 8
FastAPI + UI + reproducible deployment

Level 9
Research report + clean GitHub + strong interview narrative
```

You do not need every Level 7/8 feature to have a good project. A rigorously evaluated Level 5 project is better than a half-working Level 8 project.

---

## 12. Final “do not fake it” rule

Never use:

- invented benchmark scores;
- copied screenshots of another model;
- fake ROC curves;
- made-up dataset sizes;
- claimed publications that do not exist;
- “production-ready” language for an unvalidated prototype.

The strongest portfolio claim is a measured, reproducible result.
