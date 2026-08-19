# Stage 7 — Research Story, Interview Preparation & Portfolio Evidence

## 1. The project story

The strongest project narrative is **not**:

> “I fine-tuned ResNet on forged documents.”

The stronger story is:

> “I built an end-to-end document-forensics pipeline that generates controlled synthetic tampering, learns to classify tampered documents, localizes manipulated regions, combines classical forensic signals with deep features, and uses a multimodal model to explain evidence.”

## 2. Research questions

### RQ1
Can a transfer-learned classifier distinguish authentic and tampered synthetic documents across multiple manipulation types?

### RQ2
Do ELA and noise-residual features provide measurable gains beyond RGB-only classification?

### RQ3
Does performance generalize to unseen document templates and post-processing transformations?

### RQ4
Can a localization model identify the actual manipulated region rather than only classify the whole document?

### RQ5
Does supplying explicit forensic evidence improve VLM reasoning versus image-only VLM inspection?

## 3. Expected research artifacts

Produce:

1. baseline results;
2. ablation table;
3. robustness table;
4. per-tamper-type table;
5. localization results;
6. qualitative visual examples;
7. failure analysis;
8. VLM grounding/hallucination analysis if advanced stage is completed.

## 4. What to say in an interview

### Why this project?
“I noticed that image forgery detection is directly relevant to identity and document verification. Instead of only training a classifier, I built a controlled forensic pipeline so I could measure what each signal contributes.”

### Why synthetic data?
“It gives exact tamper masks, precise control of manipulation type, and avoids using private identity documents.”

### Why ELA?
“ELA gives a classical, interpretable forensic baseline, but I explicitly treat it as one signal rather than proof of manipulation because compression history can create false positives.”

### Why Grad-CAM?
“It gives model-attention evidence, but I do not equate attention with a ground-truth tamper mask. For actual localization I evaluate a segmentation model against synthetic ground-truth masks.”

### Why a VLM?
“The detector answers whether and where suspicious evidence exists. The VLM is used downstream to interpret OCR and forensic evidence and produce a structured explanation. That separation makes the system easier to evaluate.”

### Why LoRA?
“Full fine-tuning of a VLM is expensive. LoRA lets me adapt a small number of parameters while keeping the base model frozen, making experimentation more practical.”

## 5. Research failure questions

Be ready to answer:

### What if ELA makes performance worse?
“That is a useful result. It means the signal may not generalize under the chosen preprocessing or may introduce distribution-specific artifacts. I would retain the baseline and document the failure rather than force the feature into the final model.”

### What if high accuracy comes from template leakage?
“That is exactly why the split is performed by source document/template family. I also test unseen template families.”

### What if the VLM gives confident wrong explanations?
“I evaluate explanation grounding and hallucination separately from classification correctness. A plausible explanation is not sufficient.”

### What if external dataset performance drops?
“That demonstrates domain shift. I would inspect whether the external dataset differs in resolution, compression, tamper style, document layout, or labeling protocol, then test domain-specific adaptation.”

## 6. Portfolio README structure

Recommended README order:

1. hero statement;
2. demo GIF/image;
3. problem;
4. architecture;
5. key capabilities;
6. synthetic data generation;
7. forensic methods;
8. model training;
9. results;
10. ablations;
11. localization examples;
12. VLM extension;
13. deployment;
14. limitations;
15. reproducibility;
16. citations.

## 7. Resume bullets — fill only with real measurements

### Version A — concise
“Built IntegriDoc, an explainable document-forgery detection pipeline combining ELA, noise residual analysis, transfer learning, and tamper localization across synthetic document manipulations.”

### Version B — research-heavy
“Designed a synthetic document-tampering benchmark with pixel-level masks for text replacement, copy-move, splicing, and region manipulation; evaluated CNN-based detection with forensic feature ablations and localization metrics.”

### Version C — VLM-focused
“Built a multimodal forensic analysis pipeline that combines visual tamper detection, OCR/layout evidence, localization heatmaps, and PEFT-adapted VLM reasoning for structured document-fraud analysis.”

Do not add accuracy, latency, IoU, or model-comparison claims until the experiment runner produces the corresponding numbers.

## 8. HyperVerge alignment map

| HyperVerge area | IntegriDoc evidence |
|---|---|
| Image Forgery Detection | Core detector + localization |
| Document Understanding | document templates + OCR/layout |
| Identity Verification | document authenticity workflow without private PII |
| OCR Enhancement | OCR evidence integration |
| LLMs/VLMs | VLM evidence interpretation |
| SFT | synthetic forensic instruction data |
| PEFT/LoRA | VLM adapter fine-tuning |
| Synthetic Data | controlled tampering generator |
| Research | ablations + failure analysis |
| Benchmarking | public/synthetic evaluation protocol |
| Deployment | FastAPI + frontend + Docker |
| Inference optimization | latency/size benchmarking |

## 9. What makes the project impressive

The impressive part is the **end-to-end research loop**:

```text
problem framing
    ↓
literature review
    ↓
synthetic data generation
    ↓
classical baseline
    ↓
deep model
    ↓
localization
    ↓
ablation
    ↓
failure analysis
    ↓
VLM reasoning
    ↓
API/deployment
    ↓
measured improvement
```

That story maps much better to a research-intern role than a single notebook with one accuracy number.

## 10. Citations to maintain in the repository

Maintain a `docs/references.md` file with the exact version/date/access route for:

- HyperVerge public product/career material;
- DocTamper paper/repository;
- ResNet paper and torchvision implementation;
- ELA/image-forensics references used by the implementation;
- SAM2 paper/repository if used;
- Qwen2.5-VL paper/model documentation if used;
- PEFT/LoRA paper/documentation if used.

For every external dataset or model, record license/access restrictions separately.
