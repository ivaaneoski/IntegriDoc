# Stage 4 — VLM, PEFT/LoRA & Advanced Research Modules

## 1. Why add a VLM?

The internship role explicitly emphasizes LLMs, VLMs, SFT, PEFT/LoRA, document understanding, OCR enhancement, identity verification, image forgery detection, research reproduction, synthetic data, and end-to-end ownership.

The VLM should therefore be an **evidence interpreter**, not the sole forgery detector.

The vision pipeline should first produce structured evidence. The VLM then converts that evidence into a useful, grounded forensic report.

## 2. Recommended VLM role

Input to the VLM:

- original document image;
- OCR text + bounding boxes;
- ELA visualization;
- residual visualization;
- classifier score;
- localization mask/heatmap;
- detected suspicious regions.

Output:

```json
{
  "summary": "Likely tampered",
  "suspicious_regions": [
    {
      "field": "date field",
      "reason": "localized compression/noise inconsistency",
      "evidence_strength": "medium"
    }
  ],
  "recommended_action": "manual_review",
  "limitations": [
    "VLM explanation is not independent proof of tampering."
  ]
}
```

## 3. VLM model choice

A practical research candidate is **Qwen2.5-VL**, available in multiple parameter sizes including 3B and 7B variants. The Hugging Face documentation describes it as a multimodal vision-language model with dynamic visual handling. Use the smallest model that fits the available GPU, beginning with a 3B/7B-class checkpoint rather than a 32B/72B model. 

Keep model name/path configurable. Do not hard-code one checkpoint throughout the application.

## 4. Grounded VLM prompting

Do not ask:

> “Is this document fake?”

Instead ask for structured evidence interpretation:

```text
You are a document-forensics assistant.
You are not allowed to infer authenticity from personal identity, nationality, or visual stereotypes.
Base every conclusion only on the supplied image and forensic evidence.

Inputs:
- document image
- OCR regions
- ELA map
- residual map
- detector score
- localization heatmap

Task:
1. identify suspicious visual regions;
2. describe which supplied evidence supports each region;
3. distinguish observed evidence from inference;
4. output JSON only;
5. recommend REAL / MANUAL_REVIEW / TAMPERED according to the supplied score and evidence.
```

## 5. VLM SFT dataset

Create a synthetic instruction dataset from the project’s own generated examples.

Each record contains:

- document image;
- forensic evidence bundle;
- label;
- tamper type;
- mask/bbox;
- structured explanation.

Example training item:

```json
{
  "messages": [
    {"role": "user", "content": "<image> ... forensic evidence ..."},
    {"role": "assistant", "content": "{...structured forensic explanation...}"}
  ]
}
```

Only generate explanations that are justified by the known synthetic ground truth and model evidence.

## 6. LoRA/PEFT plan

Use Hugging Face PEFT/LoRA for the VLM fine-tuning experiment. PEFT updates a small number of adapter parameters instead of the entire model, reducing memory and optimizer-state requirements. The official documentation recommends LoRA as a common, efficient starting point. 

Initial config should be exposed as YAML:

```yaml
r: 16
lora_alpha: 32
lora_dropout: 0.05
bias: none
learning_rate: 1.0e-4
```

Target attention modules should be selected according to the exact model architecture rather than copied blindly. Inspect the model module names first.

## 7. VLM evaluation

Evaluate four capabilities independently:

### A. Classification agreement
Does the VLM recommendation agree with the actual label?

### B. Evidence grounding
Does its stated suspicious region overlap the true tamper mask/bbox?

### C. Hallucination rate
How often does it invent evidence not present in the supplied artifacts?

### D. Structured output validity
Percentage of outputs that parse cleanly into the target schema.

Do not use “sounds convincing” as a metric.

## 8. OCR integration

OCR should supply layout-aware text regions rather than only plain text.

Required OCR object:

```json
{
  "text": "DATE OF BIRTH: 14/08/2002",
  "bbox": [0.52, 0.30, 0.81, 0.40],
  "confidence": 0.97
}
```

Use OCR evidence to support field-level analysis such as:

- inconsistent font/size;
- spacing anomalies;
- altered characters;
- text-region localization mismatch.

OCR itself must not be treated as proof of tampering.

## 9. Retrieval-aware evidence library

Optional research extension:

Create a small internal knowledge base of:

- project methodology;
- forensic feature definitions;
- experiment results;
- known failure modes;
- model cards.

A retrieval layer can provide the VLM with definitions and experiment context. Do not retrieve private personal data.

## 10. SAM2-assisted localization

Meta’s Segment Anything family supports promptable segmentation; the official SAM2 repository also provides released training/fine-tuning code and checkpoints. 

Use SAM2 only as an **assistive localization experiment**.

Do not replace the learned tamper-localization model with SAM2.

Recommended experiment:

1. classifier/Grad-CAM proposes suspicious points/boxes;
2. SAM2 converts the proposal into a clean region mask;
3. compare the SAM2-assisted mask with the learned segmentation mask and ground truth;
4. report whether this actually improves IoU/Dice.

The scientific question is more important than the novelty of adding SAM2.

## 11. CLIP experiment

Optional:

- extract embeddings for authentic and tampered images;
- measure class separation;
- train a lightweight linear probe;
- compare against ResNet embeddings.

Do not present zero-shot anomaly scoring as a reliable production detector without validation.

## 12. Advanced experiments worth pursuing

### Experiment V1 — VLM with raw image only
Establish a weak baseline.

### V2 — VLM + OCR
Test whether structured OCR improves reasoning.

### V3 — VLM + forensic maps
Add ELA and residual evidence.

### V4 — VLM + localization
Add detector-generated heatmap/mask.

### V5 — LoRA fine-tuned VLM
Fine-tune on synthetic evidence-grounded explanations.

The key research claim should be:

> “Does explicit forensic evidence improve multimodal document-tampering reasoning compared with visual-only VLM inspection?”

## 13. Research hygiene

Every advanced experiment must record:

- exact model checkpoint;
- prompt version;
- dataset version;
- number of examples;
- train/validation/test split;
- LoRA config if applicable;
- decoding settings;
- evaluation script version;
- failure examples.

Avoid vague claims such as “the VLM understands forgery.”
