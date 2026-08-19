# IntegriDoc — Project Specification Pack

## Working project name
**IntegriDoc** — AI Document Forensics & Tamper Localization

> Do not use the old placeholder name “DocGuard” anywhere in the repository, UI, package names, class names, or portfolio copy.

## Why this project exists
IntegriDoc is a research-oriented document forgery detection system designed around the exact problem family relevant to HyperVerge: document understanding, identity verification, OCR-adjacent document analysis, image forgery detection, synthetic data generation, model evaluation, and production inference.

HyperVerge’s own materials describe a “Document Forgery Check” that flags tampered Aadhaar cards, edited salary slips, and fabricated bank statements. The project should therefore demonstrate a closely related defensive workflow without using real private identity documents.

## Primary outcome
Given a document image, the system should produce:

1. Authenticity verdict: `REAL`, `TAMPERED`, or `UNCERTAIN`.
2. Tamper score, explicitly labelled as a model score rather than a calibrated probability unless calibration is performed.
3. Tamper localization heatmap/mask.
4. Forensic signals: ELA, noise residual, compression/artifact cues.
5. Optional OCR-derived suspicious fields.
6. Optional VLM-generated forensic explanation grounded in the actual visual evidence.
7. Machine-readable JSON response for API use.

## Non-negotiable safety/data constraints
- Use fictional documents, public research datasets, or datasets with explicit research permissions.
- Never collect, scrape, or store real people’s Aadhaar/passports/IDs/bank statements for this project.
- Synthetic identities must be fictional.
- Keep downloaded datasets outside git.
- Record source, license/access terms, hash/version, and train/validation/test role for every external dataset.
- Do not describe synthetic examples as real-world fraud cases.

## What “done” means
The repository is complete only when all of the following exist:

- reproducible environment setup;
- synthetic document generator;
- tamper-mask generator;
- ELA and noise-residual modules;
- baseline classifier;
- strong transfer-learning classifier;
- proper group-based data split;
- evaluation suite;
- Grad-CAM/explainability;
- localization evaluation;
- inference API;
- demo UI;
- experiment tracking/results table;
- model cards and limitations;
- tests;
- documentation;
- one clear research story with ablations.

## Staged build order
Read the files in this order. Do not load all specifications into an AI coding session at once unless explicitly required.

| Stage | File | Purpose |
|---|---|---|
| 0 | `00_README_AND_EXECUTION_ORDER.md` | Scope, principles, stage order |
| 1 | `01_PRODUCT_SPEC_AND_ARCHITECTURE.md` | Product requirements and system architecture |
| 2 | `02_DATA_FORENSICS_AND_SYNTHETIC_GENERATOR.md` | Dataset design, ELA, residuals, synthetic tampering |
| 3 | `03_MODEL_TRAINING_AND_EVALUATION.md` | PyTorch models, losses, training, evaluation, ablations |
| 4 | `04_VLM_RESEARCH_AND_ADVANCED_MODULES.md` | VLM, LoRA/PEFT, OCR reasoning, SAM2, advanced research |
| 5 | `05_DEPLOYMENT_API_AND_DEMO.md` | FastAPI, inference pipeline, UI, packaging |
| 6 | `06_AI_BUILDER_EXECUTION_CONTRACT.md` | Exact instructions for Claude Code / coding agents |
| 7 | `07_RESEARCH_NOTES_AND_INTERVIEW_STORY.md` | Research framing, experiments, portfolio/interview narrative |

## Priority tiers
### P0 — must ship
Synthetic generator, forensic baselines, ResNet/EfficientNet classifier, localization, evaluation, API, demo.

### P1 — strongly recommended
EfficientNet comparison, OCR field extraction, calibration, ablation study, external document-tampering evaluation, model failure analysis.

### P2 — research bonus
VLM forensic reasoning, LoRA fine-tuning, SAM2-assisted localization, CLIP representation experiment, retrieval-aware evidence lookup.

## Core design principle
**Build the smallest complete system first. Add sophistication only after the previous stage has measured evidence that it works.**

This prevents the project from becoming “a collection of models” and turns it into an actual research/engineering project.
