# Stage 6 — AI Builder Execution Contract

This file is the operational contract for Claude Code, Cursor, Windsurf, OpenAI Codex, or another coding agent.

## 1. Mission

Build **IntegriDoc** as a complete, tested, reproducible research prototype for document forgery detection and localization.

Do not produce a mockup, fake metrics, placeholder-only implementation, or “TODO” architecture pretending to be finished.

## 2. Execution strategy

Work in checkpoints.

### CHECKPOINT 0 — repository bootstrap
Create:

- directory tree;
- `pyproject.toml`;
- `.gitignore`;
- `README.md`;
- Makefile/task runner;
- configuration system;
- basic tests;
- pre-commit/lint configuration if practical.

Success criteria: clean install + tests pass.

### CHECKPOINT 1 — forensic utilities
Implement:

- ELA;
- Gaussian high-pass residual;
- Laplacian residual;
- visualizer;
- unit tests.

Success criteria: sample synthetic images produce correct artifacts and tests pass.

### CHECKPOINT 2 — synthetic generator
Implement:

- document templates;
- seeded fictional data generation;
- tamper operators;
- mask generation;
- metadata manifest;
- dataset validator.

Success criteria: generate at least 100 source docs and several tampered variants without invalid masks or cross-split leakage.

### CHECKPOINT 3 — ResNet18 baseline
Implement:

- Dataset/DataLoader;
- group-based split;
- pretrained ResNet18;
- training loop;
- checkpointing;
- evaluation;
- confusion matrix;
- ROC/PR curves.

Success criteria: training completes in Colab and writes reproducible run artifacts.

### CHECKPOINT 4 — forensic fusion
Implement:

- RGB-only baseline;
- RGB+ELA;
- RGB+residual;
- RGB+ELA+residual;
- ablation table.

Success criteria: all experiments use the same split and report comparable metrics.

### CHECKPOINT 5 — localization
Implement:

- Grad-CAM;
- segmentation model;
- Dice/IoU evaluation;
- qualitative overlays.

Success criteria: localization report includes both visualizations and numeric metrics.

### CHECKPOINT 6 — robustness
Add:

- post-processing stress tests;
- unseen template family test;
- tamper-type breakdown;
- false-positive/false-negative report.

Success criteria: report where performance breaks.

### CHECKPOINT 7 — VLM bonus
Add only after P0/P1 works:

- OCR integration;
- Qwen2.5-VL baseline;
- structured forensic prompt;
- optional LoRA training;
- JSON parser/validator;
- hallucination/grounding evaluation.

### CHECKPOINT 8 — deployment
Implement:

- FastAPI;
- `/health`;
- `/models`;
- `/v1/analyze`;
- artifact rendering;
- frontend;
- Docker.

### CHECKPOINT 9 — polish
Create:

- final README;
- architecture diagram;
- experiment table;
- model card;
- dataset card;
- limitations;
- research report;
- portfolio images;
- resume bullets based only on measured results.

## 3. Coding rules for the agent

1. Inspect the existing repository before creating files.
2. Never overwrite working code blindly.
3. Prefer small modules with explicit interfaces.
4. Do not place training logic in notebooks only; core logic belongs in `src/`.
5. Notebooks may call reusable Python modules.
6. Every CLI command must support `--help`.
7. All random seeds must be configurable.
8. All paths must be configurable.
9. Do not hard-code local absolute paths.
10. Do not commit secrets, dataset credentials, or private files.
11. Do not fabricate evaluation results.
12. If a dependency is unavailable, document the issue and implement a CPU-safe fallback where reasonable.
13. Never use a real person’s private document to improve the project.
14. Do not call an uncalibrated score a probability.
15. Do not call Grad-CAM a ground-truth localization mask.

## 4. Expected command interface

Create commands similar to:

```bash
make setup
make test
make lint
make generate-data
make validate-data
make train MODEL=resnet18 CONFIG=configs/train_resnet18.yaml
make evaluate RUN=<run_id>
make explain IMAGE=<path>
make serve
make demo
```

If Make is inconvenient on Windows, provide equivalent Python CLI commands.

## 5. Required CLI modules

```text
python -m src.data.generate
python -m src.data.validate
python -m src.forensics.ela
python -m src.forensics.residuals
python -m src.train
python -m src.evaluate
python -m src.explain
python -m src.inference
python -m apps.api.main
```

## 6. Error handling

Errors must be explicit and actionable.

Example:

```text
MODEL_CHECKPOINT_MISSING:
Expected checkpoint at checkpoints/resnet18_fusion/best.pt
Run: make train MODEL=resnet18_fusion
```

## 7. Testing requirements

Unit tests:

- ELA output shape/range;
- residual output shape/range;
- text replacement mask correctness;
- copy-move mask correctness;
- dataset manifest parsing;
- split leakage detection;
- model forward pass;
- localization output shape;
- API schema validation.

Integration tests:

- synthetic generation → dataset load → inference;
- API upload → inference → JSON response;
- demo sample → complete evidence bundle.

## 8. No fabricated metrics

This is a hard requirement.

Never write “92% accuracy” unless the training/evaluation script actually measured 92% under a documented protocol.

In documentation, use placeholders such as:

`[FILL FROM FINAL EXPERIMENT]`

until real results exist.

## 9. Research logging

Each run must save a machine-readable record:

```json
{
  "experiment_id": "E04",
  "git_sha": "...",
  "dataset_version": "...",
  "model": "resnet18_fusion",
  "seed": 42,
  "metrics": {},
  "config": {},
  "notes": ""
}
```

## 10. Final acceptance checklist

- [ ] Repository installs cleanly.
- [ ] Unit tests pass.
- [ ] Synthetic data generator works.
- [ ] Masks are valid.
- [ ] Group-based split verified.
- [ ] ELA works.
- [ ] Residual analysis works.
- [ ] ResNet18 trains.
- [ ] Evaluation report generated.
- [ ] EfficientNet comparison attempted.
- [ ] Grad-CAM works.
- [ ] Localization model works.
- [ ] Robustness evaluation completed.
- [ ] VLM experiment documented or intentionally deferred.
- [ ] API works.
- [ ] UI works.
- [ ] Docker smoke test works.
- [ ] README contains real results only.
- [ ] Model/dataset cards exist.
- [ ] Limitations are explicit.

## 11. Agent behavior when blocked

If blocked by a missing dataset, unavailable GPU, restricted model, or dependency incompatibility:

1. continue with the local/synthetic subset;
2. create a clean adapter/interface for the unavailable component;
3. document exactly what was blocked;
4. never silently replace the experiment with a different one and claim equivalence.
