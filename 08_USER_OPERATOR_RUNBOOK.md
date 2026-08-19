# Stage 8 — User Operator Runbook

This file is written for the human building and presenting IntegriDoc. The AI coding agent builds the system; **you operate it, train it, inspect experiments, and turn the results into a research portfolio project**.

## 1. Your role

You are not expected to write every line of the system manually. Your job is to:

1. run the repository and verify the environment;
2. generate and inspect data;
3. acquire only permitted external datasets;
4. launch training experiments in Google Colab or another GPU runtime;
5. compare metrics rather than choosing the prettiest result;
6. inspect false positives/false negatives;
7. decide which experiments are worth keeping;
8. document what actually happened;
9. deploy the final checkpoint;
10. present the work as an engineering + research project.

Do **not** let an AI coding agent invent dataset downloads, metrics, benchmark claims, or research conclusions.

---

## 2. Recommended operating setup

### Local Windows PC

Use the local machine for:

- Git/GitHub;
- editing code;
- running tests;
- generating small smoke datasets;
- running CPU inference;
- building the UI/API;
- reviewing plots and reports.

Your current local hardware is adequate for this role. Serious model training should be moved to a GPU runtime.

### Google Colab / GPU runtime

Use Colab for:

- synthetic dataset generation at scale;
- ResNet/EfficientNet training;
- localization training;
- evaluation;
- VLM experiments;
- LoRA fine-tuning when feasible.

Keep the project in GitHub and store large datasets/checkpoints in Google Drive or another documented artifact store. Do not put datasets or model checkpoints into Git.

---

## 3. First-time local setup

### 3.1 Install prerequisites

Recommended:

- Git
- Python 3.11
- VS Code
- GitHub account
- Docker Desktop (optional until deployment)
- Google Chrome for Colab

Verify:

```powershell
git --version
python --version
```

Python should report a supported 3.11.x version.

### 3.2 Clone the repository

```powershell
git clone <YOUR_GITHUB_REPO_URL>
cd integridoc
```

### 3.3 Create a virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

or activate from a CMD shell:

```cmd
.venv\Scripts\activate.bat
```

### 3.4 Install the project

Prefer the project package configuration created by the AI builder:

```powershell
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

If the project uses a requirements file instead:

```powershell
pip install -r requirements.txt
```

### 3.5 Verify installation

```powershell
python -m pytest -q
python -m src.data.generate --help
python -m src.train --help
python -m src.evaluate --help
```

The repository is not ready for training if basic tests do not pass.

---

## 4. Understand the repository before running anything

Open these files first:

```text
README.md
configs/data.yaml
configs/train_resnet18.yaml
configs/train_efficientnet.yaml
configs/localization.yaml
configs/vlm.yaml
```

Then inspect:

```text
src/data/
src/forensics/
src/models/
src/localization/
src/evaluation/
src/inference/
apps/api/
apps/web/
```

The important idea is:

```text
source documents
      ↓
synthetic authentic + synthetic tampered variants
      ↓
manifest + masks
      ↓
train/val/test split
      ↓
forensic features + CNN
      ↓
classification + localization
      ↓
explainability
      ↓
API + UI
```

---

## 5. Generate your first dataset locally

Do not begin with 20,000 documents.

Start with a smoke dataset.

Example command:

```powershell
python -m src.data.generate \
  --num-sources 100 \
  --variants-per-source 2 \
  --seed 42 \
  --output data/synthetic
```

On PowerShell, the continuation character may be easier as one line:

```powershell
python -m src.data.generate --num-sources 100 --variants-per-source 2 --seed 42 --output data/synthetic
```

Then validate it:

```powershell
python -m src.data.validate --root data/synthetic
```

You should inspect:

```text
data/synthetic/
├── images/
├── masks/
├── manifests/
└── previews/
```

Open several authentic/tampered pairs manually.

### What you are checking visually

- Is the edit actually visible?
- Is the tampered field realistic?
- Does the mask cover the changed region and not the entire page?
- Are copy-move source and target regions correct?
- Are fonts and anti-aliasing believable?
- Are authentic examples also receiving normal scan/compression variation?
- Are there any obviously synthetic artifacts that make the classification trivial?

If the generated examples look unrealistic, **fix the generator before training**.

---

## 6. Build the main synthetic dataset

Once the smoke test is good, scale gradually.

Recommended first main run:

```powershell
python -m src.data.generate \
  --num-sources 5000 \
  --variants-per-source 3 \
  --seed 42 \
  --output data/synthetic_main
```

Then:

```powershell
python -m src.data.validate --root data/synthetic_main --strict
```

If the machine runs out of time/storage, generate the dataset directly in Colab/Google Drive instead.

Do not assume that more images automatically means a better model. Diversity and source separation matter more than raw count.

---

## 7. Dataset split rule you must personally verify

The split is by `source_id`.

Correct:

```text
source_0001 → train
source_0002 → train
source_0101 → validation
source_0201 → test
```

Incorrect:

```text
source_0001_authentic → train
source_0001_tampered → test
```

The second case leaks source-specific visual information.

Run the leakage checker:

```powershell
python -m src.data.validate --check-split-leakage --root data/synthetic_main
```

Do not start a final experiment until this is clean.

---

## 8. External dataset acquisition

### 8.1 DocTamper

Use DocTamper as a **domain evaluation/reference dataset**, not as a replacement for your own synthetic dataset.

The official repository currently says the dataset is available via BaiduDrive and Kaggle, is for non-commercial use, and requires an access/request process. Its official repository is the source of truth for current access requirements. citeturn932630search0

Current official repository:

https://github.com/qcf-568/DocTamper

### How you should obtain it

1. Open the official repository.
2. Read the current access/license instructions.
3. Follow the requested educational/research access procedure.
4. Download the dataset through the permitted source.
5. Keep the raw dataset outside Git.
6. Record the exact dataset version/date and license terms in `data/manifests/external_sources.yaml`.
7. Never redistribute a restricted dataset inside your GitHub repository.

If you cannot obtain the dataset under its permitted terms, **do not fabricate a substitute called DocTamper**. Continue with the synthetic project and document the external-evaluation limitation.

### 8.2 CASIA or generic forensic data

A generic image-forensics dataset may be used as a robustness experiment, but it should remain clearly labelled as generic image forensics rather than identity-document verification.

Before downloading any external dataset:

- verify license;
- verify whether redistribution is allowed;
- record the source URL;
- record download date;
- record expected checksum if supplied;
- store it outside Git;
- create a local manifest entry.

---

## 9. Recommended Colab workflow

### 9.1 Create the runtime

Open Google Colab and enable a GPU runtime.

Then check:

```python
!nvidia-smi
```

Also run:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

Do not begin a full training run if CUDA is unavailable.

### 9.2 Get the repository

Clone your GitHub repository in the Colab session:

```python
!git clone <YOUR_GITHUB_REPO_URL>
%cd integridoc
```

Install:

```python
!pip install -e ".[dev]"
```

If the repository uses `requirements.txt`, use that instead.

### 9.3 Mount Google Drive for large artifacts

Use Drive for:

```text
/content/drive/MyDrive/IntegriDoc/
├── datasets/
├── checkpoints/
├── runs/
└── exports/
```

Do not save every temporary artifact forever. Copy only meaningful runs/checkpoints.

---

## 10. Training sequence you should follow

Do not skip directly to the VLM.

### Experiment E01 — ELA baseline

Run the ELA analyzer on validation/test data.

Goal:

- understand compression artifacts;
- establish whether ELA contains useful signal;
- create a visual baseline.

### Experiment E02 — residual baseline

Run high-pass/Laplacian residual analysis.

Goal:

- establish whether high-frequency inconsistency is informative.

### Experiment E03 — RGB ResNet18

Train the standard classifier.

```bash
python -m src.train \
  --config configs/train_resnet18.yaml \
  --run-name E03_resnet18_rgb
```

Record:

- best validation F1;
- test F1;
- precision;
- recall;
- ROC-AUC;
- PR-AUC;
- confusion matrix;
- inference latency.

### Experiment E04 — RGB + ELA + residual

Train the fusion model:

```bash
python -m src.train \
  --config configs/train_resnet18_fusion.yaml \
  --run-name E04_resnet18_fusion
```

This is one of the most important experiments in the project.

The question is:

> Does explicit forensic evidence improve over the RGB-only model?

### Experiment E05 — EfficientNet-B0

Use the same split, seed policy, and evaluation protocol.

```bash
python -m src.train \
  --config configs/train_efficientnet.yaml \
  --run-name E05_efficientnet
```

### Experiment E06 — localization

Train the segmentation/localization model:

```bash
python -m src.train \
  --config configs/localization.yaml \
  --run-name E06_localization
```

### Experiment E07 — Grad-CAM

```bash
python -m src.explain \
  --checkpoint checkpoints/E04_resnet18_fusion/best.pt \
  --image <IMAGE_PATH> \
  --output results/explanations/example_01
```

### Experiment E08 — robustness

Test:

- JPEG recompression;
- resize;
- blur;
- brightness/contrast shifts;
- mixed post-processing.

### Experiment V1+ — VLM

Only start after E03–E08 have working artifacts and metrics.

---

## 11. How to choose the best checkpoint

Do **not** choose the checkpoint with the best training accuracy.

Prefer the model that has:

1. strong validation performance;
2. good test performance;
3. stable results across seeds where possible;
4. reasonable false-positive rate;
5. useful localization;
6. robustness to post-processing;
7. acceptable latency;
8. sensible failure modes.

For a fraud-like application, discuss precision/recall tradeoffs explicitly rather than treating accuracy as the only objective.

---

## 12. How to evaluate one image manually

After training:

```bash
python -m src.inference \
  --image examples/demo/tampered_text_01.png \
  --checkpoint checkpoints/E04_resnet18_fusion/best.pt \
  --output results/manual_case_01
```

Inspect all generated outputs:

```text
results/manual_case_01/
├── response.json
├── original.png
├── ela.png
├── residual.png
├── gradcam.png
└── localization.png
```

Ask yourself:

- Did the detector flag the correct document?
- Did ELA highlight the edited region?
- Did residual analysis agree?
- Did Grad-CAM focus on the tampered region?
- Did the segmentation mask make sense?
- If it failed, why?

This qualitative inspection is part of the research, not merely demo preparation.

---

## 13. How to run the API locally

After a trained checkpoint is available:

```powershell
python -m apps.api.main
```

Or, if using Uvicorn directly:

```powershell
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
```

Check:

```text
http://localhost:8000/health
```

Open interactive API documentation:

```text
http://localhost:8000/docs
```

Test the endpoint with curl:

```powershell
curl.exe -X POST "http://localhost:8000/v1/analyze" `
  -F "file=@examples/demo/tampered_text_01.png" `
  -F "run_ocr=false" `
  -F "run_vlm=false" `
  -F "return_artifacts=true"
```

---

## 14. How to run the web demo

Use the command generated by the repository's frontend setup.

Typical examples:

```bash
npm install
npm run dev
```

or, for a simple Python UI:

```bash
streamlit run apps/web/app.py
```

The final UI should show:

- uploaded document;
- verdict;
- tamper score;
- ELA;
- residual map;
- Grad-CAM;
- localization mask;
- OCR evidence when enabled;
- VLM explanation when enabled.

---

## 15. How to run the project fully end-to-end

The ideal local demo sequence is:

```bash
# 1. verify repository
make test

# 2. generate demo data
make generate-data

# 3. validate
make validate-data

# 4. start API
make serve

# 5. start frontend in another terminal
make demo
```

The exact commands should match the final repository implementation.

The AI agent must keep this runbook synchronized with the actual CLI commands instead of leaving these examples stale.

---

## 16. What artifacts you should save from every final experiment

For each meaningful run, preserve:

```text
runs/<experiment_id>/
├── config.yaml
├── metrics.json
├── predictions.csv
├── confusion_matrix.png
├── roc_curve.png
├── pr_curve.png
├── examples/
├── failures/
├── checkpoint_metadata.json
└── README.md
```

At the end, the report should be reproducible from these artifacts.

---

## 17. How to conduct failure analysis

Take the 25–50 strongest false positives and false negatives.

Create folders by failure type:

```text
reports/failure_analysis/
├── false_positive_compression/
├── false_positive_template_artifact/
├── false_negative_small_text_edit/
├── false_negative_copy_move/
└── false_negative_postprocessed/
```

For each case record:

- ground truth;
- predicted class;
- score;
- template family;
- tamper type;
- ELA;
- residual;
- Grad-CAM;
- localization output;
- likely reason for failure;
- proposed next experiment.

This section is one of the best parts to discuss in an interview because it demonstrates research thinking.

---

## 18. How to handle metrics

Never copy metrics from a paper and present them as your model's results.

Do not write:

> “Our model reaches 95.4% accuracy”

until the evaluation script actually produced that result under the documented split.

Use:

```text
Metric: F1
Split: source-aware test
Experiment: E04
Value: <actual measured value>
```

Keep the exact experiment config beside the result.

---

## 19. Recommended run order for you personally

### Session 1

- clone repo;
- run tests;
- generate 100 synthetic docs;
- inspect masks;
- fix obvious generator problems.

### Session 2

- generate 500–1,000 source docs;
- run ELA/residual visualizations;
- inspect data diversity;
- verify leakage checker.

### Session 3

- launch Colab;
- train E03 ResNet18;
- save metrics/checkpoint.

### Session 4

- train E04 fusion;
- compare directly with E03.

### Session 5

- EfficientNet;
- Grad-CAM;
- localization.

### Session 6

- robustness;
- failure analysis;
- external dataset evaluation if legally/permittedly available.

### Session 7

- OCR;
- VLM baseline;
- evidence-grounding experiment.

### Session 8

- LoRA experiment;
- API;
- UI;
- Docker.

### Session 9

- final metrics;
- README;
- research report;
- architecture diagram;
- portfolio screenshots;
- interview preparation.

---

## 20. What you should show a HyperVerge interviewer

Do not lead with:

> “I made a ResNet classifier for fake documents.”

Lead with the research/engineering pipeline:

> “I built a document-forensics system that generates controlled synthetic tampering with pixel-level ground truth, evaluates classical forensic signals like ELA and residuals, benchmarks CNN detectors, performs tamper localization, and then uses OCR and a VLM to reason over the detector's evidence.”

Then show:

1. synthetic data generator;
2. one clean vs tampered pair;
3. exact mask;
4. ELA/residual maps;
5. baseline vs fusion metrics;
6. Grad-CAM/localization;
7. failure analysis;
8. VLM evidence-grounding experiment;
9. live API/UI.

The story is much stronger when you can answer:

> “What did you test, what failed, and why did your next experiment exist?”

---

## 21. Final personal checklist

Before calling the project finished:

- [ ] I can clone and install the repo from scratch.
- [ ] I can generate synthetic data without the AI agent.
- [ ] I understand the manifest and split logic.
- [ ] I can run ELA and residual analysis.
- [ ] I can train ResNet18 in Colab.
- [ ] I can compare baseline vs forensic fusion.
- [ ] I can reproduce the final metrics.
- [ ] I understand every reported metric.
- [ ] I have inspected failure cases.
- [ ] I can explain why the model can fail.
- [ ] I can launch the API.
- [ ] I can run the demo UI.
- [ ] I can explain the VLM experiment.
- [ ] I know the license/access status of every external dataset.
- [ ] I have no private identity documents in the repository.
- [ ] I have never fabricated a metric or benchmark.

---

## 22. Most important rule

**You should be able to reproduce every major result without relying on the coding agent's memory.**

The coding agent creates the machinery. Your experiment logs, configs, checkpoints, data manifests, and reports are what make the project yours.
