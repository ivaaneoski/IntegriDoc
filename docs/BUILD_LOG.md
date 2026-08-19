# Build Log

## Phase 0 & 1 - 2026-08-19

### Implementation Summary
- Initialized repository structure (`src`, `configs`, `data`, `scripts`, `results`).
- Added configuration files (`pyproject.toml`, `default.yaml`, `.gitignore`).
- Created a procedural synthetic ID card generator using Pillow (`src/data/generator.py`).
- Implemented tamper operators for text replacement and copy-move (`src/data/tamper_ops.py`).
- Implemented Error Level Analysis (ELA) using JPEG recompression (`src/forensics/ela.py`).
- Implemented Gaussian and Laplacian high-pass residual extraction (`src/forensics/residuals.py`).
- Wrote and executed a full end-to-end smoke test script (`scripts/smoke_test.py`).

### Tests Run
- Ran manual smoke test generating 6 artifacts (authentic, tampered, mask, ELA heatmap, Gaussian residual, Laplacian residual).
- Commands: `python scripts/smoke_test.py`

### Known Limitations
- Synthetic data generator currently uses a simple text drawing routine without complex backgrounds.
- Bounding boxes are simplified; text replacement currently uses solid white blockouts before drawing new text.
- ELA uses a static percentile strategy for visualization scaling, which might need tuning on larger datasets.

### Next Step
- Phase 4: Forensic fusion experiments and ablations.

## Phase 3 (Checkpoint 3) - 2026-08-19

### Implementation Summary
- Created group-based data splitting (`src/data/split.py`) to prevent source leakage.
- Scaled up the synthetic generator script (`scripts/generate_dataset.py`) to output train/val/test JSON manifests.
- Implemented `DocumentDataset` in PyTorch with augmentations (`src/data/dataset.py`).
- Integrated torchvision's pretrained ResNet18, modifying the final layer for binary classification (`src/models/resnet.py`).
- Built training loop (`src/train.py`) that falls back to CPU if CUDA isn't available, saving the best model and metrics to `results/runs/`.
- Packaged everything for Colab in `notebooks/Colab_Training_Runner.ipynb`.

### Tests Run
- Generated 300 image dataset (100 source documents + 2 variations each).
- Ran a local 1-epoch dry-run on CPU using `python src/train.py --config configs/train_resnet18.yaml`. The model correctly downloaded the pretrained weights and saved the checkpoint, metrics, and confusion matrix.
