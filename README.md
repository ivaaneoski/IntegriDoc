# IntegriDoc
**Explainable document forgery detection and localization.**

## Overview
IntegriDoc is a research-oriented document forgery detection system. It generates synthetic documents and evaluates classical forensic signals (ELA, Noise Residuals) and deep neural network approaches.

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -e .[dev]
```

## Smoke Test
To verify the initial pipeline (Phase 0):
```bash
python scripts/smoke_test.py
```
