# Heart Disease Prediction — MLOps Assignment 01

An original, reproducible MLOps learning project for the UCI Heart Disease dataset. The goal is to demonstrate a complete path from data ingestion and exploratory analysis through tracked model training, automated testing, API packaging, Docker, local Kubernetes, and monitoring.

> **Educational use only:** this model is not a medical device and must not be used to diagnose or make clinical decisions.

## Current implementation status

- [x] Step 1 scaffolding: data download/cleaning code, EDA script, and initial unit tests
- [ ] Step 1 verified against a live UCI download and generated EDA artifacts
- [ ] Model training and cross-validation
- [ ] MLflow experiment tracking
- [ ] Model packaging and inference API
- [ ] CI workflow and automated tests
- [ ] Docker build and endpoint verification
- [ ] Minikube deployment, monitoring, screenshots, and report

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Download, clean, and inspect the dataset

```powershell
python -m src.data
python -m src.eda
```

The scripts download the official UCI Heart Disease archive, extract the Cleveland subset, convert `?` to missing values, map the original target values `0` and `1–4` to binary `0/1`, and save a cleaned CSV. Missing predictors are intentionally retained at this stage so that later training can impute them inside a fitted preprocessing pipeline, preventing train/test leakage.

## Run tests and lint

```powershell
pytest -q
ruff check src tests
```

## Dataset provenance

- Source: UCI Machine Learning Repository, Heart Disease dataset, dataset ID 45.
- Download URL used by the script: `https://archive.ics.uci.edu/static/public/45/heart+disease.zip`
- The original dataset uses `?` as a missing-value marker for some fields. The original target is an integer from 0 to 4; this project defines target 0 as absence and values 1–4 as presence of disease.

## Reproducibility notes

- Random seeds and exact dependency versions will be recorded in the modeling stage.
- Preprocessing will be fit on training folds only and packaged with the selected estimator.
- Reported scores will come from executed runs; no metric is prefilled or assumed.
