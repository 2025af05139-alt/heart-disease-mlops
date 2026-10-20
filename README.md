# Heart Disease Classification — MLOps Assignment 01

An end-to-end MLOps project for binary heart-disease classification using the UCI Heart Disease dataset. This project demonstrates data acquisition, exploratory data analysis, model training and evaluation, MLflow experiment tracking, automated testing, Docker containerization, and local Kubernetes deployment.

> **Disclaimer:** This project is for educational demonstration only. It is not a medical device and must not be used for diagnosis or treatment decisions.

## Project Architecture

```text
UCI Heart Disease Dataset
          |
          v
 Data Acquisition & Cleaning
          |
          v
 Exploratory Data Analysis
          |
          v
 Preprocessing & Model Training
          |
          +---- Logistic Regression
          |
          +---- Random Forest
          |
          v
 Cross-Validation & Evaluation
          |
          v
 MLflow Experiment Tracking
          |
          v
 Selected Model Artifact
          |
          v
 FastAPI Prediction Service
          |
          v
 Docker Container
          |
          v
 Minikube Kubernetes Deployment

GitHub Actions: Linting, tests, data preparation,
EDA, model training, and artifact uploads
```

## Technology Stack

- **Language:** Python 3.11
- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn
- **Experiment tracking:** MLflow with SQLite tracking storage
- **API:** FastAPI and Uvicorn
- **Testing and linting:** pytest and Ruff
- **Containerization:** Docker
- **Orchestration:** Kubernetes using Minikube
- **CI:** GitHub Actions

## Repository Structure

```text
heart-disease-mlops/
├── .github/
│   └── workflows/
│       └── ci.yml
├── artifacts/
│   └── models/
│       └── heart_disease_best_model.joblib
├── data/
│   ├── raw/
│   └── processed/
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── reports/
│   ├── figures/
│   ├── model_comparison.csv
│   └── model_metrics.json
├── src/
│   ├── data.py
│   ├── eda.py
│   ├── train.py
│   └── api.py
├── tests/
│   ├── test_api.py
│   └── test_data.py
├── Dockerfile
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Dataset and Preprocessing

- **Source:** UCI Machine Learning Repository — Heart Disease dataset, ID 45.
- **Download:** https://archive.ics.uci.edu/static/public/45/heart+disease.zip
- **Dataset used:** Cleveland subset.
- **Target:** Binary classification; original target value `0` represents absence of disease, while values `1–4` represent presence of disease.

The preprocessing workflow converts `?` markers to missing values, handles missing predictors within the machine-learning preprocessing pipeline, and prepares a clean dataset for model development.

The cleaned dataset contains 303 rows and 14 columns. The reported target distribution is 164 samples in class 0 and 139 in class 1.

## Installation

The following commands use a Python 3.11 environment.

### Windows PowerShell

```powershell
conda create -n mlops python=3.11 -y
conda activate mlops

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Alternatively, create a Python virtual environment and install the same requirements.

### Download and prepare the data

```powershell
python -m src.data
```

### Generate EDA reports

```powershell
python -m src.eda
```

Generated reports include class balance, feature histograms, correlation heatmaps, and an EDA summary.

### Train and evaluate the models

```powershell
python -m src.train
```

The training pipeline evaluates Logistic Regression and Random Forest using stratified cross-validation, compares their performance, selects a model based on mean cross-validation ROC-AUC, saves the selected estimator, and logs experiment information to MLflow.

## Model Evaluation

The recorded experimental run selected Logistic Regression based on mean cross-validation ROC-AUC.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Mean CV ROC-AUC | 0.9025 | 0.8964 |
| Test accuracy | 0.8852 | 0.8689 |
| Test precision | 0.8387 | 0.8125 |
| Test recall | 0.9286 | 0.9286 |
| Test ROC-AUC | 0.9665 | 0.9491 |

These are results from the recorded run, not guarantees of performance on new data.

Evaluation artifacts are saved under `reports/`, including model comparison results, metrics, and evaluation plots.

## MLflow Experiment Tracking

The experiment is named **Heart Disease Classification**.

MLflow tracks model parameters, cross-validation and test metrics, model artifacts, and evaluation outputs.

To open the local MLflow UI from the project root:

```powershell
python -m mlflow server --backend-store-uri "sqlite:///mlflow.db" --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 in your browser.

The tracking database and local tracking artifacts are generated locally and should not be committed to the public repository.

## Run the FastAPI Service

Start the local API:

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Available endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /` | Service information |
| `GET /health` | Health and model-loaded status |
| `POST /predict` | Generate a classification prediction |
| `GET /docs` | Interactive API documentation |

The prediction endpoint accepts JSON input containing the 13 model features and returns the prediction, label, predicted probability, model name, and educational-use disclaimer.

Example health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Open http://127.0.0.1:8000/docs to try the API interactively.

## Docker Deployment

Build the image:

```powershell
docker build -t heart-disease-api:1.0 .
```

Run the container:

```powershell
docker run --rm -p 8000:8000 heart-disease-api:1.0
```

Verify the API using `/health` or `/docs`.

## Kubernetes Deployment with Minikube

Start Minikube using Docker Desktop:

```powershell
minikube start --driver=docker
```

Load the existing image into Minikube:

```powershell
minikube image load heart-disease-api:1.0
```

Apply the deployment and Service manifests:

```powershell
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Check rollout and pod status:

```powershell
kubectl rollout status deployment/heart-disease-api
kubectl get deployments
kubectl get pods
kubectl get services
```

Forward a local port to the Kubernetes Service:

```powershell
kubectl port-forward service/heart-disease-api 8001:8000
```

Keep the port-forward terminal open and use http://127.0.0.1:8001/docs to access the Kubernetes-hosted API.

## Automated Testing and CI

Run unit and API tests:

```powershell
python -m pytest -q
```

Run lint checks:

```powershell
ruff check src tests
```

The project includes a GitHub Actions workflow at `.github/workflows/ci.yml`. The workflow is configured to perform linting, automated testing, data preparation, EDA, model training, and artifact uploads on pushes and pull requests targeting `main`.

Workflow runs can be inspected under the repository's **Actions** tab:

https://github.com/2025af05139-alt/heart-disease-mlops/actions

The initial GitHub Actions run completed successfully.

## Logging and Health Checks

The API records request information, including method, path, status code, and duration. The `/health` endpoint reports whether the service is running and the model is loaded.

Kubernetes readiness and liveness probes use this endpoint to check application health.

## Reproducibility

- The preprocessing and estimator are packaged together in the selected model artifact.
- Model selection is based on cross-validation rather than test-set performance alone.
- Random seeds are configured in the training pipeline.
- Dependencies are declared in `requirements.txt`.
- Training metrics, reports, and model artifacts are generated by the project scripts.
- GitHub Actions provides automated linting, testing and model-training checks.

## Disclaimer

This classifier is a learning project built using a small historical dataset. Its output is not clinically validated and must not be used to diagnose heart disease or guide treatment.
