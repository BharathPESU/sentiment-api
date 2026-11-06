# Sentiment Analysis MLOps API

An end-to-end production-grade MLOps system for sentiment analysis on movie reviews. Built with **FastAPI**, **Hugging Face Transformers (DistilBERT)**, **DVC**, **MLflow**, **Docker**, **Prometheus**, and **Prefect**.

---

## Architecture Overview

```
                      +-------------------+
                      |   IMDB Dataset    |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      | src/prepare_data  | ----> data/train.csv, test.csv
                      +---------+---------+
                                |
               +----------------+---------------+
               |                                |
               v                                v
    +--------------------+            +-------------------+
    | src/train_baseline |            |    src/train.py   | (DistilBERT)
    | (TF-IDF + LogReg)  |            +---------+---------+
    +---------+----------+                      |
              |                                 v
              +--------> [ MLflow ] <----+  models/sentiment/
                                         |  metrics.json
                                         |
                                         v
                               +-------------------+
                               |    DVC Pipeline   | ----> Remote Cache
                               +-------------------+
                                         |
                                         v
+------------------+           +-------------------+           +------------------+
|   Prometheus     | <-------- |    FastAPI App    | <-------- |    End User      |
| (:9090/metrics)  |  (scrape) |   (:8000/predict) | (request) | (Movie Reviews)  |
+------------------+           +-------------------+           +------------------+
                                         ^
                                         |
                               +-------------------+
                               | Prefect Flow      |
                               | (flows/retrain)   | <---- Drift Check (KS Test)
                               +-------------------+
```

---

## Tech Stack

- **Model**: DistilBERT (`distilbert-base-uncased`) via Hugging Face Transformers & PyTorch
- **API**: FastAPI with Pydantic request/response validation & Uvicorn
- **Tracking & Versioning**: MLflow (experiment metrics/params), DVC (data & model weights)
- **Monitoring**: Prometheus (`prometheus-fastapi-instrumentator`), Two-sample Kolmogorov-Smirnov test for data drift
- **Orchestration**: Prefect 3.0 (retraining workflow with accuracy quality gate)
- **Containerization**: Docker (multi-stage CPU-optimized image with non-root security)
- **CI/CD**: GitHub Actions (automated test runner)

---

## Project Structure

```text
sentiment-api/
├── .github/workflows/ci.yml       # GitHub Actions automated test workflow
├── data/                          # Dataset directory (DVC tracked, Git ignored)
│   └── .gitkeep
├── docker/
│   └── Dockerfile                 # Production CPU-optimized Docker image
├── docker-compose.monitoring.yml  # Prometheus monitoring stack
├── dvc.yaml                       # DVC reproducible pipeline stages (prepare, train)
├── dvc.lock                       # DVC pipeline dependency lockfile
├── flows/
│   └── retrain_flow.py            # Prefect retraining flow with quality gate
├── models/                        # Model directory (DVC tracked, Git ignored)
│   ├── .gitkeep
│   └── sentiment/                 # Trained DistilBERT model weights and tokenizer
├── monitoring/
│   ├── drift_check.py             # KS-test data drift detector
│   └── prometheus.yml             # Prometheus scraping configuration
├── params.yaml                    # Hyperparameters & data split configurations
├── pytest.ini                     # Pytest configuration & markers
├── requirements.txt               # Python package dependencies
├── src/
│   ├── prepare_data.py            # IMDB download & reproducible split generator
│   ├── train_baseline.py          # TF-IDF + Logistic Regression baseline
│   ├── train.py                   # DistilBERT fine-tuning script
│   ├── app/
│   │   ├── main.py                # FastAPI endpoints (/health, /predict, /metrics)
│   │   └── schemas.py             # Pydantic schemas for request/response
│   └── monitoring/
│       └── drift_check.py         # Drift detection utility module
└── tests/
    ├── test_api.py                # API integration tests (marked with @pytest.mark.model)
    └── test_schemas.py            # Schema validation unit tests
```

---

## Getting Started

### 1. Prerequisites & Environment Setup

```bash
# Clone the repository
git clone https://github.com/BharathPESU/sentiment-api.git
cd sentiment-api

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (CPU PyTorch)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

### 2. Data & Model Pipeline with DVC

The project uses DVC to manage datasets and large model files.

```bash
# Reproduce the entire pipeline (data preparation -> training)
dvc repro

# View metrics across runs
dvc metrics show

# Push tracked data and model artifacts to remote storage
dvc push

# Pull data and model artifacts on another machine
dvc pull
```

---

### 3. Running the FastAPI Application

```bash
# Start the API server
source .venv/bin/activate
uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

#### API Endpoints

- **`GET /health`**: Health check and model readiness status.
  ```bash
  curl http://localhost:8000/health
  # {"status":"ok","model_loaded":true}
  ```
- **`POST /predict`**: Classify review sentiment (`positive` or `negative`).
  ```bash
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "A breathtaking cinematic masterpiece with brilliant acting!"}'
  # {"label":"positive","score":0.9852}
  ```
- **`GET /metrics`**: Prometheus metrics for request count and latency histograms.

---

### 4. Running Tests

```bash
source .venv/bin/activate

# Run all unit and model tests
pytest -q

# Run fast unit tests only (without loading the model weights)
pytest -q -m "not model"
```

---

### 5. Docker Deployment & Prometheus Monitoring

#### Run API with Docker:

```bash
# Build the Docker image
docker build -t sentiment-api:latest -f docker/Dockerfile .

# Run the container
docker run -d -p 8000:8000 --name sentiment-api sentiment-api:latest
```

#### Run Monitoring Stack (Prometheus):

```bash
docker compose -f docker-compose.monitoring.yml up -d
```
Access the Prometheus dashboard at **`http://localhost:9090`** to query metrics like `http_requests_total` and `http_request_duration_seconds`.

---

### 6. Drift Detection & Automated Retraining

#### Data Drift Check (Kolmogorov-Smirnov Test)

Checks if incoming review length distributions differ significantly ($p < 0.05$) from training data:

```bash
python monitoring/drift_check.py data/current.csv
```

#### Prefect Retraining Workflow

Runs the DVC pipeline, evaluates the resulting model, and enforces an accuracy quality gate ($\ge 0.85$):

```bash
python flows/retrain_flow.py
```
