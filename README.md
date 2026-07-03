<div align="center">

# Disaster Tweet Predictor

### End-to-end NLP classification project with BiLSTM training, ONNX inference, FastAPI, Jinja, SQLite, Docker, and Render deployment

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Inference%20API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-Production%20Inference-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-BiLSTM%20Training-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jinja](https://img.shields.io/badge/Jinja-Server--Rendered%20UI-B41717?logo=jinja&logoColor=white)](https://jinja.palletsprojects.com/)
[![SQLite](https://img.shields.io/badge/SQLite-Prediction%20Logs-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Render](https://img.shields.io/badge/Render-Live%20Deployment-46E3B7?logo=render&logoColor=black)](https://render.com/)
[![Kaggle](https://img.shields.io/badge/Kaggle-NLP%20Dataset-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/nlp-getting-started)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

**Live Application**

[![Live Web App](https://img.shields.io/badge/Live%20Web%20App-Open%20on%20Render-46E3B7?logo=render&logoColor=black)](https://disaster-tweet-app.onrender.com/)
[![Classifier](https://img.shields.io/badge/Classifier-Try%20the%20App-38BDF8?logo=fastapi&logoColor=white)](https://disaster-tweet-app.onrender.com/app)
[![Health Check](https://img.shields.io/badge/API-Health%20Check-34D399?logo=checkmarx&logoColor=white)](https://disaster-tweet-app.onrender.com/health)
[![API Documentation](https://img.shields.io/badge/FastAPI-OpenAPI%20Docs-009688?logo=swagger&logoColor=white)](https://disaster-tweet-app.onrender.com/docs)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Source%20Code-181717?logo=github&logoColor=white)](https://github.com/a1mohamad/disaster-tweet-app)

**Research and Data**

[![Training Overview](https://img.shields.io/badge/Live%20App-Training%20Overview-EE4C2C?logo=pytorch&logoColor=white)](https://disaster-tweet-app.onrender.com/training)
[![Deployment Overview](https://img.shields.io/badge/Live%20App-Deployment%20Overview-2496ED?logo=docker&logoColor=white)](https://disaster-tweet-app.onrender.com/deployment)
[![Research Lab](https://img.shields.io/badge/Research%20Lab-Project%20Page-222222?logo=githubpages&logoColor=white)](https://a1mohamad.github.io/research/lung-disease-detection/index.html)
[![Kaggle Competition](https://img.shields.io/badge/Kaggle-NLP%20Getting%20Started-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/nlp-getting-started)
[![GloVe](https://img.shields.io/badge/Embeddings-GloVe%206B-8A2BE2)](https://nlp.stanford.edu/projects/glove/)

**Contact and Profiles**

[![Gmail](https://img.shields.io/badge/Gmail-a1mohamad.askari%40gmail.com-EA4335?logo=gmail&logoColor=white)](mailto:a1mohamad.askari@gmail.com)
[![iCloud](https://img.shields.io/badge/iCloud-amirmohmdaskari%40icloud.com-3693F3?logo=icloud&logoColor=white)](mailto:amirmohmdaskari@icloud.com)
[![Phone](https://img.shields.io/badge/Phone-%2B98%20901%20222%203122-25D366?logo=whatsapp&logoColor=white)](tel:+989012223122)
[![Website](https://img.shields.io/badge/Website-a1mohamad.github.io-4285F4?logo=googlechrome&logoColor=white)](https://a1mohamad.github.io)
[![GitHub](https://img.shields.io/badge/GitHub-a1mohamad-181717?logo=github&logoColor=white)](https://github.com/a1mohamad)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Amir%20Mohammad%20Askari-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/amirmohammad-askari/)
[![Kaggle](https://img.shields.io/badge/Kaggle-amirmohamadaskari-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/amirmohamadaskari)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Live System](#live-system)
- [Why This Project Matters](#why-this-project-matters)
- [System Capabilities](#system-capabilities)
- [Dataset and Results](#dataset-and-results)
- [Model Pipeline](#model-pipeline)
- [Architecture](#architecture)
- [Application Modules](#application-modules)
- [Repository Structure](#repository-structure)
- [API Reference](#api-reference)
- [Prediction Response](#prediction-response)
- [Input Validation](#input-validation)
- [Runtime Configuration](#runtime-configuration)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [ONNX Export](#onnx-export)
- [Research Workflow](#research-workflow)
- [Data and Model Artifacts](#data-and-model-artifacts)
- [Demo Persistence](#demo-persistence)
- [Responsible AI](#responsible-ai)
- [Portfolio Talking Points](#portfolio-talking-points)
- [Current Limitations](#current-limitations)
- [License](#license)

---

## Overview

**Disaster Tweet Predictor** is an end-to-end natural language processing project that classifies a tweet as either **disaster-related** or **not disaster-related**.

The repository covers the complete path from exploratory analysis and model training to optimized inference and public deployment:

- Exploratory data analysis in Jupyter notebooks
- Shared tweet preprocessing rules
- Vocabulary construction from training data
- GloVe-initialized bidirectional LSTM training
- Validation metrics and decision-threshold calibration
- PyTorch model artifact export
- ONNX conversion and numerical parity validation
- FastAPI JSON API
- Jinja-rendered web application
- SQLite prediction logging
- Full and lightweight Docker images
- Public deployment on Render

The production-oriented runtime prefers **ONNX Runtime** for lean CPU inference. A separate full image retains **PyTorch** support for local development, model export, and fallback inference.

---

## Live System

| Layer | Live Destination | Role |
|---|---|---|
| Public entry point | [Render web app](https://disaster-tweet-app.onrender.com/) | Project overview and navigation |
| Interactive classifier | [Prediction interface](https://disaster-tweet-app.onrender.com/app) | Submit tweets and review probabilities |
| Training presentation | [Training overview](https://disaster-tweet-app.onrender.com/training) | Dataset, architecture, and training results |
| Deployment presentation | [Deployment overview](https://disaster-tweet-app.onrender.com/deployment) | Runtime and container architecture |
| Health endpoint | [Service health](https://disaster-tweet-app.onrender.com/health) | Runtime availability and selected backend |
| OpenAPI interface | [FastAPI docs](https://disaster-tweet-app.onrender.com/docs) | Interactive REST API documentation |
| Source code | [GitHub repository](https://github.com/a1mohamad/disaster-tweet-app) | Research, deployment, and model-serving code |
| Dataset | [Kaggle competition](https://www.kaggle.com/competitions/nlp-getting-started) | Original train and test tweet data |

Production request flow:

```text
Browser or API client
        |
        v
Render Docker service
        |
        v
FastAPI request validation
        |
        v
Tweet cleaning + keyword merge
        |
        v
Vocabulary encoding + padding
        |
        v
ONNX Runtime BiLSTM inference
        |
        +----> probability + label + warnings
        |
        +----> SQLite demo prediction log
```

Render free-tier services may sleep after inactivity. The first request after a quiet period can therefore take longer while the container starts.

---

## Why This Project Matters

This repository demonstrates the full lifecycle of a compact applied machine learning system rather than stopping at a notebook:

- Research and deployment code are separated.
- The vocabulary is built only from training data to avoid validation leakage.
- Preprocessing behavior is mirrored between training and inference.
- Model architecture settings are explicit and reproducible.
- Validation tracks accuracy, precision, recall, and F1.
- The final classification threshold is calibrated instead of assuming `0.5`.
- ONNX output is compared against PyTorch output before deployment.
- API errors and warnings use structured response contracts.
- Predictions can be reviewed through both a REST API and a web interface.
- Docker images support different local and cloud runtime needs.

For a portfolio project, the central value is the combination of **NLP research, model-serving engineering, interface design, persistence, and cloud deployment**.

---

## System Capabilities

### Classification

- Accepts a required tweet and an optional keyword.
- Normalizes URLs, mentions, repeated punctuation, numbers, whitespace, and non-ASCII text.
- Combines keyword context with tweet text when enabled.
- Maps tokens into a fixed 10,000-token vocabulary.
- Pads or truncates sequences to 200 tokens.
- Returns a disaster probability, binary label, human-readable label, threshold, and backend.

### Inference Backends

- Uses ONNX Runtime as the preferred production backend.
- Supports direct PyTorch inference in the full local image.
- Supports automatic fallback from ONNX to PyTorch when enabled.
- Loads the calibrated threshold from a JSON or PyTorch artifact.
- Loads human-readable labels from `label_mapping.json`.

### API and Frontend

- FastAPI application with typed Pydantic request and response schemas.
- Jinja templates for project, prediction, training, and deployment pages.
- Responsive CSS interface for desktop and mobile viewports.
- Interactive OpenAPI documentation generated by FastAPI.
- Health endpoint exposing the active inference backend.

### Validation and Persistence

- Rejects empty input after normalization.
- Warns when input is very short or exceeds the model sequence limit.
- Detects likely non-English text and reports structured warnings or errors.
- Writes predictions, probabilities, labels, backend details, and warnings to SQLite.
- Exposes recent prediction records through a bounded logs endpoint.

---

## Dataset and Results

The project uses the Kaggle **Natural Language Processing with Disaster Tweets** dataset.

| Dataset | Rows | Columns |
|---|---:|---|
| Training set | 7,613 | `id`, `keyword`, `location`, `text`, `target` |
| Test set | 3,263 | `id`, `keyword`, `location`, `text` |

Training-label distribution:

| Label | Meaning | Rows |
|---:|---|---:|
| `0` | Not disaster | 4,342 |
| `1` | Disaster | 3,271 |

Recorded validation results:

| Metric | Result |
|---|---:|
| Best validation F1 | `0.7588` |
| Best validation accuracy | `0.7971` |
| Calibrated inference threshold | `0.5200` |
| Vocabulary size | `10,000` |
| Maximum sequence length | `200` |

The model is evaluated with F1, precision, and recall in addition to accuracy because disaster detection benefits from balancing missed emergency signals against false alarms.

---

## Model Pipeline

```text
Raw tweet + optional keyword
        |
        v
Text normalization
        |-- lowercase
        |-- URL -> <URL>
        |-- mention -> <USER>
        |-- integer -> <NUM>
        |-- repeated !/? normalization
        |-- whitespace cleanup
        v
Final text construction
        |
        v
Training-only vocabulary
        |
        v
Token IDs + padding/truncation
        |
        v
100-dimensional GloVe embedding
        |
        v
Two-layer bidirectional LSTM
        |
        v
Dropout + linear binary head
        |
        v
Sigmoid probability
        |
        v
Calibrated threshold
        |
        +-- not_disaster
        +-- disaster
```

### Model Configuration

| Setting | Value |
|---|---:|
| Architecture | Bidirectional LSTM |
| Embedding dimension | `100` |
| Hidden dimension | `64` |
| LSTM layers | `2` |
| Dropout | `0.28` |
| Output dimension | `1` |
| Batch size | `128` |
| Learning rate | `3e-4` |
| Maximum epochs | `50` |
| Training split | `80%` |
| Early-stopping patience | `5` |

---

## Architecture

```text
                         +----------------------+
                         | Browser / API Client |
                         +----------+-----------+
                                    |
                                    v
                         +----------+-----------+
                         | FastAPI Application  |
                         | JSON API + Jinja UI  |
                         +----------+-----------+
                                    |
                     +--------------+--------------+
                     |                             |
                     v                             v
          +----------+-----------+      +----------+-----------+
          | Input Validator      |      | HTML Templates       |
          | language + warnings  |      | CSS web interface    |
          +----------+-----------+      +----------------------+
                     |
                     v
          +----------+-----------+
          | Shared Preprocessing |
          | clean + encode + pad |
          +----------+-----------+
                     |
                     v
          +----------+-----------+
          | Predictor            |
          | ONNX first / Torch   |
          +----------+-----------+
                     |
          +----------+-----------+
          |                      |
          v                      v
+---------+----------+   +-------+----------+
| API/UI Response    |   | SQLite Log      |
| probability/label |   | request/result  |
+--------------------+   +------------------+
```

---

## Application Modules

| Module | Responsibility |
|---|---|
| `deployment/app/api.py` | FastAPI lifecycle, routes, templates, prediction orchestration, and exception responses |
| `deployment/app/app_config.py` | Environment-driven paths, model settings, preprocessing, and backend configuration |
| `deployment/app/data/preprocessing.py` | Vocabulary loading, text cleaning, language detection, encoding, and padding |
| `deployment/app/model/disaster_model.py` | PyTorch BiLSTM architecture and checkpoint loading |
| `deployment/app/model/predictor.py` | ONNX/PyTorch backends, threshold loading, label mapping, and prediction |
| `deployment/app/db.py` | SQLite connection, schema migration, inserts, and recent-log queries |
| `deployment/app/schemas.py` | Pydantic request, prediction, and log response contracts |
| `deployment/app/utils/validation.py` | Input rules and structured warnings |
| `deployment/app/utils/errors.py` | Serializable application error hierarchy |
| `deployment/app/templates` | Jinja pages for the app, training, deployment, and project overview |
| `deployment/app/static/styles.css` | Responsive visual design for the web application |
| `deployment/scripts/export_onnx.py` | PyTorch-to-ONNX export and numerical parity validation |
| `research/data_utils` | Training preprocessing, vocabulary construction, dataset encoding, and decoding |
| `research/model` | Training model and GloVe embedding loader |
| `research/training` | Training engine, metrics, class weighting, history, and early stopping |
| `research/notebooks` | EDA, model development, calibration, and saved notebook outputs |

---

## Repository Structure

```text
disaster-tweet-app/
|-- README.md
|-- LICENSE
|-- research/
|   |-- train.py
|   |-- configs/
|   |   +-- train_config.py
|   |-- data/
|   |   |-- train.csv
|   |   |-- test.csv
|   |   +-- sample_submission.csv
|   |-- data_utils/
|   |-- embeddings/
|   |-- model/
|   |-- training/
|   |-- utils/
|   |-- notebooks/
|   +-- outputs/
|-- deployment/
|   |-- app/
|   |   |-- data/
|   |   |-- model/
|   |   |-- static/
|   |   |-- templates/
|   |   +-- utils/
|   |-- artifacts/
|   |-- data/
|   |-- scripts/
|   |-- main.py
|   |-- Dockerfile
|   |-- Dockerfile.runtime
|   |-- docker-compose.yml
|   |-- docker-compose.runtime.yml
|   |-- requirements.txt
|   |-- requirements-runtime.txt
|   +-- requirements-torch.txt
```

---

## API Reference

### Health

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "backend": "onnx"
}
```

### Predict

```http
POST /predict
Content-Type: application/json
```

Request:

```json
{
  "tweet": "Forest fire near homes, residents ordered to evacuate.",
  "keyword": "wildfire"
}
```

PowerShell example:

```powershell
$body = @{
    tweet = "Forest fire near homes, residents ordered to evacuate."
    keyword = "wildfire"
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "http://localhost:8000/predict" `
    -ContentType "application/json" `
    -Body $body
```

### Prediction Logs

```http
GET /logs?limit=50
```

The requested limit is bounded between `1` and `200`.

### Web Routes

| Route | Purpose |
|---|---|
| `/` | Project landing page |
| `/app` | Interactive classifier |
| `/training` | Training and model overview |
| `/deployment` | Runtime and deployment overview |
| `/docs` | Swagger/OpenAPI interface |
| `/redoc` | ReDoc API reference |

---

## Prediction Response

```json
{
  "probability": 0.9342,
  "label": 1,
  "label_name": "disaster",
  "threshold": 0.5199999809265137,
  "backend": "onnx",
  "warnings": []
}
```

Response fields:

| Field | Meaning |
|---|---|
| `probability` | Sigmoid probability assigned to the disaster class |
| `label` | Binary decision based on the calibrated threshold |
| `label_name` | Human-readable label from `label_mapping.json` |
| `threshold` | Decision threshold loaded by the runtime |
| `backend` | Active inference backend: `onnx` or `torch` |
| `warnings` | Input-quality or preprocessing notices |

---

## Input Validation

The runtime performs validation before model inference.

| Condition | Result |
|---|---|
| Empty text after cleaning | `EMPTY_TWEET` error |
| Fewer than three tokens | `SHORT_INPUT` warning |
| More than 200 tokens | `TRIMMED_LENGTH` warning |
| Language cannot be detected | `LANGUAGE_UNDETECTED` warning |
| Uncertain non-English detection | `LANGUAGE_SUSPECT` warning |
| Confident non-English detection on sufficient text | `NON_ENGLISH` error |

Structured error example:

```json
{
  "error_type": "input_error",
  "error_code": "NON_ENGLISH",
  "message": "Detected non-English input. Model is trained on English.",
  "details": {
    "lang": "es",
    "prob": 0.99
  }
}
```

---

## Runtime Configuration

| Variable | Default | Description |
|---|---|---|
| `PORT` | `10000` in runtime image | Port injected by cloud platforms such as Render |
| `APP_PORT` | `8000` | Local Docker Compose host port |
| `DB_PATH` | `deployment/data/predictions.db` | SQLite prediction-log path |
| `ARTIFACTS_DIR` | `deployment/artifacts` | Runtime model artifact directory |
| `ONNX_MODEL_PATH` | `best_model.onnx` | ONNX model path |
| `MODEL_PATH` | `best_model.pt` | PyTorch model path |
| `VOCAB_PATH` | `vocabs.json` | Vocabulary artifact path |
| `THRESHOLD_PATH` | `best_threshold.pt` | PyTorch threshold artifact |
| `THRESHOLD_JSON_PATH` | `best_threshold.json` | Runtime-friendly threshold artifact |
| `LABEL_MAPPING_PATH` | `label_mapping.json` | Human-readable class labels |
| `INFERENCE_BACKEND` | `auto` | Selects `auto`, `onnx`, or `torch` |
| `ALLOW_TORCH_FALLBACK` | `true` | Allows PyTorch fallback when ONNX cannot load |
| `THRESHOLD` | `0.5` | Final fallback threshold |
| `MAX_LENGTH` | `200` | Maximum token sequence length |
| `DEVICE` | `auto` | PyTorch device selection |

Example configuration files:

```text
deployment/.env.example
deployment/.env.runtime.example
```

---

## Local Development

From the deployment directory:

```powershell
cd deployment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the application:

```powershell
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/
http://localhost:8000/app
http://localhost:8000/docs
```

### Command-Line Prediction

From `deployment/`:

```powershell
python main.py `
  --keyword "wildfire" `
  --tweet "Forest fire near homes, residents ordered to evacuate." `
  --use-label
```

---

## Docker Deployment

The project provides two Docker paths.

### Full Local Image

The full image installs ONNX Runtime, PyTorch, and ONNX tooling:

```powershell
cd deployment
Copy-Item .env.example .env
docker compose up --build
```

This path supports:

- ONNX inference
- PyTorch fallback
- model artifact inspection
- ONNX export tooling
- persistent SQLite volume through Docker Compose

### Lightweight Runtime Image

The runtime image excludes PyTorch and copies only the artifacts required for ONNX inference:

```powershell
cd deployment
Copy-Item .env.runtime.example .env.runtime
docker compose -f docker-compose.runtime.yml up --build
```

This path is designed for:

- Render deployment
- smaller image size
- CPU-based ONNX inference
- cloud-provided `PORT`

---

## ONNX Export

The export script loads the trained PyTorch model, exports it with dynamic batch axes, validates the ONNX graph, and compares probabilities between PyTorch and ONNX Runtime.

From `deployment/`:

```powershell
python scripts/export_onnx.py
```

Optional arguments:

```powershell
python scripts/export_onnx.py `
  --output artifacts/best_model.onnx `
  --tolerance 0.0001
```

The export process also writes `best_threshold.json`, allowing the lightweight runtime image to load the calibrated threshold without installing PyTorch.

---

## Research Workflow

The repository contains two complementary research paths:

### Notebooks

| Notebook | Purpose |
|---|---|
| `disaster-twitts-eda.ipynb` | Label balance, missingness, keyword behavior, length analysis, word frequency, n-grams, hashtags, sentiment, and linguistic exploration |
| `disaster-twitts-model.ipynb` | Cleaning, vocabulary, embeddings, BiLSTM training, checkpointing, fine-tuning, evaluation, and threshold calibration |

### Modular Training Script

From `research/`:

```powershell
python train.py
```

Training stages:

```text
Load Kaggle training CSV
        |
        v
Preprocess text and keyword
        |
        v
Stratified train/validation split
        |
        v
Build training-only vocabulary
        |
        v
Load GloVe embedding matrix
        |
        v
Train BiLSTM with weighted BCE loss
        |
        v
Track validation metrics
        |
        v
Early stopping + best checkpoint
        |
        v
Save model, vocabulary, and history
```

---

## Data and Model Artifacts

Deployment artifacts:

```text
deployment/artifacts/
|-- best_model.onnx
|-- best_model.pt
|-- best_threshold.json
|-- best_threshold.pt
|-- label_mapping.json
+-- vocabs.json
```

Artifact responsibilities:

| Artifact | Purpose |
|---|---|
| `best_model.onnx` | Lightweight production inference |
| `best_model.pt` | PyTorch state dictionary for full runtime and export |
| `best_threshold.json` | Calibrated threshold for ONNX-only containers |
| `best_threshold.pt` | Original threshold generated by research |
| `label_mapping.json` | Maps `0` and `1` to display labels |
| `vocabs.json` | Token-to-index and index-to-token mappings |

Label contract:

```json
{
  "0": "not_disaster",
  "1": "disaster"
}
```

The deployed model, research output model, and frozen-embedding notebook model are synchronized copies of the same artifact. The deployment vocabulary is likewise synchronized with the research output.

---

## Demo Persistence

The application records predictions in SQLite:

```text
prediction_logs
|-- id
|-- created_at
|-- tweet
|-- keyword
|-- final_text
|-- probability
|-- label
|-- label_name
|-- threshold
|-- backend
+-- warnings_json
```

SQLite is appropriate here because this deployment is a compact public demonstration. In the Render runtime configuration, the database is stored under `/tmp`, which is ephemeral.

Consequences:

- Prediction logs may be removed when the service restarts or is redeployed.
- The database is not intended as durable production storage.
- A production extension should use managed PostgreSQL or another persistent database.
- The public `/logs` endpoint should be protected or disabled before storing sensitive input.

---

## Responsible AI

This application is a research, education, and portfolio demonstration. It is not an emergency-response authority or a verified news source.

Important considerations:

- A positive prediction does not prove that an emergency is real.
- A negative prediction does not prove that a tweet is safe or irrelevant.
- Users should verify critical information through official local authorities and trusted sources.
- The model was trained on an English-language Kaggle dataset and may not generalize to other languages, platforms, time periods, or writing styles.
- Sarcasm, metaphor, incomplete context, copied headlines, and unusual vocabulary can produce incorrect predictions.
- Publicly submitted text may be written to the demo SQLite log.

---

## Portfolio Talking Points

This project demonstrates:

- end-to-end NLP product development
- exploratory data analysis
- reusable preprocessing contracts
- vocabulary construction without validation leakage
- pretrained GloVe embeddings
- bidirectional LSTM modeling
- class-weighted binary classification
- F1-oriented model evaluation
- early stopping and checkpoint selection
- probability-threshold calibration
- PyTorch-to-ONNX conversion
- ONNX numerical parity validation
- FastAPI lifecycle and route design
- typed Pydantic API contracts
- structured errors and warnings
- Jinja server-rendered frontend development
- responsive CSS interface design
- SQLite prediction persistence
- environment-driven application configuration
- full and lightweight Docker images
- health checks and Docker Compose deployment
- public Render deployment
- clean research-to-production separation

---

## Current Limitations

- The model is trained on a relatively small competition dataset.
- Tokenization is whitespace-based rather than subword-based.
- The vocabulary is fixed at 10,000 tokens.
- Text is truncated after 200 tokens.
- Non-ASCII text is removed during preprocessing.
- Language detection can be unreliable for very short tweets.
- The model does not retrieve live news or verify whether a reported event occurred.
- SQLite logs on the Render demo are ephemeral.
- The public logs route is not authenticated.
- The repository does not currently include an automated test suite or CI workflow.
- Model artifacts are bundled into the Docker build rather than downloaded from a versioned model registry.

Potential production upgrades include a transformer-based model, authenticated log access, managed PostgreSQL, automated tests, CI/CD, monitoring, model-version metadata, and a persistent artifact registry.

---

## License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for details.
