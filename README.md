# Voltiq ⚡

**Intelligent Renewable Energy Grid Analytics Platform**

Voltiq is a production-grade, agentic AI platform that ingests smart-meter and weather data, forecasts grid demand using PyTorch time-series models, detects anomalies via deep learning, and lets grid operators query incidents in plain English through a RAG-powered conversational agent.

---

## Architecture Overview

```
Data Sources (ENTSO-E, Open-Meteo, Smart Meters)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                  Airflow Pipelines                   │
│  Ingest → Preprocess → Train → Evaluate → Deploy    │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│              Agentic AI Layer (LangGraph)            │
│  IngestAgent → ForecastAgent → AnomalyAgent → QAAgent│
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│               FastAPI REST Layer                     │
│  /forecast  /anomalies  /query  /health  /metrics   │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────┐  ┌──────────────┐  ┌───────────────┐
│  Qdrant       │  │   MLflow     │  │  Prometheus   │
│  Vector DB    │  │   Tracking   │  │  + Grafana    │
└───────────────┘  └──────────────┘  └───────────────┘
```

## Key Features

- **Multi-agent orchestration** via LangGraph — four specialised agents collaborate on every operator query
- **PyTorch TFT (Temporal Fusion Transformer)** for 24h/48h grid demand forecasting, trainable on Colab T4
- **Deep anomaly detection** using LSTM Autoencoder on smart-meter time series
- **RAG pipeline** over ENTSO-E incident reports and grid event history (Qdrant + LangChain)
- **LLM evaluation** with DeepEval — faithfulness, answer relevancy, hallucination scoring
- **MLflow** for experiment tracking, model registry, and artifact storage
- **Airflow** DAGs for scheduled retraining and data ingestion
- **Kubernetes-ready** — Helm charts for AKS (Azure) and EKS (AWS) deployment
- **CPU-first design** — all inference and RAG runs on CPU; training notebooks target Colab T4

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep learning | PyTorch, PyTorch Lightning |
| Agent orchestration | LangGraph, LangChain |
| LLM | Claude (via Anthropic API) / local Mistral |
| Vector database | Qdrant |
| API | FastAPI, Pydantic v2, Uvicorn |
| Experiment tracking | MLflow |
| Pipeline orchestration | Apache Airflow |
| Containerisation | Docker, Docker Compose |
| Orchestration | Kubernetes, Helm |
| Cloud | Azure AKS + Blob / AWS EKS + S3 |
| Monitoring | Prometheus, Grafana |
| LLM evaluation | DeepEval, RAGAS |
| Testing | Pytest, Testcontainers |

---

## Quick Start

```bash
# Clone
git clone https://github.com/yourname/voltiq.git && cd voltiq

# Environment
cp .env.example .env        # fill in API keys
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Spin up backing services (Qdrant, MLflow, Airflow, Prometheus)
docker compose up -d

# Run the API
uvicorn api.main:app --reload

# Trigger a manual pipeline run
python scripts/run_pipeline.py --dag ingest_entso_e
```

## Project Structure

```
voltiq/
├── agents/             # LangGraph multi-agent graph definitions
├── api/                # FastAPI application — routers, middleware, schemas
├── core/               # Shared config, logging, constants
├── data/               # Raw, processed, and external dataset storage
├── forecasting/        # TFT model, training loop, evaluation metrics
├── anomaly/            # LSTM Autoencoder anomaly detection
├── rag/                # Document loaders, chunkers, Qdrant retriever
├── pipelines/          # Airflow DAGs and task definitions
├── infrastructure/     # Docker, Kubernetes manifests, Terraform
├── monitoring/         # Prometheus config, Grafana dashboards
├── mlflow_tracking/    # MLflow helpers, model registry utils
├── evaluation/         # DeepEval & RAGAS LLM evaluation suites
├── notebooks/          # Colab-ready training and EDA notebooks
├── tests/              # Unit, integration, e2e tests
├── scripts/            # CLI helpers
└── docs/               # Architecture docs, ADRs
```

## Data Sources (all free / open)

- **ENTSO-E Transparency Platform** — EU-wide electricity load, generation, and cross-border flows
- **Open-Meteo API** — historical + forecast weather (temperature, solar irradiance, wind)
- **Smart meter synthetic data** — generated via `scripts/generate_synthetic_meters.py`

## Roadmap

- [x] Project scaffold
- [ ] Data ingestion pipeline (ENTSO-E + Open-Meteo)
- [ ] TFT forecasting model + Colab training notebook
- [ ] LSTM Autoencoder anomaly detector
- [ ] RAG pipeline over incident history
- [ ] LangGraph agent graph
- [ ] FastAPI REST layer
- [ ] MLflow experiment tracking
- [ ] Airflow DAG orchestration
- [ ] LLM evaluation suite
- [ ] Kubernetes deployment manifests
- [ ] Grafana dashboards

---

Built as a portfolio project targeting **EU AI Engineer** roles. Covers the full MLOps lifecycle from raw data ingestion to production deployment.
