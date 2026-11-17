# Olist MLOps – Late Delivery Prediction

An end-to-end Machine Learning and MLOps project that predicts whether an Olist order will be delivered late.

The project covers data validation, model development, experiment tracking, model registry, API deployment, containerization, and basic monitoring.

## Project Overview

The goal of this project is to predict late order deliveries using the Olist dataset and an XGBoost classification model.

The project implements an ML workflow that integrates data validation, model management, model serving, and CI/CD automation.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Machine Learning | XGBoost, Scikit-learn |
| Data Processing | Pandas |
| Database | PostgreSQL |
| Data Versioning | DVC |
| Experiment Tracking | MLflow |
| Model Registry | MLflow Model Registry |
| Data Validation | Great Expectations |
| API Development | FastAPI |
| Containerization | Docker, Docker Compose |
| Artifact Storage | SeaweedFS (S3-compatible storage) |
| CI/CD | GitHub Actions |
| Container Registry | GitHub Container Registry (GHCR) |

## Project Architecture

The project follows an end-to-end ML workflow:

1. Data versioning and management using DVC.
2. Data validation using Great Expectations.
3. Model development and evaluation using Python and XGBoost.
4. Experiment tracking and model registration using MLflow.
5. Model artifact storage using S3-compatible storage.
6. Model loading through MLflow Model Registry using the `champion` alias.
7. Prediction serving through FastAPI.
8. Containerization using Docker and Docker Compose.
9. Automated testing and Docker image publishing using GitHub Actions.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/model` | Model information |
| GET | `/monitoring` | Monitoring metrics |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |

### Example Prediction Request

```json
{
  "purchase_year": 2018,
  "purchase_month": 8,
  "purchase_dayofweek": 2,
  "approval_delay_hours": 2.5,
  "estimated_delivery_days": 20,
  "count_item": 1,
  "total_price": 100.0,
  "total_freight": 15.0,
  "count_sequential": 1,
  "sum_pyament_value": 115.0
}
```

### Example Prediction Response

```json
{
  "prediction": 0,
  "probability": 0.375,
  "model_version": "champion"
}
```

Prediction classes:
- `0`: On-time delivery
- `1`: Late delivery

## Monitoring and Logging

The API provides basic monitoring metrics, including:

- Successful predictions.
- Failed predictions.
- Total prediction requests.
- Error rate.
- Average prediction latency.
- Model version.

Prediction requests are also recorded in a JSONL log file containing timestamps, input data, prediction results or errors, latency, and model version.

Logs are persisted using a Docker volume mapping to the local `logs/` directory.

Note: Monitoring counters are maintained in memory and reset when the API container restarts.

## CI/CD Pipeline

GitHub Actions automates the following workflow:

1. Install project dependencies.
2. Run validation and incoming data validation tests.
3. Build the Docker image.
4. Publish the image to GitHub Container Registry (GHCR).

The Docker image is published with `latest` and commit SHA tags.

## Running the Project

### Prerequisites

- Python 3.12
- Docker Desktop
- Git
- A running MLflow tracking server
- A registered model named `OlistLatePrediction` with the `champion` alias
- Model artifacts accessible through the configured S3-compatible storage

### Environment Variables

Create a `.env` file in the project root:

```env
MINIO_ROOT_USER=your_minio_username
MINIO_ROOT_PASSWORD=your_minio_password
POSTGRES_PASSWORD=your_postgres_password
```

Use your actual credentials. Do not commit the `.env` file to GitHub.

### Start Services

Build and start the Docker Compose services:

```bash
docker compose up -d --build
```

The services include:

- FastAPI application
- PostgreSQL
- S3-compatible artifact storage

The MLflow tracking server must be running separately and accessible at the configured tracking URI.

### Access the API

Once the services are running:

- API: http://localhost:8001
- Swagger documentation: http://localhost:8001/docs
- Health check: http://localhost:8001/health
- Model information: http://localhost:8001/model
- Monitoring: http://localhost:8001/monitoring

## Testing

Run the validation tests using:

```bash
python -m pytest tests/test_validation.py tests/test_incoming_validation.py -q
```

## Repository

GitHub: https://github.com/suhaib511/Olist-MLOps

---

**Developed as part of my Machine Learning and MLOps learning journey.**