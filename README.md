# MLOps Assignment 2 - Group 52

## 1. Project Overview

This project implements a complete end-to-end MLOps pipeline for a binary image classification use case: **Cats vs Dogs** for a pet adoption platform. The pipeline covers the full lifecycle of a machine learning system, including data versioning, model development, experiment tracking, containerization, CI/CD automation, deployment, and monitoring using open-source tools.

Dataset used: Kaggle Cats vs Dogs dataset.
All images were preprocessed into 224×224 RGB format and split into train, validation, and test sets.

---

## 2. System Architecture

The implemented pipeline follows this workflow:

Data Ingestion → Data Versioning → Preprocessing → Model Training → Experiment Tracking → Model Packaging → API Service → Containerization → CI Pipeline → Deployment → Monitoring

---

## 3. Tools and Technologies Used

* Version Control: Git
* Data Versioning: DVC
* Model Development: PyTorch
* Experiment Tracking: MLflow
* API Framework: FastAPI
* Containerization: Docker
* CI/CD: GitHub Actions
* Deployment: Docker Compose
* Testing: Pytest

---

# M1: Model Development and Experiment Tracking

## Objective

Build a baseline model, track experiments, and version all artifacts.

---

## 1. Data and Code Versioning

Git was used to manage source code, project structure, scripts, and configuration files.

DVC was used to track both:

* Raw dataset
* Preprocessed dataset

This ensures reproducibility and enables tracking dataset changes across experiments.

---

## 2. Model Building

A baseline Convolutional Neural Network (CNN) was implemented using PyTorch.

Steps performed:

* Images resized to 224×224 RGB format
* Dataset split into 80% training, 10% validation, and 10% testing
* Data loading implemented using PyTorch ImageFolder
* Model trained using cross-entropy loss and Adam optimizer

The trained model was saved as a serialized artifact:

```
cnn_model.pt
```

---

## 3. Experiment Tracking

MLflow was used to track experiments, including:

* Hyperparameters (batch size, epochs, learning rate)
* Training metrics (loss values per epoch)
* Model artifacts
* Experiment runs for reproducibility

MLflow UI was used to visualize experiment history.

---

# M2: Model Packaging and Containerization

## Objective

Package the trained model into a reproducible, containerized inference service.

---

## 1. Inference Service

A REST API was implemented using FastAPI.

Endpoints provided:

### Health Check

```
GET /health
```

Returns service status and monitoring information.

### Prediction Endpoint

```
POST /predict
```

Accepts an uploaded image and returns:

* Predicted class label
* Confidence score

---

## 2. Environment Specification

All dependencies required for training and inference were defined in:

```
requirements.txt
```

This ensures reproducibility across environments.

---

## 3. Containerization

A Dockerfile was created to package the inference service.

Steps performed:

* Base Python image used
* Dependencies installed via requirements.txt
* Model and API service included in container
* Service exposed on port 8000

The Docker image was successfully built and tested locally.

---

# M3: CI Pipeline for Build, Test and Image Creation

## Objective

Automate testing and Docker image creation using Continuous Integration.

---

## 1. Automated Testing

Unit tests were implemented using Pytest to validate:

* Processed dataset availability
* Trained model artifact existence

Smoke tests were also created to verify API health endpoint.

---

## 2. CI Setup

GitHub Actions was used to implement CI automation.

On every push to the main branch, the pipeline performs:

* Repository checkout
* Dependency installation
* Unit test execution
* Docker image build

---

## 3. Artifact Publishing

The CI pipeline successfully builds the Docker image automatically.
The workflow demonstrates readiness for publishing to container registries.

---

# M4: CD Pipeline and Deployment

## Objective

Deploy the containerized model service automatically.

---

## 1. Deployment Target

Docker Compose was used as the deployment target.

A deployment configuration file:

```
docker-compose.yml
```

defines:

* Service configuration
* Port mapping
* Automatic restart behavior

---

## 2. Continuous Deployment Flow

Deployment automation ensures that updated images can be rebuilt and redeployed seamlessly using Docker Compose.

---

## 3. Smoke Tests and Health Checks

A smoke test script verifies that the deployed service is reachable by:

* Calling the health endpoint
* Ensuring a valid response is returned

This ensures deployment reliability.

---

# M5: Monitoring, Logging and Post-Deployment Tracking

## Objective

Monitor the deployed system and track runtime metrics.

---

## 1. Basic Monitoring and Logging

The inference service includes:

* Request logging using Python logging module
* Tracking of API request count
* Measurement of prediction latency

Monitoring information is visible through the health endpoint.

---

## 2. Model Performance Tracking

Post-deployment, the system supports evaluation using:

* Real or simulated prediction requests
* Comparison of predicted labels with ground truth
* Performance monitoring through logs

---

# 4. How to Run the Project

## Install Dependencies

```
pip install -r requirements.txt
```

## Run the Service using Docker Compose

```
docker-compose up --build
```

## Access API

Health endpoint:

```
http://localhost:8000/health
```

Interactive API documentation:

```
http://localhost:8000/docs
```

---

# 5. Deliverables Included

* Complete source code
* DVC configuration files
* Trained model artifact
* Dockerfile and deployment configuration
* CI/CD pipeline configuration
* Monitoring-enabled inference service

---

# 6. Conclusion

This project demonstrates a complete industry-style MLOps workflow, covering all stages from data management and model training to deployment, automation, and monitoring. The pipeline ensures reproducibility, scalability, and maintainability of machine learning systems.
