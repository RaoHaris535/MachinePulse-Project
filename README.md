# MachinePulse

MachinePulse is a reproducible predictive-maintenance project built on the
[MetroPT-3](https://archive.ics.uci.edu/dataset/791/metropt%2B3%2B) industrial
sensor dataset. Phase 1 focuses on a local, end-to-end machine learning
pipeline with DVC data versioning and remote MLflow experiment tracking on
DagsHub.

The planned modeling workflow compares a Logistic Regression failure-window
baseline with a PyTorch LSTM Autoencoder for multivariate time-series anomaly
detection. Failure labels used for evaluation will be derived from the event
intervals published with the otherwise unlabeled dataset.

## Phase 1 scope

- Reproducible ingestion of MetroPT-3 from its official source
- Schema, missing-value, timestamp, and range validation
- Chronological preprocessing and train/validation/test splits
- Logistic Regression baseline and LSTM Autoencoder
- Evaluation using classification and anomaly-detection metrics
- Remote MLflow run tracking and model registration on DagsHub
- A named DVC pipeline reproducible with `dvc repro`

Docker, APIs, dashboards, cloud deployment, CI/CD, monitoring, and automated
retraining are intentionally deferred to later phases.

## Project status

The repository structure and Python environment are initialized. Data and
model pipeline stages will be added incrementally in subsequent milestones.

## Prerequisites

- Git
- [`uv`](https://docs.astral.sh/uv/)
- A DagsHub account connected to this GitHub repository

Python 3.11 is used for compatibility across PyTorch, MLflow, and DVC.

## Setup

```bash
git clone https://github.com/RaoHaris535/MachinePulse-Project.git MachinePulse_Project
cd MachinePulse_Project
uv python install 3.11
uv sync --group dev
source .venv/bin/activate
cp .env.example .env
```

Update `.env` with your DagsHub repository details. Never commit `.env` or an
access token.

Verify the environment:

```bash
python --version
python -c "import machinepulse; print(machinepulse.__version__)"
pytest
```

Pipeline commands will be documented as their corresponding stages are
implemented.

## Repository structure

```text
MachinePulse_Project/
├── configs/             # Versioned pipeline and model configuration
├── data/                # DVC-managed raw and processed data (not in Git)
├── models/              # Generated model files (not in Git)
├── reports/             # Generated metrics and figures
├── src/machinepulse/    # Application and ML pipeline source code
├── tests/               # Automated tests
├── .env.example         # Safe environment-variable template
├── .gitignore
├── .python-version
└── pyproject.toml       # Dependencies and development-tool configuration
```

## Reproducibility policy

- Raw data, processed data, model weights, credentials, and local MLflow files
  are excluded from Git.
- Random seeds and pipeline parameters live in versioned configuration files.
- Data and generated pipeline outputs will be tracked with DVC.
- Experiments and registered models will be stored on the remote DagsHub
  MLflow server.

## Team contributions

Every contributor should configure their own Git name and email and commit a
real, reviewable change. Commit history should reflect actual contributions;
credentials and generated datasets must never be committed.

## Dataset citation

Davari, N., Veloso, B., Ribeiro, R., and Gama, J. (2021). *MetroPT-3 Dataset*.
UCI Machine Learning Repository. https://doi.org/10.24432/C5VW3R
