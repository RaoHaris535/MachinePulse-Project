# Phase 1 submission evidence

This directory contains the evidence used in the MachinePulse Phase 1 report.
The screenshots were captured from the remote DagsHub MLflow server and from
the repository's live DVC pipeline on 2026-11-08.

The submission document is `MachinePulse-Phase1-Report.pdf`. It is generated
reproducibly by `scripts/build_phase1_report.py` and contains four pages.

## Links

- GitHub repository: https://github.com/RaoHaris535/MachinePulse-Project
- DagsHub repository: https://dagshub.com/RaoHaris535/MachinePulse-Project
- MLflow experiment: https://dagshub.com/RaoHaris535/MachinePulse-Project.mlflow/#/experiments/0
- Successful LSTM run: https://dagshub.com/RaoHaris535/MachinePulse-Project.mlflow/#/experiments/0/runs/6698f594cbea484c9f56f6d3584cfbd3
- Registered model version: https://dagshub.com/RaoHaris535/MachinePulse-Project.mlflow/#/models/MachinePulse-LSTM-Autoencoder/versions/1

## Evidence files

- `evidence/mlflow-experiment-runs.png`: remote experiment table with at least
  two completed runs.
- `evidence/mlflow-lstm-run.png`: successful main-model run with parameters,
  metrics, tags, and registered-model association.
- `evidence/mlflow-model-staging.png`: registered model version 1 in Staging.
- `evidence/dvc-dag.png`: actual output of `dvc dag`, rendered in a legible
  terminal frame by `scripts/render_dvc_dag.py`.

No access tokens, passwords, raw data, or model weights are included here.
