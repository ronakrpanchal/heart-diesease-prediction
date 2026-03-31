<div align="center">

# Heart Disease Prediction

Machine Learning workflow for heart disease risk classification using preprocessing, feature engineering, baseline model benchmarking, and advanced ensemble strategies.

</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-green)
![LightGBM](https://img.shields.io/badge/LightGBM-Enabled-brightgreen)
![CatBoost](https://img.shields.io/badge/CatBoost-Enabled-yellowgreen)
![GitHub stars](https://img.shields.io/github/stars/ronakrpanchal/heart-diesease-prediction)
![GitHub forks](https://img.shields.io/github/forks/ronakrpanchal/heart-diesease-prediction)
![GitHub issues](https://img.shields.io/github/issues/ronakrpanchal/heart-diesease-prediction)
![GitHub last commit](https://img.shields.io/github/last-commit/ronakrpanchal/heart-diesease-prediction)
![Made with Love](https://img.shields.io/badge/Made%20with-Love-ff69b4)
![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-purple)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-blue.svg)
![Status](https://img.shields.io/badge/Status-Research%20Workflow-informational)

</div>

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Data](#data)
- [Evaluation Metrics](#evaluation-metrics)
- [Modeling Pipeline](#modeling-pipeline)
- [Results](#results)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Notes](#notes)

## Overview

This repository contains an end-to-end heart disease prediction project that includes:

- Exploratory analysis and preprocessing notebooks
- Baseline model comparisons across classical and boosting models
- Hyperparameter tuning with cross-validation
- Ensemble experimentation including stacking and voting variants
- Exported evaluation outputs and LaTeX-ready result tables

The workflow is notebook-driven and designed for iterative experimentation.

## Data

- Kaggle Dataset: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- Keep the downloaded dataset at `data/heart.csv` to run notebooks without path changes.

## Evaluation Metrics

Model evaluation visuals are stored in the `figures/` directory.

### Methodology 

![Methodology](figures/methodology.png)

### Confusion Matrix

![Confusion Matrix](figures/confusion%20matrix.png)

### ROC-AUC Curve

![ROC AUC](figures/roc%20auc.png)


## How to Run

### 1) Clone repository

```bash
git clone <your-repo-url>
cd "Heart Diesease Prediction"
```

### 2) Create environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirement.txt
```

### 4) Launch notebooks

```bash
jupyter notebook
```

### 5) Suggested execution order

1. `notebooks/overview.ipynb`
2. `notebooks/preprocess.ipynb`
3. `notebooks/model_comparision.ipynb`
4. `notebooks/cv_hyperparameter_tuned/*`
5. `notebooks/Ensemble/*`

## Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn, Plotly
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Imbalanced-learn
- SHAP, LIME

## Notes

- Main outputs for reporting are in `outputs/`
- Utility plotting and feature engineering functions are in `helper.py`
- The project is experimentation-focused and notebook-first

## Activity
![Alt](https://repobeats.axiom.co/api/embed/e4cc1975e9fe4ffa6a9fc0d61616f21f73102a4d.svg "Repobeats analytics image")

## Contributors

<a href="https://github.com/ronakrpanchal/heart-diesease-prediction/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ronakrpanchal/heart-diesease-prediction"/>
</a>

## Support
if you find the project useful, please consider giving it a star ⭐ 💫

Thank you 🤩