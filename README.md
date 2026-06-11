# Improving ML-Based IPC Prediction with Ratio Features

ECE 228 final project reproducibility package.

## Overview

This project reproduces the Random Forest IPC prediction workflow from
``Accelerating Computer Architecture Simulation through Machine Learning'' and
extends it with ratio-based feature engineering. The package evaluates four
model scripts on the same prepared train/test split:

- Reproduced Random Forest baseline
- Ratio-feature Random Forest
- XGBoost
- Ratio-feature MLP

## Data

The prepared input files are stored at the repository root:

- `training_data.csv`
- `testing_data.csv`

All model scripts read these two files by default.

## Setup

```bash
pip install -r requirements.txt
```

## Reproducing Results

Run each command from the repository root:

```bash
python models/reproduce_random_forest.py
python models/ratio_features_random_forest.py
python models/xgboost_regression.py
python models/mlp_regression.py
```

## Output Directories

Each model writes its generated metrics, predictions, and plots to a dedicated
directory:

- `paper_reproduction_results/`
- `ratio_feature_random_forest_results/`
- `xgboost_reproduction_results/`
- `mlp_results/`

## Current Results

| Model | RMSE | MAE | R2 |
|---|---:|---:|---:|
| Reproduced Original Random Forest | 0.0703 | 0.0317 | 0.9158 |
| Normalized Random Forest | 0.0534 | 0.0265 | 0.9515 |
| Raw-Feature RF Control | 0.0671 | 0.0311 | 0.9235 |
| Ratio-Feature Random Forest | 0.0538 | 0.0269 | 0.9508 |
| Original XGBoost | 0.0737 | 0.0383 | 0.9075 |
| Ratio-Feature XGBoost | 0.0564 | 0.0313 | 0.9458 |
| Ratio-Feature MLP | 0.0768 | 0.0403 | 0.8995 |

## Authors

Haoyue Zhang  
Zhiyuan Yao
