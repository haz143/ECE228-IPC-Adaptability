# Improving ML-based IPC Prediction with Ratio-Based Features

ECE228 Final Project

## Overview

This project reproduces the Random Forest IPC prediction framework proposed in:

"Accelerating Computer Architecture Simulation through Machine Learning"

and extends it with ratio-based feature engineering to investigate model adaptability.

## Contributions

* Reproduced the original Random Forest IPC prediction framework.
* Evaluated prediction performance using RMSE, MAE, and R².
* Introduced ratio-based workload-normalized features.
* Performed adaptability stress testing.
* Improved RMSE from 0.0666 to 0.0540.
* Improved R² from 0.9244 to 0.9504.

## Repository Structure

* reproduce_paper_random_forest.py
  Reproduces the original Random Forest model.

* stress_test_ratio_features.py
  Implements ratio-based feature engineering and adaptability evaluation.

* results/
  Generated figures and evaluation results.

* report/
  Final report and presentation slides.

## Main Results

| Model            | RMSE   | R²     |
| ---------------- | ------ | ------ |
| Raw Feature RF   | 0.0666 | 0.9244 |
| Ratio Feature RF | 0.0540 | 0.9504 |

## Authors

Haoyue Zhang
Zhiyuan Yao

UC San Diego ECE228 Final Project



