# Predicting Smartphone Addiction

This repository documents experiments for Kaggle's [Playground Series — Season 6, Episode 8](https://www.kaggle.com/competitions/playground-series-s6e8).

The task is binary classification: predict the probability of `addicted_label` from twelve smartphone-usage and lifestyle features. Submissions are ranked by ROC AUC.

## Data

The competition provides 691,369 training rows and 296,302 test rows. The features include screen time, social-media use, gaming, work or study time, sleep, notifications, app opens, weekend screen time, gender, stress level, and academic or work impact. Every feature contains missing values.

Competition files are not redistributed here. See [data/README.md](data/README.md) for the schema and aggregate quality checks, then download the files from Kaggle after accepting the competition rules.

## Validation

Experiments use stratified out-of-fold ROC AUC. Candidate submissions must preserve the official test IDs, contain one finite probability per row, and stay within `[0, 1]`.

The opening analysis found that:

- daily and weekend screen time and social-media use carry most of the standalone signal;
- missingness rates differ between train and test, even though missingness alone has little target signal;
- a time-allocation residual, `daily_screen_time - social_media - gaming - work_study`, captures useful joint structure;
- diverse out-of-fold model predictions are more useful near the public frontier than adding many highly correlated tree models.

## Experiments

Public results are recorded in [experiments/experiment_log.csv](experiments/experiment_log.csv). Scores are copied from the official Kaggle submissions table after evaluation.

Current best public ROC AUC: **0.97068** (`c03`).

## Reproducibility and credits

The ensemble work builds on publicly released S6E8 out-of-fold predictions and notebooks, especially the [public OOF library](https://www.kaggle.com/datasets/szymonkapiski/s6e8-oof-library-47-models), [Naji's aligned predictions](https://www.kaggle.com/datasets/najiama/predicting-smartphone-addiction-oof-submission-csv), and the public missingness-aware blend analyses by [Riponce](https://www.kaggle.com/code/riponce/1-public-lb-0-97068-honest-55-model-stack) and [Rayk Kretzschmar](https://www.kaggle.com/code/raykkretzschmar/s6e8-missingness-aware-55-model-blend). Each published experiment records its source predictions and transformation.
