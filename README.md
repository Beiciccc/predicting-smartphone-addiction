# Predicting Smartphone Addiction

This repository documents experiments for Kaggle's [Playground Series — Season 6, Episode 8](https://www.kaggle.com/competitions/playground-series-s6e8).

The task is binary classification: predict the probability of `addicted_label` from twelve smartphone-usage and lifestyle features. Submissions are ranked by ROC AUC.

## Data

The competition provides 691,369 training rows and 296,302 test rows. The features include screen time, social-media use, gaming, work or study time, sleep, notifications, app opens, weekend screen time, gender, stress level, and academic or work impact. Every feature contains missing values.

Competition files are not redistributed here. See [data/README.md](data/README.md) for the schema and aggregate quality checks, then download the files from Kaggle after accepting the competition rules.

## Validation

Experiments use stratified out-of-fold ROC AUC. Candidate submissions must preserve the official test IDs, contain one finite probability per row, and stay within `[0, 1]`.

Residual candidates now use two validation gates: they must first improve the fixed structural anchor, then beat the strongest comparable accepted candidate on the same partitions in both overall OOF AUC and fold-level head-to-head stability. The incumbent is refreshed from the official submissions list immediately before upload; a stronger newly accepted aligned OOF result invalidates an earlier pass, and candidates are re-certified serially. A candidate that clears only the anchor gate is retained as a diagnostic result rather than described as an improvement.

The opening analysis found that:

- daily and weekend screen time and social-media use carry most of the standalone signal;
- missingness rates differ between train and test, even though missingness alone has little target signal;
- a time-allocation residual, `daily_screen_time - social_media - gaming - work_study`, captures useful joint structure;
- diverse out-of-fold model predictions are more useful near the public frontier than adding many highly correlated tree models.

## Experiments

Public results are recorded in [experiments/experiment_log.csv](experiments/experiment_log.csv). Scores are copied from the official Kaggle submissions table after evaluation.

The competition-specific GPU model is published as [Fold-Safe Lattice Target Encoding](https://www.kaggle.com/code/beicicc/s6e8-fold-safe-lattice-target-encoding). Exact scored candidates are documented in [Frontier Blend Experiments](https://www.kaggle.com/code/beicicc/s6e8-frontier-blend-experiments), the [Seed Diversity Residual Audit](https://www.kaggle.com/code/beicicc/s6e8-seed-diversity-residual-audit), the [Strict RealMLP Residual Audit](https://www.kaggle.com/code/beicicc/s6e8-strict-realmlp-residual-audit), the [Strict Neural Residual Audit](https://www.kaggle.com/code/beicicc/s6e8-strict-neural-residual-audit), the [August 5 Frontier Provenance Audit](https://www.kaggle.com/code/beicicc/s6e8-aug-5-frontier-provenance-audit), and the [Strict Seed-Average Meta Audit](https://www.kaggle.com/code/beicicc/s6e8-realmlp-seed01-strict-meta-20260805).

The independent-seed residual improved all five fixed OOF folds but scored `0.97067`, slightly below the `0.97069` anchor. This suggests that diversity created by changing the outer partition did not transfer fully to the fold-averaged test prediction, so later residual tests keep the anchor's fixed partition.

The fixed-partition lattice-plus-RealMLP meta residual also improved all five OOF folds, reaching `0.969561426` versus the `0.969512388` C04 anchor. Its public score was `0.97069`, matching the anchor but not separating at the displayed leaderboard precision. A later source audit found that the lattice component selected its stopping round on each outer validation fold, so the result is retained as a diagnostic and subsequent strict candidates exclude that component.

The strict native-float64 RealMLP-plus-TabNet meta residual excludes the lattice component and uses fixed-epoch neural predictions on the same five partitions. It improved all five folds against C04 and reached `0.969552734`, but trailed C09 by `0.000008692` overall and on all five matched folds. Its public score of `0.97068` was one displayed leaderboard step below the `0.97069` project best, consistent with that head-to-head ordering. This result motivated the second validation gate above.

The August 4 public 74-member missingness-regime stack reached `0.969687` at the meta OOF level and improved the project public score to `0.97084`. Adding three factorization-machine views through a fixed one-third rank mixture improved all five aligned OOF folds to `0.969697`, but C12 tied C11 at the displayed public precision. A separate 25% rank residual from Naji's public version-14 artifact also improved all five aligned folds, reaching `0.969694841`, and C13 again scored `0.97084`. Source audits found outer-validation checkpoint selection in the lookup and factorization-machine members and incomplete model-selection provenance for Naji14, so these results remain public-provenance diagnostics rather than strict fixed-epoch evidence. The formulas, scored hashes, source links, and audit distinctions are preserved in the public notebook.

A locked Naji14 residual sweep selected 50% as the smallest weight that improved C12 overall while winning four of five aligned folds. C15 reached `0.969700304` OOF and scored `0.97084`, confirming a small complementary OOF signal without improving the displayed public score. It remains a diagnostic result because the underlying Naji14 recipe and full-OOF selection provenance are not published.

The two-seed fixed-epoch RealMLP average improved every individual model fold and produced a strict meta candidate at `0.969562914`. It beat C04, C10, and C09 on all five aligned folds, but scored `0.97068`. Because C11-C13 had become stronger accepted aligned results while this experiment was running, the C09-based pass was stale at upload time. C14 is retained as a reproducible negative result and motivated the live incumbent refresh rule above.

Current best public ROC AUC: **0.97084** (`c11`, `c12`, and `c13`).

## Reproducibility and credits

The ensemble work builds on publicly released S6E8 out-of-fold predictions and notebooks, especially the [public OOF library](https://www.kaggle.com/datasets/szymonkapiski/s6e8-oof-library-47-models), [Naji's aligned predictions](https://www.kaggle.com/datasets/najiama/predicting-smartphone-addiction-oof-submission-csv), and the public missingness-aware blend analyses by [Riponce](https://www.kaggle.com/code/riponce/1-public-lb-0-97068-honest-55-model-stack) and [Rayk Kretzschmar](https://www.kaggle.com/code/raykkretzschmar/s6e8-missingness-aware-55-model-blend). Each published experiment records its source predictions and transformation.
