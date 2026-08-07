# Predicting Smartphone Addiction

This repository documents experiments for Kaggle's [Playground Series — Season 6, Episode 8](https://www.kaggle.com/competitions/playground-series-s6e8).

The task is binary classification: predict the probability of `addicted_label` from twelve smartphone-usage and lifestyle features. Submissions are ranked by ROC AUC.

## Data

The competition provides 691,369 training rows and 296,302 test rows. The features include screen time, social-media use, gaming, work or study time, sleep, notifications, app opens, weekend screen time, gender, stress level, and academic or work impact. Every feature contains missing values.

Competition files are not redistributed here. See [data/README.md](data/README.md) for the schema and aggregate quality checks, then download the files from Kaggle after accepting the competition rules.

## Validation

Experiments use stratified out-of-fold ROC AUC. Candidate submissions must preserve the official test IDs, contain one finite probability per row, and stay within `[0, 1]`.

Residual candidates use two validation gates: they must first improve the fixed structural anchor, then beat the strongest comparable aligned OOF candidate on the same partitions in both overall ROC AUC and fold-level head-to-head stability. A candidate that clears only the anchor gate is retained as a diagnostic result rather than described as an improvement.

The opening analysis found that:

- daily and weekend screen time and social-media use carry most of the standalone signal;
- missingness rates differ between train and test, even though missingness alone has little target signal;
- a time-allocation residual, `daily_screen_time - social_media - gaming - work_study`, captures useful joint structure;
- diverse out-of-fold model predictions are more useful near the public frontier than adding many highly correlated tree models.

## Experiments

Public results are recorded in [experiments/experiment_log.csv](experiments/experiment_log.csv). Scores are copied from the official Kaggle submissions table after evaluation.

The competition-specific GPU model is published as [Fold-Safe Lattice Target Encoding](https://www.kaggle.com/code/beicicc/s6e8-fold-safe-lattice-target-encoding). Exact scored candidates are documented in [Frontier Blend Experiments](https://www.kaggle.com/code/beicicc/s6e8-frontier-blend-experiments), the [Seed Diversity Residual Audit](https://www.kaggle.com/code/beicicc/s6e8-seed-diversity-residual-audit), the [Strict RealMLP Residual Audit](https://www.kaggle.com/code/beicicc/s6e8-strict-realmlp-residual-audit), the [Strict Neural Residual Audit](https://www.kaggle.com/code/beicicc/s6e8-strict-neural-residual-audit), the [August 5 Frontier Provenance Audit](https://www.kaggle.com/code/beicicc/s6e8-aug-5-frontier-provenance-audit), the [August 6 Fixed-Schedule Lookup Audit](https://www.kaggle.com/code/beicicc/s6e8-aug-6-fixed-schedule-lookup-audit), and the [Strict Seed-Average Meta Audit](https://www.kaggle.com/code/beicicc/s6e8-realmlp-seed01-strict-meta-20260805).

The independent-seed residual improved all five fixed OOF folds but scored `0.97067`, slightly below the `0.97069` anchor. This suggests that diversity created by changing the outer partition did not transfer fully to the fold-averaged test prediction, so later residual tests keep the anchor's fixed partition.

The fixed-partition lattice-plus-RealMLP meta residual also improved all five OOF folds, reaching `0.969561426` versus the `0.969512388` C04 anchor. Its public score was `0.97069`, matching the anchor but not separating at the displayed leaderboard precision. A later source audit found that the lattice component selected its stopping round on each outer validation fold, so the result is retained as a diagnostic and subsequent strict candidates exclude that component.

The strict native-float64 RealMLP-plus-TabNet meta residual excludes the lattice component and uses fixed-epoch neural predictions on the same five partitions. It improved all five folds against C04 and reached `0.969552734`, but trailed C09 by `0.000008692` overall and on all five matched folds. Its public score of `0.97068` was one displayed leaderboard step below the `0.97069` project best, consistent with that head-to-head ordering. This result motivated the second validation gate above.

The August 4 public 74-member missingness-regime stack reached `0.969687` at the meta OOF level and improved the project public score to `0.97084`. Adding three factorization-machine views through a fixed one-third rank mixture improved all five aligned OOF folds to `0.969697`, but C12 tied C11 at the displayed public precision. A separate 25% rank residual from Naji's public version-14 artifact also improved all five aligned folds, reaching `0.969694841`, and C13 again scored `0.97084`. Source audits found outer-validation checkpoint selection in the lookup and factorization-machine members and incomplete model-selection provenance for Naji14, so these results remain public-provenance diagnostics rather than strict fixed-epoch evidence. The formulas, scored hashes, source links, and audit distinctions are preserved in the public notebook.

A locked Naji14 residual sweep selected 50% as the smallest weight that improved C12 overall while winning four of five aligned folds. C15 reached `0.969700304` OOF and scored `0.97084`, confirming a small complementary OOF signal without improving the displayed public score. It remains a diagnostic result because the underlying Naji14 recipe and full-OOF selection provenance are not published.

The next locked 65% Naji14 weight improved C15 on all five aligned OOF folds to `0.969702443`, but C16 scored `0.97083`. Because the remaining OOF gain was small and test transfer moved in the opposite direction, further escalation on this single-source residual axis was stopped in favor of orthogonal candidates.

Replacing the C11 side with C12's factorization-machine views produced a more stable orthogonal blend. C17 used 65% C12 and 35% Naji14, reached `0.969704758` OOF with four of five fold gains versus C16, and restored the public score to `0.97084`.

The first predeclared strict C09 residual point replaced 2.5% of C12 in C17. C18 improved all five aligned folds to `0.969706359` OOF and again scored `0.97084`; larger points in that residual sequence were not inspected after the first point passed.

A CC0 ordered-CatBoost artifact was then used as a small negative corrector. C19 reranked C18 after subtracting 1% of Golem Strategy D, reached `0.969708634` OOF, and improved the public score to `0.97085`. Because several corrector weights had already been inspected before the 1% artifact was frozen, this result is explicitly recorded as a source-informed post-hoc diagnostic; the source arrays also omit IDs, so their documented original row order was audited through exact lengths and the reproduced solo OOF AUC.

One additional locked point increased the strict C09 allocation from 2.5% to 5% while preserving the frozen Golem correction. C20 improved all five aligned folds over C19 to `0.969710125` OOF and tied the project-best public score of `0.97085`.

An independent fixed-schedule Lookup Transformer audit then completed with 24 epochs per fold, final EMA checkpoints, and no outer-fold checkpoint selection or early stopping. The standalone model reached `0.966051350` OOF. C21 combines 99% of globally ranked C19 with a 1% Lookup rank residual, reaches `0.969713851` aligned OOF, improves all five folds over C20, and scores `0.97085`. Increasing the total Lookup weight to 2.5% gives C22 an aligned OOF score of `0.969720033`, five fold wins over C21, and a public score of `0.97086`. C23 raises the total Lookup allocation to 5%, reaches `0.969725829` aligned OOF with another five fold wins, and improves the project-best public score to `0.97087`. The architecture and epoch count are source-informed, and the target-free preprocessing uses combined train and test covariates. Prediction arrays, fold assignments, metrics, and hashes are published in the [fixed-schedule Lookup artifacts](https://www.kaggle.com/datasets/beicicc/s6e8-fixed-schedule-lookup-transformer-artifacts).

An independently trained exact-value CatBoost model uses the 12 official predictors plus nine round-trip float64 keys, exactly 4,000 GPU boosting iterations per fold, and no early stopping or outer-fold checkpoint selection. Its standalone OOF score is `0.967297977`. C24 adds a 1% ranked residual from this model to C23, reaches `0.969729536` aligned OOF with five fold wins, and scores `0.97087` in official submission `55324592`. Prediction arrays, fold assignments, metrics, and the fixed training contract are published in the [exact-value CatBoost artifacts](https://www.kaggle.com/datasets/beicicc/s6e8-fixed-schedule-exact-value-catboost-artifacts), while the scored reconstruction is recorded in version 6 of the public residual audit.

A second fixed-schedule Lookup initialization changes only the model and training random seeds while preserving the five partitions, architecture, fixed 24-epoch schedule, and final-EMA checkpoint rule. The second member reaches `0.966057605` standalone OOF. C25 adds a 1% residual from the equal-rank average of the two Lookup initializations to C24, reaches `0.969732452` aligned OOF with five fold wins, and scores `0.97087` in official submission `55324880`. The [second-seed Lookup artifacts](https://www.kaggle.com/datasets/beicicc/s6e8-second-seed-fixed-schedule-lookup-artifacts) publish its prediction arrays, fold assignments, metrics, and fixed training contract; version 7 of the public residual audit reproduces the exact scored C25 file.

A controlled LightGBM feature ablation then trains raw-12 and enhanced-103 models on the same folds, seed, and fixed 900-round schedule. The enhanced model adds `other_screen` plus rounding and decimal-identity features, improving standalone OOF from `0.963393817` to `0.965745977` with gains on all five folds. C26 adds coefficient `0.012` of their normalized-rank contrast to C25, reaches `0.969734839` aligned OOF with five fold wins, and improves the public score to `0.97088` in official submission `55325064`. The [identity and digit LightGBM artifacts](https://www.kaggle.com/datasets/beicicc/s6e8-fixed900-identity-digit-lightgbm-artifacts) publish both prediction pairs, fold assignments, metrics, and the training contract; version 8 of the public residual audit reproduces the exact scored file.

Revisiting the independently trained exact-value CatBoost member on the new C26 anchor, a fresh locked incremental sequence selects its first point at 0.75%. C27 reaches `0.969737414` aligned OOF, improves all five folds as well as the even- and odd-ID slices, and scores `0.97088` in official submission `55325352`. Version 9 of the public residual audit reconstructs the exact scored file from the previously published CatBoost artifacts.

A strict RealMLP member averages two initializations trained for exactly four epochs per outer fold, using inner-fold target encoding and no outer-validation checkpoint selection. Its standalone OOF score is `0.968258398`. C28 adds a 3.2% ranked residual to C27, reaches `0.969741014` aligned OOF with five fold wins, but scores `0.97087` in official submission `55325553`, one displayed step below the project best. This is retained as a reproducible negative transfer result. The [fixed-4 two-seed RealMLP artifacts](https://www.kaggle.com/datasets/beicicc/s6e8-fixed4-realmlp-two-seed-artifacts) publish predictions, folds, metrics, and the training contract; version 10 of the public residual audit reproduces the exact scored file.

The two-seed fixed-epoch RealMLP average improved every individual model fold and produced a strict meta candidate at `0.969562914`. It beat C04, C10, and C09 on all five aligned folds, but scored `0.97068` and trailed the stronger C11-C13 aligned results. C14 is retained as a reproducible negative result.

Current best public ROC AUC: **0.97088** (`c27`, tied with `c26` at displayed precision; `c28` scores `0.97087`).

## Reproducibility and credits

The ensemble work builds on publicly released S6E8 out-of-fold predictions and notebooks, especially the [public OOF library](https://www.kaggle.com/datasets/szymonkapiski/s6e8-oof-library-47-models), [Naji's aligned predictions](https://www.kaggle.com/datasets/najiama/predicting-smartphone-addiction-oof-submission-csv), and the public missingness-aware blend analyses by [Riponce](https://www.kaggle.com/code/riponce/1-public-lb-0-97068-honest-55-model-stack) and [Rayk Kretzschmar](https://www.kaggle.com/code/raykkretzschmar/s6e8-missingness-aware-55-model-blend). Each published experiment records its source predictions and transformation.
