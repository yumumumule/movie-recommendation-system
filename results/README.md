# Recorded experiment artifacts

Completed on **2026-10-28 (Asia/Shanghai)** using the official MovieLens Latest Small
September 2018 snapshot, seed 42, Python 3.12.9 and NumPy 2.2.6 on Windows.
All values are from actual execution. No model settings were changed after test scoring.

| File | Contents |
| --- | --- |
| `experiment_plan.json` | Grid and protocol written before training |
| `selection.json` | Model choice and settings frozen before test evaluation |
| `metrics.csv` | Validation RMSE and final test RMSE/MAE for five models |
| `bias_trials.csv` | Validation-only regularization search for additive biases |
| `mf_trials.csv` | All 18 validation-only MF trials, best epochs and durations |
| `training_history.csv` | Per-epoch objective, train RMSE and validation RMSE for selected MF |
| `ranking_metrics.csv` | Full-catalogue Precision/Recall/NDCG@10 for all five models |
| `user_activity.csv` | Selected model's test error by fixed training-history bins |
| `cold_start.csv` | Test errors by training-ID coverage; empty segments have blank metrics |
| `recommendations_user_1.csv` | Real Top-10 CLI example, excluding all source-history interactions |
| `provenance.json` | Input/split hashes, data sizes, versions, runtime, and base commit |
| `figures/` | Six PNGs rendered directly from the measured CSVs |

The `git_base_commit` is the repository HEAD before implementation files were committed;
the source and these artifacts are subsequently committed together. Split fingerprints
are normalized to LF line endings for cross-platform comparison; this normalization
does not modify split membership, models, or measured metrics.

Best validation choice: MF with 32 factors, learning rate 0.01, per-interaction L2 0.1,
batch size 256, restored at epoch 58. Final test RMSE **0.857811**, MAE **0.657072**.
The selected trial reached the 60-epoch cap; full convergence is not claimed.
The run took about 72 seconds for search and evaluation, excluding download and plotting.

Ranking uses test ratings >=4 as binary relevant items and the full catalogue excluding
train+validation interactions. It includes 585 eligible users and excludes 20 with no
positive test labels. Unobserved items are treated as nonrelevant. No negative sampling
or test-based tuning is used. MF's NDCG@10 is **0.007008**, below the constant-score
movie-ID ordering; the README discusses this limitation explicitly.

To rerun without overwriting these artifacts:

```bash
python -m experiments.run_experiments --output-dir results/reproduced --artifacts-dir artifacts/reproduced
```

Derived MovieLens artifacts follow [DATA_LICENSE.md](DATA_LICENSE.md). This directory
contains aggregate measurements and a small recommendation example, not raw ratings,
course data, or fitted parameters.
