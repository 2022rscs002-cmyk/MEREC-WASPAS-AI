# MEREC-WASPAS-AI: leakage-controlled benchmark for smart infrastructure assessment

Code, data and outputs for the manuscript *An AI-Integrated MEREC–WASPAS Model for Smart Infrastructure Optimization under Uncertainty: A Simulation-Based Proof-of-Concept Study*.

## Structure
| Folder | Content |
|---|---|
| `data/` | Semi-synthetic corpus (CSV, Parquet), corpus before artefact injection, `anchors.csv` (Stage 1 city–year generator parameters), copula correlation matrix, generation parameters, full-corpus weights, data dictionary |
| `code/` | `generate_corpus.py` (Section 5.1); `core.py` (fold-internal MEREC, neutrosophic memberships, WASPAS, tertile labels, learners); `tune.py`; `exp_seeds.py` (P1/P5, ablation, Tables 9 and 11); `exp_protocols.py` (P2–P4); `exp_misc.py` (Tables 4–6 and 8, cross-validation, λ objective, sensitivity grid, worked example, proxy correlations); `aggregate.py`; figure scripts |
| `outputs/` | `hpo_space.json`, `hpo_results.csv`, `hpo_selected.json`, `seed_results.csv`, `loco_results.csv`, `lambda_objective.csv`, `eda_tests.csv`, `weights_p1_fold.csv`, `sensitivity_grid.csv`, `p1_test_predictions.csv` (record-level predictions and class probabilities of every configuration on the P1 test partition), `confusion_matrix_p1.csv`, `results.json` (all table values) |
| `figures/` | Figures 1–13, 600 dpi TIFF |

## Reproduction
```
conda env create -f environment.yml
conda activate merec-waspas-ai
python run_all.py
```
All randomness is seeded (corpus seed 42; P5 seeds listed in `code/exp_seeds.py`).

## Notes
* The corpus is semi-synthetic. `anchors.csv` contains generator parameters of plausible magnitude, not statistics extracted from official sources.
* `WASPAS_score` and `optimization_class` in the corpus file are computed on the full corpus for description only; in every fold the weights, memberships, normalisation extrema and tertile cut points are recomputed from the training partition.
* The leakage-controlled specification never supplies `WASPAS_score` or the weight block to a classifier.
