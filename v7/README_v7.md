# Fraud Threshold Optimisation — v7

Version 7 of the modelling notebook for my dissertation, *"The Threshold Dilemma: Optimising Fraud Detection Thresholds in UK Payments under PSR Reimbursement and the FCA Consumer Duty."* Notebook file: `dissertation_5753132_v7.ipynb`.

**This one replaces v6** — if you're pulling a copy of this repo, use v7, not v6-then-v7 back to back. Everything from v6 is already in here (PaySim as a third robustness dataset, the extra IEEE-CIS feature engineering pass) plus one new piece.

## What's new: a stacked ensemble (cell 22)

My proposal is built around three separate models (XGBoost, Random Forest, Logistic Regression) compared side by side — that comparison is still the core of the dissertation and isn't going anywhere. What I added in v7 is a clearly-separate extra experiment: does combining the three models' outputs beat the best one on its own?

I went with this specifically because it was the one lever left with real headroom on AUC-PR. Calibration wasn't going to do it — AUC-PR is a ranking metric and doesn't change under the monotonic isotonic recalibration already in the pipeline (which is already good, ECE < 0.003, it's just not going to move this number). Feature engineering and hyperparameter tuning had both clearly plateaued by v4→v5→v6. Ensembling the three model types is the standard remaining move, and it's actually what got the 0.891 AUC-PR figure I'm benchmarking against in the literature — so it seemed worth actually trying rather than assuming it wouldn't help.

How it works: a small logistic regression learns how to weight the three models' *already-calibrated* probabilities, fit on `val_thr` — a split none of the three base models ever touched during their own training or calibration. AUC-PR and ROC-AUC are then computed purely on the held-out test set, which the stacker never sees while fitting. The one simplification worth flagging: the stacker's own threshold is also picked from `val_thr`, which is a minor leak risk in theory but negligible in practice for a 4-parameter linear combiner — there's a comment in the notebook explaining the reasoning if you want the detail.

Being realistic about what to expect here: stacking a strong model (XGBoost, AUC-PR somewhere around 0.85–0.87 depending on how v6's features landed) with two considerably weaker ones (Random Forest ~0.66, Logistic Regression ~0.45) doesn't automatically help much — it's entirely possible the stacker just learns to put nearly all its weight on XGBoost and lands close to XGBoost's own number anyway. `table16_stacked_ensemble.csv` reports this either way, including a `beats_best_single_model` column that just tells you straight whether it worked.

## Testing I did before calling this done

- Ran the full cell sequence on synthetic data end-to-end, including the stacker itself this time — no errors.
- Unit-tested the stacker specifically with synthetic probability streams of known, different skill levels (a deliberately weak "logistic" signal, a mid-strength "random_forest" one, a strong "xgboost" one) — confirmed the fitted stacker puts its largest weight on the strongest input, and confirmed the reported AUC-PR is reproducible purely from applying the val_thr-fit stacker to test-set inputs, i.e. no test-set leakage into how it's fit.
- Haven't generated real numbers on the actual datasets in this session yet — next step is running it in Colab.

## Running it in Colab

Same as v6 — CPU runtime, no accelerator needed, `FETCH_PAYSIM = True` in cell 3 if you want PaySim included.

## How long it takes

Cell 22 (the stacker) is basically instant — it's just fitting a 4-parameter logistic regression on probabilities that are already computed. Total time is essentially the same as v6.

| Cell | What | Roughly |
|---|---|---|
| 19 | Imbalance method comparison | 15–25 min |
| 20 | Hyperparameter search (150k-row subsample) | 35–55 min |
| 21 | Main run — full 590,540 rows, tuned, 6 aggregate groups | 45–65 min |
| 22 | **Stacked ensemble (new)** | seconds |
| 23 | Sample-size ablation | 30–40 min |
| 24 | Regime comparison | a few seconds |
| 25 | Robustness — temporal + ULB (full) + PaySim (if fetched) + 5 seeds | 65–90 min |
| 26–30 | Sensitivity, significance testing, SHAP, figures, export | 10–15 min |
| **Total** | | **~3.5–5 hrs with PaySim / ~3–4.5 hrs without** |

If you're short on time: cells 1–22 give you the full-dataset headline result *and* the stacked-ensemble comparison — which is really the number I was trying to see against the 0.891 benchmark. Everything after cell 22 is robustness depth, not required to answer that specific question.

## Reading `table16_stacked_ensemble.csv`

One row: the ensemble's AUC-PR, ROC-AUC, precision/recall/F1 at its own cost-optimal threshold, the three learned weights, and a `beats_best_single_model` column (True/False) with the margin. Whichever way it comes out, it's a real finding either way — if stacking *doesn't* beat XGBoost alone, that's not a failed experiment, it just tells me the three models are picking up overlapping signal rather than complementary signal, which is worth a line in the discussion chapter.
