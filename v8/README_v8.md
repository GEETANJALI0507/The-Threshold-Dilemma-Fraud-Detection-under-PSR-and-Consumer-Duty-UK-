# Fraud Threshold Optimisation — v8 pipeline

`dissertation_5753132_v8.ipynb`.
Primary dataset: IEEE-CIS, full 590,540 rows. Secondary: ULB Credit Card
Fraud, full 284,807 rows. **Third: PaySim**, mobile-money fraud, 300,000-row
sample — **on by default this time.**

## Read this first — v8 changes exactly one thing

v8 is **identical to v7 in every line of modelling code.** Features,
hyperparameter search, the primary XGBoost/RF/LR fit, and Cell 22's stacked
ensemble are all untouched. The only change, in Cell 3:

```python
FETCH_PAYSIM = True   # was False in both v6 and v7
```

Both v6 and v7 built the full PaySim wiring — download, leak-safe
balance-delta feature engineering, its own train/val/test split, its own row
in `table3_robustness.csv` — but the flag was left off on both actual runs,
so it never got used. Nothing about that was a bug in the code; it was just
missed twice when starting the notebook. v8 exists so it doesn't get missed
a third time. If you want to skip PaySim on a given run, set `FETCH_PAYSIM =
False` in Cell 3 as before — it still degrades gracefully either way.

**What to expect when it runs:** a `"PaySim, stratified"` row appears in
`table3_robustness.csv` alongside the existing `"IEEE, stratified"`,
`"IEEE, temporal"`, and `"ULB, stratified"` rows. PaySim **cannot** become
more training rows for the IEEE-CIS model — its columns (`type`,
`oldbalanceOrg`, `newbalanceDest`, …) share nothing with IEEE-CIS's ~430
card/device/email columns, so there is no way to merge the two schemas into
one classifier's training set. What it buys you is a genuinely independent
generalisability check on a different fraud type (mobile-money push
payments), closing the "IEEE-CIS/ULB are both card fraud, not APP fraud"
limitation flagged in every audit so far. It does **not** move IEEE-CIS's
own AUC-PR, precision, or recall by any amount — those numbers should come
out statistically identical to v7's (0.869 AUC-PR, 0.974 ROC-AUC), modulo
the same small run-to-run variation already seen between v6 and v7.

**On the precision/recall ceiling, stated again because it doesn't change
with this version either:** no configuration of this pipeline — no dataset,
no feature, no imbalance method, no amount of tuning — delivers precision
and recall both ≥90% at a single threshold on IEEE-CIS. `precision_target_90`
and `recall_target_90` remain the honest way to report both, each ≥90%
individually, never together.

## Verification performed

- Diffed line-by-line against v7's source: the only functional change is the
  `FETCH_PAYSIM` default; everything else — including the already-unit-tested
  velocity feature, aggregate groups, and PaySim balance-delta arithmetic —
  is byte-for-byte the same code that v6/v7 already verified.
- Notebook JSON validated with `nbformat` (31 cells, matches v7's cell count
  exactly since no cells were added or removed).
- Every code cell's Python syntax checked directly (magic lines like
  `!pip install` stripped first) — zero syntax errors across all 31 cells.
- Real IEEE-CIS/ULB/PaySim numbers have not been generated in this
  session — run the notebook in Colab, then share results back for the same
  evaluation done for v1 through v7.

## How to run (Google Colab)

Identical to v7 — CPU runtime, no accelerator.

1. Cells 1–2: environment, imports/config.
2. Cell 3: data acquisition. Kaggle token as before. `FETCH_PAYSIM` is now
   `True` by default — leave it unless you specifically want to skip the
   ~470MB PaySim download for a faster run.
3. Cells 4–18 in order; Cell 18 is the smoke test.
4. Cell 19 onward as before, including Cell 22's stacked ensemble.

## Time expectations

Same as v7, plus the PaySim fetch/fit this time since the flag is now on.

| Cell | What | Estimate |
|---|---|---|
| 19 | Imbalance method comparison | ~15–25 min |
| 20 | Hyperparameter search, 150k-row subsample | ~35–55 min |
| 21 | Primary run: full 590,540 rows, tuned, 6 aggregate groups | ~45–65 min |
| 22 | Stacked ensemble | seconds |
| 23 | Sample-size ablation | ~30–40 min |
| 24 | Regime comparison | seconds |
| 25 | Robustness: temporal + ULB (both full) + **PaySim (300k, on by default)** + 5 seeds | ~65–90 min |
| 26–30 | Sensitivity, significance, SHAP, figures, export | ~10–15 min |
| **Total** | | **~3.5–5 hours** |

If that's more than one sitting: Cells 1–22 give you the full-dataset, tuned
headline result and the stacked-ensemble comparison — everything after Cell
22, including the PaySim fetch inside Cell 25, is robustness depth and can
run as a later session without touching the primary numbers.

## Outputs produced

Same set as v7 (16 tables incl. `table16_stacked_ensemble.csv`, 7 figures),
plus this time an actual `"PaySim, stratified"` row in
`table3_robustness.csv` instead of a skip message.
