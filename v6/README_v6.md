# Fraud Threshold Optimisation — v6

This is version 6 of the modelling notebook for my dissertation, *"The Threshold Dilemma: Optimising Fraud Detection Thresholds in UK Payments under PSR Reimbursement and the FCA Consumer Duty."* Notebook file: `dissertation_5753132_v6.ipynb`.

Datasets used:
- **IEEE-CIS Fraud Detection** (primary) — full 590,540 rows
- **ULB Credit Card Fraud** (secondary, robustness check) — full 284,807 rows
- **PaySim** (new in this version) — mobile-money fraud, 300,000-row sample

## Why v6 exists

By v5 I'd more or less hit the ceiling of what feature engineering and imbalance handling could do on IEEE-CIS alone, and every version of my audit kept flagging the same limitation: both IEEE-CIS and ULB are *card* fraud, not the authorised push-payment (APP) fraud that PSR reimbursement rules are actually about. PaySim is the closest public dataset to that — it's mobile-money transfers, which is structurally much closer to a push payment than a card swipe. So v6 adds it as a third, fully independent robustness dataset: its own split, its own fit, its own row in the results table, evaluated exactly the way ULB already was.

To be clear about what this **isn't**: PaySim's columns (`type`, `oldbalanceOrg`, `newbalanceDest`, etc.) don't overlap at all with IEEE-CIS's ~430 card/device/email columns, so there's no way to fold it into the IEEE-CIS training set even if I wanted to. It's a separate generalisability test, not extra training data. It doesn't move the main IEEE-CIS numbers at all.

The second change is on the feature engineering side. I added three more aggregate-encoding groups on top of the ones from v3/v4 (`card2`, `P_emaildomain`, and the `card1`×`P_emaildomain` combination), and — the more meaningful one — an actual **time-windowed velocity feature**: how many times a given card transacted in the last hour, rather than just a running lifetime count. A card with 50 transactions spread over three years and a card with 50 transactions in the last hour look identical to the old lifetime-count feature; they look completely different to this one, and a sudden burst is a pretty standard fraud signal. Realistically this might nudge AUC-PR up into the high 0.85s/low 0.87s from 0.851 — it's not going to push it above 0.90.

**Still true in v6, same as every version before it:** there's no combination of dataset, feature, or tuning that gets precision and recall both above 90% at a single threshold on IEEE-CIS. I checked this directly off the actual precision-recall curve, not just assumed it, and it lines up with what the published literature reports too. `precision_target_90` and `recall_target_90` are still the honest way to report this — each hits 90% on its own, just never at the same time.

## Testing I did before calling this done

- Ran the full cell sequence on synthetic data end-to-end — no errors.
- Unit-tested the velocity feature specifically, not just smoke-tested it: fed it a synthetic burst of 5 transactions 100 seconds apart and got the count sequence `[0,1,2,3,4]` as expected; added a 2-hour gap partway through and got `[0,1,2,0,1]` — confirming the window actually forgets old transactions once they age out, rather than just accumulating.
- Checked the three new aggregate-encoding groups populate all 18 expected columns and compute cleanly on a held-out sample.
- Unit-tested the PaySim balance-delta arithmetic against a synthetic frame with known balances.
- I haven't run this on the real datasets yet in this environment — next step is to actually run it in Colab and pull the real numbers.

## Running it in Colab

Same setup as v5 — CPU runtime is fine, you don't need a GPU/TPU.

1. Cells 1–2: environment + imports/config.
2. Cell 3: data download. Needs a Kaggle token, same as before. If you want PaySim included, set `FETCH_PAYSIM = True` near the top of this cell — it's optional, and the robustness step will just skip it gracefully if you leave it off.
3. Cells 4–18 run in order; cell 18 is the smoke test.
4. Cell 19 onward (imbalance comparison, main fit, etc.) as before.

## How long it takes

Longer than v5 because there are three extra aggregate groups to compute and, if you fetch it, a whole extra dataset to fit.

| Cell | What | Roughly |
|---|---|---|
| 19 | Imbalance method comparison | 15–25 min |
| 20 | Hyperparameter search (150k-row subsample) | 35–55 min |
| 21 | Main run — full 590,540 rows, tuned, 6 aggregate groups | 45–65 min |
| 22 | Sample-size ablation | 30–40 min |
| 23 | Regime comparison | a few seconds |
| 24 | Robustness — temporal + ULB (full) + PaySim (if fetched) + 5 seeds | 65–90 min |
| 25–29 | Sensitivity, significance testing, SHAP, figures, export | 10–15 min |
| **Total** | | **~3.5–5 hrs with PaySim / ~3–4.5 hrs without** |

If you don't have that much time in one go: cells 1–21 alone give you the full-dataset headline result on IEEE-CIS. PaySim and the rest of cell 24 can wait for a second session without affecting the main numbers.

## What comes out

Same outputs as v5, plus a `"PaySim, stratified"` row added to `table3_robustness.csv` when `FETCH_PAYSIM` was set to `True`.
