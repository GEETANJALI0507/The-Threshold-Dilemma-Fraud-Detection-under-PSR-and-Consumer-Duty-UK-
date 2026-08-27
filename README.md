# The-Threshold-Dilemma-Fraud-Detection-under-PSR-and-Consumer-Duty-UK-
This is the code behind my dissertation, "The Threshold Dilemma: Optimising Fraud Detection Thresholds in UK Payments under PSR Reimbursement and the FCA Consumer Duty."

**The question I'm trying to answer**: when a bank sets the probability threshold above which a transaction gets flagged as fraud, that single number has to satisfy two things that pull in opposite directions. The FCA's Consumer Duty pushes toward catching more fraud and protecting customers — which means a lower threshold, more alerts, more false positives, more friction for genuine customers. The PSR's mandatory APP fraud reimbursement rules mean every fraud that gets through now has a direct cost to the bank — which pushes toward a higher threshold too, but for a completely different reason (cost control, not customer protection), and the two don't automatically agree on where that threshold should sit. I wanted to actually model that trade-off with real cost numbers and real data, rather than argue it in the abstract.

**Approach:** three models (Logistic Regression, Random Forest, XGBoost) trained on transaction data with realistic class imbalance (~3.5% fraud), then a battery of threshold-selection strategies (cost-optimal, F1-optimal, Youden's J, a couple of literature-standard closed-form approaches) evaluated against a cost model that reflects the actual regulatory environment — not just accuracy, which is close to meaningless here (a model that flags nothing still scores ~96%). Thresholds are then compared across three regulatory "regimes" (a literature baseline, current UK regulation, and the UK's statutory reimbursement cap) to see how much the right answer actually moves depending on which rules apply.

**Data:** IEEE-CIS Fraud Detection (primary, ~590k transactions) and the ULB Credit Card Fraud dataset (secondary, for cross-dataset robustness) — both card fraud. Later versions add PaySim, a mobile-money dataset that's structurally closer to the authorised push-payment (APP) fraud the PSR rules actually target, as an independent generalisability check.

**Key findings so far:**

The threshold that's "correct" changes meaningfully depending on which regulatory regime you evaluate under — this is the central result the dissertation is built around.
There's a hard ceiling in this data: no combination of model, feature set, or imbalance-handling technique gets precision and recall both above 90% at a single threshold. That's reported honestly as precision_target_90 / recall_target_90 rather than papered over.
Feature engineering (leak-safe aggregate encodings, transaction velocity) and imbalance-handling method both matter less than expected past a certain point — most of the achievable performance comes from the model choice itself (XGBoost consistently ahead of Random Forest and Logistic Regression).
Explored whether stacking the three models beats the best single one — a genuine open question rather than an assumed win, reported either way.

**Repo structure**: the notebook went through several iterations (v1 through v8, v1 is the first version of the code and v8 the final) as I added hyperparameter tuning, more robust cost-sensitivity testing, bootstrap significance testing, SHAP explainability, additional datasets, and finally the stacked-ensemble comparison. Each version folder has its own README explaining what changed and why. v8 is the current/final pipeline. I have uploaded from v6, as it's the model that I wanted to archive as spoken in the proposal. 

