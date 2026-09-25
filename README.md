# Predicting long-term response to Drug X from early-treatment data

[![tests](https://github.com/Wrlog/remission-prediction-shap/actions/workflows/tests.yml/badge.svg)](https://github.com/Wrlog/remission-prediction-shap/actions/workflows/tests.yml)
[![dashboard](https://github.com/Wrlog/remission-prediction-shap/actions/workflows/pages.yml/badge.svg)](https://wrlog.github.io/remission-prediction-shap/)

Everything here runs on simulated data. It's a research and teaching demo, it hasn't been validated, and it
must not be used for patient care.

Results dashboard: [wrlog.github.io/remission-prediction-shap](https://wrlog.github.io/remission-prediction-shap/)

The question: using what's known in the first weeks of treatment with a hypothetical IV drug (Drug X), can I
predict whether a patient's biomarker will be at target about a year later, and do individual PK/PD estimates
help beyond the raw clinical data? This follows the general approach of my PhD work on model-informed
precision dosing. The real data and results from that work aren't public, so the cohort, the numbers and the
design choices here are all made up for this repo. Because the data come from a known PK/PD model, I can check
the models against the true probability of reaching target, and check SHAP against the true mechanism.

## Results

Baseline data alone give an out-of-fold AUROC of about 0.80 with all four algorithms, against 0.98 for the
true simulated probabilities. Re-estimating PK/PD with data up to the first maintenance infusion (day 98) is
the one step that clearly helps (+0.043 to +0.087, corrected p < 0.02 for three of the four algorithms), but
feeding the same day-98 data in as raw values does just as well, so in this simulation the PK/PD model mostly
repackages what the raw data already carry. The best configuration is elastic-net logistic regression on L4:
AUROC 0.925 (bootstrap 95% CI 0.893 to 0.960), ahead of the three tree ensembles.

| Step | Known by | Elastic net | Random forest | XGBoost | CatBoost |
|---|---|---|---|---|---|
| L1 baseline | day 0 | 0.806 | 0.805 | 0.798 | 0.800 |
| L4 PK/PD refitted | day 98 | 0.925 | 0.897 | 0.899 | 0.894 |
| L4r raw day-98 data | day 98 | 0.916 | 0.915 | 0.912 | 0.918 |

The dashboard has the rest: every ladder step with error bars, per-fold changes with corrected and naive
tests, leave-one-study-out, the missingness and study-only checks, imputation sensitivity, calibration,
decision curves, SHAP per algorithm and against the true mechanism, feature selection frequency, and the
simulated cohort. The matplotlib versions of the figures are in [`figures/`](figures/) and every table is in
[`results/`](results/).

![Out-of-fold AUROC by feature group and algorithm](figures/auroc_ladder.png)

## The simulated studies

All four of my demo repos share one made-up PK/PD model for Drug X: two-compartment PK with a 2 h infusion
and weight-based dosing (CL 0.30 L/day with weight, albumin, INF and ADA covariates, 30% IIV and 15% IOV per
dosing interval; V1 3.2 L, Q 0.50 L/day, V2 2.0 L), and an indirect response in which drug inhibits
production of the biomarker, driven by the average concentration over each dosing interval (Cave), with
BASE 600 units and IC50 3.0 mg/L (60% and 90% IIV) and kout 0.04/day. Residual error is 15% on drug levels
and 25% on the biomarker. I haven't changed any of those numbers. Here Cave is the exact interval average
from the closed-form PK solution, which is what a fine grid converges to.

What this repo adds:

- Regimen: 5 mg/kg on days 0, 21 and 42, then every 56 days from day 98, with visit jitter of up to 2 days
  (induction) or 3 days (maintenance).
- ADA onset: from the second dose on, each dosing interval has a chance of antibodies appearing, with
  logit p = -2.6 - 1.5 log(Cave of the previous interval / 20) - 1.2 x co-medication. CL goes up 1.8-fold
  from onset.
- Labs after baseline follow the biomarker (INF and PLT fall and albumin rises as it drops). PK uses the
  baseline covariates.
- Three studies with different patients and panels. A (78 patients) measures all labs at every visit; B (62,
  lighter and more inflamed) has PLT at baseline only; C (50, the most inflamed) has albumin and PLT at
  baseline only, no biomarker or drug level at infusion 2, and the only early ADA tests. Scheduled samples
  after screening go missing 8% of the time.

The outcome is the first recorded biomarker value between day 315 and day 399, at target if it's below 200
units. One patient had no value in the window, which leaves 189 patients, 72% at target.

## Features and the ladder

Clinical features at infusion k are the pre-dose values at that visit (albumin, INF, PLT, the biomarker, its
log change from baseline, the drug level, the ADA test); the baseline block adds age, sex, weight, an
activity score and co-medication. Each PK/PD block holds MAP estimates of CL, V1, BASE and IC50 plus the
predicted Cave, fitted sequentially (PK first, then PD with Cave from the PK estimates) to everything
recorded up to the next infusion. The prior is the shared population model, treated as a published model, so
nothing is fitted on this cohort.

| Step | Contents | Features | Known by |
|---|---|---:|---|
| L1 | baseline clinical | 9 | day 0 |
| L2 | L1 + PK/PD block 1 | 14 | day 21 |
| L3 | L2 + clinical at infusions 2 and 3 | 28 | day 42 |
| L4 | L3 with PK/PD block 3 in place of block 1 | 28 | day 98 |
| L5 | everything: clinical to infusion 3 and all three PK/PD blocks | 38 | day 98 |
| L4r (check) | L3 + raw clinical values at day 98 | 35 | day 98 |

Every block declares a cutoff visit and its builder only receives records up to that visit's infusion time.
[`tests/test_leakage.py`](tests/test_leakage.py) scrambles every record after each cutoff and checks that
nothing moves, and that later blocks do react to earlier data.

## Models and validation

- Elastic-net logistic regression, random forest, XGBoost and CatBoost. No voting or stacking.
- Repeated nested CV: 3 repeats of stratified 5-fold outer CV, with a stratified 4-fold inner randomized
  search over 6 configurations. All algorithms and steps share the same splits.
- Tree models use recursive feature elimination inside the inner loop (6, 12 or 18 features kept, tuned);
  the elastic net selects through its L1 part.
- Missing data, handled inside each training fold: drop columns more than 70% missing, then per-visit median.
- Ladder contrasts use the Nadeau and Bengio corrected resampled t-test on the 15 paired per-fold differences.
- Leave-one-study-out, a missing-indicators-only model, a study-only model, five imputers on L5, calibration
  slope and intercept, precision-recall and a decision curve.
- SHAP on the best configuration of each algorithm, refitted on the full cohort (L5), compared with an exact
  Shapley split of the true late biomarker into exposure, baseline level and drug sensitivity.

## Running it

Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install --no-deps -e .
python -m pytest                   # about a minute
python scripts/run_pipeline.py     # full run, writes results/ and data/
python scripts/make_figures.py     # figures/ from results/
pip install -r dashboard/requirements.txt
python -m dashboard.build          # site/index.html from results/
```

The full pipeline took 11 minutes on a 12-core laptop (`--jobs` sets the number of worker processes).
`--quick` runs a smaller cohort with one repeat and a tiny search; CI runs it with the tests on every push.
The dashboard build only needs plotly, pandas and numpy and never retrains; the Pages workflow runs it on
every push to main. To look at it locally, run `python -m http.server -d site` and open
http://localhost:8000. Settings are in [`src/remission/config.py`](src/remission/config.py). Other package
versions than those in `requirements.txt` may shift the third decimal.

## Layout

```
src/remission/   config, PK/PD, simulation, MAP fit, features and ladder, models, validation, SHAP
scripts/         run_pipeline.py, make_figures.py
tests/           PK/PD against an ODE solver, simulation structure, leakage, validation helpers
results/         every table the dashboard and README use
figures/         matplotlib figures drawn from results/
dashboard/       build.py and the plotly figures for the Pages site
```

## Limitations

- The MAP prior is the same population model that generated the data, apart from IOV and ADA timing. With
  real data it would have to come from outside the cohort or be refitted inside each training fold.
- 3 repeats and 6 search configurations keep the run on a laptop. The repeat SD is small because every repeat
  uses the same 189 patients; the fold SD (0.05 to 0.09, 38 patients per test fold) and the bootstrap interval
  are the better guide.
- The L3 to L4 gain is about time as much as about the model, which is why L4r is there.
- With 189 patients the trees' feature elimination drops the weaker exposure features, so SHAP gives exposure
  less than its true share.
- The cohort size, study panels, ADA rule, windows and search settings are my choices for this demo.

## License

MIT, see [LICENSE](LICENSE).
