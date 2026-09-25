"""The simulated cohort has the structure the analysis relies on."""

import numpy as np
import pytest

from remission import config
from remission.features import build_features
from remission.simulate import simulate_cohort


@pytest.fixture(scope="module")
def cohort():
    return simulate_cohort(seed=5, scale_n=1.0)


def test_study_sizes_and_ids(cohort):
    baseline, _, outcome, truth = cohort
    assert baseline.groupby("study").size().to_dict() == {k: v["n"] for k, v in config.STUDIES.items()}
    assert baseline.id.is_unique and set(baseline.id) == set(outcome.id) == set(truth.id)


def test_outcome_is_first_biomarker_in_window(cohort):
    _, records, outcome, _ = cohort
    lo, hi = config.OUTCOME_WINDOW
    bio = records[records.analyte == "bio"]
    for _, row in outcome.dropna().sample(25, random_state=0).iterrows():
        mine = bio[(bio.id == row.id) & (bio.time >= lo) & (bio.time <= hi)].sort_values("time")
        assert mine.time.iloc[0] == row.outcome_time
        assert mine.value.iloc[0] == row.bio_outcome
        assert row.at_target == float(row.bio_outcome < config.TARGET)


def test_studies_differ_in_outcome_rate(cohort):
    _, _, outcome, _ = cohort
    rates = outcome.groupby("study").at_target.mean()
    assert rates.max() - rates.min() > 0.15


def test_missingness_follows_the_panels(cohort):
    baseline, records, _, _ = cohort
    X = build_features(baseline, records, n_jobs=1)
    study = baseline.set_index("id").loc[X.index, "study"]
    miss = X.isna().groupby(study.to_numpy()).mean()
    # platelets after baseline are only drawn in study A
    assert miss.loc["A", "v2_plt"] < 0.2 and miss.loc["B", "v2_plt"] == 1.0 and miss.loc["C", "v2_plt"] == 1.0
    # study C skips the biomarker and the drug level at infusion 2
    assert miss.loc["C", "v2_bio"] == 1.0 and miss.loc["C", "v2_conc"] == 1.0
    # early ADA tests only in study C
    assert miss.loc["C", "v2_ada"] < 0.2 and miss.loc["A", "v2_ada"] == 1.0
    # screening labs are complete
    assert X[["v1_alb", "v1_inf", "v1_plt", "v1_bio"]].notna().all().all()


def test_truth_is_consistent(cohort):
    _, _, outcome, truth = cohort
    t = truth.dropna(subset=["p_true"])
    assert ((t.p_true >= 0) & (t.p_true <= 1)).all()
    # the Shapley parts add up to -log B relative to the reference patient
    total = t[["phi_exposure", "phi_base", "phi_ic50"]].sum(axis=1)
    assert np.corrcoef(total, -np.log(t.bio_true_outcome))[0, 1] > 0.99
    # higher true probability should go with being at target
    o = outcome.set_index("id").loc[t.id]
    assert t.p_true.to_numpy()[o.at_target.to_numpy() == 1].mean() > t.p_true.to_numpy()[o.at_target.to_numpy() == 0].mean()
