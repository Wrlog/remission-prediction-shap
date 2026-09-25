"""Pipelines, the missingness rule and the statistics helpers."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from remission import validation as V
from remission.models import ALGOS, CarryForward, HighMissingDropper, make_pipeline, rfe_grid, selected_features


def test_corrected_ttest_is_wider_than_naive():
    rng = np.random.default_rng(0)
    d = rng.normal(0.02, 0.05, 20)
    mean, t, p = V.corrected_ttest(d, n_train=150, n_test=38)
    naive_t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    assert np.isclose(mean, d.mean())
    assert abs(t) < abs(naive_t)
    # hand formula
    expected = d.mean() / np.sqrt((1 / 20 + 38 / 150) * d.var(ddof=1))
    assert np.isclose(t, expected) and 0 < p < 1


def test_dropper_uses_fit_data_only():
    X = pd.DataFrame({"a": [1.0, np.nan, np.nan, np.nan], "b": [1.0, 2.0, np.nan, 4.0]})
    drop = HighMissingDropper(threshold=0.7).fit(X)
    assert drop.keep_ == ["b"] and drop.dropped_ == ["a"]
    assert list(drop.transform(pd.DataFrame({"a": [1.0], "b": [2.0]})).columns) == ["b"]


def test_carry_forward_fills_from_previous_visit():
    X = pd.DataFrame({"v1_bio": [6.0, 6.0], "v2_bio": [np.nan, 5.0], "v2_bio_change": [np.nan, -1.0], "v3_bio": [np.nan, np.nan]})
    out = CarryForward().fit_transform(X)
    assert out.loc[0, "v2_bio"] == 6.0 and out.loc[0, "v2_bio_change"] == 0.0
    assert out.loc[1, "v3_bio"] == 5.0


def test_calibration_of_a_calibrated_model():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 20000)
    y = rng.binomial(1, p)
    slope, intercept = V.calibration_slope_intercept(y, p)
    assert abs(slope - 1) < 0.08 and abs(intercept) < 0.05
    # overconfident predictions give a slope below 1
    lp = np.log(p / (1 - p)) * 2
    slope2, _ = V.calibration_slope_intercept(y, 1 / (1 + np.exp(-lp)))
    assert slope2 < 0.6


def test_net_benefit_edges():
    y = np.array([1, 0, 1, 0, 1])
    p = np.array([0.9, 0.1, 0.8, 0.3, 0.6])
    nb = V.net_benefit(y, p, [0.5])
    assert np.isclose(nb.model.iloc[0], 3 / 5)
    assert np.isclose(nb.treat_all.iloc[0], 3 / 5 - 2 / 5)


def test_rfe_grid_caps_at_feature_count():
    assert rfe_grid(9) == [6, 9]
    assert rfe_grid(3) == [3]


@pytest.mark.parametrize("algo", ALGOS)
def test_pipelines_fit_with_missing_values(algo):
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(80, 14)), columns=[f"f{i}" for i in range(14)])
    y = (X.f0 + 0.5 * rng.normal(size=80) > 0).astype(int).to_numpy()
    X.loc[rng.random(80) < 0.3, "f3"] = np.nan
    X.loc[rng.random(80) < 0.9, "f4"] = np.nan  # above the drop threshold
    search = V.fit_search(X, y, algo, seed=0, n_search=2, inner_folds=3)
    fitted = search.best_estimator_
    assert "f4" in fitted.named_steps["drop"].dropped_
    assert "f4" not in selected_features(fitted)
    assert roc_auc_score(y, search.predict_proba(X)[:, 1]) > 0.7


def test_outer_folds_are_the_same_for_every_algorithm():
    y = np.array([0, 1] * 30)
    a = V.outer_folds(y, repeat=2)
    b = V.outer_folds(y, repeat=2)
    assert all(np.array_equal(x[1], z[1]) for x, z in zip(a, b))
    assert not np.array_equal(a[0][1], V.outer_folds(y, repeat=3)[0][1])
