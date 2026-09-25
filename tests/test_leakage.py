"""Nothing after a block's cutoff may reach that block's features."""

import numpy as np
import pandas as pd
import pytest

from remission import config
from remission import features as F
from remission.estimate import fit_block
from remission.simulate import simulate_cohort


@pytest.fixture(scope="module")
def cohort():
    return simulate_cohort(seed=11, scale_n=0.35)


@pytest.fixture(scope="module")
def features(cohort):
    baseline, records, _, _ = cohort
    return F.build_features(baseline, records, n_jobs=1)


def _scramble_after(records, cutoff_visit, rng):
    """Rewrite every measurement taken after the infusion at cutoff_visit."""
    vt = F.visit_times(records)[cutoff_visit]
    late = records.time > records.id.map(vt)
    out = records.copy()
    vals = out.loc[late, "value"].to_numpy()
    out.loc[late, "value"] = rng.permutation(vals) * rng.uniform(0.2, 5.0, len(vals))
    return out, int(late.sum())


def test_ladder_passes_the_guard():
    F.check_ladder()


def test_guard_rejects_a_late_block():
    with pytest.raises(ValueError, match="cutoff visit 5"):
        F.check_ladder({"bad": ["base", "pkpd_v4"]}, {**F.BLOCK_CUTOFF_VISIT, "pkpd_v4": 5})
    with pytest.raises(ValueError, match="no declared cutoff"):
        F.check_ladder({"bad": ["outcome_block"]})


def test_pkpd_blocks_use_data_to_the_next_infusion():
    for k in (1, 2, 3):
        assert F.BLOCK_CUTOFF_VISIT[f"pkpd_v{k}"] == k + 1
    assert F.availability("L1") == 0
    assert F.availability("L2") == 21
    assert F.availability("L3") == 42
    assert F.availability("L4") == F.availability("L5") == config.MAINT_START
    assert config.MAINT_START < config.OUTCOME_WINDOW[0]


@pytest.mark.parametrize("block", list(F.BLOCK_CUTOFF_VISIT))
def test_each_block_ignores_data_after_its_cutoff(block, cohort, features):
    baseline, records, _, _ = cohort
    cut = F.BLOCK_CUTOFF_VISIT[block]
    scrambled, n_late = _scramble_after(records, cut, np.random.default_rng(3))
    assert n_late > 0
    after = F.build_features(baseline, scrambled, n_jobs=1)
    cols = F.BLOCK_COLUMNS[block]
    pd.testing.assert_frame_equal(features[cols], after[cols])


def test_scrambling_the_outcome_period_changes_nothing(cohort, features):
    baseline, records, _, _ = cohort
    scrambled, _ = _scramble_after(records, F.PREDICTION_VISIT, np.random.default_rng(9))
    pd.testing.assert_frame_equal(features, F.build_features(baseline, scrambled, n_jobs=1))


def test_later_data_does_change_later_blocks(cohort, features):
    # the scramble test is only meaningful if the blocks react to data
    baseline, records, _, _ = cohort
    scrambled, _ = _scramble_after(records, 1, np.random.default_rng(4))
    after = F.build_features(baseline, scrambled, n_jobs=1)
    assert not np.allclose(features["v3_IC50"], after["v3_IC50"])
    assert not features[F.BLOCK_COLUMNS["clin_v3"]].equals(after[F.BLOCK_COLUMNS["clin_v3"]])


def test_fit_block_refuses_late_records(cohort):
    baseline, records, _, _ = cohort
    pid = baseline.id.iloc[0]
    rec = records[records.id == pid]
    with pytest.raises(ValueError, match="after the cutoff"):
        fit_block(float(baseline.weight.iloc[0]), rec, cutoff=30.0)


def test_no_outcome_or_truth_columns(cohort, features):
    _, _, outcome, truth = cohort
    banned = (set(outcome.columns) | set(truth.columns)) - {"id"}
    assert not banned & set(features.columns)
