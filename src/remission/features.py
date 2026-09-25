"""Features by infusion visit, the leakage guard, and the feature ladder.

Features come in blocks. Each block has a cutoff visit, and its builder only
ever sees records up to that visit's infusion time (pre-dose samples at that
visit count; anything after the infusion does not):

  base      visit 1  demographics and screening labs
  clin_v2   visit 2  labs, biomarker, drug level and ADA test before infusion 2
  clin_v3   visit 3  the same before infusion 3
  clin_v4   visit 4  the same before infusion 4 (only used in a check)
  pkpd_v1   visit 2  MAP estimates from everything up to infusion 2
  pkpd_v2   visit 3  ... up to infusion 3
  pkpd_v3   visit 4  ... up to the first maintenance infusion

So a PK/PD block always uses the data up to the next infusion. Skewed
quantities (INF, biomarker, drug levels, PK/PD estimates) enter on the log
scale.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config
from .estimate import fit_block

BLOCK_CUTOFF_VISIT = {"base": 1, "clin_v2": 2, "clin_v3": 3, "clin_v4": 4, "pkpd_v1": 2, "pkpd_v2": 3, "pkpd_v3": 4}
# Latest visit whose data may reach any feature. Its nominal day must sit
# well before the outcome window.
PREDICTION_VISIT = 4
NOMINAL_VISIT_DAY = {1: 0, 2: 21, 3: 42, 4: config.MAINT_START}

BASE_COLS = ["age", "female", "weight", "activity", "imm", "v1_alb", "v1_inf", "v1_plt", "v1_bio"]
CLIN_NAMES = ["alb", "inf", "plt", "bio", "bio_change", "conc", "ada"]
PKPD_NAMES = ["CL", "V1", "BASE", "IC50", "Cave"]

BLOCK_COLUMNS = {
    "base": BASE_COLS,
    "clin_v2": [f"v2_{n}" for n in CLIN_NAMES],
    "clin_v3": [f"v3_{n}" for n in CLIN_NAMES],
    "clin_v4": [f"v4_{n}" for n in CLIN_NAMES],
    "pkpd_v1": [f"v1_{n}" for n in PKPD_NAMES],
    "pkpd_v2": [f"v2_{n}" for n in PKPD_NAMES],
    "pkpd_v3": [f"v3_{n}" for n in PKPD_NAMES],
}
PKPD_COLUMNS = [c for b in ("pkpd_v1", "pkpd_v2", "pkpd_v3") for c in BLOCK_COLUMNS[b]]

LADDER = {
    "L1": ["base"],
    "L2": ["base", "pkpd_v1"],
    "L3": ["base", "clin_v2", "clin_v3", "pkpd_v1"],
    "L4": ["base", "clin_v2", "clin_v3", "pkpd_v3"],
    "L5": ["base", "clin_v2", "clin_v3", "pkpd_v1", "pkpd_v2", "pkpd_v3"],
}
LADDER_LABELS = {
    "L1": "Baseline clinical",
    "L2": "+ baseline PK/PD",
    "L3": "+ clinical to infusion 3",
    "L4": "PK/PD updated at infusion 4",
    "L5": "Everything",
}
CONTRASTS = [("L1", "L2"), ("L2", "L3"), ("L3", "L4"), ("L4", "L5")]

# Not part of the ladder: the same information time as L4, but the data up to
# infusion 4 enter as raw values instead of through the PK/PD model.
CHECK_STEPS = {"L4r": ["base", "clin_v2", "clin_v3", "clin_v4", "pkpd_v1"]}
LADDER_LABELS["L4r"] = "Raw data to infusion 4"
LOGGED = {"inf", "bio", "conc", "CL", "V1", "BASE", "IC50", "Cave"}


def _blocks(step):
    return LADDER[step] if step in LADDER else CHECK_STEPS[step]


def ladder_columns(step):
    return [c for b in _blocks(step) for c in BLOCK_COLUMNS[b]]


def check_ladder(ladder=None, cutoffs=None):
    """Raise if any block in the ladder reads data past the prediction visit,
    or if a ladder step uses a block without a declared cutoff."""
    ladder = {**LADDER, **CHECK_STEPS} if ladder is None else ladder
    cutoffs = BLOCK_CUTOFF_VISIT if cutoffs is None else cutoffs
    bad = []
    for step, blocks in ladder.items():
        for b in blocks:
            if b not in cutoffs:
                bad.append((step, b, "no declared cutoff"))
            elif cutoffs[b] > PREDICTION_VISIT:
                bad.append((step, b, f"cutoff visit {cutoffs[b]}"))
    if NOMINAL_VISIT_DAY[PREDICTION_VISIT] >= config.OUTCOME_WINDOW[0]:
        bad.append(("all", "prediction visit", "inside the outcome window"))
    if bad:
        raise ValueError(f"leaky feature blocks: {bad}")


def availability(step):
    """Nominal day on which every feature in a ladder step is known."""
    return max(NOMINAL_VISIT_DAY[BLOCK_CUTOFF_VISIT[b]] for b in _blocks(step))


def visit_times(records):
    d = records[records.analyte == "dose"]
    return d.pivot_table(index="id", columns="visit", values="time", aggfunc="first")


def censor(records, cutoff_visit):
    """Records known by the infusion at `cutoff_visit`, patient by patient."""
    vt = visit_times(records)[cutoff_visit]
    t = records["id"].map(vt)
    return records[records.time <= t].copy()


def _value(rec, analyte, visit):
    sub = rec[(rec.analyte == analyte) & (rec.visit == visit)].set_index("id")
    val = sub["value"].copy()
    if analyte == "conc":
        val[sub["blq"].astype(bool)] = config.PK["lloq"] / 2.0
    return val.groupby(level=0).first()


def _log(x):
    return np.log(x.astype(float))


def build_base(baseline, rec):
    X = baseline.set_index("id")[["age", "female", "weight", "activity", "imm"]].astype(float)
    X["v1_alb"] = _value(rec, "alb", 1)
    X["v1_inf"] = _log(_value(rec, "inf", 1))
    X["v1_plt"] = _value(rec, "plt", 1)
    X["v1_bio"] = _log(_value(rec, "bio", 1))
    return X[BASE_COLS]


def build_clinical(baseline, rec, k):
    X = pd.DataFrame(index=baseline["id"].to_numpy())
    X.index.name = "id"
    X[f"v{k}_alb"] = _value(rec, "alb", k)
    X[f"v{k}_inf"] = _log(_value(rec, "inf", k))
    X[f"v{k}_plt"] = _value(rec, "plt", k)
    X[f"v{k}_bio"] = _log(_value(rec, "bio", k))
    X[f"v{k}_bio_change"] = X[f"v{k}_bio"] - _log(_value(rec, "bio", 1))
    X[f"v{k}_conc"] = _log(_value(rec, "conc", k))
    X[f"v{k}_ada"] = _value(rec, "ada", k)
    return X[[f"v{k}_{n}" for n in CLIN_NAMES]]


def _fit_patient(pid, weight, rec, cutoff):
    return pid, fit_block(weight, rec, cutoff)


def build_pkpd(baseline, rec, k, n_jobs=1):
    """MAP block k. `rec` is already censored at visit k + 1."""
    vt = visit_times(rec)
    wt = baseline.set_index("id")["weight"]
    groups = dict(tuple(rec.groupby("id")))
    jobs = [delayed(_fit_patient)(pid, float(wt[pid]), groups[pid], float(vt.at[pid, k + 1])) for pid in baseline["id"]]
    out = dict(Parallel(n_jobs=n_jobs)(jobs))
    est = pd.DataFrame.from_dict(out, orient="index")
    est.index.name = "id"
    X = pd.DataFrame(index=est.index)
    for n in PKPD_NAMES:
        X[f"v{k}_{n}"] = np.log(est[n])
    return X, est


def build_features(baseline, records, n_jobs=1, return_estimates=False):
    """All feature blocks, each built from its own censored view."""
    check_ladder()
    parts, estimates = [], {}
    for block, cut in BLOCK_CUTOFF_VISIT.items():
        rec = censor(records, cut)
        if block == "base":
            parts.append(build_base(baseline, rec))
        elif block.startswith("clin"):
            parts.append(build_clinical(baseline, rec, cut))
        else:
            k = int(block[-1])
            X, est = build_pkpd(baseline, rec, k, n_jobs=n_jobs)
            parts.append(X)
            estimates[block] = est
    X = pd.concat(parts, axis=1).loc[baseline["id"].to_numpy()]
    X = X[[c for b in BLOCK_COLUMNS.values() for c in b]]
    return (X, estimates) if return_estimates else X


def column_block(col):
    for b, cols in BLOCK_COLUMNS.items():
        if col in cols:
            return b
    raise KeyError(col)


def pretty(col):
    """Readable label for plots."""
    if col in ("age", "female", "weight", "activity", "imm"):
        return {"imm": "co-medication", "activity": "activity score"}.get(col, col)
    v, name = col.split("_", 1)
    label = {"bio": "biomarker", "bio_change": "biomarker change", "conc": "drug level", "alb": "ALB", "inf": "INF", "plt": "PLT", "ada": "ADA"}.get(name, name)
    if name in PKPD_NAMES:
        return f"{label} (MAP to inf {int(v[1]) + 1})"
    return f"{label} (inf {v[1]})"


def is_pkpd(col):
    return col in PKPD_COLUMNS



BLOCK_LABELS = {
    "base": "baseline clinical",
    "clin_v2": "clinical at infusion 2",
    "clin_v3": "clinical at infusion 3",
    "clin_v4": "clinical at infusion 4",
    "pkpd_v1": "PK/PD fit to infusion 2",
    "pkpd_v2": "PK/PD fit to infusion 3",
    "pkpd_v3": "PK/PD fit to infusion 4",
}


def design_tables():
    """The ladder, the feature list and the design constants as plain tables,
    so the dashboard can show them without importing this package."""
    ladder = pd.DataFrame(
        [
            {
                "step": step,
                "label": LADDER_LABELS[step],
                "blocks": " + ".join(BLOCK_LABELS[b] for b in _blocks(step)),
                "n_features": len(ladder_columns(step)),
                "known_day": availability(step),
                "kind": "ladder" if step in LADDER else "check",
            }
            for step in [*LADDER, *CHECK_STEPS]
        ]
    )
    feats = pd.DataFrame(
        [
            {
                "feature": c,
                "label": pretty(c),
                "block": b,
                "pkpd": is_pkpd(c),
                "known_day": NOMINAL_VISIT_DAY[BLOCK_CUTOFF_VISIT[b]],
            }
            for b, cols in BLOCK_COLUMNS.items()
            for c in cols
        ]
    )
    design = {
        "target": config.TARGET,
        "outcome_window": list(config.OUTCOME_WINDOW),
        "visit_days": {str(k): v for k, v in NOMINAL_VISIT_DAY.items()},
        "maint_every": config.MAINT_EVERY,
        "dose_mg_per_kg": config.DOSE_MG_PER_KG,
    }
    return ladder, feats, design
