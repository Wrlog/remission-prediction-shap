"""Simulate the pooled cohort: three studies, one Drug X regimen.

Each patient gets the standard regimen (5 mg/kg on days 0, 21, 42, then
every 56 days from day 98). The true PK/PD model produces drug levels and
biomarker values; each study only records what its panel says it measured.
The outcome is read off the noisy biomarker record, so nothing about it is
drawn from a separate risk model.

Returns four tables:
  baseline  one row per patient, things known before the first dose
  records   long table of every measurement: id, time, visit, analyte, value
  outcome   the first biomarker in the outcome window and the at-target flag
  truth     true parameters and quantities the analysis must never see
"""

import math

import numpy as np
import pandas as pd

from . import config
from .pkpd import biomarker, pk_solve, typical_pd, typical_pk

LABS = ("alb", "inf", "plt")


def _visit_times(rng):
    """Nominal schedule with jitter. Visits 1-3 are induction, 4 on are
    maintenance. Every visit is also an infusion."""
    j = config.VISIT_JITTER
    t = [0.0, 21.0 + rng.integers(-j, j + 1), 42.0 + rng.integers(-j, j + 1)]
    nominal = config.MAINT_START
    while nominal <= config.LAST_DAY - 10:
        t.append(nominal + rng.integers(-config.MAINT_JITTER, config.MAINT_JITTER + 1))
        nominal += config.MAINT_EVERY
    return np.array(t, dtype=float)


def _in_panel(panel, analyte, visit):
    wanted = panel[analyte]
    return visit in wanted or (visit >= 5 and "maint" in wanted)


def _draw_covariates(rng, s, n):
    z = rng.normal(s["z_shift"], 1.0, n)
    inf = np.clip(5.0 * np.exp(0.9 * z + rng.normal(0, 0.35, n)), 0.5, 60.0)
    alb = np.clip(3.9 - 0.25 * z + rng.normal(0, 0.28, n), 2.5, 5.0)
    plt = np.clip(300.0 * np.exp(0.20 * z + rng.normal(0, 0.30, n)), 150.0, 600.0)
    wt = np.clip(s["wt_median"] * np.exp(rng.normal(0, 0.25, n)), 20.0, 100.0)
    age = np.clip(rng.normal(*s["age"], n), 6, 80).round()
    activity = np.clip(7 + 2.2 * z + rng.normal(0, 2.0, n), 0, 20).round()
    return {
        "age": age,
        "female": rng.binomial(1, 0.5, n).astype(float),
        "weight": wt.round(1),
        "activity": activity,
        "imm": rng.binomial(1, s["p_imm"], n).astype(float),
        "alb0": alb.round(2),
        "inf0": inf.round(2),
        "plt0": plt.round(0),
    }


def _ada_onset(rng, dose_t, caves, imm):
    """Antibodies can appear in any interval from the second dose on, more
    often after a low-exposure interval and less often on co-medication."""
    a = config.ADA
    for j in range(1, len(dose_t)):
        logit = a["a0"] + a["a_cave"] * math.log(max(caves[j - 1], 1e-3) / 20.0) + a["a_imm"] * imm
        if rng.random() < 1.0 / (1.0 + math.exp(-logit)):
            end = dose_t[j + 1] if j + 1 < len(dose_t) else config.LAST_DAY
            return rng.uniform(dose_t[j], end)
    return math.inf


def _intervals(dose_t):
    ends = np.append(dose_t[1:], config.LAST_DAY)
    return list(zip(dose_t, ends))


def _shapley3(f, x, ref):
    """Exact Shapley values for a 3-input function against a reference."""
    from itertools import combinations

    players = range(3)
    w = {0: 1 / 3, 1: 1 / 6, 2: 1 / 3}
    phi = np.zeros(3)
    for i in players:
        others = [p for p in players if p != i]
        for size in range(3):
            for S in combinations(others, size):
                with_i = [x[p] if (p in S or p == i) else ref[p] for p in players]
                without = [x[p] if p in S else ref[p] for p in players]
                phi[i] += w[size] * (f(*with_i) - f(*without))
    return phi


def simulate_cohort(seed=config.SEED, scale_n=1.0):
    rng = np.random.default_rng(seed)
    pk, pd_ = config.PK, config.PD
    base_rows, rec_rows, out_rows, truth_rows = [], [], [], []
    pid = 0
    for study, s in config.STUDIES.items():
        n = max(12, int(round(s["n"] * scale_n)))
        cov = _draw_covariates(rng, s, n)
        for i in range(n):
            pid += 1
            c = {k: v[i] for k, v in cov.items()}
            tcl, tv1, q, v2 = typical_pk(c["weight"], c["alb0"], c["inf0"])
            tbase, tic50 = typical_pd(c["inf0"], c["plt0"])
            cl = float(tcl) * math.exp(rng.normal(0, config.omega(pk["cv_cl"])))
            v1 = float(tv1) * math.exp(rng.normal(0, config.omega(pk["cv_v1"])))
            base = float(tbase) * math.exp(rng.normal(0, config.omega(pd_["cv_base"])))
            ic50 = float(tic50) * math.exp(rng.normal(0, config.omega(pd_["cv_ic50"])))
            q, v2 = float(q), float(v2)

            visits = _visit_times(rng)
            dose_mg = config.DOSE_MG_PER_KG * c["weight"]
            doses = np.full(len(visits), dose_mg)
            kappa = rng.normal(0, config.omega(pk["cv_iov_cl"]), len(visits))
            cl_occ = cl * np.exp(kappa)
            iv = _intervals(visits)

            _, cave0 = pk_solve(visits, doses, visits, cl_occ, v1, q, v2, intervals=iv)
            onset = _ada_onset(rng, visits, cave0, c["imm"])
            if math.isfinite(onset):
                j = int(np.searchsorted(visits, onset, side="right") - 1)
                cl_t = np.concatenate([visits[: j + 1], [onset], visits[j + 1 :]])
                cl_v = np.concatenate([cl_occ[: j + 1], cl_occ[j:] * pk["cl_ada"]])
            else:
                cl_t, cl_v = visits, cl_occ

            # sampling plan
            panel = s["panel"]
            samples = []  # (time, visit, analyte)
            for v_idx, t in enumerate(visits, start=1):
                for a in ("bio", "conc", "ada") + LABS:
                    if v_idx == 1 and a in ("conc", "ada"):
                        continue
                    if _in_panel(panel, a, v_idx):
                        samples.append((t, v_idx, a))
                if v_idx in panel["peak"]:
                    samples.append((t + pk["t_inf"] + 1.0 / 24.0, v_idx, "peak"))

            conc_times = [t for t, _, a in samples if a in ("conc", "peak")]
            conc_true, caves = pk_solve(visits, doses, cl_t, cl_v, v1, q, v2, obs_times=conc_times, intervals=iv)
            conc_map = dict(zip(conc_times, conc_true))
            bio_times = sorted({t for t, _, a in samples if a in ("bio",) + LABS})
            bio_true = dict(zip(bio_times, biomarker(bio_times, visits, caves, base, ic50)))

            base_rows.append({"id": pid, "study": study, **{k: c[k] for k in ("age", "female", "weight", "activity", "imm")}})
            for t, v_idx, a in samples:
                # screening labs and the baseline biomarker are always there
                if not (v_idx == 1 and a in ("bio",) + LABS) and rng.random() < config.P_MISSING:
                    continue
                blq = False
                if a in ("conc", "peak"):
                    val = conc_map[t] * (1 + pk["sigma_conc"] * rng.normal())
                    if val < pk["lloq"]:
                        val, blq = np.nan, True
                elif a == "bio":
                    val = max(1.0, bio_true[t] * (1 + pd_["sigma_bio"] * rng.normal()))
                elif a == "ada":
                    val = float(onset < t)
                else:
                    ratio = bio_true[t] / base
                    if a == "inf":
                        val = c["inf0"] if v_idx == 1 else c["inf0"] * ratio**0.5 * math.exp(rng.normal(0, 0.2))
                    elif a == "alb":
                        val = c["alb0"] if v_idx == 1 else c["alb0"] + 0.35 * (1 - ratio) + rng.normal(0, 0.15)
                    else:
                        val = c["plt0"] if v_idx == 1 else c["plt0"] * ratio**0.15 * math.exp(rng.normal(0, 0.1))
                rec_rows.append({"id": pid, "time": t, "visit": v_idx, "analyte": a, "value": val, "blq": blq})
            # dosing records are data too
            for v_idx, t in enumerate(visits, start=1):
                rec_rows.append({"id": pid, "time": t, "visit": v_idx, "analyte": "dose", "value": dose_mg, "blq": False})

            truth_rows.append(
                {
                    "id": pid,
                    "CL": cl,
                    "V1": v1,
                    "BASE": base,
                    "IC50": ic50,
                    "ada_onset": onset,
                    "_visits": visits,
                    "_caves": caves,
                }
            )

    baseline = pd.DataFrame(base_rows)
    records = pd.DataFrame(rec_rows)
    truth = pd.DataFrame(truth_rows)

    # outcome: first recorded biomarker in the window
    lo, hi = config.OUTCOME_WINDOW
    win = records[(records.analyte == "bio") & (records.time >= lo) & (records.time <= hi)]
    first = win.sort_values("time").groupby("id").head(1).set_index("id")
    outcome = pd.DataFrame({"id": baseline["id"]}).set_index("id")
    outcome["study"] = baseline.set_index("id")["study"]
    outcome["outcome_time"] = first["time"]
    outcome["bio_outcome"] = first["value"]
    outcome["at_target"] = (outcome["bio_outcome"] < config.TARGET).astype(float)
    outcome.loc[outcome["bio_outcome"].isna(), "at_target"] = np.nan

    # truth at the outcome time, and the true Shapley split of -log B
    t_idx = truth.set_index("id")
    ref_caves = _reference_caves(truth)
    ref = (np.median(truth["BASE"]), np.median(truth["IC50"]), None)
    rows = []
    for pid_, r in t_idx.iterrows():
        t_out = outcome.at[pid_, "outcome_time"]
        if not np.isfinite(t_out):
            rows.append({"id": pid_})
            continue
        visits, caves = r["_visits"], r["_caves"]
        k = len(caves)
        ref_path = ref_caves[:k]

        def f(b, ic, cv, _t=t_out, _v=visits):
            return -math.log(biomarker([_t], _v, cv, b, ic)[0])

        phi = _shapley3(f, (r["BASE"], r["IC50"], caves), (ref[0], ref[1], ref_path))
        b_true = math.exp(-f(r["BASE"], r["IC50"], caves))
        z = (config.TARGET / b_true - 1.0) / config.PD["sigma_bio"]
        rows.append(
            {
                "id": pid_,
                "bio_true_outcome": b_true,
                "p_true": 0.5 * math.erfc(-z / math.sqrt(2)),
                "cave_before_outcome": float(caves[np.searchsorted(visits, t_out, side="right") - 2]),
                "phi_base": phi[0],
                "phi_ic50": phi[1],
                "phi_exposure": phi[2],
            }
        )
    extra = pd.DataFrame(rows).set_index("id")
    truth = t_idx.drop(columns=["_visits", "_caves"]).join(extra).reset_index()
    truth["ada_by_outcome"] = (truth["ada_onset"] < outcome["outcome_time"].reindex(truth["id"]).to_numpy()).astype(float)
    return baseline, records, outcome.reset_index(), truth


def _reference_caves(truth):
    """Median Cave for each interval index across patients."""
    k = min(len(c) for c in truth["_caves"])
    stack = np.vstack([c[:k] for c in truth["_caves"]])
    med = np.median(stack, axis=0)
    longest = max(len(c) for c in truth["_caves"])
    return np.concatenate([med, np.repeat(med[-1], longest - k)])
