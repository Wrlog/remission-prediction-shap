"""Repeated nested CV, leave-one-study-out, and the statistics on top.

The outer splits depend only on the repeat number, so every algorithm and
every ladder step sees the same folds. That makes per-fold differences
between ladder steps paired, which the corrected resampled t-test needs.
"""

import math
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from . import config
from .models import make_pipeline, search_space, selected_features

# cost weights so the slowest jobs start first
_COST = {"cat": 4.0, "rf": 3.0, "xgb": 2.0, "enet": 1.0}


def fit_search(X, y, algo, seed, n_search, inner_folds, imputer="median"):
    search = RandomizedSearchCV(
        make_pipeline(algo, seed, imputer),
        search_space(algo, X.shape[1]),
        n_iter=n_search,
        scoring="roc_auc",
        cv=StratifiedKFold(inner_folds, shuffle=True, random_state=seed),
        refit=True,
        random_state=seed,
        n_jobs=1,
        error_score="raise",
    )
    search.fit(X, y)
    return search


def _fold_job(key, X, y, train, test, algo, seed, n_search, inner_folds, imputer):
    t0 = time.time()
    search = fit_search(X.iloc[train], y[train], algo, seed, n_search, inner_folds, imputer)
    p = search.predict_proba(X.iloc[test])[:, 1]
    best = search.best_estimator_
    return {
        **key,
        "auc": roc_auc_score(y[test], p),
        "inner_auc": search.best_score_,
        "test_idx": test,
        "pred": p,
        "n_train": len(train),
        "n_test": len(test),
        "selected": selected_features(best),
        "dropped": list(best.named_steps["drop"].dropped_),
        "columns": list(X.columns),
        "params": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v)) for k, v in search.best_params_.items()},
        "seconds": time.time() - t0,
    }


def outer_folds(y, repeat, n_folds=config.OUTER_FOLDS):
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=config.SEED + 101 * repeat)
    return list(skf.split(np.zeros(len(y)), y))


def nested_cv_jobs(tag, X, y, algos, n_repeats, n_search, inner_folds=config.INNER_FOLDS, imputer="median", extra=None):
    """Job list for repeated nested CV on one feature matrix."""
    jobs = []
    for r in range(n_repeats):
        for f, (train, test) in enumerate(outer_folds(y, r)):
            for algo in algos:
                key = {"tag": tag, "algo": algo, "repeat": r, "fold": f, "imputer": imputer, **(extra or {})}
                seed = 1000 * r + f
                jobs.append((_COST[algo], delayed(_fold_job)(key, X, y, train, test, algo, seed, n_search, inner_folds, imputer)))
    return jobs


def _loso_job(key, X, y, study, held, algo, seed, n_search, inner_folds):
    train, test = np.where(study != held)[0], np.where(study == held)[0]
    search = fit_search(X.iloc[train], y[train], algo, seed, n_search, inner_folds)
    p = search.predict_proba(X.iloc[test])[:, 1]
    return {**key, "held_out": held, "auc": roc_auc_score(y[test], p), "n_test": len(test), "prevalence": float(y[test].mean())}


def loso_jobs(tag, X, y, study, algos, n_search, inner_folds=config.INNER_FOLDS):
    jobs = []
    for held in sorted(set(study)):
        for algo in algos:
            key = {"tag": tag, "algo": algo}
            jobs.append((_COST[algo], delayed(_loso_job)(key, X, y, study, held, algo, 17, n_search, inner_folds)))
    return jobs


def run_jobs(jobs, n_jobs):
    jobs = sorted(jobs, key=lambda j: -j[0])
    return Parallel(n_jobs=n_jobs, verbose=0)(j for _, j in jobs)


# ---------------------------------------------------------------- tables


def fold_table(results):
    keep = ["tag", "algo", "imputer", "repeat", "fold", "auc", "inner_auc", "n_train", "n_test", "seconds"]
    return pd.DataFrame([{k: r[k] for k in keep if k in r} for r in results])


def oof_table(results, ids, y):
    rows = []
    for r in results:
        for i, p in zip(r["test_idx"], r["pred"]):
            rows.append({"tag": r["tag"], "algo": r["algo"], "imputer": r["imputer"], "repeat": r["repeat"], "id": ids[i], "y": y[i], "pred": p})
    return pd.DataFrame(rows)


def selection_table(results):
    rows = []
    for r in results:
        sel, drop = set(r["selected"]), set(r["dropped"])
        for c in r["columns"]:
            rows.append({"tag": r["tag"], "algo": r["algo"], "repeat": r["repeat"], "fold": r["fold"], "feature": c, "selected": int(c in sel), "dropped": int(c in drop)})
    return pd.DataFrame(rows)


def summarize(folds, oof):
    """Grand mean and SD of the per-repeat mean AUROC (seed sensitivity,
    not a confidence interval), plus the pooled out-of-fold AUROC."""
    run = folds.groupby(["tag", "algo", "imputer", "repeat"]).auc.mean().rename("run_auc").reset_index()
    pooled = oof.groupby(["tag", "algo", "imputer", "repeat"]).apply(lambda d: roc_auc_score(d.y, d.pred), include_groups=False).rename("pooled_auc").reset_index()
    run = run.merge(pooled, on=["tag", "algo", "imputer", "repeat"])
    fold_sd = folds.groupby(["tag", "algo", "imputer"]).auc.std().rename("fold_sd")
    out = run.groupby(["tag", "algo", "imputer"]).agg(
        auc_mean=("run_auc", "mean"), auc_run_sd=("run_auc", "std"), pooled_auc=("pooled_auc", "mean"), n_repeats=("run_auc", "size")
    )
    return out.join(fold_sd).reset_index()


def corrected_ttest(d, n_train, n_test):
    """Nadeau and Bengio corrected resampled t-test on paired per-fold
    differences from repeated k-fold CV."""
    d = np.asarray(d, dtype=float)
    j = len(d)
    var = np.var(d, ddof=1)
    if var == 0:
        return float(d.mean()), float("nan"), float("nan")
    t = d.mean() / math.sqrt((1.0 / j + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), j - 1)
    return float(d.mean()), float(t), float(p)


def contrast_table(folds, contrasts):
    rows = []
    for a, b in contrasts:
        for algo, g in folds.groupby("algo"):
            fa = g[g.tag == a].set_index(["repeat", "fold"]).auc
            fb = g[g.tag == b].set_index(["repeat", "fold"]).auc
            d = (fb - fa.reindex(fb.index)).dropna()
            n_tr = g.n_train.mean()
            n_te = g.n_test.mean()
            mean, t, p = corrected_ttest(d.to_numpy(), n_tr, n_te)
            naive_p = stats.ttest_1samp(d.to_numpy(), 0).pvalue if d.std() > 0 else float("nan")
            rows.append({"from": a, "to": b, "algo": algo, "delta_mean": mean, "delta_sd": d.std(), "t_corrected": t, "p_corrected": p, "p_naive": naive_p, "n_folds": len(d)})
    return pd.DataFrame(rows)


def per_fold_deltas(folds, contrasts):
    rows = []
    for a, b in contrasts:
        for algo, g in folds.groupby("algo"):
            fa = g[g.tag == a].set_index(["repeat", "fold"]).auc
            fb = g[g.tag == b].set_index(["repeat", "fold"]).auc
            for (r, f), v in (fb - fa.reindex(fb.index)).dropna().items():
                rows.append({"contrast": f"{a}->{b}", "algo": algo, "repeat": r, "fold": f, "delta": v})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- calibration and decisions


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_slope_intercept(y, p):
    """Slope from logistic recalibration on the logit of p; intercept is
    calibration-in-the-large (logit of p as an offset, slope fixed at 1)."""
    lp = _logit(p)
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(lp[:, None], y).coef_[0, 0]

    def score(a):
        return np.mean(1 / (1 + np.exp(-(a + lp)))) - np.mean(y)

    intercept = brentq(score, -10, 10)
    return float(slope), float(intercept)


def calibration_curve(y, p, n_bins=5):
    q = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    q[0], q[-1] = -np.inf, np.inf
    b = np.digitize(p, q[1:-1])
    return pd.DataFrame({"bin": b, "y": y, "p": p}).groupby("bin").agg(mean_pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")).reset_index()


def net_benefit(y_event, p_event, thresholds):
    y_event = np.asarray(y_event)
    n = len(y_event)
    rows = []
    prev = y_event.mean()
    for pt in thresholds:
        treat = p_event >= pt
        tp = np.sum(treat & (y_event == 1)) / n
        fp = np.sum(treat & (y_event == 0)) / n
        w = pt / (1 - pt)
        rows.append({"threshold": pt, "model": tp - fp * w, "treat_all": prev - (1 - prev) * w, "treat_none": 0.0})
    return pd.DataFrame(rows)


def bootstrap_auc(y, p, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        out.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(out, [2.5, 97.5])


def pr_summary(y, p):
    return {"auprc": float(average_precision_score(y, p)), "prevalence": float(np.mean(y))}
