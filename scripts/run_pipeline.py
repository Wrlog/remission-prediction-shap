"""Simulate the cohort, build features, run every validation, fit the SHAP
models and write the result tables to results/.

    python scripts/run_pipeline.py              # full run
    python scripts/run_pipeline.py --quick      # small cohort and search, for CI
"""

import argparse
import json
import os
import platform
import time
import warnings
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import delayed  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_auc_score  # noqa: E402

from remission import config  # noqa: E402
from remission import explain  # noqa: E402
from remission import features as F  # noqa: E402
from remission import validation as V  # noqa: E402
from remission.models import ALGOS, IMPUTERS  # noqa: E402
from remission.simulate import simulate_cohort  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*max_iter.*")


def _refit_job(algo, X, y, n_search, inner_folds):
    search = V.fit_search(X, y, algo, seed=4242, n_search=n_search, inner_folds=inner_folds)
    return {"kind": "refit", "algo": algo, "pipe": search.best_estimator_, "inner_auc": search.best_score_, "params": {k: str(v) for k, v in search.best_params_.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    s = {k: getattr(config, k) for k in ("N_REPEATS", "N_SEARCH", "INNER_FOLDS", "SENS_REPEATS", "N_BOOT")}
    s["SENS_IMPUTERS"] = tuple(IMPUTERS[1:])
    scale_n = 1.0
    if args.quick:
        s.update({k: v for k, v in config.QUICK.items() if k in s})
        scale_n = config.QUICK["scale_n"]
    out, data_dir = Path(args.out), Path(args.data)
    out.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---------------------------------------------------------------- data
    baseline, records, outcome, truth = simulate_cohort(args.seed, scale_n)
    for name, df in [("baseline", baseline), ("records", records), ("outcome", outcome), ("truth", truth)]:
        df.to_csv(data_dir / f"{name}.csv", index=False)
    X_all, est = F.build_features(baseline, records, n_jobs=args.jobs, return_estimates=True)
    keep = outcome.set_index("id").at_target.reindex(X_all.index).notna()
    n_excluded = int((~keep).sum())
    X_all = X_all[keep]
    ids = X_all.index.to_numpy()
    y = outcome.set_index("id").loc[ids, "at_target"].to_numpy().astype(int)
    study = baseline.set_index("id").loc[ids, "study"].to_numpy()
    tt = truth.set_index("id").loc[ids]
    X_all.to_csv(out / "features.csv")
    print(f"{len(ids)} patients ({n_excluded} without an outcome value), at target {y.mean():.3f}  [{time.time() - t0:.0f}s]")

    # cohort and missingness tables
    coh = pd.DataFrame({"study": study, "y": y, "ada": tt["ada_by_outcome"].to_numpy(), "weight": X_all["weight"].to_numpy(), "bio0": np.exp(X_all["v1_bio"]).to_numpy()})
    coh_sum = coh.groupby("study").agg(n=("y", "size"), at_target=("y", "mean"), ada_by_outcome=("ada", "mean"), weight_median=("weight", "median"), baseline_bio_median=("bio0", "median"))
    coh_sum.loc["All"] = [len(y), y.mean(), coh.ada.mean(), coh.weight.median(), coh.bio0.median()]
    coh_sum.reset_index().to_csv(out / "cohort_summary.csv", index=False)
    miss = X_all.isna().groupby(study).mean().T
    miss["All"] = X_all.isna().mean()
    miss.reset_index(names="feature").to_csv(out / "missingness_by_study.csv", index=False)

    # biomarker trajectories for the cohort figure (observed values only)
    bio = records[(records.analyte == "bio") & records.id.isin(ids)].merge(outcome[["id", "study", "at_target"]], on="id")
    bio[["id", "study", "time", "value", "at_target"]].to_csv(out / "biomarker_records.csv", index=False)

    # MAP estimates against the truth
    # MAP estimates against the truth; shrinkage is 1 - SD(eta hat) / omega
    rows = []
    for block, e in est.items():
        e = e.loc[ids]
        for p, cv in [("CL", config.PK["cv_cl"]), ("V1", config.PK["cv_v1"]), ("BASE", config.PD["cv_base"]), ("IC50", config.PD["cv_ic50"])]:
            r = np.corrcoef(np.log(e[p]), np.log(tt[p]))[0, 1]
            shrink = 1 - e[f"eta_{p}"].std() / config.omega(cv)
            rows.append({"block": block, "parameter": p, "r_log": r, "shrinkage": shrink})
    pd.DataFrame(rows).to_csv(out / "map_vs_truth.csv", index=False)
    pd.concat({b: e.loc[ids] for b, e in est.items()}, names=["block", "id"]).to_csv(out / "map_estimates.csv")

    # ---------------------------------------------------------------- jobs
    jobs = []
    for step in list(F.LADDER) + list(F.CHECK_STEPS):
        Xs = X_all[F.ladder_columns(step)]
        jobs += V.nested_cv_jobs(step, Xs, y, ALGOS, s["N_REPEATS"], s["N_SEARCH"], s["INNER_FOLDS"], extra={"kind": "cv"})
        if step in F.LADDER:
            jobs += V.loso_jobs(step, Xs, y, study, ALGOS, s["N_SEARCH"], s["INNER_FOLDS"])
    X5 = X_all[F.ladder_columns("L5")]
    for imp in s["SENS_IMPUTERS"]:
        jobs += V.nested_cv_jobs("L5", X5, y, ALGOS, s["SENS_REPEATS"], s["N_SEARCH"], s["INNER_FOLDS"], imputer=imp, extra={"kind": "sens"})
    ind = X5.isna().astype(float)
    ind = ind.loc[:, ind.std() > 0].add_prefix("missing_")
    study_x = pd.get_dummies(pd.Series(study, index=X5.index), prefix="study").astype(float)
    for tag, Xc in [("missing_only", ind), ("study_only", study_x)]:
        jobs += V.nested_cv_jobs(tag, Xc, y, ["enet", "xgb"], s["N_REPEATS"], s["N_SEARCH"], s["INNER_FOLDS"], extra={"kind": "check"})
    for algo in ALGOS:
        jobs.append((10.0, delayed(_refit_job)(algo, X5, y, s["N_SEARCH"], s["INNER_FOLDS"])))
    print(f"running {len(jobs)} jobs on {args.jobs} workers")
    results = V.run_jobs(jobs, args.jobs)
    print(f"jobs done [{time.time() - t0:.0f}s]")

    cv = [r for r in results if r.get("kind") in ("cv", "sens", "check")]
    loso = [r for r in results if "held_out" in r]
    refits = {r["algo"]: r for r in results if r.get("kind") == "refit"}

    # ---------------------------------------------------------------- CV tables
    folds = V.fold_table(cv)
    folds["kind"] = [r["kind"] for r in cv]
    oof = V.oof_table(cv, ids, y)
    folds.to_csv(out / "cv_folds.csv", index=False)
    oof.to_csv(out / "cv_oof.csv", index=False)
    summary = V.summarize(folds, oof)
    summary.to_csv(out / "cv_summary.csv", index=False)

    main_folds = folds[(folds.kind == "cv")]
    V.contrast_table(main_folds, F.CONTRASTS + [("L4r", "L4")]).to_csv(out / "ladder_contrasts.csv", index=False)
    V.per_fold_deltas(main_folds, F.CONTRASTS).to_csv(out / "ladder_deltas.csv", index=False)

    sel = V.selection_table([r for r in cv if r["kind"] == "cv"])
    freq = sel.groupby(["tag", "algo", "feature"]).agg(selected=("selected", "mean"), dropped=("dropped", "mean")).reset_index()
    freq.to_csv(out / "selection_frequency.csv", index=False)
    pd.DataFrame([{"tag": r["tag"], "algo": r["algo"], "repeat": r["repeat"], "fold": r["fold"], **r["params"]} for r in cv if r["kind"] == "cv"]).to_csv(out / "best_params.csv", index=False)

    lo = pd.DataFrame(loso)
    lo.to_csv(out / "loso.csv", index=False)

    # imputation sensitivity: compare on the same repeats
    sens = folds[(folds.tag == "L5") & (folds.kind.isin(["cv", "sens"])) & (folds.repeat < s["SENS_REPEATS"])]
    sens_tab = sens.groupby(["imputer", "algo"]).auc.agg(["mean", "std", "size"]).reset_index()
    sens_tab.to_csv(out / "imputation_sensitivity.csv", index=False)

    # ---------------------------------------------------------------- champion diagnostics
    main = summary[summary.tag.isin(list(F.LADDER)) & (summary.imputer == "median")]
    champ = main.sort_values("auc_mean", ascending=False).iloc[0]
    cstep, calgo = champ.tag, champ.algo

    cal_rows = []
    for (tag, algo), g in oof[oof.tag.isin(list(F.LADDER)) & (oof.imputer == "median")].groupby(["tag", "algo"]):
        vals = [V.calibration_slope_intercept(d.y.to_numpy(), d.pred.to_numpy()) for _, d in g.groupby("repeat")]
        cal_rows.append({"tag": tag, "algo": algo, "slope": np.mean([v[0] for v in vals]), "intercept": np.mean([v[1] for v in vals])})
    pd.DataFrame(cal_rows).to_csv(out / "calibration_stats.csv", index=False)

    def averaged(tag, algo):
        g = oof[(oof.tag == tag) & (oof.algo == algo) & (oof.imputer == "median")]
        return g.groupby("id").agg(y=("y", "first"), pred=("pred", "mean")).loc[ids]

    avg = averaged(cstep, calgo)
    lo_ci, hi_ci = V.bootstrap_auc(avg.y.to_numpy(), avg.pred.to_numpy(), s["N_BOOT"], seed=1)
    V.calibration_curve(avg.y.to_numpy(), avg.pred.to_numpy()).to_csv(out / "calibration_curve.csv", index=False)

    pr_rows = []
    for tag in ("L1", cstep):
        a = averaged(tag, calgo)
        prec, rec, _ = precision_recall_curve(a.y, a.pred)
        pr_rows.append(pd.DataFrame({"tag": tag, "algo": calgo, "precision": prec, "recall": rec, **V.pr_summary(a.y, a.pred)}))
    pd.concat(pr_rows).to_csv(out / "pr_curve.csv", index=False)

    # Decision curve. The action is to intensify dosing for a patient the
    # model says will not reach target, so the event is "not at target".
    thr = np.round(np.arange(0.05, 0.61, 0.01), 2)
    dc = []
    for tag in ("L1", cstep):
        a = averaged(tag, calgo)
        d = V.net_benefit(1 - a.y.to_numpy(), 1 - a.pred.to_numpy(), thr)
        d["tag"] = tag
        dc.append(d)
    pd.concat(dc).to_csv(out / "decision_curve.csv", index=False)

    ceiling = roc_auc_score(y, tt["p_true"].to_numpy())
    champion = {
        "step": cstep,
        "algo": calgo,
        "auc_mean": champ.auc_mean,
        "auc_run_sd": champ.auc_run_sd,
        "pooled_auc": champ.pooled_auc,
        "averaged_oof_auc": roc_auc_score(avg.y, avg.pred),
        "bootstrap_ci": [lo_ci, hi_ci],
        "ceiling_auc_true_probability": ceiling,
    }

    # ---------------------------------------------------------------- SHAP
    imps, shap_groups = {}, []
    for algo, r in refits.items():
        sv, scale = explain.shap_matrix(r["pipe"], X5)
        sv.to_csv(out / f"shap_values_{algo}.csv")
        imp = explain.importance(sv)
        imp.insert(0, "algo", algo)
        imp["scale"] = scale
        imps[algo] = imp
        g, grouped = explain.grouped_vs_truth(sv, truth)
        g.insert(0, "algo", algo)
        shap_groups.append(g)
        grouped.to_csv(out / f"shap_grouped_{algo}.csv")
    pd.concat(imps.values()).to_csv(out / "shap_importance.csv", index=False)
    explain.consensus(imps).to_csv(out / "shap_consensus.csv", index=False)
    pd.concat(shap_groups).to_csv(out / "shap_vs_truth.csv", index=False)
    tt[["CL", "V1", "BASE", "IC50", "phi_exposure", "phi_base", "phi_ic50", "p_true", "bio_true_outcome"]].to_csv(out / "truth_components.csv")
    pd.DataFrame([{"algo": a, "inner_auc": r["inner_auc"], **r["params"]} for a, r in refits.items()]).to_csv(out / "refit_params.csv", index=False)

    info = {
        "quick": args.quick,
        "seed": args.seed,
        "n_patients": int(len(ids)),
        "n_excluded_no_outcome": n_excluded,
        "prevalence": float(y.mean()),
        "settings": {**s, "OUTER_FOLDS": config.OUTER_FOLDS, "RFE_STEP": config.RFE_STEP, "RFE_KEEP": config.RFE_KEEP, "MISSING_DROP": config.MISSING_DROP},
        "champion": champion,
        "n_jobs": len(jobs),
        "workers": args.jobs,
        "runtime_seconds": round(time.time() - t0, 1),
        "python": platform.python_version(),
        "versions": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "xgboost", "catboost", "shap", "matplotlib"]},
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=float))
    print(json.dumps(champion, indent=2, default=float))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
