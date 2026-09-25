"""SHAP for the refitted top configuration of each algorithm, and the check
against the known simulation mechanism."""

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.feature_selection import RFE

# Which part of the true mechanism each feature mostly carries. The
# biomarker and labs after baseline already mix all three drivers, so they
# get their own group.
MECH_GROUPS = {
    "exposure": ["weight", "imm", "v1_alb", "v2_conc", "v3_conc", "v2_ada", "v3_ada"]
    + [f"v{k}_{n}" for k in (1, 2, 3) for n in ("CL", "V1", "Cave")],
    "baseline level": ["activity", "v1_inf", "v1_bio"] + [f"v{k}_BASE" for k in (1, 2, 3)],
    "drug sensitivity": ["v1_plt"] + [f"v{k}_IC50" for k in (1, 2, 3)],
    "early response": [f"v{k}_{n}" for k in (2, 3) for n in ("bio", "bio_change", "alb", "inf", "plt")],
    "no effect": ["age", "female"],
}
TRUTH_FOR_GROUP = {"exposure": "phi_exposure", "baseline level": "phi_base", "drug sensitivity": "phi_ic50"}


def group_of(col):
    for g, cols in MECH_GROUPS.items():
        if col in cols:
            return g
    raise KeyError(col)


def shap_matrix(pipe, X):
    """SHAP values for every column of X (zero for columns the fitted
    pipeline dropped or did not select). Tree models: TreeExplainer on the
    RFE-selected features. Logistic regression: LinearExplainer on the
    scaled, imputed features. Returns (DataFrame, scale label)."""
    Xi = pipe.named_steps["impute"].transform(pipe.named_steps["drop"].transform(X))
    model = pipe.named_steps["model"]
    out = pd.DataFrame(0.0, index=X.index, columns=X.columns)
    if isinstance(model, RFE):
        cols = [c for c, s in zip(Xi.columns, model.support_) if s]
        est = model.estimator_
        Xs = Xi[cols].to_numpy()
        name = type(est).__name__
        if name == "XGBClassifier":
            try:
                vals = shap.TreeExplainer(est).shap_values(Xs)
            except Exception:
                import xgboost as xgb

                vals = est.get_booster().predict(xgb.DMatrix(Xs), pred_contribs=True)[:, :-1]
            scale = "log-odds"
        else:
            vals = shap.TreeExplainer(est).shap_values(Xs)
            scale = "probability" if name == "RandomForestClassifier" else "log-odds"
        vals = np.asarray(vals)
        if vals.ndim == 3:
            vals = vals[:, :, 1]
        out[cols] = vals
    else:
        Xs = pipe.named_steps["scale"].transform(Xi)
        expl = shap.LinearExplainer(model, shap.maskers.Independent(Xs.to_numpy(), max_samples=len(Xs)))
        out[list(Xs.columns)] = expl.shap_values(Xs.to_numpy())
        scale = "log-odds"
    return out, scale


def importance(sv):
    imp = sv.abs().mean()
    return pd.DataFrame({"feature": imp.index, "mean_abs_shap": imp.to_numpy(), "share": (imp / imp.sum()).to_numpy()})


def consensus(importances):
    """importances: {algo: importance table}. Share of total |SHAP| per
    feature for each algorithm, the mean share, and how many algorithms put
    the feature in their top 10."""
    wide = pd.DataFrame({a: t.set_index("feature")["share"] for a, t in importances.items()})
    ranks = wide.rank(ascending=False)
    wide["mean_share"] = wide[list(importances)].mean(axis=1)
    wide["n_top10"] = (ranks <= 10).sum(axis=1)
    return wide.sort_values("mean_share", ascending=False).reset_index(names="feature")


def grouped_vs_truth(sv, truth):
    """Sum SHAP within each mechanism group and compare with the true
    Shapley split of -log(biomarker) at the outcome visit."""
    t = truth.set_index("id").loc[sv.index]
    grouped = pd.DataFrame({g: sv[[c for c in cols if c in sv]].sum(axis=1) for g, cols in MECH_GROUPS.items()})
    total_true = t[["phi_exposure", "phi_base", "phi_ic50"]].sum(axis=1)
    rows = []
    for g in MECH_GROUPS:
        share = grouped[g].abs().mean() / grouped.abs().mean().sum()
        if g in TRUTH_FOR_GROUP:
            ref = t[TRUTH_FOR_GROUP[g]]
            true_share = ref.abs().mean() / t[list(TRUTH_FOR_GROUP.values())].abs().mean().sum()
        elif g == "early response":
            ref, true_share = total_true, np.nan
        else:
            ref, true_share = None, 0.0
        rho = spearmanr(grouped[g], ref).statistic if ref is not None and grouped[g].std() > 0 else np.nan
        rows.append({"group": g, "shap_share": share, "true_share": true_share, "spearman": rho, "compared_with": "total" if g == "early response" else TRUTH_FOR_GROUP.get(g, "")})
    return pd.DataFrame(rows), grouped
