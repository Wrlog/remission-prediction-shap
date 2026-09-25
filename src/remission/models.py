"""Model pipelines and their search spaces.

Every pipeline starts with the missingness rule and an imputer, both fitted
on whatever data the pipeline is fitted on, so inside CV they only ever see
the training fold. Tree models then go through recursive feature elimination
(RFE) with the same model type as the ranking estimator, and the number of
features kept is tuned with the other hyperparameters.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import RFE
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config

set_config(transform_output="pandas")

ALGOS = ["enet", "rf", "xgb", "cat"]
ALGO_LABELS = {"enet": "Elastic-net LR", "rf": "Random forest", "xgb": "XGBoost", "cat": "CatBoost"}
IMPUTERS = ["median", "mean", "knn", "iterative", "locf"]
IMPUTER_LABELS = {
    "median": "Per-visit median",
    "mean": "Per-visit mean",
    "knn": f"KNN (k={config.KNN_K})",
    "iterative": "Iterative (chained)",
    "locf": "Carry forward + median",
}


class HighMissingDropper(BaseEstimator, TransformerMixin):
    """Drop columns whose share of missing values in the fit data is above
    `threshold`."""

    def __init__(self, threshold=config.MISSING_DROP):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        share = X.isna().mean()
        self.keep_ = [c for c in X.columns if share[c] <= self.threshold]
        self.dropped_ = [c for c in X.columns if share[c] > self.threshold]
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.keep_, dtype=object)


class CarryForward(BaseEstimator, TransformerMixin):
    """Fill a missing visit value from the previous visit of the same
    patient (a within-row operation, nothing is learned)."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(pd.DataFrame(X).columns, dtype=object)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for name in ("alb", "inf", "plt", "bio", "conc", "ada"):
            for k in (2, 3):
                col, prev = f"v{k}_{name}", f"v{k - 1}_{name}"
                if col in X and prev in X:
                    carried = X[col].isna() & X[prev].notna()
                    X.loc[carried, col] = X.loc[carried, prev]
                    if name == "bio" and f"v{k}_bio_change" in X:
                        X.loc[carried, f"v{k}_bio_change"] = X.loc[carried, prev] - X.loc[carried, "v1_bio"]
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def make_imputer(kind="median", seed=0):
    if kind == "median":
        return SimpleImputer(strategy="median")
    if kind == "mean":
        return SimpleImputer(strategy="mean")
    if kind == "knn":
        return Pipeline([("scale", StandardScaler()), ("knn", KNNImputer(n_neighbors=config.KNN_K))])
    if kind == "iterative":
        return IterativeImputer(max_iter=10, random_state=seed)
    if kind == "locf":
        return Pipeline([("carry", CarryForward()), ("median", SimpleImputer(strategy="median"))])
    raise ValueError(kind)


def _tree(algo, seed):
    if algo == "rf":
        return RandomForestClassifier(n_estimators=80, n_jobs=1, random_state=seed)
    if algo == "xgb":
        return XGBClassifier(n_jobs=1, tree_method="exact", random_state=seed, eval_metric="logloss")
    if algo == "cat":
        return CatBoostClassifier(
            verbose=0, thread_count=1, random_seed=seed, allow_writing_files=False, boosting_type="Plain", border_count=32
        )
    raise ValueError(algo)


def make_pipeline(algo, seed=0, imputer="median"):
    steps = [("drop", HighMissingDropper()), ("impute", make_imputer(imputer, seed))]
    if algo == "enet":
        steps += [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(solver="saga", l1_ratio=0.5, C=1.0, max_iter=4000, tol=1e-3, random_state=seed)),
        ]
    else:
        steps.append(("model", RFE(_tree(algo, seed), n_features_to_select=config.RFE_KEEP[0], step=config.RFE_STEP)))
    return Pipeline(steps)


def rfe_grid(n_features):
    return sorted({min(k, n_features) for k in config.RFE_KEEP})


def search_space(algo, n_features):
    if algo == "enet":
        return {"model__C": loguniform(0.005, 5.0), "model__l1_ratio": uniform(0.0, 1.0)}
    space = {"model__n_features_to_select": rfe_grid(n_features)}
    if algo == "rf":
        space.update(
            {
                "model__estimator__max_depth": [2, 3, 4, 6, None],
                "model__estimator__min_samples_leaf": randint(2, 13),
                "model__estimator__max_features": ["sqrt", 0.33, 0.5],
                "model__estimator__class_weight": [None, "balanced"],
            }
        )
    elif algo == "xgb":
        space.update(
            {
                "model__estimator__n_estimators": randint(40, 161),
                "model__estimator__learning_rate": loguniform(0.02, 0.3),
                "model__estimator__max_depth": [1, 2, 3, 4],
                "model__estimator__subsample": uniform(0.6, 0.4),
                "model__estimator__colsample_bytree": uniform(0.5, 0.5),
                "model__estimator__min_child_weight": loguniform(1.0, 10.0),
                "model__estimator__reg_lambda": loguniform(0.5, 20.0),
            }
        )
    elif algo == "cat":
        space.update(
            {
                "model__estimator__iterations": [60, 120],
                "model__estimator__depth": [2, 3, 4],
                "model__estimator__learning_rate": loguniform(0.02, 0.3),
                "model__estimator__l2_leaf_reg": loguniform(1.0, 30.0),
            }
        )
    return space


def selected_features(fitted):
    """Columns the fitted pipeline actually uses."""
    kept = list(fitted.named_steps["drop"].keep_)
    model = fitted.named_steps["model"]
    if isinstance(model, RFE):
        return [c for c, s in zip(kept, model.support_) if s]
    return [c for c, w in zip(kept, model.coef_.ravel()) if abs(w) > 1e-8]
