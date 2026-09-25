"""Individual PK/PD estimates (MAP) from the data seen up to a cutoff.

Sequential, as in a standard PK-then-PD workflow: CL and V1 are fitted to
the drug levels first, then BASE and IC50 are fitted to the biomarker with
Cave per interval computed from those PK estimates.

The prior is the population model in config (treated here as a published
model, so nothing is fitted on this cohort and there is no train/test
issue). The fitting model has no IOV, and it does not know when antibodies
appeared: if a test is positive, onset is placed halfway between that test
and the visit before it. Residuals are on the log scale.
"""

import math

import numpy as np
from scipy.optimize import minimize

from . import config
from .pkpd import biomarker, pk_solve, typical_pd, typical_pk

W_PK = (config.omega(config.PK["cv_cl"]), config.omega(config.PK["cv_v1"]))
W_PD = (config.omega(config.PD["cv_base"]), config.omega(config.PD["cv_ic50"]))
SD_CONC = config.omega(config.PK["sigma_conc"])
SD_BIO = config.omega(config.PD["sigma_bio"])


def _first(rec, analyte, visit):
    v = rec.loc[(rec.analyte == analyte) & (rec.visit == visit), "value"]
    return float(v.iloc[0]) if len(v) else np.nan


def _ada_schedule(rec, dose_t, cl):
    """CL schedule under the fitting model's ADA assumption."""
    ada = rec[rec.analyte == "ada"].sort_values("time")
    pos = ada[ada.value > 0.5]
    if pos.empty:
        return np.array([dose_t[0]]), np.array([cl])
    t_pos = float(pos.time.iloc[0])
    before = dose_t[dose_t < t_pos]
    t_prev = float(before[-1]) if len(before) else 0.0
    onset = 0.5 * (t_prev + t_pos)
    return np.array([dose_t[0], onset]), np.array([cl, cl * config.PK["cl_ada"]])


def fit_block(weight, rec, cutoff):
    """MAP estimates from `rec`, one patient's records up to `cutoff`.

    `rec` must already be censored; the function also refuses anything
    later than the cutoff so it cannot be misused.
    """
    if (rec.time > cutoff).any():
        raise ValueError("records after the cutoff were passed to fit_block")
    alb0, inf0, plt0 = _first(rec, "alb", 1), _first(rec, "inf", 1), _first(rec, "plt", 1)
    tcl, tv1, q, v2 = (float(x) for x in typical_pk(weight, alb0, inf0))
    tbase, tic50 = (float(x) for x in typical_pd(inf0, plt0))

    doses = rec[(rec.analyte == "dose") & (rec.time < cutoff)].sort_values("time")
    dose_t, dose_amt = doses.time.to_numpy(), doses.value.to_numpy()
    iv = list(zip(dose_t, np.append(dose_t[1:], cutoff)))

    conc = rec[rec.analyte.isin(["conc", "peak"]) & (rec.value > 0)]
    t_c, log_y = conc.time.to_numpy(), np.log(conc.value.to_numpy())

    def pk_pred(eta, obs_t=(), intervals=()):
        cl = tcl * math.exp(eta[0])
        cl_t, cl_v = _ada_schedule(rec, dose_t, cl)
        return pk_solve(dose_t, dose_amt, cl_t, cl_v, tv1 * math.exp(eta[1]), q, v2, obs_t, intervals)

    def pk_obj(eta):
        prior = 0.5 * ((eta[0] / W_PK[0]) ** 2 + (eta[1] / W_PK[1]) ** 2)
        if len(t_c) == 0:
            return prior
        pred, _ = pk_pred(eta, t_c)
        r = (log_y - np.log(np.maximum(pred, 1e-8))) / SD_CONC
        return prior + 0.5 * float(r @ r)

    eta_pk = np.zeros(2)
    if len(t_c):
        eta_pk = minimize(pk_obj, np.zeros(2), method="L-BFGS-B", bounds=[(-2.5, 2.5)] * 2).x
    _, caves = pk_pred(eta_pk, (), iv)

    bio = rec[(rec.analyte == "bio") & (rec.value > 0)]
    t_b, log_b = bio.time.to_numpy(), np.log(bio.value.to_numpy())

    def pd_obj(eta):
        prior = 0.5 * ((eta[0] / W_PD[0]) ** 2 + (eta[1] / W_PD[1]) ** 2)
        pred = biomarker(t_b, dose_t, caves, tbase * math.exp(eta[0]), tic50 * math.exp(eta[1]))
        r = (log_b - np.log(pred)) / SD_BIO
        return prior + 0.5 * float(r @ r)

    eta_pd = minimize(pd_obj, np.zeros(2), method="L-BFGS-B", bounds=[(-4.0, 4.0)] * 2).x
    return {
        "CL": tcl * math.exp(eta_pk[0]),
        "V1": tv1 * math.exp(eta_pk[1]),
        "BASE": tbase * math.exp(eta_pd[0]),
        "IC50": tic50 * math.exp(eta_pd[1]),
        "Cave": float(caves[-1]),
        "eta_CL": float(eta_pk[0]),
        "eta_V1": float(eta_pk[1]),
        "eta_BASE": float(eta_pd[0]),
        "eta_IC50": float(eta_pd[1]),
        "n_conc": int(len(t_c)),
        "n_bio": int(len(t_b)),
    }
