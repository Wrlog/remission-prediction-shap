"""Settings for the simulated cohort and the analysis.

The PK/PD numbers are the shared Drug X model used across my demo repos.
Everything else (study sizes, panels, ADA onset, the outcome window, the
search settings) is specific to this repo. Time is in days, doses in mg,
concentrations in mg/L and the biomarker in units.
"""

import math

# ---------------------------------------------------------------- PK (shared)
PK = {
    "cl": 0.30,  # L/day for 70 kg, albumin 4.0, INF 5, ADA negative
    "cl_wt": 0.75,
    "cl_alb": -1.0,
    "cl_inf": 0.10,
    "cl_ada": 1.8,  # CL multiplier once anti-drug antibodies appear
    "v1": 3.2,
    "v1_wt": 1.0,
    "q": 0.50,
    "q_wt": 0.75,
    "v2": 2.0,
    "v2_wt": 1.0,
    "cv_cl": 0.30,
    "cv_v1": 0.20,
    "cv_iov_cl": 0.15,
    "sigma_conc": 0.15,  # proportional
    "lloq": 0.5,
    "t_inf": 2.0 / 24.0,  # 2 h infusion
}

# ---------------------------------------------------------------- PD (shared)
PD = {
    "base": 600.0,
    "base_inf": 0.30,
    "ic50": 3.0,
    "ic50_plt": 0.40,
    "kout": 0.04,
    "cv_base": 0.60,
    "cv_ic50": 0.90,
    "sigma_bio": 0.25,  # proportional
}
TARGET = 200.0  # biomarker below this is "at target"


def omega(cv):
    """Log-normal SD that gives a coefficient of variation `cv`."""
    return math.sqrt(math.log(1.0 + cv**2))


# ---------------------------------------------------------------- regimen
DOSE_MG_PER_KG = 5.0
INDUCTION_DAYS = (0, 21, 42)
MAINT_START = 98  # first maintenance infusion
MAINT_EVERY = 56
LAST_DAY = 440
VISIT_JITTER = 2  # induction visits within +/- 2 days
MAINT_JITTER = 3

# The outcome: first observed biomarker in this window, below TARGET.
OUTCOME_WINDOW = (315.0, 399.0)

# ---------------------------------------------------------------- ADA onset
# Per-interval chance of antibodies appearing, from the second dose on:
# logit p = a0 + a_cave * log(Cave of previous interval / 20) + a_imm * IMM
ADA = {"a0": -2.6, "a_cave": -1.5, "a_imm": -1.2}

# ---------------------------------------------------------------- studies
# Three made-up studies. They differ in who they enrolled (weight, how
# inflamed patients were, co-medication) and in what they measured.
# Panels list which analytes are drawn at which visit (1-3 induction,
# 4 = first maintenance visit, "maint" = later maintenance visits).
STUDIES = {
    "A": {
        "n": 78,
        "wt_median": 64.0,
        "z_shift": 0.0,
        "p_imm": 0.6,
        "age": (38, 12),
        "panel": {
            "alb": (1, 2, 3, 4),
            "inf": (1, 2, 3, 4),
            "plt": (1, 2, 3, 4),
            "bio": (1, 2, 3, 4, "maint"),
            "conc": (2, 3, 4, "maint"),
            "peak": (1, 3),
            "ada": (4, "maint"),
        },
    },
    "B": {
        "n": 62,
        "wt_median": 40.0,
        "z_shift": 0.7,
        "p_imm": 0.3,
        "age": (19, 6),
        "panel": {
            "alb": (1, 2, 3, 4),
            "inf": (1, 2, 3, 4),
            "plt": (1,),
            "bio": (1, 2, 3, 4, "maint"),
            "conc": (2, 3, 4, "maint"),
            "peak": (),
            "ada": (4, "maint"),
        },
    },
    "C": {
        "n": 50,
        "wt_median": 58.0,
        "z_shift": 1.5,
        "p_imm": 0.05,
        "age": (45, 14),
        "panel": {
            "alb": (1,),
            "inf": (1, 2, 3, 4),
            "plt": (1,),
            "bio": (1, 3, 4, "maint"),
            "conc": (3, 4, "maint"),
            "peak": (),
            "ada": (2, 3, 4, "maint"),
        },
    },
}
P_MISSING = 0.08  # chance a scheduled sample is not available

SEED = 7310

# ---------------------------------------------------------------- analysis
MISSING_DROP = 0.70  # drop a column if more than this share is missing in the training fold
N_REPEATS = 3
OUTER_FOLDS = 5
INNER_FOLDS = 4
N_SEARCH = 6
RFE_STEP = 1 / 3  # share of the starting features removed per RFE step
RFE_KEEP = (6, 12, 18)
KNN_K = 7
SENS_REPEATS = 1  # repeats used for the imputation sensitivity runs
N_BOOT = 1000

QUICK = {
    "N_REPEATS": 1,
    "N_SEARCH": 2,
    "INNER_FOLDS": 3,
    "SENS_REPEATS": 1,
    "SENS_IMPUTERS": ("mean",),
    "N_BOOT": 200,
    "scale_n": 0.5,
}
