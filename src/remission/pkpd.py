"""Two-compartment PK with 2 h infusions and the indirect response PD model.

The PK is solved exactly between events (dose start, infusion end, a change
in CL). Within a segment the system is linear with constant input, so the
2x2 matrix exponential has a closed form. The same routine gives the
integral of the central amount, which is what Cave needs. Cave is therefore
the exact interval average (the limit of a fine grid).
"""

import math

import numpy as np

from . import config


def typical_pk(weight, alb, inf):
    """Typical CL, V1, Q, V2 for arrays of covariates (ADA negative)."""
    p = config.PK
    wt = np.asarray(weight, dtype=float) / 70.0
    cl = p["cl"] * wt ** p["cl_wt"] * (np.asarray(alb) / 4.0) ** p["cl_alb"] * (np.asarray(inf) / 5.0) ** p["cl_inf"]
    return cl, p["v1"] * wt ** p["v1_wt"], p["q"] * wt ** p["q_wt"], p["v2"] * wt ** p["v2_wt"]


def typical_pd(inf, plt):
    p = config.PD
    base = p["base"] * (np.asarray(inf, dtype=float) / 5.0) ** p["base_inf"]
    ic50 = p["ic50"] * (np.asarray(plt, dtype=float) / 300.0) ** p["ic50_plt"]
    return base, ic50


def _segment(x1, x2, rate, dt, k10, k12, k21):
    """Propagate amounts (x1 central, x2 peripheral) over dt with a constant
    infusion rate into the central compartment. Returns the new amounts and
    the integral of x1 over the segment."""
    s = k10 + k12 + k21
    disc = math.sqrt(s * s - 4.0 * k10 * k21)
    l1, l2 = (-s + disc) / 2.0, (-s - disc) / 2.0
    # A = [[a11, a12], [a21, a22]]
    a11, a12, a21, a22 = -(k10 + k12), k21, k12, -k21
    d = l1 - l2

    def mat(g1, g2):
        # Sylvester: f(A) = (f(l1)(A - l2 I) - f(l2)(A - l1 I)) / (l1 - l2)
        return (
            (g1 * (a11 - l2) - g2 * (a11 - l1)) / d,
            (g1 * a12 - g2 * a12) / d,
            (g1 * a21 - g2 * a21) / d,
            (g1 * (a22 - l2) - g2 * (a22 - l1)) / d,
        )

    e1, e2 = math.exp(l1 * dt), math.exp(l2 * dt)
    i1, i2 = (e1 - 1.0) / l1, (e2 - 1.0) / l2
    j1, j2 = (i1 - dt) / l1, (i2 - dt) / l2
    m_e = mat(e1, e2)
    m_i = mat(i1, i2)
    m_j = mat(j1, j2)
    n1 = m_e[0] * x1 + m_e[1] * x2 + m_i[0] * rate
    n2 = m_e[2] * x1 + m_e[3] * x2 + m_i[2] * rate
    auc1 = m_i[0] * x1 + m_i[1] * x2 + m_j[0] * rate
    return n1, n2, auc1


def pk_solve(dose_times, doses, cl_times, cl_values, v1, q, v2, obs_times=(), intervals=()):
    """Concentrations at obs_times and average concentrations over intervals.

    dose_times, doses: infusion start times and amounts (mg), infusion
    length config.PK["t_inf"]. cl_times, cl_values: CL is cl_values[i] from
    cl_times[i] on (cl_times[0] must be <= the first dose). intervals: list
    of (start, end) pairs. Returns (conc array, cave array).
    """
    t_inf = config.PK["t_inf"]
    dose_times = np.asarray(dose_times, dtype=float)
    obs_times = np.asarray(obs_times, dtype=float)
    iv = np.asarray(intervals, dtype=float).reshape(-1, 2)
    events = set(dose_times.tolist()) | set((dose_times + t_inf).tolist()) | set(np.asarray(cl_times, float).tolist())
    events |= set(obs_times.tolist()) | set(iv.ravel().tolist())
    grid = np.array(sorted(events))
    cl_times = np.asarray(cl_times, dtype=float)
    cl_values = np.asarray(cl_values, dtype=float)

    x1 = x2 = 0.0
    cum = 0.0
    amount = {grid[0]: (x1, cum)}
    k12, k21 = q / v1, q / v2
    for a, b in zip(grid[:-1], grid[1:]):
        mid = 0.5 * (a + b)
        on = (dose_times <= mid) & (mid < dose_times + t_inf)
        rate = float(np.sum(np.asarray(doses)[on])) / t_inf
        idx = np.searchsorted(cl_times, mid, side="right") - 1
        cl = cl_values[max(idx, 0)]
        x1, x2, auc = _segment(x1, x2, rate, b - a, cl / v1, k12, k21)
        cum += auc
        amount[b] = (x1, cum)
    conc = np.array([amount[t][0] / v1 for t in obs_times])
    cave = np.array([(amount[e][1] - amount[s][1]) / v1 / (e - s) for s, e in iv])
    return conc, cave


def biomarker(times, interval_starts, caves, base, ic50):
    """Indirect response with Imax = 1 and Cave held constant per interval.

    interval_starts[j] is when interval j (with average concentration
    caves[j]) begins; before the first start the biomarker sits at BASE.
    """
    kout = config.PD["kout"]
    starts = np.asarray(interval_starts, dtype=float)
    caves = np.asarray(caves, dtype=float)
    bss = base * ic50 / (ic50 + caves)
    # value at each interval start
    b_start = np.empty(len(starts))
    b = base
    for j in range(len(starts)):
        if j > 0:
            dt = starts[j] - starts[j - 1]
            b = bss[j - 1] + (b - bss[j - 1]) * math.exp(-kout * dt)
        b_start[j] = b
    times = np.asarray(times, dtype=float)
    out = np.full(times.shape, float(base))
    j = np.searchsorted(starts, times, side="right") - 1
    ok = j >= 0
    jj = j[ok]
    out[ok] = bss[jj] + (b_start[jj] - bss[jj]) * np.exp(-kout * (times[ok] - starts[jj]))
    return out
