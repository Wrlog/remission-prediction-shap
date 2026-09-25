"""The closed-form PK and PD against a numerical ODE solve."""

import numpy as np
from scipy.integrate import solve_ivp

from remission import config
from remission.pkpd import biomarker, pk_solve, typical_pd, typical_pk


def _ode_conc(dose_t, dose, cl_t, cl_v, v1, q, v2, t_end):
    t_inf = config.PK["t_inf"]

    def cl_at(t):
        return cl_v[np.searchsorted(cl_t, t, side="right") - 1]

    def rhs(t, x):
        rate = sum(d / t_inf for s, d in zip(dose_t, dose) if s <= t < s + t_inf)
        cl = cl_at(t)
        return [rate - cl / v1 * x[0] - q / v1 * x[0] + q / v2 * x[1], q / v1 * x[0] - q / v2 * x[1]]

    grid = np.linspace(0, t_end, 20001)
    sol = solve_ivp(rhs, (0, t_end), [0, 0], t_eval=grid, max_step=t_inf / 4, rtol=1e-8, atol=1e-10)
    return grid, sol.y[0] / v1


def test_pk_matches_ode_with_a_clearance_change():
    cl, v1, q, v2 = (float(x) for x in typical_pk(60.0, 3.6, 8.0))
    dose_t = np.array([0.0, 21.0, 42.0])
    dose = np.full(3, 300.0)
    cl_t, cl_v = np.array([0.0, 30.0]), np.array([cl, cl * 1.8])
    obs = np.array([1.0, 10.0, 21.0, 29.0, 35.0, 42.0, 56.0])
    conc, cave = pk_solve(dose_t, dose, cl_t, cl_v, v1, q, v2, obs, [(0, 21), (21, 42), (42, 70)])
    grid, ref = _ode_conc(dose_t, dose, cl_t, cl_v, v1, q, v2, 70.0)
    np.testing.assert_allclose(conc, np.interp(obs, grid, ref), rtol=2e-3)
    for (a, b), c in zip([(0, 21), (21, 42), (42, 70)], cave):
        m = (grid >= a) & (grid <= b)
        np.testing.assert_allclose(c, np.trapezoid(ref[m], grid[m]) / (b - a), rtol=2e-3)


def test_sample_at_dose_time_is_predose():
    cl, v1, q, v2 = (float(x) for x in typical_pk(70.0, 4.0, 5.0))
    before, _ = pk_solve([0.0, 21.0], [350.0, 350.0], [0.0], [cl], v1, q, v2, [21.0])
    only_first, _ = pk_solve([0.0], [350.0], [0.0], [cl], v1, q, v2, [21.0])
    np.testing.assert_allclose(before, only_first)


def test_typical_values_at_reference_covariates():
    cl, v1, q, v2 = typical_pk(70.0, 4.0, 5.0)
    assert np.isclose(cl, 0.30) and np.isclose(v1, 3.2) and np.isclose(q, 0.50) and np.isclose(v2, 2.0)
    base, ic50 = typical_pd(5.0, 300.0)
    assert np.isclose(base, 600.0) and np.isclose(ic50, 3.0)


def test_biomarker_goes_to_the_inhibited_steady_state():
    t = np.array([0.0, 400.0])
    b = biomarker(t, [0.0], [9.0], 600.0, 3.0)
    assert np.isclose(b[0], 600.0)
    assert np.isclose(b[1], 600.0 * 3.0 / (3.0 + 9.0), rtol=1e-4)


def test_biomarker_matches_ode():
    starts, caves = [0.0, 21.0, 42.0], [30.0, 12.0, 20.0]
    base, ic50, kout = 800.0, 4.0, config.PD["kout"]

    def rhs(t, b):
        j = np.searchsorted(starts, t, side="right") - 1
        c = caves[j]
        return [kout * base * (1 - c / (ic50 + c)) - kout * b[0]]

    t_obs = np.array([5.0, 21.0, 30.0, 60.0])
    sol = solve_ivp(rhs, (0, 60), [base], t_eval=t_obs, max_step=0.5, rtol=1e-9)
    np.testing.assert_allclose(biomarker(t_obs, starts, caves, base, ic50), sol.y[0], rtol=1e-4)
