import numpy as np
import pandas as pd
from scipy.integrate import odeint
from magnetar import *


################################################################################
# Tests for functions in magnetar/funcs.py                                     #
################################################################################


def test_init_conds():
    """
Test function for init_conds in magnetar library.
    """
    Msol = 1.99e33  # Solar mass in grams

    MdiscI = 0.001  # Solar masses
    P = 1.0         # milliseconds

    Mdisc = MdiscI * Msol                 # Convert to grams
    omega = (2.0 * np.pi) / (1.0e-3 * P)  # Convert to angular frequency

    assert ((init_conds(MdiscI, P)[0] == Mdisc) &
            (init_conds(MdiscI, P)[1] == omega))


def test_odes_integrated_by_odeint():
    """
Function to test that odes is consistently being integrated by ODEINT.
    """
    expected_data = pd.read_csv("tests/test_data/odes_integrated_by_odeint.csv",
                                index_col=False)
    t = expected_data["t"]

    MdiscI = 0.001
    y0 = [MdiscI * 1.99e33, (2.0 * np.pi) / (1.0e-3 * 1.0)]
    B = 1.0
    RdiscI = 100.0
    epsilon = 1.0
    delta = 10.0

    soln = odeint(odes, y0, t, args=(B, MdiscI, RdiscI, epsilon, delta))
    Mdisc = soln[:, 0]
    omega = soln[:, 1]

    assert (np.isclose(Mdisc, expected_data["Mdisc"]).all() &
            np.isclose(omega, expected_data["omega"]).all())


def test_model_light_curve():
    expected_data = pd.read_csv("tests/test_data/model_light_curve.csv",
                                index_col=False)

    pars = [1.0, 5.0, 0.001, 100.0, 0.1, 1.0]
    t, Ltot, Lprop, Ldip = model_lc(pars)

    cond_t = np.isclose(t, expected_data["t"]).all()
    cond_Ltot = np.isclose(Ltot, expected_data["Ltot"]).all()
    cond_Lprop = np.isclose(Lprop, expected_data["Lprop"]).all()
    cond_Ldip = np.isclose(Ldip, expected_data["Ldip"]).all()

    assert cond_t & cond_Ltot & cond_Lprop & cond_Ldip


################################################################################
# Integration validation tests for ODEINT                                      #
################################################################################


def dy_dx(y, x):
    # Define a function which calculates the derivative
    return x - y


def test_first_order_odeint():
    """
Function to test the integration of a simple first order Ordinary Differential
Equation with ODEINT is consistent.
    """
    expected_data = pd.read_csv("tests/test_data/first_order_ode.csv")
    xs = expected_data["xs"]
    y0 = 1.0

    ys = odeint(dy_dx, y0, xs)
    ys = np.array(ys).flatten()

    assert np.isclose(ys, expected_data["ys"].values).all()


def dU_dx(U, x):
    """
U is a vector such that y=U[0] and z=U[1]. This function should return [y', z'].
z is the substitution z==y'
    """
    return [U[1], -2*U[1] - 2*U[0] + np.cos(2*x)]


def test_damped_harmonic_oscillator_ode():
    """
Function to test that a set of second order ordinary differential equations are
consistently integrated by ODEINT.

y'' + 2y' + 2y = cos(2x), y(0)=0, y'(0)=0
Use substition z==y':
z' + 2z + 2y = cos(2x), z(0)=y(0)=0
    """
    expected_data = pd.read_csv("tests/test_data/damped_harmonic_oscillator.csv")
    xs = expected_data["xs"]
    U0 = [0, 0]

    Us = odeint(dU_dx, U0, xs)
    ys = Us[:, 0]

    assert np.isclose(ys, expected_data["ys"]).all()


def dP_dt(P, t, a=1.0, b=1.0, c=1.0, d=1.0):
    """
AKA Lotka-Volterra equations: pair of first order, non-linear ordinary
    differential equations representing a simplified model of the change in
    populations of two species which interact via predation.

dx/dt = x*(a - b*y); dy/dt = -y*(c -d*x)

a, b, c, d are parameters which are assumed to be positive.
    """
    return [P[0]*(a - b*P[1]), -P[1]*(c - d*P[0])]


def test_predator_prey_ode():
    """
Function to test that the predator-prey set of ordinary differential equations
integrated by ODEINT are consistent.
    """
    expected_data = pd.read_csv("tests/test_data/predator_vs_prey.csv")
    ts = expected_data["ts"]
    P0 = [1.5, 1.0]

    Ps = odeint(dP_dt, P0, ts)
    prey = Ps[:, 0]
    predator = Ps[:, 1]

    assert ((np.isclose(prey, expected_data["prey"]).all()) &
            (np.isclose(predator, expected_data["predators"]).all()))


################################################################################
# Tests for functions in magnetar/fit_stats.py                                 #
################################################################################


def test_redchisq():
    """
Function to test the chi squared statistic calculation in fit_stats.py
    """
    x = np.arange(50.0)
    ymod = 2.0 * x + 3.0
    yerr = np.random.normal(0.0, scale=0.05, size=len(ymod))
    ydata = ymod + yerr

    chisq = np.sum(((ydata - ymod) / yerr) ** 2.0)

    assert redchisq(ydata, ymod, sd=yerr) == chisq


def test_akaike_info_criterion():
    """
Function to test the calculation of the corrected Akaike Information Criterion.
    """
    expected_data = pd.read_csv("tests/test_data/noisy_gaussian.csv")
    ydata = expected_data["ydata"]
    yerr = expected_data["yerr"]
    ymod = expected_data["ymod"]
    Npars = 2

    a = -1.0 * np.sum(((ydata - ymod) / yerr) ** 2.0)
    b = 2.0 * Npars
    c = ((2.0 * Npars) * (Npars + 1.0)) / (ydata.size - Npars - 1.0)
    expected_aicc = a + b + c

    assert aicc(ydata, ymod, yerr, Npars) == expected_aicc
