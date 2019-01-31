import numpy as np
from magnetar.funcs import init_conds
from magnetar.fit_stats import redchisq


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


def test_redchisq():
    """
Function to test the chi squared statistic calculation in fit_stats.py
    """
    x = np.arange(50.0)
    ymod = 2.0 * x + 3.0
    yerr = np.random.normal(0.75845, scale=0.25, size=len(ymod))

    ydata = np.zeros_like(ymod)
    for i in range(len(ymod)):
        ydata[i] = ymod[i] + np.random.normal(0.0, scale=yerr[i])

    chisq = np.sum(((ydata - ymod) ** 2.0) / yerr)

    assert redchisq(ydata, ymod, sd=yerr) == chisq
