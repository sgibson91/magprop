import numpy as np
from magnetar.funcs import init_conds
from magnetar.fit_stats import redchisq


def geomean(list_a, list_b):
    """
Function to calculate the geometric mean of pairs of numbers.

    :param list_a: list or array of positive uncertainties
    :param list_b: list or array of negative uncertainties
    :return: list or array of the geometric mean of the pairs of numbers between
             list_a and list_b
    """
    return [np.prod([i, np.abs(j)]) ** 0.5 for i, j in zip(list_a, list_b)]


def test_geomean():
    """
Test function for geomean function.
    """
    list_a = [1.0, 2.0, 3.0]
    list_b = [4.0, 5.0, 6.0]

    ans = []
    for x, y in zip(list_a, list_b):
        ans.append((x * y) ** 0.5)

    assert geomean(list_a, list_b) == ans


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
