import numpy as np

def redchisq(ydata, ymod, deg=None, sd=None):
    """
Returns the reduced chi-square statistic for an arbitrary model, chisq/nu,
where nu is the number of degrees of freedom. If individual standard deviations
(array sd) are supplied, then the chi-square error statistic is computed as the
sum of squared errors divided by the standard deviations.

ydata, ymod, sd assumed to be Numpy arrays. deg integer.

Usage: >>> chisq = redchisq(ydata, ymod, n, sd)
ydata : data [array] (required)
 ymod : model evaluated at the same x points as ydata [array] (required)
  deg : number of free parameters in the model [integer] (optional)
   sd : uncertainties in ydata [array] (optional)
    """
    # Chi-square statistic
    if sd == None:
        chisq = np.sum((ydata - ymod) ** 2.0)
    else:
        chisq = np.sum(((ydata - ymod) / sd) ** 2.0)

    if deg == None:
        return chisq
    else:
        # Number of degrees of freedom assuming 2 free parameters
        nu = ydata.size - 1 - deg
        return chisq / nu


def aicc(ydata, ymod, yerr, Npars):
    """
Function to calculate the corrected Akaike Information Criterion.

ydata, ymod and yerr must all be of the same length.

    :param ydata: list or array of y data points (float)
    :param ymod: list or array of y model points (float)
    :param yerr: list or array of errors on ydata (float)
    :param Npars: number of fitting parameters (int)
    :return:
    """
    cond1 = ydata.size == ymod.size
    cond2 = ydata.size == yerr.size
    cond3 = ymod.size == yerr.size

    if (not cond1) or (not cond2) or (not cond3):
        print "ydata.size == ymod.size:", cond1
        print "ydata.size == yerr.size:", cond2
        print "ymod.size == yerr.size:", cond3
        raise ValueError("ydata, ymod and yerr should all be the same length")

    a = -1.0 * np.sum(((ydata - ymod) / yerr) ** 2.0)
    b = 2.0 * Npars
    c = ((2.0 * Npars) * (Npars + 1.0)) / (ydata.size - Npars - 1.0)

    return a + b + c
