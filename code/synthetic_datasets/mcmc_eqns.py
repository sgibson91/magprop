import numpy as np
from funcs import model_lum


def lnlike(pars, x, y, yerr):
    """
This function calculates the Gaussian Log-Likelihood probability of a model fit
to data. Returns a float.

Usage >>> lnprob(pars, x, y, yerr)
pars : a list of parameters to be parsed to model_lum
   x : x-axis data to be fitted to (array of floats)
   y : y-axis data to be fitted to (aray of floats)
yerr : error on y-data (array of floats)
    """
    arr = np.array(pars)  # Copy pars into array
    arr[2:] = 10.0 ** arr[2:]  # Unlog required parameters

    # Calculate model
    mod = model_lum(arr, xdata=x)

    if mod == "flag":  # Flags models that break odeint
        return -1.0 * np.inf
    else:
        return -0.5 * np.sum(((y - mod) / yerr) ** 2.0)


def lnprior(pars):
    """
Function to calculate the Prior probability function. Actually checks if
parameters are within allowed range (l < pars < u), returns 0.0 if True, -inf
if False.

Usage >>> lnprior(pars, u, l)
pars : a list of parameters to be parsed to model_lum

upper : upper limits for parameters (array of floats)
lower : lower limits for parameters (array of floats)
    """
    upper = np.array([10.0, 10.0, -2.0, np.log10(2000.0), 2.0, 3.0])
    lower = np.array([1.0e-3, 0.69, -6.0, np.log10(50.0), -2.0, -1.0])

    upper_cond = pars <= upper
    lower_cond = pars >= lower

    if np.all(upper_cond) and np.all(lower_cond):
        return 0.0
    else:
        return -np.inf


def lnprob(pars, x, y, yerr, fbad):
    """
Function to calculate Posterior probability function. It checks that lnlike and
lnprior have returned finite numbers and returns the summation of them. Any
parameter sets that have flagged an error from odeint are written to a file.

Usage >>> lnprob(pars, x, y, yerr, u, l, fbad)
pars : list of parameters to be parsed to model_lum
   x : x-axis data to be fitted to (array of floats)
   y : y-axis data to be fitted to (array of floats)
yerr : error on y-data (array of floats)
fbad : filename for bad parameter sets to be written to (string)
    """
    # Calculate prior
    lp = lnprior(pars)
    if not np.isfinite(lp):
        return -1.0 * np.inf

    # Calculate likelihood
    ll = lnlike(pars, x, y, yerr)
    if (not np.isfinite(ll)) and (fbad is not None):
        with open(fbad, "a") as f:
            for i, k in enumerate(pars):
                if i == (len(pars) - 1):
                    f.write(f"{k}\n")
                else:
                    f.write(f"{k}, ")
        return -1.0 * np.inf

    return ll + lp
