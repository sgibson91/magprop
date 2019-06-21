import numpy as np
import pandas as pd
from . import model_lc


def lnlike(pars, data, GRBtype):
    """
Function to calculate the log-likelihood of a model given input data based on
the chi square goodness-of-fit statstic.

    :param pars: list or array of fitting parameters (float)
    :param data: dataframe of observed GRB (pandas.DataFrame)
    :param GRBtype: string indicating the GRB type (str)
    :return: log-likelihood value (float)
    """
    # Separate data
    x = data["t"]
    y = data["Lum50"]
    yerr = data["Lum50err"]

    # Calculate the model
    if len(pars) == 6:
        ymod = model_lc(pars, xdata=x, GRBtype=GRBtype)
    elif len(pars) == 7:
        ymod = model_lc(pars[:6], xdata=x, GRBtype=GRBtype, f_beam=pars[6])
    elif len(pars) == 8:
        ymod = model_lc(
            pars[:6], xdata=x, GRBtype=GRBtype, dipeff=pars[6], propeff=pars[7]

    elif len(pars) == 9:
        ymod = model_lc(
            pars[:6], xdata=x, GRBtype=GRBtype, dipeff=pars[6],
            propeff=pars[7], f_beam=pars[8]
        )

    # Return the log-likelihood
    return -0.5 * np.sum(((y - ymod) / yerr) ** 2.0)


def lnprior(pars, custom_lims=None):
    """
Function to calculate the log-prior function. If the parameters are within the
prescribed limits, 0 value is returned. Otherwise, negative infinity is
returned.

A CSV file of custom limits can be passed via the custom_lims argument.

    :param pars: list or array of parameters (float)
    :param custom_lims: path to a CSV file containing custom limits
    :return: 0.0 or negative infinity depending on whether parameters are within
             limits or not
    """
    # Import limits
    if custom_lims is None:
        lims = pd.read_csv("magnetar/mcmc_limits.csv", index_col="pars")
    else:
        try:
            lims = pd.read_csv(custom_lims, index_col="pars")
        except ValueError:
            raise ValueError("Please provide a valid file path.")

    # Determine if parameters are within the prescribed limits
    # Special handling for Npars == 7 case
    if len(pars) == 7:
        seven_lims_lo = np.zeros_like(pars)
        seven_lims_hi = np.zeros_like(pars)

        seven_lims_lo[:6] = lims["lower"].values[:6]
        seven_lims_lo[-1] = lims["lower"].values[-1]

        seven_lims_hi[:6] = lims["upper"].values[:6]
        seven_lims_hi[-1] = lims["upper"].values[-1]

        cond_lo = np.all(pars >= seven_lims_lo)
        cond_hi = np.all(pars <= seven_lims_hi)

    else:
        cond_lo = np.all(pars >= lims["lower"].values[:len(pars)])
        cond_hi = np.all(pars <= lims["upper"].values[:len(pars)])

    if (not cond_lo) or (not cond_hi):
        return -np.inf
    else:
        return 0.0


def lnprob(pars, data, GRBtype, custom_lims=None):
    """
Calculate the log-probability of the model given input parameters and observed
data.

Calculate log-prior first so that computational time is not wasted calculating
models that do not fulfill the prior requirement.

    :param pars: list or array of input parameters (float)
    :param data: dataframe of observed GRB (pandas.DataFrame)
    :param GRBtype: string indicating the GRB type (str)
    :param custom_lims: path to a CSV file containing custom parameter limits
    :return: the log-probability value of the model compared to the data (float)
    """
    # Calculate log-prior
    if custom_lims is not None:
        lp = lnprior(pars, custom_lims=custom_lims)
    else:
        lp = lnprior(pars)

    # Determine if log-prior is finite
    if not np.isfinite(lp):
        return -np.inf

    # Calculate log-likelihood
    ll = lnlike(pars, data, GRBtype)

    # Determine if log-likelihood is finite
    if not np.isfinite(ll):
        return -np.inf

    # Return log-probability
    return lp + ll
