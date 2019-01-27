import numpy as np


def redchisq(ydata, ymod, deg=None, sd=None):
    """
Returns the reduced chi-square error statistic for an arbitrary model, chisq/nu,
where nu is the number of degrees of freedom. If individual standard deviations
(array sd) are supplied, then the chi-square error statistic is computed as the
sum of squared errors divided by the standard deviations.

    :param ydata: array containing data (float)
    :param ymod: array containing model (float)
    :param deg: number of free parameter in the model (int)
    :param sd: array of uncertainties in ydata (float)
    :return: (reduced) chi-square statistic (float)
    """
    # Chi-Square statistic
    if sd is not None:
        chisq = np.sum(((ydata - ymod) / sd) ** 2.0)

    else:
        chisq = np.sum((ydata - ymod) ** 2.0)

    if deg is not None:
        # Number of degrees of freedom assuming 2 free parameters
        nu = ydata.size - 1.0 - deg

        return chisq / nu

    else:
        return chisq


def conv(samples, N, Nc, Ns):
    """
Function to calculate the R test of convergence of MC chains for multiple
dimensions. See Daniel Mortlock's notes on Comuptational Statistics from the
ICIC workshop or Gelman, A. and Rubin, D. B. (1992) for details.

    :param samples: array of shape (Nc, Ns, N) that has already had burn-in
                    removed
    :param N: int number of fitting parameters (Npars)
    :param Nc: int number of chain (Nwalk)
    :param Ns: int number of MC steps (Nstep)
    :return: array of convergence ratios for each parameter
    """
    # 1) Calculate the mean of each chain for each parameter
    # 2) Calculate the variance of each chain for each parameter
    xbarc = np.ndarray((Nc, N))    # (1)
    sigma2c = np.ndarray((Nc, N))  # (2)
    for i in range(N):
        for j in range(Nc):
            xbarc[j, i] = (1.0 / Ns) * np.sum(samples[j, :, i])  # (1)
            sigma2c[j, i] = ((1.0 / (Ns - 1.0)) * np.sum((samples[j, :, i] -
                             xbarc[j, i]) ** 2.0))  # (2)

    # 3) Calculate the mean of all the chains for each parameter
    # 4) Calculate the average of the individual chains' variances for each
    #    parameter
    # 5) Calculate the variance of the chains' means for each parameter
    # 6) Calculate the ratio of each parameter
    xbar = np.zeros(N)          # (3)
    sigma2_chain = np.zeros(N)  # (4)
    sigma2_mean = np.zeros(N)   # (5)
    ratio = np.zeros(N)         # (6)
    for i in range(N):
        xbar[i] = (1.0 / Nc) * np.sum(xbarc[:, i])                         # (3)
        sigma2_chain[i] = (1.0 / Nc) * np.sum(sigma2c[:, i])               # (4)
        sigma2_mean[i] = (1.0 / Nc) * np.sum((xbarc[:,i] - xbar[i]) ** 2.0)# (5)
        ratio[i] = (((((Nc - 1.0) / Nc) * sigma2_chain[i]) + ((1.0 / Nc) *
                    sigma2_mean[i])) / sigma2_chain[i])                    # (6)

    return ratio


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
        raise ValueError("ydata, ymod and yerr should all be the same length")

    from mcmc_eqns import lnlike

    a = 2.0 * lnlike(ydata, ymod, yerr)
    b = 2.0 * Npars
    c = ((2.0 * Npars) * (Npars + 1.0)) / (ydata.size - Npars - 1.0)

    return a + b + c
