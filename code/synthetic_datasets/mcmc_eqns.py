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
    arr = np.array(pars)       # Copy pars into array
    arr[2:] = 10.0 ** arr[2:]  # Unlog required parameters

    # Calculate model
    mod = model_lum(arr, xdata=x)

    if mod == 'flag':          # Flags models that break odeint
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
        with open(fbad, 'a') as f:
            for i, k in enumerate(pars):
                if i == (len(pars)-1):
                    f.write("{0}\n".format(k))
                else:
                    f.write("{0}, ".format(k))
        return -1.0 * np.inf

    return ll + lp


def conv(samples, N, Nc, Ns):
    """
    Function to calculate the R test of convergence of MC chains for multiple
    dimensions. See Daniel Mortlock's notes on Computational Statistics from
    the ICIC workshop or Gelman, A. and Robin, D. B. (1992) for full details.

    Usage >>> conv(samples, N, Nc, Ns)
    samples : array of shape (Nc, Ns, N) that has already had burn-in removed
          N : number of fitting parameters - integer (equivalent to Npars)
         Nc : number of chains - integer (equiv. to Nwalk)
         Ns : number of MC steps - integer (equiv. to Nstep)
    """
    # STEP 1) Calculate the mean of each chain for each parameter
    # STEP 2)  Calculate the variance of each chain for each parameter
    xbarc = np.ndarray((Nc, N))    # (1)
    sigma2c = np.ndarray((Nc, N))  # (2)
    for i in range(N):
        for j in range(Nc):
            xbarc[j,i] = (1.0 / Nc) * np.sum(samples[j,:,i])             # (1)
            sigma2c[j,i] = ((1.0 / (Ns - 1.0)) * np.sum((samples[j,:,i] -
                            xbarc[j,i]) ** 2.0))                         # (2)

    # STEP 3) Calculate the mean of all the chains for each parameter
    # STEP 4) Calculate the average of the individual chains' variances for
    #         each parameter
    # STEP 5) Calculate the variance of the chains' means for each parameter
    # STEP 6) Calculate the ratio for each parameter
    xbar = np.zeros(N)          # (3)
    sigma2_chain = np.zeros(N)  # (4)
    sigma2_mean = np.zeros(N)   # (5)
    ratio = np.zeros(N)         # (6)
    for i in range(N):
        xbar[i] = (1.0 / Nc) * np.sum(xbarc[:,i])                        # (3)
        sigma2_chain[i] = (1.0 / Nc) * np.sum(sigma2c[:,i])              # (4)
        sigma2_mean[i] = (1.0 / Nc) * np.sum((xbarc[:,i] - xbar[i])**2.0)# (5)
        ratio[i] = (((((Nc - 1.0) / Nc) * sigma2_chain[i]) + ((1.0 / Nc) *
                    sigma2_mean[i])) / sigma2_chain[i])                  # (6)

    return ratio
