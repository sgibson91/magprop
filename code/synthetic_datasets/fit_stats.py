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


def AIC(ymod, ydata, yerr, k):
    """
Function to calculate the Akaike Information Criterion. This is a statistic
for model comparison with different numbers of fitting parameters. It can be
derived from the Chi Square statistic and includes a penalty for 'overfitting'
by increasing the number of free parameters. The minimum AIC value reveals
the most statistically significant model.

Equation: AIC = -2ln(L) + 2k + (2k(k+1))/(N-k-1)
where: L is the maximised likelihood
       k is the number of free parameters in the model
       N is the number of observations (i.e. data points).

Usage: >>> AIC(ymod, ydata, yerr, k)
 ymod : Model data at same x-values as data. [array] (required)
ydata : Data. [array] (required)
 yerr : Errors on data points. [array] (required)
    k : Number of free parameters. [integer] (required)
    """
    # Calculate Chi Square term from above function
    chisq = redchisq(ydata, ymod, sd=yerr)

    # Calculate third term
    numerator = 2 * k * (k + 1)
    denominator = len(ydata) - k - 1
    term = numerator / denominator

    # Calculate and return AIC
    return chisq + (2 * k) + term


def rel_llhood(aic_i, aic_min):
    """
This function calculates the relative likelihood of two models described by a
corrected Akaike Information Criterion (see above function).

Usage: >>> rel_llhood(AICcs, AICcmin)
  aic_i : AICc value of model to be compared [float] (required)
aic_min : minimum AICc value of potential models [float] (required)
    """
    return np.exp((aic_min - aic_i) / 2.0)
