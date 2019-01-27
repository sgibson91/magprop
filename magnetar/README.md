# The Magnetar Python Library

## `funcs.py`

This script contains all the functions and constants used to generate model light curves of Gamma-Ray Burst X-ray afterglows.

### `init_conds`

This function converts input values into appropriate units for the model.

**Input variables:**
* `MdiscI` - initial disc mass, solar masses
* `P` - initial spin period, milliseconds

**Output variables:**
* `MdiscI` - initial disc mass, grams
* `omega` - initial angular frequency, per second

### `odes`

This function calculates the differential equations which will be integrated by `ODEINT`.

**Input variables:**
* `B` - magnetic field strength, 10^15 G
* `omega` - initial angular frequency, per second
* `MdiscI` - initial disc mass, grams
* `RdiscI` - disc radius, km
* `epsilon` - timescale ratio
* `delta` - mass ratio

**Optional variables:**
* `n` - propeller "switch-on" efficiency, default value = 1.0
* `alpha` - sound speed prescription, default value = 0.1
* `cs7` - sound speed in disc, default value = 1.0 x 10^7 cm/s
* `k` - capping fraction, default value = 0.9

**Output variables:**
* `Mdotdisc(t)` - time derivative of the disc mass, grams
* `omegadot(t)` - time derivative of the angular frequency, per second

### `model_lc`

This function calls `init_conds` to generate the initial conditions, solve `odes` using `ODEINT` for the input parameters, and calculates the total, dipole and propeller luminosities of the afterglow in units of 10^50 erg/s.
If an array of time points is provided as `xdata`, the light curve is interpolated for those time points and the total luminosity is returned for those time values in 10^50 erg/s.

**Input variables:**

* `pars` - an array or list containing the 6 core parameters: `B`, `P`, `MdiscI`, `RdiscI`, `epsilon`, `delta`

**Optional variables:**

* `xdata` - the time points of GRB data, default value = `None` 
* `dipeff` - dipole energy-to-luminosity conversion efficiency, default value = 0.05
* `propeff` - propeller energy-to-luminosity conversion efficiency, default value = 0.4
* `f_beam` - beaming fraction, default value = 1.0
* `n` - propeller "switch-on" efficiency, default value = 1.0
* `alpha` - sound speed prescription, default value = 0.1
* `cs7` - sound speed in disc, default value = 1.0 x 10^7 cm/s
* `k` - capping fraction, default value = 0.9

**Output variables:**
```
if xdata == None:
    return array containing:
      * t - time array
      * Ltot - total luminosity, 10^50 erg/s
      * Lprop - propeller luminosity, 10^50 erg/s
      * Ldip - dipole luminosity, 10^50 erg/s
else:
    return array containing:
      * L - total luminosity at time points given in xdata, 10^50 erg/s
```

### Running `funcs.py` as a script

`funcs.py` can be run as a script to plot a light curve for given values of the 6 core parameters.

**Required arguments:**

* `-B` - magnetic field strength between 10^-3 - 10 x 10^15 G
* `-P` - initial spin period between 10^-3 - 10 milliseconds
* `-M` - initial disc mass between 10^-5 - 0.1 solar masses
* `-R` - initial disc radius between 50 - 1500 km
* `-eps` - timescale ratio between 0.1 - 100
* `-delt` - mass ratio between 10^-3 - 1000

**Recommended usage:**
```
python magnetar/funcs.py -B 1.0 -P 1.0 -M 0.001 -R 100.0 -eps 1.0 -delt 1.0
```

## `fit_stats.py`

This script contains various statistical calculations used to verify the models created by the `funcs.py` script.

### `redchisq`

This function calculates the (reduced) chi square statistic of a model given input data.

**Input variables:**
* `ydata` - an array or list of the observed `y`-data
* `ymod` - an array or list the modelled `y`-data

**Optional variables:**
* `deg` - the number of fitting parameters in the model (integer)
* `sd` - a list or array of the uncertainties in `ydata`

**Output variables:**
* `chisq` - the (reduced) chi square goodness-of-fit statistic

### `conv`

This function calculates the convergence of a set of Monte Carlo Markov chains.

**Input variables:**
* `samples` - an array of shape `(Nc, Ns, N)` containing post-burn-in Monte Carlo samples
* `N` - the number of fitting parameters (integer)
* `Nc` - the number of Markov chains (integer)
* `Ns` - the number of Monte Carlo steps (integer)

**Output variables:**
* `ratios` - an array of convergence ratios for each fitting parameter

Chains are considered to have converged when the ratios `~1`
