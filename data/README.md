# Samples of Gamma-Ray Burst data

This folder contains samples of Gamma-Ray Bursts (GRBs) that were interrogated by the Magnetar Propeller Model with Fallback Accretion.

The data are available at the [UK Swift Science Data Centre website](http://www.swift.ac.uk/) and the observations were made by the [Neil Gehrels Swift Observatory](https://swift.gsfc.nasa.gov/).

* [Data Collection Process](#data-collection-process)
* [Data Format](#data-format)
  * [Raw format](#raw-format)
  * [Cleaned Format](#cleaned-format)
  * [_k_-corrected Format](#_k_-corrected-format)
* [GRB Samples](#grb-samples)
  * [Short GRBs with Extended Emission (SGRBEEs)](#short-grbs-with-extended-emission-sgrbees)

---

## Data Collection Process


## Data Format

### Raw Format

The raw data files have filepaths fitting the pattern: `data/<GRB-sample>/<GRB-name>_raw.txt`.

These files are in `.txt` format and have six columns of floating point data.
The columns in order from left to right are: observed arrival time of photon, positive time uncertainty, negative time uncertainty, observed photon flux, positive flux uncertainty, negative flux uncertainty.

There are also rows in the raw datafiles pertaining to the plotting package used by the host website.
Such rows begin with `!`, `READ` or `NO`.
`code/clean_data.py` will remove these rows, convert the data to CSV and create folders under `data/<GRB-sample>/` named after each GRB in the sample.

### Cleaned Format

After applying `code/clean_data.py`, the data files will now have filepaths following the pattern: `data/<GRB-sample>/<GRB-name>/<GRB_name>.csv`.

These files are in CSV format and have 6 columns of floating point data with headers (from left to right): `flux`, `fluxneg`, `fluxpos`, `t`, `tneg`, `tpos`.
These relate to the observed flux, negative flux uncertainty, positive flux uncertainty, observed photon arrival time, negative time uncertainty and positive flux uncertainty, respectively.

### _k_-corrected Format

The last stage of the data processing pipeline is to run `code/kcorr.py` on the GRB samples.
This code performs a _k_-correction on the data which accounts for the distance from Earth the GRBs occurred at the the energy bandpass of the instruments on-board the Swift satellite.

The _k_-corrected data files will have filepaths that follow the pattern: `data/<GRB-sample>/<GRB-name>/<GRB-name>_k.csv`.
These files are in CSV format and have 7 columns of floating point data with headers: `Lum50`, `Lum50neg`, `Lum50pos`, `t`, `tneg`, `tpos`, `Lum50err`.
These relate to the luminosity, negative luminosity uncertainty, positive luminosity uncertainty, red-shift corrected time, negative time uncertainty, positive time uncertainty and the geometric mean of `Lum50pos` and `Lum50neg`, respectively.
All luminosities are reported in units of `10^50 ergs s^-1`.

## GRB Samples

### Short GRBs with Extended Emission (SGRBEEs)

These is the Short GRB with Extended Emission sample used in the 2017 paper.

* 050724
* 051016B
* 051227
* 060614
* 061210
* 070714B
* 071227
* 080123
* 080503
* 100212A
* 100522A
* 111121A
* 150424A
* 160410A
