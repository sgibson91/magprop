# Samples of Gamma-Ray Burst data

This folder contains samples of Gamma-Ray Bursts (GRBs) that were interrogated by the Magnetar Propeller Model with Fallback Accretion.

The data are available at the [UK Swift Science Data Centre website](http://www.swift.ac.uk/) and the observations were made by the [Neil Gehrels Swift Observatory](https://swift.gsfc.nasa.gov/).

* [Data Collection Process](#data-collection-process)
  * [Combined BAT and XRT GRB data](#combined-bat-and-xrt-grb-data)
  * [XRT GRB data](#xrt-grb-data)
  * [_k_-correction Variables](#_k_-correction-variables)
* [Data Format](#data-format)
  * [Raw format](#raw-format)
  * [Cleaned Format](#cleaned-format)
  * [Format of _k_-correction Variables](#format-of-_k_-correction-variables)
  * [_k_-corrected Format](#_k_-corrected-format)
* [GRB Samples](#grb-samples)
  * [Short GRBs with Extended Emission (SGRBEEs)](#short-grbs-with-extended-emission-sgrbees)

---

## Data Collection Process

### Combined BAT and XRT GRB Data

**BAT:** Burst Alert Telescope

**XRT:** X-Ray Telescope

1. Go to the [Swift website](http://www.swift.ac.uk)

2. Search for the GRB you want to collect the data for.
   (Search box is in the top right of the page.)

3. Click on "Burst Analyser".

4. Find the graph titled: "BAT-XRT unabsorbed flux light curves".

5. In the box to the right of the graph, select the following options:

   * **BAT binning:** SNR 5
   * **Subplot?:** no
   * **Display which bands?:** Uncheck "Flux density at 10 keV" and select "0.3-10 keV flux".

6. Underneath the graph, click "Data file".
   This will show the raw data which can be copy/pasted into the `data/<GRB-sample>/<GRB-name>_raw.txt` file.

### XRT GRB data

**XRT:** X-Ray Telescope

1. Go to the [Swift website](http://www.swift.ac.uk)

2. Search for the GRB you want to collect the data for.
   (Search box is in the top right of the page.)

3. Click on "XRT lightcurve".

4. Find the graph titled: "Basic Light Curve".
   (It should have y-axis units of "Count Rate (0.3-10 keV) (/s)".)

5. Above the graph, click "Data file".
   This will show the raw data which can be copy/pasted into the `data/<GRB-sample>/<GRB-name>_raw.txt` file.

### _k_-correction Variables

1. Go to the [Swift website](http://www.swift.ac.uk)

2. Search for the GRB you want to collect the data for.
   (Search box is in the top right of the page.)

3. Click on "XRT spectrum".

4. Scroll down to the "Late Time spectrum".

5. Use the table at the bottom of the page to gather the following information:

   * **Redshift:** will be labelled "z of absorber".
     If the value given is zero, check the literature or use the average of the sample with red-shift values.
   * **Photon Index:** will be labelled "Photon Index" and its uncertainties given.
   * **Absorption coefficient:** divide the number in "Flux (0.3 - 10 keV) (unabsorbed)" by the number in "Flux (0.3 - 10 keV) (absorbed)".

## Data Format

### Raw Format

The raw data files have filepaths fitting the pattern: `data/<GRB-sample>/<GRB-name>_raw.txt`.

These files are in `.txt` format and have six columns of floating point data.
The columns in order from left to right are: observed arrival time of photon, positive time uncertainty, negative time uncertainty, observed photon flux, positive flux uncertainty, negative flux uncertainty.
Temporal values are reported in units of seconds and flux values are reported in units of `ergs s^-1 cm^-2`.

There are also rows in the raw datafiles pertaining to the plotting package used by the host website.
Such rows begin with `!`, `READ` or `NO`.
`code/clean_data.py` will remove these rows, convert the data to CSV and create folders under `data/<GRB-sample>/` named after each GRB in the sample.

### Cleaned Format

After applying `code/clean_data.py`, the data files will now have filepaths following the pattern: `data/<GRB-sample>/<GRB-name>/<GRB_name>.csv`.

These files are in CSV format and have 6 columns of floating point data with headers (from left to right): `flux`, `fluxneg`, `fluxpos`, `t`, `tneg`, `tpos`.
These relate to the observed flux, negative flux uncertainty, positive flux uncertainty, observed photon arrival time, negative time uncertainty and positive flux uncertainty, respectively.
Temporal values are reported in units of seconds and flux values are reported in units of `ergs s^-1 cm^-2`.

### Format of _k_-correction Variables

The variables required for performing a _k_correction ([see next section](#_k_-corrected-format)) are stored in: `data/kcorr-<GRB-sample>.csv`.

These files are CSV format with 1 column of string data, relating to the names of the GRBs in the sample, and 5 columns of floating point data with headers: `Gamma`, `Gamma+err`, `Gamma-err`, `sigma`, `z`.
This relates to the photon index, positive uncertainy and negative uncertainty on the photon index, the absorption coeffiction and the red-shift, respectively.

### _k_-corrected Format

The last stage of the data processing pipeline is to run `code/kcorr.py` on the GRB samples.
This code performs a _k_-correction on the data which accounts for the distance from Earth the GRBs occurred at the the energy bandpass of the instruments on-board the Swift satellite.

The _k_-corrected data files will have filepaths that follow the pattern: `data/<GRB-sample>/<GRB-name>/<GRB-name>_k.csv`.
These files are in CSV format and have 7 columns of floating point data with headers: `Lum50`, `Lum50neg`, `Lum50pos`, `t`, `tneg`, `tpos`, `Lum50err`.
These relate to the luminosity, negative luminosity uncertainty, positive luminosity uncertainty, red-shift corrected time, negative time uncertainty, positive time uncertainty and the geometric mean of `Lum50pos` and `Lum50neg`, respectively.
Luminosities are reported in units of `10^50 ergs s^-1` and temporal values are reported in units of seconds.

## GRB Samples

### Short GRBs with Extended Emission (SGRBEEs)

This is the Short GRB with Extended Emission sample used in the 2017 paper.
The data contain combined BAT and XRT observations.

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
