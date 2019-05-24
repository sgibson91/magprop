:construction: :construction: This repo uses Python 2.7 which will be [deprecated on 01/01/2020](https://legacy.python.org/dev/peps/pep-0373/#id4). The code needs checking for compatibility with and updating to Python 3. See [#38](https://github.com/sgibson91/magprop/issues/38) :construction: :construction:

# Magnetar Propeller Model with Fallback Accretion

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Suite of code that models fallback accretion onto a magnetar and uses Markov Chain Monte Carlo to fit this to samples of Gamma-Ray Bursts.

[![Build Status](https://travis-ci.org/sgibson91/magprop.svg?branch=master)](https://travis-ci.org/sgibson91/magprop) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sgibson91/magprop/master?urlpath=lab)

## Installation

To clone this repo:

```
git clone https://github.com/sgibson91/magprop.git
cd magprop
```

To install the requirements:

```
pip install -r requirements.txt
```

To install the `magnetar` library:

```
python setup.py install
```

## Usage

To run the scripts in this repo, first click the badge below.
This will launch a JupyterLab environment containing the repo, via Binder.
You may wish to right-click the badge and select "Open Link in New Tab" (or whichever variant your browser provides) so you can still refer to these notes.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sgibson91/magprop/master/?urlpath=lab)

From the menu, select a Python 2 Console.
(The second icon on the second row down.)

To run a figure script, type the following:
```
python code/figure_<number>.py
```
SHIFT-RETURN executes the command.

**NOTES:**
* It is recommended that you run all the scripts from the root of the repo (i.e. do not `cd code | python <script_name>.py`).
* You will not be able to run the MCMC algorithms from within the Binder instance due to a computational cap.
  The Binder link is provided for demonstration purposes only.

* Figures will be saved to the sub-directory `plots/`.

### List of Scripts in `code/`

* `figure_1.py`: Reproduces figure 1
* `figure_2.py`: Reproduces figure 2
* `figure_3.py`: Reproduces figure 3
* `figure_4.py`: Reproduces figure 4
* `figure_5.py`: Reproduces figure 5


* `kcorr.py`: Performs a k-correction on a GRB dataset.
  Takes command line argument `-t S` for the short GRB sample.

## Running Tests

To run tests after launching the Binder, select `terminal` from the launcher window in JupyterLab and execute the following commands.
```bash
# Activate the environment
$ source activate kernel

# Run the tests with verbose output
$ python -m pytest -vvv
```

To see the code coverage of the test suite, run the following commands.
```bash
$ coverage run -m pytest -vvv
$ coverge report
```

## Citing this work

Please quote the following citation when referring to this work.

### Paper

* [*Fallback Accretion on to a Newborn Magnetar: Short GRBs with Extended Emission*](https://arxiv.org/abs/1706.04802)

### Citation

```
@article{doi:10.1093/mnras/stx1531,
author = {Gibson, S. L. and Wynn, G. A. and Gompertz, B. P. and O'Brien, P. T.},
title = {Fallback accretion on to a newborn magnetar: short GRBs with extended emission},
journal = {Monthly Notices of the Royal Astronomical Society},
volume = {470},
number = {4},
pages = {4925-4940},
year = {2017},
doi = {10.1093/mnras/stx1531},
URL = {http://dx.doi.org/10.1093/mnras/stx1531},
eprint = {/oup/backfile/content_public/journal/mnras/470/4/10.1093_mnras_stx1531/1/stx1531.pdf}
}
```

### License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This work is published under the MIT license.
Please see the `LICENSE` file for further information.
