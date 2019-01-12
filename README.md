# Magnetar Propeller Model with Fallback Accretion

Suite of code that models fallback accretion onto a magnetar and uses Markov Chain Monte Carlo to fit this to samples of Gamma-Ray Bursts.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sgibson91/magprop/master/?urlpath=lab)

## Usage

To run the scripts in this repo, first click the badge below.
This will launch a JupyterLab environment containing the repo, via Binder.
You may wish to right-click the badge and select "Open Link in New Tab" (or whichever variant your browser provides) so you can still refer to these notes.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sgibson91/magprop/fig2-script/?urlpath=lab)

From the menu, select a Python 2 Console.
(The second icon on the second row down.)

To run a script, type the following:
```
python code/<script_name>.py
```
SHIFT-RETURN executes the command.

It is recommended that you run all the scripts from the root of the repo (i.e. do not `cd code | python <script_name>.py`).

Figures will be saved to the sub-directory `plots/`.

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

This work is published under the MIT license.
Please see the `LICENSE` file for further information.
