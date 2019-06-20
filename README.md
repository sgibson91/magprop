:construction: :construction: This repo uses Python 2.7 which will be [deprecated on 01/01/2020](https://legacy.python.org/dev/peps/pep-0373/#id4). The code needs checking for compatibility with and updating to Python 3. See [#38](https://github.com/sgibson91/magprop/issues/38) :construction: :construction:

# Magnetar Propeller Model with Fallback Accretion

Suite of code that models fallback accretion onto a magnetar and uses Markov Chain Monte Carlo to fit this to samples of Gamma-Ray Bursts.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Build Status](https://travis-ci.org/sgibson91/magprop.svg?branch=master)](https://travis-ci.org/sgibson91/magprop)

- [Magnetar Propeller Model with Fallback Accretion](#magnetar-propeller-model-with-fallback-accretion)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Reproducing Model Figures 1-5](#reproducing-model-figures-1-5)
    - [Running MCMC on Synthetic Datasets](#running-mcmc-on-synthetic-datasets)
    - [Preparing the GRB samples](#preparing-the-grb-samples)
    - [Binder](#binder)
  - [Running Tests](#running-tests)
  - [Citing this work](#citing-this-work)
    - [Paper](#paper)
    - [Citation](#citation)
    - [License](#license)

---

## Installation

Begin by cloning this repo.

```
git clone https://github.com/sgibson91/magprop.git
cd magprop
```

Install the requirements using `pip`.

```
pip install -r requirements.txt
```

Use the `setup.py` to install the `magnetar` library.

```
python setup.py install
```

## Usage

### Reproducing Model Figures 1-5

Execute a figure script by running:

```
python code/figure_<number>.py
```

These scripts will reproduce the model figures 1-5 in the Short GRBs paper.
The figures will be saved to the `plots/` directory.

### Running MCMC on Synthetic Datasets

An MCMC simulation can be run on a synthetic dataset of one of the four GRB types in order to evaluate the performance of the model and MCMC algorithm.
The four GRB types are: Humped, Classic, Sloped, and Stuttering.

First off, generate a dataset by running the following script.

```
python code/synthetic_dataset/generate_synthetic_dataset.py --grb <GRB-type>
```

The dataset will be saved to `data/synthetic_datasets/<GRB-type>/<GRB-type>.csv`.

Then run the MCMC simulation on the synthetic dataset.

```
python code/synthetic_dataset/mcmc_synthetic.py --grb <GRB-type> --n-walk <Nwalk> --n-step <Nstep>
```

where:
* `Nwalk` is the number of MCMC walkers to use, and
* `Nstep` is the number of MCMC steps to take.

This will optimise for 6 parameters: `B`, `P`, `MdiscI`, `RdiscI`, `epsilon` and `delta`.
Generated datafiles will be saved to `data/synthetic_datasets/<GRB-type>` and figures will be saved to `plots/synthetic_datasets/<GRB-type>`.

If you need to re-run an anlysis with the same input random seed, parse the `--re-run` flag.

Once the MCMC is completed, then run the analysis script to generate figures and fitting statistics.

```
python code/synthetic_datasets/plot_synthetic.py --grb <GRB-type> --n-burn <Nburn>
```
where `Nburn` is the number of steps to remove as burn-in.

The optimal model will be saved to `data/synthetic_datasets/<GRB-type>/<GRB-type>_model.csv` and `plots/synthetic_datasets/<GRB-type>/<GRB-type>_model.png`.
Another important file to check is `data/synthetic_datasets/<GRB-type>/<GRB-type>_stats.json` which will contain the optimised parameters and fitting statistics.

### Preparing the GRB samples

The raw datafiles for the Short GRB sample are stored in `data/SGRBS/`.
The dataset needs cleaning first to remove comments generated by the website that hosts the data and convert it to CSV format.

```
python code/clean_data.py
```

The last stage of preparing the dataset involves performing a _k_-correction.
A _k_-correction accounts for the distance the GRB exploded at and the energy bandwidth of the telescope that captured the data in order to make it compatible with the magnetar model.
See [this paper](https://iopscience.iop.org/article/10.1086/321093/fulltext/) for more detail.

Run the _k_-correction on the Short GRB sample by running the following command.

```
python code/kcorr -t S
```

### Binder

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sgibson91/magprop/master?urlpath=lab)

To run this repo in Binder, click the launch button above.
When your server launches, you will see a JupyterLab interface.

Running a script in the terminal will be the same as running the scripts locally.
If you run a script in the Python Console, then you'll need to modify the command to the following.

```
%run code/figure_<number>.py
```

You will **NOT** be able to run the MCMC simulations inside the Binder instance as the servers are limited to `1G` memory and `0.5` CPU.
Please follow the instructions in [Installation](#installation) in order to run the MCMC simulations locally.

## Running Tests

To run tests, execute the following command.

```
python -m pytest -vvv
```

To see the code coverage of the test suite, run the following commands.

```
coverage run -m pytest -vvv
coverage report
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
Please see the [`LICENSE`](./LICENSE) file for further information.
