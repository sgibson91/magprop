from __future__ import print_function

import os
import json
import corner
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc

# Use LaTeX in plots
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)

# Names for plotting
names = [r"$B$", r"$P_i$", r"$\log_{10}(M_{\rm D,i})$",
         r"$\log_{10}(R_{\rm D})$", r"$\log_{10}(\epsilon)$",
         r"$\log_{10}(\delta)$", r"$\eta_{\rm dip}$", r"$\eta_{\rm prop}$",
         r"$1/f_{\rm B}$"]
names_7 = [r"$B$", r"$P_i$", r"$\log_{10}(M_{\rm D,i})$",
           r"$\log_{10}(R_{\rm D})$", r"$\log_{10}(\epsilon)$",
           r"$\log_{10}(\delta)$", r"$1/f_{\rm B}$"]

# Names for columns and dataframes
col_names = ["B", "P", "Md", "Rd", "eps", "delt", "dipeff", "propeff", "fbeam"]
col_names_7 = ["B", "P", "Md", "Rd", "eps", "delt", "fbeam"]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to make analysis plots from MCMC chains"
    )

    parser.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["S", "L"],
        default=None,
        help="Define a Long or Short type GRB"
    )
    parser.add_argument(
        "--grb",
        required=True,
        type=str,
        help="GRB name"
    )
    parser.add_argument(
        "-l",
        "--label",
        required=True,
        type=str,
        help="Identifying label of MCMC run"
    )
    parser.add_argument(
        "-b",
        "--n-burn",
        type=int,
        default=1000,
        help="Number of burn-in steps to remove"
    )
    parser.add_argument(
        "--re-run",
        action="store_true",
        help="Re-run an analysis"
    )

    return parser.parse_args()

def create_filenames(grb_name, run_label, grb_type=None):
    """Function to generate filepaths"""
    #=== Data directory ===#
    # Base directory
    if grb_type is None:
        base_dir = "data"
    else:
        base_dir = os.path.join("data", f"{grb_type}GRBS")

    # Data dir
    data_dir = os.path.join(base_dir, grb_name)

    # Filenames
    fdata = os.path.join(data_dir, f"{grb_name}_k.csv")
    finfo = os.path.join(data_dir, f"{grb_name}_{run_label}_stats.json")
    fchain = os.path.join(data_dir, f"{grb_name}_{run_label}_chain.csv")
    fresult = os.path.join(data_dir, f"{grb_name}_{run_label}_result.csv")

    #=== Plotting directory ==#
    # Base directory
    if grb_type is None:
        base_dir = "plots"
    else:
        base_dir = os.path.join("plots", f"{grb_type}GRBS")

    # Plot directory
    plot_dir = os.path.join(base_dir, grb_name)

    # Filenames
    fmodel = os.path.join(plot_dir, f"{grb_name}_{run_label}_model.png")
    fcorner = os.path.join(plot_dir, f"{grb_name}_{run_label}_corner.png")

    return fdata, finfo, fchain, fresult, fmodel, fcorner

def create_corner_plot(samples, Npars, fcorner):
    """Create a corner plot"""
    if Npars == 7:
        fig = corner.corner(
            samples,
            labels=names_7,
            label_kwargs={"fontsize": 12},
            quantiles=[0.025, 0.5, 0.975],
            plot_contours=True
        )
    else:
        fig = corner.corner(
            samples,
            labels=names,
            label_kwargs={"fontsize": 12},
            quantiles=[0.025, 0.5, 0.975],
            plot_contours=True
        )
    fig.savefig(fcorner)
    plt.clf()

def main():
    args = parse_args()

    fdata, finfo, fchain, fresult, fmodel, fcorner = create_filenames(
        args.grb, args.label, grb_type=args.type
    )

    # Read in stats file
    with open(finfo, "r") as f:
        info = json.load(f)

    # Retrieve MC parameters
    Npars = info["Npars"]
    Nstep = info["Nstep"]
    Nwalk = info["Nwalk"]

    if args.re_run:
        Nburn = info["Nburn"]
    else:
        Nburn = args.n_burn

    # Read in chain & probability
    cols = list(range(Npars))
    skip = Nburn * Nwalk
    if Npars == 7:
        samples = pd.read_csv(
            fchain,
            header=None,
            names=col_names_7,
            usecols=cols,
            skiprows=skip
        )
    else:
        samples = pd.read_csv(
            fchain,
            header=None,
            names=col_names[:Npars],
            usecols=cols,
            skiprows=skip
        )

    # Create a corner plot
    create_corner_plot(samples, Npars, fcorner)

if __name__ == "__main__":
    main()
