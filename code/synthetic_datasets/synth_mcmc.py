from __future__ import print_function

import os
import sys
import json
import argparse
import numpy as np
import emcee as em
import pandas as pd
import matplotlib.pyplot as plt
from funcs import model_lum
from mcmc_eqns import lnprob
from multiprocessing import Pool
from matplotlib.ticker import MaxNLocator

truths = {
    'Humped': np.array([1.0, 5.0, -3.0, 2.0, -1.0, 0.0]),
    'Classic': np.array([1.0, 5.0, -3.0, 3.0, -1.0, 0.0]),
    'Sloped': np.array([1.0, 1.0, -3.0, 2.0,  1.0, 1.0]),
    'Stuttering': np.array([1.0, 5.0, -5.0, 2.0, -1.0, 2.0])
}

names = ['$B$', '$P$', '$\log_{10} (M_{\\rm D,i})$', '$\log_{10} (R_{\\rm D})$',
         '$\log_{10} (\epsilon)$', '$\log_{10} (\delta)$']


def sort_on_runtime(pos):
    """
Function to sort chain runtimes at execution.
    """
    p = np.atleast_2d(pos)
    idx = np.argsort(p[:, 0])[::-1]

    return p[idx], idx


def parse_args():
    parser = argparse.ArgumentParser(
        description=""
    )

    parser.add_argument(
        "--grb",
        required=True,
        choices=["Humped", "Classic", "Sloped", "Stuttering"],
        help=""
    )
    parser.add_argument(
        "-s",
        "--n-step",
        type=int,
        nargs="?",
        help="Number of MCMC steps to take"
    )
    parser.add_argument(
        "-w",
        "--n-walk",
        type=int,
        nargs="?",
        help="Number of ensemble walkers to deploy. Must be even!"
    )
    parser.add_argument(
        "--re-run",
        action="store_true",
        help="Re-run an analysis by reading in MCMC parameters and random seed"
    )

    return parser.parse_args()


def create_filenames(GRB):
    # Data basename
    data_basename = os.path.join("data", "synthetic_datasets")
    if not os.path.exists(data_basename):
        raise FileNotFoundError("Please make sure your chosen dataset exists.")
        sys.exit(1)

    # Directory name
    data_dirname = os.path.join(data_basename, GRB)
    if not os.path.exists(data_dirname):
        raise FileNotFoundError("Please make sure your chosen dataset exists.")
        sys.exit(1)

    # Plot basename
    plot_basename = os.path.join("plots", "synthetic_datasets")
    if not os.path.exists(plot_basename):
        os.makedirs(plot_basename)

    # Plot directory name
    plot_dirname = os.path.join(plot_basename, GRB)
    if not os.path.exists(plot_dirname):
        os.mkdir(plot_dirname)

    # Construct filenames
    fdata = os.path.join(data_dirname, f"{GRB}.csv")
    fchain = os.path.join(data_dirname, f"{GRB}_chain.csv")
    fbad = os.path.join(data_dirname, f"{GRB}_bad.csv")
    finfo = os.path.join(data_dirname, f"{GRB}_info.json")
    fplot = os.path.join(plot_dirname, f"{GRB}_trace.png")

    # Initialise bad parameter file
    f = open(fbad, "w")
    f.close()

    return fdata, fchain, fbad, finfo, fplot, data_dirname


def create_trace_plot(sampler, Npars, Nstep, Nwalk, fplot):
    fig, axes = plt.subplots(Npars+1, 1, sharex=True, figsize=(6,8))

    for i in range(Nwalk):
        axes[0].plot(range(Nstep), sampler.get_log_prob()[:, i], c='gray',
                     alpha=0.4)
    axes[0].yaxis.set_major_locator(MaxNLocator(4, prune='lower'))
    axes[0].tick_params(axis='both', which='major', labelsize=10)
    axes[0].set_ylabel('$\ln (p)$', fontsize=12)

    for i in range(Npars):
        for j in range(Nwalk):
            axes[i + 1].plot(range(Nstep), sampler.get_chain()[:, j, i],
                             c='gray', alpha=0.4)
        axes[i + 1].yaxis.set_major_locator(MaxNLocator(4, prune='lower'))
        axes[i + 1].tick_params(axis='both', which='major', labelsize=10)
        axes[i + 1].set_ylabel(names[i], fontsize=12)

    axes[-1].set_xlabel('Model Number', fontsize=12)
    fig.tight_layout(h_pad=0.1)
    fig.savefig('{0}'.format(fplot))
    plt.clf()


def main():
    # Parse command line arguments
    args = parse_args()

    # Build filenames
    fdata, fchain, fbad, finfo, fplot, fn = create_filenames(args.grb)

    if args.re_run:
        with open(finfo, "r") as stream:
            mc_pars = json.load(stream)

        # Set seed
        np.random.seed(mc_pars["seed"])

        # Retrieve MCMC parameters
        Npars = mc_pars["Npars"]
        Nstep = mc_pars["Nstep"]
        Nwalk = mc_pars["Nwalk"]

    else:
        # Get current seed
        seed = np.random.get_state()[1][0]

        # MCMC parameters
        Npars = 6            # Number of fitting parameters
        Nwalk = args.n_walk  # Number of walkers
        Nstep = args.n_step  # Number of MCMC steps

        # Write MCMC parameters to JSON file
        info = {
            "Npars": Npars,
            "Nwalk": Nwalk,
            "Nstep": Nstep,
            "seed": seed
        }

    # Read in data
    data = pd.read_csv(fdata)
    x = data["x"]
    y = data["y"]
    yerr = data["yerr"]

    # Calculate initial position
    p0 = np.array(truths[args.grb])
    pos = [p0 + 1.0e-4 * np.random.randn(Npars) for i in range(Nwalk)]

    with Pool() as pool:  # context management
        # Initialise Ensemble Sampler
        sampler = em.EnsembleSampler(
            Nwalk, Npars, lnprob, args=(x, y, yerr, fbad), pool=pool,
            runtime_sortingfn=sort_on_runtime
        )
        # Run MCMC
        sampler.run_mcmc(pos, Nstep, progress=True)

    # Write full MCMC to file
    with open(fchain, 'w') as f:
        f.write(f"{Npars}, {Nwalk}, {Nstep}\n")
        for j in range(Nstep):
            for i in range(Nwalk):
                for k in range(Npars):
                    f.write(f"{sampler.chain[i, j, k]:.6f}, ")
                f.write(f"{sampler.lnprobability[i, j]:.6f}\n")

    # Write each individual parameter to it's own file
    for k in range(Npars):
        with open(f"{fn}_{k}.csv", 'w') as f:
            for j in range(Nstep):
                for i in range(Nwalk):
                    if i == (Nwalk-1):
                        f.write(f"{sampler.chain[i, j, k]:.6f}\n")
                    else:
                        f.write(f"{sampler.chain[i,j,k]:.6f}, ")

    # Write probability to it's own file
    with open(f"{fn}_lnp.csv", 'w') as f:
        for j in range(Nstep):
            for i in range(Nwalk):
                if i == (Nwalk-1):
                    f.write(f"{sampler.lnprobability[i, j]:.6f}\n")
                else:
                    f.write(f"{sampler.lnprobability[i,j]:.6f}, ")

    # Acceptance fraction and autocorrelation time
    tau = sampler.get_autocorr_time()
    print(
        f"{args.grb}\n" +
        f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction)}\n" +
        f"Average auto-correlation time: {np.mean(tau):.3f}"
    )

    info["acceptance_fraction"] = np.mean(sampler.acceptance_fraction)
    info["tau"] = tau
    with open(finfo, "w") as f:
        json.dump(info, f)

    # Time series
    create_trace_plot(sampler, Npars, Nstep, Nwalk, fplot)


if __name__ == "__main__":
    main()
