from __future__ import print_function

import os
import json
import argparse
import numpy as np
import emcee as em
import pandas as pd
import matplotlib.pyplot as plt
from magnetar import lnprob
from multiprocessing import Pool
from matplotlib.ticker import MaxNLocator

# Parameter names
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
        description="Script to optimise the magnetar model to GRB data using MCMC"
    )

    parser.add_argument(
        "-t",
        "--type",
        required=True,
        type=str,
        choices=["L", "S"],
        help="GRB type"
    )
    parser.add_argument(
        "--grb",
        required=True,
        type=str,
        help="GRB name. Format YYMMDD[:ABC:]."
    )
    parser.add_argument(
        "-l",
        "--label",
        required=True,
        type=str,
        help="MCMC run label for provenance"
    )
    parser.add_argument(
        "-p",
        "--n-pars",
        type=int,
        choices=[6, 7, 8, 9],
        nargs="?",
        help="Number of parameters to optimise for"
    )
    parser.add_argument(
        "-s",
        "--n-step",
        type=int,
        nargs="?",
        help="Number os MCMC steps to take"
    )
    parser.add_argument(
        "-w",
        "--n-walk",
        type=int,
        nargs="?",
        help="Number of ensemble walkers to deploy"
    )
    parser.add_argument(
        "--burn",
        action="store_true",
        help="Parse this flag if it is a burn-in run"
    )
    parser.add_argument(
        "--re-run",
        action="store_true",
        help="Parse this flag to re-fun a previous analysis"
    )
    parser.add_argument(
        "-c",
        "--custom-limits",
        default=None,
        help="Path to a CSV file containing custom parameter limits"
    )
    parser.add_argument(
        "-d",
        "--n-threads",
        type=int,
        default=4,
        help="Number of threads to parallelise across"
    )

    return parser.parse_args()


def create_filenames(args):
    # Data directory
    data_dir = os.path.join("data", f"{args.type}GRBS", f"{args.grb}")

    # Plot directory
    plot_dir = os.path.join("plots", f"{args.type}GRBS", f"{args.grb}")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Construct filenames
    fdata = os.path.join(data_dir, f"{args.grb}_k.csv")
    fstats = os.path.join(data_dir, f"{args.grb}_{args.label}_stats.json")

    if args.burn:
        fchain = os.path.join(data_dir, f"{args.grb}_{args.label}_burn.csv")
        fn = os.path.join(data_dir, f"{args.grb}_{args.label}_burn")
        fplot = os.path.join(plot_dir, f"{args.grb}_{args.label}_burn_trace.png")
    else:
        fchain = os.path.join(data_dir, f"{args.grb}_{args.label}_chain.csv")
        fn = os.path.join(data_dir, f"{args.grb}_{args.label}_chain")
        fplot = os.path.join(plot_dir, f"{args.grb}_{args.label}_chain_trace.png")

    return fdata, fstats, fchain, fplot, fn


def write_output(sampler, Npars, Nstep, Nwalk, fchain, fn):
    filenames = ["B", "P", "Md", "Rd", "eps", "delt"]

    # Write full MCMC to file
    with open(fchain, 'w') as f:
        for j in range(Nstep):
            for i in range(Nwalk):
                for k in range(Npars):
                    f.write(f"{sampler.chain[i, j, k]:.6f}, ")
                f.write(f"{sampler.lnprobability[i,j]:.6f}\n")

    # Write each individual parameter to it's own file
    for k in range(Npars):
        with open(f"{fn}_{filenames[k]}.csv", 'w') as f:
            for j in range(Nstep):
                for i in range(Nwalk):
                    if i == (Nwalk-1):
                        f.write(f"{sampler.chain[i, j, k]:.6f}\n")
                    else:
                        f.write(f"{sampler.chain[i, j, k]:.6f}, ")

    # Write probability to it's own file
    with open(f"{fn}_lnP.csv", 'w') as f:
        for j in range(Nstep):
            for i in range(Nwalk):
                if i == (Nwalk-1):
                    f.write(f"{sampler.lnprobability[i, j]:.6f}\n")
                else:
                    f.write(f"{sampler.lnprobability[i, j]:.6f}, ")


def main():
    # Get command line args
    args = parse_args()

    # Create filenames
    fdata, fstats, fchain, fplot, fn = create_filenames(args)

    if args.re_run:
        # Load info file
        with open(fstats, "r") as stream:
            mc_pars = json.load(stream)

        # Set random seed
        np.random.seed(mc_pars["seed"])

        # Set MCMC parameters
        Npars = mc_pars["Npars"]
        Nstep = mc_pars["Nstep"]
        Nwalk = mc_pars["Nwalk"]

    else:
        # Retrieve random seed
        seed = np.random.get_state()[1][0]

        # MCMC parameters
        Npars = args.n_pars
        Nstep = args.n_step
        Nwalk = args.n_walk

        info = {
            "seed": seed,
            "Npars": Npars,
            "Nstep": Nstep,
            "Nwalk": Nwalk
        }

    # Read in data
    data = pd.read_csv(fdata)

    # Initial position
    if args.burn:

        if args.custom_limits is not None:
            mcmc_limits = pd.read_csv(args.custom_limits)
        else:
            mcmc_limits = pd.read_csv("magnetar/mcmc_limits.csv")

        pos = np.ndarray((Nwalk, Npars))
        for i in range(Npars):
            pos[:, i] = np.random.uniform(
                low=mcmc_limits["lower"].values[i],
                high=mcmc_limits["upper"].values[i],
                size=Nwalk
            )

    else:
        pos = np.loadtxt(f"{fn}_bestlnp.csv", delimiter=",")

    with Pool(args.n_threads) as pool:  # context management
        # Initialise the sampler
        sampler = sampler = em.EnsembleSampler(
            Nwalk, Npars, lnprob, args=(data, args.type), pool=pool,
            runtime_sortingfn=sort_on_runtime
        )
        # Run MCMC
        sampler.run_mcmc(pos, Nstep, progress=True)

    # Acceptance fraction and autocorrelation time
    tau = sampler.get_autocorr_time()
    print(
        f"{args.grb}\n" +
        f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction)}\n" +
        f"Average auto-correlation time: {np.mean(tau):.3f}"
    )

    if args.burn:
        info["burn"] = {
            "acceptance_fraction": np.mean(sampler.acceptance_fraction),
            "tau": tau
        }
    else:
        info["chain"] = {
            "acceptance_fraction": np.mean(sampler.acceptance_fraction),
            "tau": tau
        }
    with open(finfo, "w") as f:
        json.dump(info, f)

    # Write output to files
    write_output(sampler, Npars, Nstep, Nwalk, fchain, fn)


if __name__ == "__main__":
    main()

# #=== Plotting ===#
# # Time series
# fig, axes = plt.subplots(Npars+1, 1, sharex=True, figsize=(6,8))

# for i in range(Nwalk):
#     axes[0].plot(range(Nstep), sampler.lnprobability[i,:], c='gray',
#                  alpha=0.4)
# axes[0].yaxis.set_major_locator(MaxNLocator(4, prune='lower'))
# axes[0].tick_params(axis='both', which='major', labelsize=10)
# axes[0].set_ylabel('$\ln (p)$', fontsize=12)

# for i in range(Npars):
#     for j in range(Nwalk):
#         axes[i+1].plot(range(Nstep), sampler.chain[j,:,i], c='gray',
#                        alpha=0.4)
#     axes[i+1].yaxis.set_major_locator(MaxNLocator(4, prune='lower'))
#     axes[i+1].tick_params(axis='both', which='major', labelsize=10)
#     axes[i+1].set_ylabel(names[i], fontsize=12)

# axes[-1].set_xlabel('Model Number', fontsize=12)
# fig.tight_layout(h_pad=0.1)
# fig.savefig('{0}_timeseries.png'.format(fn), dpi=720)
# plt.clf()
