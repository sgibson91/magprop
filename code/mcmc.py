from __future__ import print_function

import os
import json
import argparse
import numpy as np
import emcee as em
import matplotlib.pyplot as plt
from magnetar import lnprob
from matplotlib.ticker import MaxNLocator

# Parameter names
names = ['$B$', '$P$', '$\log_{10} (M_{\\rm D,i})$', '$\log_{10} (R_{\\rm D})$',
         '$\log_{10} (\epsilon)$', '$\log_{10} (\delta)$']


def parse_args():
    parser = argparse.ArgumentParser(
        description=""
    )

    parser.add_argument(
        "-t",
        "--type",
        required=True,
        type=str,
        choices=["L", "S"],
        help=""
    )
    parser.add_argument(
        "--grb",
        required=True,
        type=str,
        help=""
    )
    parser.add_argument(
        "-l",
        "--label",
        required=True,
        type=str,
        help=""
    )
    parser.add_argument(
        "-p",
        "--n-pars",
        type=int,
        nargs="?",
        help=""
    )
    parser.add_argument(
        "-s",
        "--n-step",
        type=int,
        nargs="?",
        help=""
    )
    parser.add_argument(
        "-w",
        "--n-walk",
        type=int,
        nargs="?",
        help=""
    )
    parser.add_argument(
        "--burn",
        action="store_true",
        help=""
    )
    parser.add_argument(
        "--re-run",
        action="store_true",
        help=""
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
    fbad = os.path.join(data_dir, f"{args.grb}_bad.csv")
    fstats = os.path.join(data_dir, f"{args.grb}_stats.json")

    if args.burn:
        fchain = os.path.join(data_dir, f"{args.grb}_burn.csv")
        fn = os.path.join(data_dir, f"{args.grb}_burn")
        fplot = os.path.join(plot_dir, f"{args.grb}_burn_trace.png")
    else:
        fchain = os.path.join(data_dir, f"{args.grb}_chain.csv")
        fn = os.path.join(data_dir, f"{args.grb}_chain")
        fplot = os.path.join(plot_dir, f"{args.grb}_chain_trace.png")

    # Initialise bad parameter file
    f = open(fbad, "w")
    f.close()

    return fdata, fbad, fstats, fchain, fplot, fn


def main():
    # Get command line args
    args = parse_args()

    # Create filenames
    fdata, fbad, fstats, fchain, fplot, fn = create_filenames(args)

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


if __name__ == "__main__":
    main()


# # Read in data
# data = np.loadtxt(fdata)
# data = data[np.argsort(data[:,0])]
# x = data[:,0]
# y = data[:,1]
# yerr = data[:,2]

# # Calculate initial position
# p0 = np.array(truths[tag])
# pos = [p0 + 1.e-4 * np.random.randn(Npars) for i in range(Nwalk)]

# # Initialise Ensemble Sampler
# sampler = em.EnsembleSampler(Nwalk, Npars, lnprob, args=(x, y, yerr, u[:Npars],
#                              l[:Npars], fbad), threads=3)

# # Run MCMC
# sampler.run_mcmc(pos, Nstep)

# # Write full MCMC to file
# with open(fchain, 'w') as f:
#     f.write("{0}, {1}, {2}\n".format(Npars, Nwalk, Nstep))
#     for j in range(Nstep):
#         for i in range(Nwalk):
#             for k in range(Npars):
#                 f.write("{0:.6f}, ".format(sampler.chain[i,j,k]))
#             f.write("{0:.6f}\n".format(sampler.lnprobability[i,j]))

# # Write each individual parameter to it's own file
# for k in range(Npars):
#     with open("{0}_walk_{1}.csv".format(fn, k), 'w') as f:
#         for j in range(Nstep):
#             for i in range(Nwalk):
#                 if i == (Nwalk-1):
#                     f.write("{0:.6f}\n".format(sampler.chain[i,j,k]))
#                 else:
#                     f.write("{0:.6f}, ".format(sampler.chain[i,j,k]))

# # Write probability to it's own file
# with open("{0}_walk_lnp.csv".format(fn), 'w') as f:
#     for j in range(Nstep):
#         for i in range(Nwalk):
#             if i == (Nwalk-1):
#                 f.write("{0:.6f}\n".format(sampler.lnprobability[i,j]))
#             else:
#                 f.write("{0:.6f}, ".format(sampler.lnprobability[i,j]))

# # Acceptance fraction and convergence ratios
# body = """{0}
# Mean acceptance fraction: {1}
# Convergence ratios: {2}
# """.format(tag, np.mean(sampler.acceptance_fraction),
#            conv(sampler.chain[:,Nburn:,:], Npars, Nwalk, Nstep))
# print body
# with open(fout, 'a') as f:
#     f.write(body)

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
