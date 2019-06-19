from __future__ import print_function

import os
import sys
import argparse
import numpy as np
import emcee as em
import pandas as pd
import matplotlib.pyplot as plt
from funcs import model_lum
from mcmc_eqns import lnprob, conv
from matplotlib.ticker import MaxNLocator

truths = {
    'Humped': np.array([1.0, 5.0, -3.0, 2.0, -1.0, 0.0]),
    'Classic': np.array([1.0, 5.0, -3.0, 3.0, -1.0, 0.0]),
    'Sloped': np.array([1.0, 1.0, -3.0, 2.0,  1.0, 1.0]),
    'Stuttering': np.array([1.0, 5.0, -5.0, 2.0, -1.0, 2.0])
}

names = ['$B$', '$P$', '$\log_{10} (M_{\\rm D,i})$', '$\log_{10} (R_{\\rm D})$',
         '$\log_{10} (\epsilon)$', '$\log_{10} (\delta)$']


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

    return parser.parse_args()


def create_filenames(GRB):
    # Data basename
    basename = os.path.join("data", "synthetic_datasets")
    if not os.path.exists(basename):
        raise FileNotFoundError("Please make sure your chosen dataset exists.")
        sys.exit(1)

    # Directory name
    dirname = os.path.join(basename, GRB)
    if not os.path.exists(dirname):
        raise FileNotFoundError("Please make sure your chosen dataset exists.")
        sys.exit(1)

    # Construct filenames
    fdata = os.path.join(dirname, "{0}.csv".format(GRB))
    fchain = os.path.join(dirname, "{0}_chain.csv".format(GRB))
    fbad = os.path.join(dirname, "{0}_bad.csv".format(GRB))
    fstat = os.path.join(dirname, "{0}_stats.txt")
    fout = os.path.join(dirname, "{0}_out.txt".format(GRB))

    # Initialise bad parameter file
    f = open(fbad, "w")
    f.close()

    # Plot basename
    basename = os.path.join("plots", "synthetic_datasets")
    if not os.path.exists(basename):
        os.mkdir(basename)

    # Plot directory name
    dirname = os.path.join(basename, GRB)
    if not os.path.join(dirname):
        os.mkdir(dirname)

    fplot = os.path.join(dirname, "{0}_trace.png")

    return fdata, fchain, fbad, fstat, fout, fplot, dirname


def create_trace_plot(sampler, Npars, Nstep, Nwalk, fplot):
    fig, axes = plt.subplots(Npars+1, 1, sharex=True, figsize=(6,8))

    for i in range(Nwalk):
        axes[0].plot(range(Nstep), sampler.lnprobability[i,:], c='gray',
                    alpha=0.4)
    axes[0].yaxis.set_major_locator(MaxNLocator(4, prune='lower'))
    axes[0].tick_params(axis='both', which='major', labelsize=10)
    axes[0].set_ylabel('$\ln (p)$', fontsize=12)

    for i in range(Npars):
        for j in range(Nwalk):
            axes[i+1].plot(range(Nstep), sampler.chain[j,:,i], c='gray',
                        alpha=0.4)
        axes[i+1].yaxis.set_major_locator(MaxNLocator(4, prune='lower'))
        axes[i+1].tick_params(axis='both', which='major', labelsize=10)
        axes[i+1].set_ylabel(names[i], fontsize=12)

    axes[-1].set_xlabel('Model Number', fontsize=12)
    fig.tight_layout(h_pad=0.1)
    fig.savefig('{0}'.format(fplot))
    plt.clf()


def main():
    # Parse command line arguments
    args = parse_args()

    # Build filenames
    fdata, fchain, fbad, fstat, fout, fplot, fn = create_filenames(args.grb)

    # Read in data
    data = pd.read_csv(fdata)
    x = data["x"]
    y = data["y"]
    yerr = data["yerr"]

    # MCMC parameters
    Npars = 6      # Number of fitting parameters
    Nwalk = 50     # Number of walkers
    Nstep = 20000  # Number of MCMC steps

    # Calculate initial position
    p0 = np.array(truths[GRB])
    pos = [p0 + 1.e-4 * np.random.randn(Npars) for i in range(Nwalk)]

    # Initialise Ensemble Sampler
    sampler = em.EnsembleSampler(Nwalk, Npars, lnprob, args=(x, y, yerr, fbad),
                                 threads=3)

    # Run MCMC
    sampler.run_mcmc(pos, Nstep)

    # Write full MCMC to file
    with open(fchain, 'w') as f:
        f.write("{0}, {1}, {2}\n".format(Npars, Nwalk, Nstep))
        for j in range(Nstep):
            for i in range(Nwalk):
                for k in range(Npars):
                    f.write("{0:.6f}, ".format(sampler.chain[i,j,k]))
                f.write("{0:.6f}\n".format(sampler.lnprobability[i,j]))

    # Write each individual parameter to it's own file
    for k in range(Npars):
        with open("{0}_{1}.csv".format(fn, k), 'w') as f:
            for j in range(Nstep):
                for i in range(Nwalk):
                    if i == (Nwalk-1):
                        f.write("{0:.6f}\n".format(sampler.chain[i,j,k]))
                    else:
                        f.write("{0:.6f}, ".format(sampler.chain[i,j,k]))

    # Write probability to it's own file
    with open("{0}_lnp.csv".format(fn), 'w') as f:
        for j in range(Nstep):
            for i in range(Nwalk):
                if i == (Nwalk-1):
                    f.write("{0:.6f}\n".format(sampler.lnprobability[i,j]))
                else:
                    f.write("{0:.6f}, ".format(sampler.lnprobability[i,j]))

    # Acceptance fraction and convergence ratios
    body = """{0}
    Mean acceptance fraction: {1}
    Convergence ratios: {2}
    """.format(GRB, np.mean(sampler.acceptance_fraction),
            conv(sampler.chain[:,Nburn:,:], Npars, Nwalk, Nstep))
    print(body)
    with open(fout, 'a') as f:
        f.write(body)

    # Time series
    create_trace_plot(sampler, Npars, Nstep, Nwalk, fplot)


if __name__ == "__main__":
    main()
