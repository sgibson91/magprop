from __future__ import print_function

import os
import json
import corner
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from funcs import model_lum
from fit_stats import aicc, redchisq
from matplotlib.ticker import MaxNLocator

# Dictionary of true values for burst types
truths = {
    'Humped': [1.0, 5.0, -3.0, 2.0, -1.0, 0.0],
    'Classic': [1.0, 5.0, -3.0, 3.0, -1.0, 0.0],
    'Sloped': [1.0, 1.0, -3.0, 2.0, 1.0, 1.0],
    'Stuttering': [1.0, 5.0, -5.0, 2.0, -1.0, 2.0]
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Name of synthetic GRB and number of burn-in samples to parse"
    )

    parser.add_argument(
        "--grb",
        required=True,
        choices=["Humped", "Classic", "Sloped", "Stuttering"],
        help="Name of synthetic GRB dataset"
    )
    parser.add_argument(
        "-b",
        "--n-burn",
        type=int,
        required=True,
        help="Number of burn-in samples to remove"
    )

    return parser.parse_args()


def create_filenames(GRB):
    # Data basename and directory
    data_basename = os.path.join("data", "synthetic_datasets")
    data_dirname = os.path.join(data_basename, GRB)
    if not os.path.exists(data_dirname):
        raise FileNotFoundError(
            "Please ensure your synthetic dataset and MCMC files have been generated."
        )

    # Plot basename as directory
    plot_basename = os.path.join("plots", "synthetic_datasets")
    if not os.path.exists(plot_basename):
        os.mkdir(plot_basename)

    plot_dirname = os.path.join(plot_basename, GRB)
    if not os.path.exists(plot_dirname):
        os.mkdir(plot_dirname)

    # Construct filenames
    fdata = os.path.join(data_dirname, "{0}.csv".format(GRB))
    fchain = os.path.join(data_dirname, "{0}_chain.csv".format(GRB))
    fstats = os.path.join(data_dirname, "{0}_stats.json".format(GRB))
    fres = os.path.join(data_dirname, "{0}_model.csv".format(GRB))
    finfo = os.path.join(data_dirname, "{0}_info.json".format(GRB))
    fplot_corner = os.path.join(plot_dirname, "{0}_corner.png".format(GRB))
    fplot_model = os.path.join(plot_dirname, "{0}_model.png".format(GRB))

    return fdata, fchain, fstats, fres, finfo, fplot_corner, fplot_model


def create_corner_plot(s, GRB, fplot):
    # Latex strings for labels
    names = [r'$B$', r'$P$', r'$\log_{10} (M_{\rm D,i})$',
             r'$\log_{10} (R_{\rm D})$', r'$\log_{10} (\epsilon)$',
             r'$\log_{10} (\delta)$']

    fig = corner.corner(
        s,
        labels=names,
        truths=np.array(truths[GRB]),
        label_kwargs={'fontsize':12},
        quantiles=[0.025, 0.5, 0.975],
        plot_contours=True
    )
    fig.savefig("{0}".format(fplot))
    plt.clf()


def create_model_plot(model, data, GRB, fplot):
    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111)

    ax.errorbar(data["x"], data["y"], yerr=data["yerr"], fmt='.r', capsize=0.0)

    try:
        ax.loglog(model[0, :], model[1, :], c='k')
        ax.loglog(model[0, :], model[2, :], c='k', ls='--')
        ax.loglog(model[0, :], model[3, :], c='k', ls=':')
    except:
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.set_xlim(1.0e0, 1.0e6)
    ax.set_ylim(1.0e-8, 1.0e2)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_xlabel(r'Time (s)', fontsize=12)
    ax.set_ylabel(r'Luminosity $\left(10^{50} {\rm erg} {\rm s}^{-1}\right)$',
                fontsize=12)
    ax.set_title(r'{0}'.format(GRB), fontsize=12)

    fig.tight_layout()
    fig.savefig("{0}".format(fplot))
    plt.clf()


def main():
    # Get command line arguments
    args = parse_args()

    # Create filenames
    fdata, fchain, fstats, fres, finfo, fplot_corner, fplot_model = \
        create_filenames(args.grb)

    # Read in MCMC parameters
    with open(finfo, "r") as stream:
        mc_pars = json.load(stream)

    Npars = mc_pars["Npars"]
    Nwalk = mc_pars["Nwalk"]
    Nstep = mc_pars["Nstep"]

    # Read in data
    data = pd.read_csv(fdata)

    # Read in chain & probability and reshape
    cols = tuple(range(Npars))
    skip = (args.n_burn * Nwalk) + 1
    samples = np.loadtxt(fchain, delimiter=',', skiprows=1, usecols=cols)

    stats = {"Nburn": args.n_burn}

    # Create corner plot
    create_corner_plot(samples, args.grb, fplot_corner)

    # Correlations
    corrs = []
    for i in range(Npars):
        j = i + 1
        while j < Npars:
            corrs.append(np.corrcoef(samples[:,i], samples[:,j])[0,1])
            j += 1
    stats["correlations"] = corrs

    # Results
    samples[:, 2:] = 10.0 ** samples[:, 2:]  # Convert out of log-space

    B, P, Md, Rd, eps, delt = map(
        lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
        zip(*np.percentile(samples, [2.5, 50.0, 97.5], axis=0))
    )

    pars = [B[0], P[0], Md[0], Rd[0], eps[0], delt[0]]
    stats["pars"] = {
        "B": B,
        "P_i": P,
        "MdiscI": Md,
        "RdiscI": Rd,
        "epsilon": eps,
        "delta": delt,
        "truths": truths[args.grb]
    }


    # Calculate model and fit statistics and write to file
    ymod = model_lum(pars, xdata=data["x"])
    try:
        # Calculate AIC
        AIC = aicc(ymod, data["y"], data["yerr"], Npars)
        stats["stats"] = {"aicc": AIC}
    except:
        chisq = redchisq(data["y"], ymod, sd=data["yerr"])
        stats["stats"] = {"chi_square": chisq}

    chisq_r = redchisq(data["y"], ymod, deg=Npars, sd=data["yerr"])
    stats["stats"] = {"chi_square_red": chisq_r}

    stats["latex"] = (
        '\n\n{0}'.format(args.grb) + ' & $' +
        '{0}'.format(B[0]) + '^{+' +
        '{0}'.format(B[1]) + '}_{-' +
        '{0}'.format(B[2]) + '}$ & $' +
        '{0}'.format(P[0]) + '^{+' +
        '{0}'.format(P[1]) + '}_{-' +
        '{0}'.format(P[2]) + '}$ & $' +
        '{0}'.format(Md[0]) + '^{+' +
        '{0}'.format(Md[1]) + '}_{-' +
        '{0}'.format(Md[2]) + '}$ & $' +
        '{0}'.format(Rd[0]) + '^{+' +
        '{0}'.format(Rd[1]) + '}_{-' +
        '{0}'.format(Rd[2]) + '}$ & $' +
        '{0}'.format(eps[0]) + '^{+' +
        '{0}'.format(eps[1]) + '}_{-' +
        '{0}'.format(eps[2]) + '}$ & $' +
        '{0}'.format(delt[0]) + '^{+' +
        '{0}'.format(delt[1]) + '}_{-' +
        '{0}'.format(delt[2]) + '}$ & $' +
        '{0}'.format(chisq_r) + '$ \\\\ [2pt]'
    )

    # Smoothed model
    fit = model_lum(pars)

    # Write to a file
    with open(fstats, 'w') as f:
        json.dump(stats, f)
    print("Results written to: {0}".format(fstats))

    # Plot the smoothed model
    create_model_plot(fit, data, args.grb, fplot_model)

    # Write smoothed model to a file
    model = pd.DataFrame({
        "t": fit[0, :],
        "Ltot": fit[1, :],
        "Lprop": fit[2, :],
        "Ldip": fit[3, :]
    })
    model.to_csv(fres)


if __name__ == "__main__":
    main()
