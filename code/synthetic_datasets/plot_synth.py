from __future__ import print_function

import os
import corner
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from matplotlib import rc
from funcs import model_lum
from fit_stats import AIC, redchisq
from matplotlib.ticker import MaxNLocator


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
        "-b",
        "--n-burn",
        type=int,
        required=True,
        help=""
    )

    return parser.parse_args()


def create_filenames(GRB):
    # Data basename and directory
    data_basename = os.path.join("data", "synthetic_datasets")
    data_dirname = os.path.join(data_basename, GRB)

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
    fstats = os.path.join(data_dirname, "{0}_stats.txt".format(GRB))
    fres = os.path.join(data_basename, "{0}_results.csv".format(GRB))
    fplot_corner = os.path.join(plot_dirname, "{0}_corner.png".format(GRB))
    fplot_model = os.path.join(plot_dirname, "{0}_model.png".format(GRB))

    return fdata, fchain, fstats, fres, fplot_corner, fplot_model


# Dictionary of true values for burst types
truths = {
    'Humped': [1.0, 5.0, -3.0, 2.0, -1.0, 0.0],
    'Classic': [1.0, 5.0, -3.0, 3.0, -1.0, 0.0],
    'Sloped': [1.0, 1.0, -3.0, 2.0, 1.0, 1.0],
    'Stuttering': [1.0, 5.0, -5.0, 2.0, -1.0, 2.0]
}

# Use LaTeX in plots
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

# Arrays for upper and lower limits on parameters
u = np.array([10.0, 10.0, -2.0, np.log10(2000.0), 2.0, 3.0])
l = np.array([1.e-3, 0.69, -6.0, np.log10(50.0), -2.0, -1.0])

# Get command line arguments
args = parse_args()

# Create filenames
fdata, fchain, fstats, fres, fplot_corner, fplot_model = \
    create_filenames(args.grb)

# Read in data
data = np.loadtxt(fdata, delimiter=",", skiprows=1)
data = data[np.argsort(data[:,0])]
x = data[:,0]
y = data[:,1]
yerr = data[:,2]

# MCMC parameters
Npars = 6      # Number of fitting parameters
Nwalk = 50     # Number of walkers
Nstep = 20000  # Number of MCMC steps

# Read in chain & probability and reshape
cols = tuple(range(Npars))
skip = (args.n_burn * Nwalk) + 1
s = np.loadtxt(fchain, delimiter=',', skiprows=1, usecols=cols)

body = """{0}

MC Parameters:
Npars: {1}
Nwalk: {2}
Nstep: {3}
Number of samples burned: {4}
""".format(args.grb, Npars, Nwalk, Nstep, args.n_burn)

#=== Plotting ===#
# Latex strings for labels
names = [r'$B$', r'$P$', r'$\log_{10} (M_{\rm D,i})$',
         r'$\log_{10} (R_{\rm D})$', r'$\log_{10} (\epsilon)$',
         r'$\log_{10} (\delta)$']

# Corner plot
fig = corner.corner(
    s,
    labels=names,
    truths=np.array(truths[args.grb]),
    label_kwargs={'fontsize':12},
    quantiles=[0.025, 0.5, 0.975],
    plot_contours=True
)
fig.savefig("{0}".format(fplot_corner))
plt.clf()

# Correlations
corrs = []
for i in range(Npars):
    j = i + 1
    while j < Npars:
        corrs.append(np.corrcoef(s[:,i], s[:,j])[0,1])
        j += 1

# Results
s[:, 2:] = 10.0 ** s[:, 2:]  # Convert out of log-space

B, P, Md, Rd, eps, delt = map(
    lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
    zip(*np.percentile(s, [2.5, 50.0, 97.5], axis=0))
)

pars = [B[0], P[0], Md[0], Rd[0], eps[0], delt[0]]

# Write results
body += """\nResults:
           B (x10^15 G) = {0:.4f} (+{1:.4f}, -{2:.4f})\t(Truth: {3:.1f})
                 P (ms) = {4:.3f} (+{5:.3f}, -{6:.3f})\t(Truth: {7:.1f})
           Mdisc (Msol) = {8:.4} (+{9:.4}, -{10:.4})\t(Truth: {11:.6f})
             Rdisc (km) = {12:.1f} (+{12:.1f}, -{14:.1f})\t(Truth: {15:.1f})
                epsilon = {16:.2f} (+{17:.2f}, -{18:.2f})\t(Truth: {19:.2f})
                  delta = {20:.4} (+{21:.4}, -{22:.4})\t(Truth: {23:.1f})
""".format(B[0], B[1], B[2], truths[args.grb][0], P[0], P[1], P[2],
           truths[args.grb][1], Md[0], Md[1], Md[2], 10.0**truths[args.grb][2],
           Rd[0], Rd[1], Rd[2], 10.0**truths[args.grb][3], eps[0], eps[1],
           eps[2], 10.0**truths[args.grb][4], delt[0], delt[1], delt[2],
           10.0**truths[args.grb][5])

# Calculate model and fit statistics and write to file
ymod = model_lum(pars, xdata=x)
try:
    # Calculate AIC
    AIC = AIC(ymod, y, yerr, Npars)
    body += "\nAIC = {0:.3f}".format(AIC)
except:
    chisq = redchisq(y, ymod, sd=yerr)
    body += """\nChi Square = {0:.3f}
N = {1}
k = {2}""".format(chisq, len(y), Npars)

chisq_r = redchisq(y, ymod, deg=Npars, sd=yerr)
body += "\nReduced Chi Square = {0:.3f}".format(chisq_r)

body += ('\n\n{0}'.format(args.grb) + ' & $' +
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
         '{0}'.format(chisq_r) + '$ \\\\ [2pt]')

try:
    body += """\n\nCorrelations:
             B->P: {0:.3f}
     B->log10(Md): {1:.3f}
     B->log10(Rd): {2:.3f}
B->log10(epsilon): {3:.3f}
  B->log10(delta): {4:.3f}

     P->log10(Md): {5:.3f}
     P->log10(Rd): {6:.3f}
P->log10(epsilon): {7:.3f}
  P->log10(delta): {8:.3f}

     log10(Md)->log10(Rd): {9:.3f}
log10(Md)->log10(epsilon): {10:.3f}
  log10(Md)->log10(delta): {11:.3f}

log10(Rd)->log10(epsilon): {12:.3f}
  log10(Rd)->log10(delta): {13:.3f}

log10(epsilon)->log10(delta): {14:.3f}
""".format(corrs[0], corrs[1], corrs[2], corrs[3], corrs[4], corrs[5],
           corrs[6], corrs[7], corrs[8], corrs[9], corrs[10], corrs[11],
           corrs[12], corrs[13], corrs[14])
except:
    pass

# Smoothed model
fit = model_lum(pars)

# Write to a file
with open(fstats, 'w') as f:
    f.write(body)
print("Results written to: {0}".format(fstats))

# Plot the smoothed model
fig = plt.figure(figsize=(6, 4.5))
ax = fig.add_subplot(111)

ax.errorbar(x, y, yerr=yerr, fmt='.r', capsize=0.0)

try:
    ax.loglog(fit[0,:], fit[1,:], c='k')
    ax.loglog(fit[0,:], fit[2,:], c='k', ls='--')
    ax.loglog(fit[0,:], fit[3,:], c='k', ls=':')
except:
    ax.set_xscale('log')
    ax.set_yscale('log')

ax.set_xlim(1.0e0, 1.0e6)
ax.set_ylim(1.0e-8, 1.0e2)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.set_xlabel(r'Time (s)', fontsize=12)
ax.set_ylabel(r'Luminosity $\left(10^{50}~{\rm erg}~{\rm s}^{-1}\right)$',
              fontsize=12)
ax.set_title(r'{0}'.format(args.grb), fontsize=12)

fig.tight_layout()
fig.savefig("{0}".format(fplot_model))
plt.clf()

# Write smoothed model to a file
with open(fres, 'w') as f:
    for i in range(fit.shape[1]):
        f.write('{0}, {1}, {2}, {3}\n'.format(fit[0,i], fit[1,i], fit[2,i],
                fit[3,i]))
