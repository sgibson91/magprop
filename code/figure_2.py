import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Get filepaths
HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.split(HERE)[0]

# Global constants
G = 6.674e-8                      # Gravitational constant (cgs)
c = 3.0e10                        # Speed of light (cm/s)
R = 1.0e6                         # Magnetar radius (cm)
Msol = 1.99e33                    # Solar mass (cgs)
M = 1.4 * Msol                    # Magnetar mass
I = (4.0 / 5.0) * M * (R ** 2.0)  # Moment of Inertia
GM = G * M
tarr = np.logspace(0.0, 6.0, num=10001, base=10.0)


# Calculate initial conditions to pass to odeint
def init_conds(MdiscI, P):
    """
Function to convert a disc mass from solar masses to grams and an initial spin
period in milliseconds into an angular frequency.

    :param MdiscI: disc mass - solar masses
    :param P: initial spin period - milliseconds
    :return: an array containing the disc mass in grams and the angular freq.
    """
    Mdisc0 = MdiscI * Msol                 # Disc mass
    omega0 = (2.0 * np.pi) / (1.0e-3 * P)  # Angular frequency

    return np.array([Mdisc0, omega0])


# Model to be passed to odeint to calculate Mdisc and omega
def odes(y, t, B, MdiscI, RdiscI, epsilon, delta, n=1.0, alpha=0.1, cs7=1.0,
         k=0.9):
    """
Function to be passed to ODEINT to calculate the disc mass and angular frequency
over time.

    :param y: output from init_conds
    :param t: time points to solve equations for
    :param B: magnetic field strength - 10^15 G
    :param MdiscI: initial disc mass - solar masses
    :param RdiscI: disc radius - km
    :param epsilon: timescale ratio
    :param delta: mass ratio
    :param n: propeller "switch-on"
    :param alpha: sound speed prescription
    :param cs7: sound speed in disc - 10^7 cm/s
    :param k: capping fraction
    :return: time derivatives of disc mass and angular frequency to be integrated
             by ODEINT
    """
    # Initial conditions
    Mdisc, omega = y

    # Constants
    Rdisc = RdiscI * 1.0e5                 # Disc radius
    tvisc = Rdisc / (alpha * cs7 * 1.0e7)  # Viscous timescale
    mu = 1.0e15 * B * (R ** 3.0)           # Magnetic Dipole Moment
    M0 = delta * MdiscI * Msol             # Global Mass Budget
    tfb = epsilon * tvisc                  # Fallback timescale

    # Radii -- Alfven, Corotation, Light Cylinder
    Rm = ((mu ** (4.0 / 7.0)) * (GM ** (-1.0 / 7.0)) * (((3.0 * Mdisc) / tvisc)
          ** (-2.0 / 7.0)))
    Rc = (GM / (omega ** 2.0)) ** (1.0 / 3.0)
    Rlc = c / omega
    # Cap Alfven radius
    if Rm >= (k * Rlc):
        Rm = k * Rlc

    w = (Rm / Rc) ** (3.0 / 2.0)  # Fastness Parameter

    bigT = 0.5 * I * (omega ** 2.0)  # Rotational energy
    modW = (0.6 * M * (c ** 2.0) * ((GM / (R * (c ** 2.0))) / (1.0 - 0.5 * (GM /
            (R * (c ** 2.0))))))     # Binding energy
    rot_param = bigT / modW          # Rotation parameter

    # Dipole torque
    Ndip = (-1.0 * (mu ** 2.0) * (omega ** 3.0)) / (6.0 * (c ** 3.0))

    # Mass flow rates
    eta2 = 0.5 * (1.0 + np.tanh(n * (w - 1.0)))
    eta1 = 1.0 - eta2
    Mdotprop = eta2 * (Mdisc / tvisc)  # Propelled
    Mdotacc = eta1 * (Mdisc / tvisc)   # Accretion
    Mdotfb = (M0 / tfb) * (((t + tfb) / tfb) ** (-5.0 / 3.0))
    Mdotdisc = Mdotfb - Mdotprop - Mdotacc

    if rot_param > 0.27:
        Nacc = 0.0  # Prevents magnetar break-up
    else:
        # Accretion torque
        if Rm >= R:
            Nacc = ((GM * Rm) ** 0.5) * (Mdotacc - Mdotprop)
        else:
            Nacc = ((GM * R) ** 0.5) * (Mdotacc - Mdotprop)

    omegadot = (Nacc + Ndip) / I  # Angular frequency time derivative

    return np.array([Mdotdisc, omegadot])


# Check if plots folder exists
if not (os.path.exists(os.path.join(ROOT, "plots"))):
    os.mkdir(os.path.join(ROOT, "plots"))

# Variable set-up
B = 1.0         # Magnetic field strength - 10^15 G
P = 5.0         # Initial spin period - milliseconds
MdiscI = 0.001  # Disc mass - solar masses
RdiscI = 100.0  # Disc radius - km

mu = 1.0e15 * B * (R ** 3.0)         # Magnetic dipole moment
Rdisc = RdiscI * 1.0e5               # Disc radius in km
tvisc = Rdisc / (0.1 * 1.0 * 1.0e7)  # Viscous timescale

eps_vals = [1.0, 1.0, 10.0, 10.0]   # Epsilon values
delt_vals = [1.0, 10.0, 1.0, 10.0]  # Delta values
vals = zip(eps_vals, delt_vals)

y = init_conds(MdiscI, P)  # Calculate initial conditions

colours = ['r', 'r', 'g', 'g']      # For plotting
linestyle = ['-', '--', '-', '--']  # For plotting

# Initialise plots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax3.axhline(1.0, ls='--', c='k')

# Loop over constants
for i, consts in enumerate(vals):
    epsilon, delta = consts

    soln = odeint(odes, y, tarr, args=(B, MdiscI, RdiscI, epsilon, delta))
    Mdisc = np.array(soln[:, 0])
    omega = np.array(soln[:, 1])

    Rm = ((mu ** (4.0 / 7.0)) * (GM ** (-1.0 / 7.0)) * (((3.0 * Mdisc) / tvisc)
          ** (-2.0 / 7.0)))
    Rc = (GM / (omega ** 2.0)) ** (1.0 / 3.0)

    ax1.loglog(tarr, Mdisc/Msol, ls=linestyle[i], c=colours[i],
               label='$\epsilon$ = {0:.0f}; $\delta$ = {1:.0f}'.format(
               epsilon, delta))
    ax2.semilogx(tarr, omega, ls=linestyle[i], c=colours[i])
    ax3.semilogx(tarr, Rm/Rc, ls=linestyle[i], c=colours[i])

ax1.set_xlim(1.0e0, 1.0e6)
ax1.set_ylim(bottom=1.0e-8)
ax2.set_xlim(1.0e0, 1.0e6)
ax3.set_xlim(1.0e0, 1.0e6)
ax3.set_ylim(0.0, 2.0)
ax1.tick_params(axis='both', which='major', labelsize=8)
ax2.tick_params(axis='both', which='major', labelsize=8)
ax3.tick_params(axis='both', which='major', labelsize=8)

ax1.legend(loc='upper right', fontsize=8)
ax3.set_xlabel('Time (s)', fontsize=10)

ax1.set_ylabel('$M_{\\rm D}$ (${\\rm M}_{\odot}$)', fontsize=10)
ax2.set_ylabel('$\omega$ (${\\rm s}^{-1}$)', fontsize=10)
ax3.set_ylabel('$r_{\\rm m}/r_{\\rm c}$', fontsize=10)

fig.tight_layout(h_pad=0.2)
fig.savefig(os.path.join(ROOT, "plots/figure_2.png"))
