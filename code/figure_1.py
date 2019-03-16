import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


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
def odes(y, t, B, MdiscI, RdiscI, epsilon, delta, n, alpha, cs7, k):
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


# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output-file', required=True,
                    help="Output file to save figure to.")
args = parser.parse_args()

# Variable set- up
B = 1.0         # Magnetic field strength - 10^15 G
P = 1.0         # Initial spin period - milliseconds
MdiscI = 0.1    # Disc mass - solar masses
RdiscI = 100.0  # Disc radius - km
epsilon = 1.0   # Timescale ratio
delta = 1.0e-6  # Mass ratio
alpha = 0.1     # Sound speed prescription
cs7 = 1.0       # Sound speed in disc - 10^7 cm/s
k = 0.9         # Capping fraction

y0 = init_conds(MdiscI, P)  # Calculate initial conditions

mu = 1.0e15 * B * (R ** 3.0)           # Magnetic dipole moment
Rdisc = RdiscI * 1.0e5                 # Disc radius - cm
tvisc = Rdisc / (alpha * cs7 * 1.0e7)  # Viscous timescale

n_vals = [1.0, 10.0, 50.0]  # Propeller "switch-on" values
lines = [':', '--', '-']    # Linestyles for plotting

for n, ln in zip(n_vals, lines):
    pars = [B, P, MdiscI, RdiscI, epsilon, delta, n]
    soln = odeint(odes, y0, tarr, args=(B, MdiscI, RdiscI, epsilon, delta, n,
                  alpha, cs7, k))

    Mdisc = np.array(soln[:, 0])
    omega = np.array(soln[:, 1])

    Rm = ((mu ** (4.0 / 7.0)) * (GM ** (-1.0 / 7.0)) * (((3.0 * Mdisc) / tvisc)
          ** (-2.0 / 7.0)))
    Rc = (GM / (omega ** 2.0)) ** (1.0 / 3.0)
    Rlc = c / omega
    Rm = np.where(Rm >= (0.9 * Rlc), (0.9 * Rlc), Rm)

    w = np.sort((Rm / Rc) ** (3.0 / 2.0))
    eta2 = 0.5 * (1.0 + np.tanh(n * (w - 1.0)))

    plt.plot(w, eta2, c='k', ls=ln)

plt.xlim(0.8, 1.2)
plt.ylim(-0.1, 1.1)
plt.xticks([0.8, 0.9, 1.0, 1.1, 1.2])
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('Fastness Parameter, $\Omega$', fontsize=12)
plt.ylabel('Propeller Efficiency, $\eta_2$', fontsize=12)
plt.tight_layout()
plt.savefig(args.output_file)
