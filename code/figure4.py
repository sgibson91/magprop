import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.ticker import MaxNLocator


# Check if sub-directory 'plots' exists - if not, create it
directory = os.path.join(os.getcwd(), "plots")
if not os.path.exists(directory):
    os.mkdir(directory)


# Global constants
G = 6.674e-8                      # Gravitational constant (cgs)
c = 3.0e10                        # Speed of light (cm/s)
R = 1.0e6                         # Magnetar radius (cm)
Msol = 1.99e33                    # Solar mass (cgs)
M = 1.4 * Msol                    # Magnetar mass
I = (4.0 / 5.0) * M * (R ** 2.0)  # Moment of Inertia
alpha = 0.1                       # Sound speed prescription
cs7 = 1.0                         # Sound speed in disc - 10^7 cm/s
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
def odes(y, t, B, MdiscI, RdiscI, epsilon, delta, n, alpha=0.1, cs7=1.0, k=0.9):
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
    Rm = ((mu ** (4.0 / 7.0)) * (GM ** (-1.0 / 7.0)) * (((3.0 * Mdisc) / tvisc) **
          (-2.0 / 7.0)))
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
    Mdotfb = (M0 / tfb) * ((t + tfb) / tfb) ** (-5.0 / 3.0)  # Fallback
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


fig, axes = plt.subplots(4, 3, sharex=True, figsize=(11, 11))

params = {"Humped": [1.0, 5.0, 0.001, 100.0, 0.1, 1.0, 10.0],
          "Classic": [1.0, 5.0, 0.001, 1000.0, 0.1, 1.0, 10.0],
          "Sloped": [1.0, 1.0, 0.001, 100.0, 10.0, 10.0, 1.0],
          "Stuttering": [1.0, 5.0, 1.e-5, 100.0, 0.1, 100.0, 10.0]}

grbs = ["Humped", "Classic", "Sloped", "Stuttering"]

for z, grb in enumerate(grbs):

    B, P, MdiscI, RdiscI, epsilon, delta, n = params[grb]

    # Constants and convert units to cgs
    Rdisc = RdiscI * 1.0e5                 # Convert disc radius to cm
    tvisc = Rdisc / (alpha * cs7 * 1.0e7)  # Viscous timescale
    mu = 1.0e15 * B * (R ** 3.0)           # Magnetic Dipole Moment
    M0 = delta * MdiscI * Msol             # Global Mass Budget
    tfb = epsilon * tvisc                  # Fallback timescale

    y0 = init_conds(MdiscI, P)  # Calculate initial conditions

    # Solve equations
    soln = odeint(odes, y0, tarr, args=(B, MdiscI, RdiscI, epsilon, delta, n))
    Mdisc = np.array(soln[:, 0])
    omega = np.array(soln[:, 1])

    Rm = ((mu ** (4.0 / 7.0)) * (GM ** (-1.0 / 7.0)) * (((3.0 * Mdisc) / tvisc) **
          (-2.0 / 7.0)))
    Rc = (GM / (omega ** 2.0)) ** (1.0 / 3.0)
    Rlc = c / omega
    Rm = np.where(Rm >= (0.9 * Rlc), 0.9 * Rlc, Rm)

    w = (Rm / Rc) ** (3.0 / 2.0)     # Fastness parameter
    bigT = 0.5 * I * (omega ** 2.0)  # Rotational energy
    modW = (0.6 * M * (c ** 2.0) * ((GM / (R * (c ** 2.0))) / (1.0 - 0.5 * (GM /
            (R * (c ** 2.0))))))     # Binding energy
    rot_param = bigT / modW          # Rotation parameter

    # Efficiencies and mass flow rates
    eta2 = 0.5 * (1.0 + np.tanh(n * (w - 1.0)))
    eta1 = 1.0 - eta2
    Mdotprop = eta2 * (Mdisc / tvisc)  # Propelled
    Mdotacc = eta1 * (Mdisc / tvisc)   # Accreted

    Nacc = np.zeros_like(Mdisc)
    for i in range(len(Nacc)):
        if rot_param[i] > 0.27:  # Prevents magnetar break-up
            Nacc[i] = 0.0
        else:
            # Accretion torque
            if Rm[i] >= R:
                Nacc[i] = ((GM * Rm[i]) ** 0.5) * (Mdotacc[i] - Mdotprop[i])
            else:
                Nacc[i] = ((GM * R) ** 0.5) * (Mdotacc[i] - Mdotprop[i])

    # Luminosities
    Ldip = ((mu ** 2.0) * (omega ** 4.0)) / (6.0 * (c ** 3.0))
    Ldip = np.where(Ldip <= 0.0, 0.0, Ldip)
    Ldip = np.where(np.isfinite(Ldip), Ldip, 0.0)

    Lprop = (-1.0 * Nacc * omega) - ((GM / Rm) * eta2 * (Mdisc / tvisc))
    Lprop = np.where(Lprop <= 0.0, 0.0, Lprop)
    Lprop = np.where(np.isfinite(Lprop), Lprop, 0.0)

    Ltot = Lprop + Ldip

    # Plotting
    axes[z, 0].loglog(tarr, Ldip/1.0e50, c='k', ls=':')
    axes[z, 0].loglog(tarr, Lprop/1.0e50, c='k', ls='--')
    axes[z, 0].loglog(tarr, Ltot/1.0e50, c='k', ls='-')
    axes[z, 0].set_xlim(1.0e0, 1.0e6)
    axes[z, 0].set_ylim(1.0e-7, 1.0e0)
    axes[z, 0].tick_params(axis='both', which='major', labelsize=10)
    axes[z, 0].set_ylabel('Luminosity ($10^{50}$ erg ${\\rm s}^{-1}$)', fontsize=12)

    axes[z, 1].semilogx(tarr, 1.0e4*(Mdotacc/Msol), c='k')
    axes[z, 1].semilogx(tarr, 1.0e4*(Mdotprop/Msol), c='k', ls='--')
    axes[z, 1].set_xlim(1.0e0, 1.0e6)
    axes[z, 1].tick_params(axis='both', which='major', labelsize=10)
    axes[z, 1].set_ylabel('$10^{-4}$ $M_{\odot}$ ${\\rm s}^{-1}$', fontsize=12)

    axes[z, 2].axhline(RdiscI, c='k', ls='-.')
    axes[z, 2].axhline(10.0, c='k', ls='-.')
    axes[z, 2].loglog(tarr, Rc/1.0e5, c='k', ls=':')
    axes[z, 2].loglog(tarr, Rm/1.0e5, c='k', ls='--')
    axes[z, 2].loglog(tarr, Rlc/1.0e5, c='k')
    axes[z, 2].set_xlim(1.0e0, 1.0e6)
    axes[z, 2].set_ylim(1.0e0, 1.0e4)
    axes[z, 2].tick_params(axis='both', which='major', labelsize=10)
    axes[z, 2].set_ylabel('Radial distance (km)', fontsize=12)

# Tidying up plots
for i in range(4):
    axes[i, 0].set_yticks([1.0e-6, 1.0e-4, 1.0e-2, 1.0e0])
    axes[i, 1].yaxis.set_major_locator(MaxNLocator(5, prune='lower'))
    axes[i, 2].set_yticks([1.0e1, 1.0e2, 1.0e3, 1.0e4])

for i in range(3):
    axes[-1, i].set_xticks([1.0e0, 1.0e2, 1.0e4, 1.0e6])
    axes[-1, i]. set_xlabel('Time (s)', fontsize=12)

axes[2, 0].set_ylim(1.0e-7, 1.0e2)
axes[2, 0].set_yticks([1.0e-6, 1.0e-4, 1.0e-2, 1.0e0, 1.0e2])

fig.tight_layout(h_pad=0.0)

fig.savefig(os.path.join(directory, "figure4.png"))
