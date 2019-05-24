import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.ticker import MaxNLocator


# Global variables
G = 6.674e-8                      # Newtons constant in cgs
c = 3.0e10                        # Speed on light in cm/s
R = 1.0e6                         # Radius of Magnetar (10km)
Msol = 1.99e33                    # Solar mass in grams
M = 1.4 * Msol                    # Mass of Magnetar in grams
I = (4.0 / 5.0) * M * (R ** 2.0)  # Moment of Inertia
n = 10.0                          # Propeller "switch-on"
alpha = 0.1                       # Sound speed prescription
cs7 = 1.0                         # Sound speed in disc - 10^7 cm/s
k = 0.9                           # Capping fraction
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


# Piro & Ott Model
def piroott(y, t, B, MdiscI, RdiscI, epsilon, delta):
    """
Function to be integrated by ODEINT following the model in Piro & Ott (2011).

    :param y: output from init_conds
    :param t: time points to solve for
    :param B: magnetic field strength - 10^15 G
    :param MdiscI: initial disc mass - solar masses
    :param RdiscI: disc radius - km
    :param epsilon: timescale ratio
    :param delta: mass ratio
    :return: array containing time derivatives of disc mass and angular
             frequency to be integrated by ODEINT
    """
    Mdisc, omega = y

    # Constants
    Rdisc = RdiscI * 1.0e5                 # Convert disc radius to cm
    tvisc = Rdisc / (alpha * cs7 * 1.0e7)  # Viscous timescale
    mu = 1.0e15 * B * (R ** 3.0)           # Magnetic Dipole Moment
    M0 = delta * MdiscI * Msol             # Global Mass Budget
    tfb = epsilon * tvisc                  # Fallback timescale

    # Radii - Alfven, Corotation, Light Cylinder
    Rm = ((mu ** (4.0 / 7.0)) * (GM ** (-1.0 / 7.0)) * (((3.0 * Mdisc) / tvisc)
          ** (-2.0 / 7.0)))
    Rc = (GM / (omega ** 2.0)) ** (1.0 / 3.0)
    Rlc = c / omega
    if Rm >= (k * Rlc):
        Rm = k * Rlc

    w = (Rm / Rc) ** (3.0 / 2.0)     # Fastness parameter
    bigT = 0.5 * I * (omega ** 2.0)  # Rotational energy
    modW = (0.6 * M * (c ** 2.0) * ((GM / (R * (c ** 2.0))) / (1.0 - 0.5 * (GM /
            (R * (c ** 2.0))))))     # Binding energy
    rot_param = bigT / modW          # Rotation parameter

    # Dipole torque
    Ndip = (-1.0 * (mu ** 2.0) * (omega ** 3.0)) / (6.0 * (c ** 3.0))

    # Efficiencies
    eta2 = 0.5 * (1.0 + np.tanh(n * (w - 1.0)))
    eta1 = 1.0 - eta2

    # Mass flow rates
    Mdotprop = eta2 * (Mdisc / tvisc)                        # Propelled
    Mdotacc = eta1 * (Mdisc / tvisc)                         # Accreted
    Mdotfb = (M0 / tfb) * ((t + tfb) / tfb) ** (-5.0 / 3.0)  # Fallback
    Mdotdisc = Mdotfb - Mdotprop - Mdotacc                   # Total

    # Accretion torque
    if rot_param > 0.27:
        Nacc = 0.0
    else:
        if Rm >= R:
            Nacc = ((GM * Rm) ** 0.5) * (Mdotacc - Mdotprop)
        else:
            Nacc = ((GM * R) ** 0.5) * (Mdotacc - Mdotprop)

    omegadot = (Nacc + Ndip) / I

    return np.array([Mdotdisc, omegadot])


# Bucciantini Model
def bucciantini(y, t, B, MdiscI, RdiscI, epsilon, delta):
    """
Function to be integrated by ODEINT following model in Bucciantini et al. (2006).
    :param y: output from init_conds
    :param t: time points to solve for
    :param B: magnetic field strength - 10^15 G
    :param MdiscI: initial disc mass - solar masses
    :param RdiscI: disc radius - km
    :param epsilon: timescale ratio
    :param delta: mass ratio
    :return: array containing disc mass and angular frequency time derivatives
             to be integrated by ODEINT
    """
    Mdisc, omega = y

    # Constants
    Rdisc = RdiscI * 1.0e5                 # Convert disc radius to cm
    tvisc = Rdisc / (alpha * cs7 * 1.0e7)  # Viscous timescale
    mu = 1.0e15 * B * (R ** 3.0)           # Magnetic Dipole Moment
    M0 = delta * MdiscI * Msol             # Global Mass Budget
    tfb = epsilon * tvisc                  # Fallback timescale

    # Radii - Alfven, Corotation, Light Cylinder
    Rm = ((mu ** (4.0 / 7.0)) * (GM ** (-1.0 / 7.0)) * (((3.0 * Mdisc) / tvisc)
          ** (-2.0 / 7.0)))
    Rc = (GM / (omega ** 2.0)) ** (1.0 / 3.0)
    Rlc = c / omega
    if Rm >= (k * Rlc):
        Rm = k * Rlc

    w = (Rm / Rc) ** (3.0 / 2.0)     # Fastness parameter
    bigT = 0.5 * I * (omega ** 2.0)  # Rotational energy
    modW = (0.6 * M * (c ** 2.0) * ((GM / (R * (c ** 2.0))) / (1.0 - 0.5 * (GM /
            (R * (c ** 2.0))))))     # Binding energy
    rot_param = bigT / modW          # Rotation parameter

    # Dipole torque
    Ndip = ((-2.0 / 3.0) * (((mu ** 2.0) * (omega ** 3.0)) / (c ** 3.0)) * ((Rlc
            / Rm) ** 3.0))

    # Efficiencies and Mass flow rates
    eta2 = 0.5 * (1.0 + np.tanh(n * (w - 1.0)))
    eta1 = 1.0 - eta2
    Mdotprop = eta2 * (Mdisc / tvisc)                        # Propelled
    Mdotacc = eta1 * (Mdisc / tvisc)                         # Accreted
    Mdotfb = (M0 / tfb) * ((t + tfb) / tfb) ** (-5.0 / 3.0)  # Fallback
    Mdotdisc = Mdotfb - Mdotprop - Mdotacc                   # Total

    # Accretion torque
    if rot_param > 0.27:
        Nacc = 0.
    else:
        if Rm >= R:
            Nacc = ((GM * Rm) ** 0.5) * (Mdotacc - Mdotprop)
        else:
            Nacc = ((GM * R) ** 0.5) * (Mdotacc - Mdotprop)

    omegadot = (Nacc + Ndip) / I

    return np.array([Mdotdisc, omegadot])


# Check if plots folder exists
if not (os.path.exists("plots")):
    os.mkdir("plots")

# === Calculations === #

# Variables
B = 1.0          # Magnetic field in 10^15 G
P = 5.0          # Spin period in milliseconds
MdiscI = 0.001   # Disc mass in solar masses
RdiscI = 1000.0  # Disc radius in km
epsilon = 0.1    # Timescale ratio
delta = 1.0      # Fallback mass ratio

# Constants and convert units to cgs
Rdisc = RdiscI * 1.0e5                 # Convert disc radius to cm
tvisc = Rdisc / (alpha * cs7 * 1.0e7)  # Viscous timescale
mu = 1.0e15 * B * (R ** 3.0)           # Magnetic Dipole Moment in G cm^3
M0 = delta * MdiscI * Msol             # Global Mass Budget
tfb = epsilon * tvisc                  # Fallback timescale

y0 = init_conds(MdiscI, P)  # Calculate initial conditions

# === Integrate the model === #

# Piro & Ott
po_soln = odeint(piroott, y0, tarr, args=(B, MdiscI, RdiscI, epsilon, delta))
po_Mdisc = np.array(po_soln[:, 0])
po_omega = np.array(po_soln[:, 1])

# Bucciantini
b_soln = odeint(bucciantini, y0, tarr, args=(B, MdiscI, RdiscI, epsilon, delta))
b_Mdisc = np.array(b_soln[:, 0])
b_omega = np.array(b_soln[:, 1])

# Recover radii, Mdotprop, and Mdotacc from returned Mdisc and omega
modW = (0.6 * M * (c ** 2.0) * ((GM / (R * (c ** 2.0))) / (1.0 - 0.5 * (GM / (R
        * (c ** 2.0))))))

# Piro & Ott
po_Rm = ((mu ** (4.0 / 7.0)) * (GM ** (-1.0 / 7.0)) * (((3.0 * po_Mdisc) / tvisc)
         ** (-2.0 / 7.0)))
po_Rc = (GM / (po_omega ** 2.0)) ** (1.0 / 3.0)
po_Rlc = c / po_omega

po_Rm = np.where((po_Rm >= k * po_Rlc), (k * po_Rlc), po_Rm)

po_w = (po_Rm / po_Rc) ** (3.0 / 2.0)
po_bigT = 0.5 * I * (po_omega ** 2.0)
po_beta = po_bigT / modW

po_Ndip = (-1.0 * (mu ** 2.0) * (po_omega ** 3.0)) / (6.0 * (c ** 3.0))

po_eta2 = 0.5 * (1.0 + np.tanh(n * (po_w - 1.0)))
po_eta1 = 1.0 - po_eta2
po_Mdotprop = po_eta2 * (po_Mdisc / tvisc)
po_Mdotacc = po_eta1 * (po_Mdisc / tvisc)

po_Nacc = np.zeros_like(po_Mdisc)
for i in range(len(po_Nacc)):
    if po_beta[i] > 0.27:
        po_Nacc[i] = 0.0
    else:
        if po_Rm[i] >= R:
            po_Nacc[i] = (((GM * po_Rm[i]) ** 0.5) * (po_Mdotacc[i] -
                          po_Mdotprop[i]))
        else:
            po_Nacc[i] = ((GM * R) ** 0.5) * (po_Mdotacc[i] - po_Mdotprop[i])

po_Ldip = ((mu ** 2.0) * (po_omega ** 4.0)) / (6.0 * (c ** 3.0))
po_Ldip = np.where((po_Ldip <= 0.0), 0.0, po_Ldip)
po_Ldip = np.where(np.isfinite(po_Ldip), po_Ldip, 0.0)

po_Lprop = ((-1.0 * po_Nacc * po_omega) - ((GM / po_Rm) * po_eta2 * (po_Mdisc /
            tvisc)))
po_Lprop = np.where((po_Lprop <= 0.0), 0.0, po_Lprop)
po_Lprop = np.where(np.isfinite(po_Lprop), po_Lprop, 0.0)

po_Ltot = po_Lprop + po_Ldip

# Bucciantini
b_Rm = ((mu ** (4.0 / 7.0)) * (GM ** (-1.0 / 7.0)) * (((3.0 * b_Mdisc) / tvisc)
        ** (-2.0 / 7.0)))
b_Rc = (GM / (b_omega ** 2.0)) ** (1.0 / 3.0)
b_Rlc = c / b_omega

b_Rm = np.where(b_Rm >= (k * b_Rlc), (k * b_Rlc), b_Rm)

b_w = (b_Rm / b_Rc) ** (3.0 / 2.0)
b_bigT = 0.5 * I * (b_omega ** 2.0)
b_beta = b_bigT / modW

b_Ndip = ((-2.0 / 3.0) * (((mu ** 2.0) * (b_omega ** 3.0)) / (c ** 3.0)) *
          ((b_Rlc / b_Rm) ** 3.0))

b_eta2 = 0.5 * (1.0 + np.tanh(n * (b_w - 1.0)))
b_eta1 = 1.0 - b_eta2
b_Mdotprop = b_eta2 * (b_Mdisc / tvisc)
b_Mdotacc = b_eta1 * (b_Mdisc / tvisc)

b_Nacc = np.zeros_like(b_Mdisc)
for i in range(len(b_Nacc)):
    if b_beta[i] > 0.27:
        b_Nacc[i] = 0.0
    else:
        if b_Rm[i] >= R:
            b_Nacc[i] = ((GM * b_Rm[i]) ** 0.5) * (b_Mdotacc[i] - b_Mdotprop[i])
        else:
            b_Nacc[i] = ((GM * R) ** 0.5) * (b_Mdotacc[i] - b_Mdotprop[i])

b_Ldip = ((mu ** 2.0) * (b_omega ** 4.0)) / (6.0 * (c ** 3.0))
b_Ldip = np.where(b_Ldip <= 0.0, 0.0, b_Ldip)
b_Ldip = np.where(np.isfinite(b_Ldip), b_Ldip, 0.0)

b_Lprop = ((-1.0 * b_Nacc * b_omega) - ((GM / b_Rm) * b_eta2 * (b_Mdisc /
           tvisc)))
b_Lprop = np.where(b_Lprop <= 0.0, 0.0, b_Lprop)
b_Lprop = np.where(np.isfinite(b_Lprop), b_Lprop, 0.0)

b_Ltot = b_Lprop + b_Ldip

# === Plotting === #

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7.5, 7))

ax1.semilogx(tarr, po_omega, c='k')
ax1.semilogx(tarr, b_omega, c='k', ls='--')
ax1.set_xlim(1.0e0, 1.0e6)
ax1.yaxis.set_major_locator(MaxNLocator(5))
ax1.tick_params(axis='both', which='major', labelsize=10)
ax1.set_xticks([])
ax1.set_ylabel('$\omega$ (${\\rm s}^{-1}$)', fontsize=12)
ax1.set_title("(a)", fontsize=10)

ax2.semilogx(tarr, po_Ndip/1.0e42, c='k')
ax2.set_xlim(1.0e0, 1.0e6)
ax2.yaxis.set_major_locator(MaxNLocator(5))
ax2.tick_params(axis='both', which='major', labelsize=10)
ax2.set_xticks([])
ax2.set_ylabel('$N_{\\rm dip}$ ($10^{42}$ ${\\rm erg}$ ${\\rm G}^{-1}$ '
               '${\\rm cm}^{-3}$) (solid line)', fontsize=12)
ax2.set_title("(b)", fontsize=10)

ax2twin = ax2.twinx()
ax2twin.semilogx(tarr, b_Ndip/1.0e44, c='k', ls='--')
ax2twin.tick_params(axis='both', which='major', labelsize=10)
ax2twin.set_xticks([])
ax2twin.set_ylabel('$N_{\\rm dip}$ ($10^{44}$ ${\\rm erg}$ ${\\rm G}^{-1}$ '
                   '${\\rm cm}^{-3}$) (dashed line)', fontsize=12)

ax3.loglog(tarr, po_Ldip/1.0e50, c='k')
ax3.loglog(tarr, b_Ldip/1.0e50, c='k', ls='--')
ax3.set_xlim(1.0e0, 1.0e6)
ax3.set_ylim(1.0e-8, 1.0e0)
ax3.set_xticks([1.0e0, 1.0e2, 1.0e4, 1.0e6])
ax3.set_yticks([1.0e-6, 1.0e-4, 1.0e-2, 1.0e0])
ax3.tick_params(axis='both', which='major', labelsize=10)
ax3.set_xlabel('Time (s)', fontsize=12)
ax3.set_ylabel('Dipole Luminosity ($10^{50}$ ${\\rm erg}$ ${\\rm s}^{-1}$)',
               fontsize=12)
ax3.set_title("(c)", fontsize=10)

ax4.loglog(tarr, po_Ltot/1.0e50, c='k')
ax4.loglog(tarr, b_Ltot/1.0e50, c='k', ls='--')
ax4.set_xlim(1.0e0, 1.0e6)
ax4.set_ylim(1.0e-8, 1.0e0)
ax4.set_xticks([1.0e0, 1.0e2, 1.0e4, 1.0e6])
ax4.set_yticks([1.0e-6, 1.0e-4, 1.0e-2, 1.0e0])
ax4.tick_params(axis='both', which='major', labelsize=10)
ax4.set_xlabel('Time (s)', fontsize=12)
ax4.set_ylabel('Total Luminosity ($10^{50}$ ${\\rm erg}$ ${\\rm s}^{-1}$)',
               fontsize=12)
ax4.set_title("(d)", fontsize=10)

fig.tight_layout()
fig.savefig("plots/figure_3.png")
