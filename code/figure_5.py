import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Get filepaths
HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.split(HERE)[0]

# Global constants
G = 6.674e-8  # Gravitational constant - cgs units
c = 3.0e10  # Light speed - cm/s
R = 1.0e6  # Magnetar radius - cm
Rkm = 10.0  # Magnetar radius - km
omass = 1.4  # Magnetar mass - Msol
Msol = 1.99e33  # Solar mass - grams
M = omass * Msol  # Magnetar mass - grams
I = (4.0 / 5.0) * M * (R ** 2.0)  # Moment of inertia
alpha = 0.1  # Sound speed prescription
cs7 = 1.0  # Sound speed in disc - 10^7 cm/s
k = 0.9  # Capping fraction
j = 1.0e6  # Duration of plot
propeff = 1.0  # Propeller energy-to-luminosity conversion efficiency
dipeff = 1.0  # Dipole energy-to-luminosity conversion efficiency
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
    Mdisc0 = MdiscI * Msol  # Disc mass
    omega0 = (2.0 * np.pi) / (1.0e-3 * P)  # Angular frequency

    return np.array([Mdisc0, omega0])


# Model to be passed to odeint to calculate Mdisc and omega
def odes(y, t, B, MdiscI, RdiscI, epsilon, delta, n=1.0, alpha=0.1, cs7=1.0, k=0.9):
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
    Rdisc = RdiscI * 1.0e5  # Disc radius
    tvisc = Rdisc / (alpha * cs7 * 1.0e7)  # Viscous timescale
    mu = 1.0e15 * B * (R ** 3.0)  # Magnetic Dipole Moment
    M0 = delta * MdiscI * Msol  # Global Mass Budget
    tfb = epsilon * tvisc  # Fallback timescale

    # Radii -- Alfven, Corotation, Light Cylinder
    Rm = (
        (mu ** (4.0 / 7.0))
        * (GM ** (-1.0 / 7.0))
        * (((3.0 * Mdisc) / tvisc) ** (-2.0 / 7.0))
    )
    Rc = (GM / (omega ** 2.0)) ** (1.0 / 3.0)
    Rlc = c / omega
    # Cap Alfven radius
    if Rm >= (k * Rlc):
        Rm = k * Rlc

    w = (Rm / Rc) ** (3.0 / 2.0)  # Fastness Parameter

    bigT = 0.5 * I * (omega ** 2.0)  # Rotational energy
    modW = (
        0.6
        * M
        * (c ** 2.0)
        * ((GM / (R * (c ** 2.0))) / (1.0 - 0.5 * (GM / (R * (c ** 2.0)))))
    )  # Binding energy
    rot_param = bigT / modW  # Rotation parameter

    # Dipole torque
    Ndip = (-1.0 * (mu ** 2.0) * (omega ** 3.0)) / (6.0 * (c ** 3.0))

    # Mass flow rates
    eta2 = 0.5 * (1.0 + np.tanh(n * (w - 1.0)))
    eta1 = 1.0 - eta2
    Mdotprop = eta2 * (Mdisc / tvisc)  # Propelled
    Mdotacc = eta1 * (Mdisc / tvisc)  # Accretion
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


def model_lc(
    pars, dipeff=1.0, propeff=1.0, f_beam=1.0, n=1.0, alpha=0.1, cs7=1.0, k=0.9
):
    """
Function to calculate the model light curve for a given set of parameters.

    :param pars: list of input parameters including:
                   * B: magnetic field strenght - 10^15 G
                   * P: initial spin period - milliseconds
                   * MdiscI: initial disc mass - solar masses
                   * RdiscI: disc radius - km
                   * epsilon: timescale ratio
                   * delta: mass ratio
    :param dipeff: dipole energy-to-luminosity conversion efficiency
    :param propeff: propeller energy-to-luminosity conversion efficiency
    :param f_beam: beaming factor
    :param n: propeller "switch-on"
    :param alpha: sound speed prescription
    :param cs7: sound speed in disc - 10^7 cm/s
    :param k: capping fraction
    :return: an array containing total, dipole and propeller luminosities in
             units of 10^50 erg/s
    """
    B, P, MdiscI, RdiscI, epsilon, delta = pars  # Separate out variables
    y0 = init_conds(MdiscI, P)  # Calculate initial conditions

    # Solve equations
    soln, info = odeint(
        odes, y0, tarr, args=(B, MdiscI, RdiscI, epsilon, delta), full_output=True
    )
    if info["message"] != "Integration successful.":
        return "flag"

    # Split solution
    Mdisc = np.array(soln[:, 0])
    omega = np.array(soln[:, 1])

    # Constants
    Rdisc = RdiscI * 1.0e5  # Disc radius - cm
    tvisc = Rdisc / (alpha * cs7 * 1.0e7)  # Viscous timescale - s
    mu = 1.0e15 * B * (R ** 3.0)  # Magnetic dipole moment

    # Radii -- Alfven, Corotation and Light Cylinder
    Rm = (
        (mu ** (4.0 / 7.0))
        * (GM ** (-1.0 / 7.0))
        * (((3.0 * Mdisc) / tvisc) ** (-2.0 / 7.0))
    )
    Rc = (GM / (omega ** 2.0)) ** (1.0 / 3.0)
    Rlc = c / omega
    Rm = np.where(Rm >= (k * Rlc), (k * Rlc), Rm)

    w = (Rm / Rc) ** (3.0 / 2.0)  # Fastness parameter
    bigT = 0.5 * I * (omega ** 2.0)  # Rotational energy
    modW = (
        0.6
        * M
        * (c ** 2.0)
        * ((GM / (R * (c ** 2.0))) / (1.0 - 0.5 * (GM / (R * (c ** 2.0)))))
    )  # Binding energy
    rot_param = bigT / modW  # Rotational parameter

    # Efficiencies and Mass Flow Rates
    eta2 = 0.5 * (1.0 + np.tanh(n * (w - 1.0)))
    eta1 = 1.0 - eta2
    Mdotprop = eta2 * (Mdisc / tvisc)  # Propelled
    Mdotacc = eta1 * (Mdisc / tvisc)  # Accreted

    Nacc = np.zeros_like(Mdisc)
    for i in range(len(Nacc)):
        if rot_param[i] > 0.27:
            Nacc[i] = 0.0
        else:
            if Rm[i] >= R:
                Nacc[i] = ((GM * Rm[i]) ** 0.5) * (Mdotacc[i] - Mdotprop[i])
            else:
                Nacc[i] = ((GM * R) ** 0.5) * (Mdotacc[i] - Mdotprop[i])

    # Dipole luminosity
    Ldip = (mu ** 2.0 * omega ** 4.0) / (6.0 * (c ** 3.0))
    Ldip = np.where(Ldip <= 0.0, 0.0, Ldip)
    Ldip = np.where(np.isfinite(Ldip), Ldip, 0.0)

    # Propeller luminosity
    Lprop = (-1.0 * Nacc * omega) - ((GM / Rm) * eta2 * (Mdisc / tvisc))
    Lprop = np.where(Lprop <= 0.0, 0.0, Lprop)
    Lprop = np.where(np.isfinite(Lprop), Lprop, 0.0)

    # Total luminosity
    Ltot = f_beam * ((dipeff * Ldip) + (propeff * Lprop))

    return np.array([Ltot, Lprop, Ldip]) / 1.0e50


# Check if plots folder exists
if not (os.path.exists(os.path.join(ROOT, "plots"))):
    os.mkdir(os.path.join(ROOT, "plots"))

grbs = {
    "Humped": [1.0, 5.0, 1.0e-3, 100.0, 1.0, 1.0e-6],
    "Classic": [1.0, 5.0, 1.0e-4, 1000.0, 1.0, 1.0e-6],
    "Sloped": [10.0, 5.0, 1.0e-4, 1000.0, 1.0, 1.0e-6],
    "Stuttering": [5.0, 5.0, 1.0e-2, 500.0, 1.0, 1.0e-6],
}

grbs_list = ["Humped", "Classic", "Sloped", "Stuttering"]

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(5, 4.5))
pltx = 0
plty = 0

for z, grb in enumerate(grbs_list):

    B, P, MdiscI, RdiscI, epsilon, delta = grbs[grb]
    ax = axes[pltx, plty]

    # === My model === #
    Ltot_sg, Lprop_sg, Ldip_sg = model_lc(grbs[grb])

    # === Ben's model === #
    # Define constants and convert units
    spin = P * 1.0e-3  # Convert to seconds
    Rdisc = RdiscI * 1.0e5  # Convert to cm
    visc = alpha * cs7 * 1.0e7 * Rdisc  # Viscosity
    tvisc = (Rdisc ** 2.0) / visc  # Viscous timescale
    mu = 1.0e15 * B * (R ** 3.0)  # Magnetic Dipole Moment
    omegazero = (2.0 * np.pi) / spin  # Angular frequency of magnetar

    # Create arrays
    Mdot = np.zeros(int(j))  # Accretion rate
    Mdisc = np.zeros(int(j))  # Disc mass
    Msum = np.zeros(int(j))  # Accreted mass
    Rm = np.zeros(int(j))  # Alfven radius
    Rc = np.zeros(int(j))  # Corotation radius
    M_bg = np.zeros(int(j))  # Magnetar mass
    Ndip = np.zeros(int(j))  # Dipole torque
    w = np.zeros(int(j))  # Fastness parameter
    n = np.zeros(int(j))  # Dimensionless torque
    Nacc = np.zeros(int(j))  # Accretion torque
    beta = np.zeros(int(j))  # Rotation parameter
    inertia = np.zeros(int(j))  # Moment of inertia
    bigT = np.zeros(int(j))  # Rotational energy
    modW = np.zeros(int(j))  # Binding energy
    omegadot = np.zeros(int(j))  # Spin down rate
    Lprop = np.zeros(int(j))  # Propeller luminosity
    Ldip = np.zeros(int(j))  # Dipole luminosity
    Pms = np.zeros(int(j))  # Spin period evolution
    t = np.zeros(int(j))  # Time array
    omega = np.zeros(int(j))  # Angular frequency
    lightradius = np.zeros(int(j))  # Light cylinder

    # Setting initial conditions
    omega[0] = omegazero
    Mdisc[0] = MdiscI * Msol
    t[0] = 1.0
    M_bg[0] = omass * Msol
    Mdot[0] = (3.0 * Mdisc[0] * visc) / (Rdisc ** 2.0)
    Rm[0] = (
        (mu ** (4.0 / 7.0))
        * ((G * M_bg[0]) ** (-1.0 / 7.0))
        * (Mdot[0] ** (-2.0 / 7.0))
    )
    Rc[0] = ((G * M_bg[0]) / (omega[0] ** 2.0)) ** (1.0 / 3.0)
    lightradius[0] = c / omega[0]

    if Rm[0] >= (k * lightradius[0]):
        Rm[0] = k * lightradius[0]

    Ndip[0] = (
        (-2.0 / 3.0)
        * (((mu ** 2.0) * (omega[0] ** 3.0)) / (c ** 3.0))
        * ((lightradius[0] / Rm[0]) ** 3.0)
    )

    w[0] = (Rm[0] / Rc[0]) ** (3.0 / 2.0)
    n[0] = 1.0 - w[0]
    inertia[0] = 0.35 * M_bg[0] * (R ** 2.0)
    bigT[0] = 0.5 * inertia[0] * (omega[0] ** 2.0)

    modW[0] = (
        0.6
        * M_bg[0]
        * (c ** 2.0)
        * (
            ((G * M_bg[0]) / (R * (c ** 2.0)))
            / (1.0 - 0.5 * ((G * M_bg[0]) / (R * (c ** 2.0))))
        )
    )

    beta[0] = bigT[0] / modW[0]

    # First iteration
    if beta[0] > 0.27:
        Nacc[0] = 0.0
    else:
        if Rm[0] >= R:
            Nacc[0] = n[0] * ((G * M_bg[0] * Rm[0]) ** 0.5) * Mdot[0]
            if not np.isfinite(Nacc[0]):
                Nacc[0] = 0.0
        else:
            Nacc[0] = (
                (1.0 - (omega[0] / (((G * M_bg[0]) / (R ** 3.0)) ** 0.5)))
                / ((G * M_bg[0] * R) ** 0.5)
                * Mdot[0]
            )
            if not np.isfinite(Nacc[0]):
                Nacc[0] = 0.0

    omegadot[0] = (Ndip[0] + Nacc[0]) / inertia[0]
    Lprop[0] = (-1.0 * Nacc[0] * omega[0]) - ((G * M_bg[0] * Mdot[0]) / Rm[0])
    Ldip[0] = ((mu ** 2.0) * (omega[0] ** 4.0)) / (6.0 * (c ** 3.0))

    if Rc[0] >= Rm[0]:
        Msum[0] = Mdot[0]
    else:
        Msum[0] = 0.0

    # Main loop
    for i in range(1, int(j)):

        t[i] = t[i - 1] + 1.0
        omega[i] = omega[i - 1] + omegadot[i - 1]

        M_bg[i] = M_bg[i - 1] + Msum[i - 1]
        Mdisc[i] = Mdisc[i - 1] - Mdot[i - 1]
        Mdot[i] = Mdot[0] * np.exp((-3.0 * visc * t[i]) / (Rdisc ** 2.0))
        Rm[i] = (
            (mu ** (4.0 / 7.0))
            * ((G * M_bg[i]) ** (-1.0 / 7.0))
            * (Mdot[i] ** (-2.0 / 7.0))
        )
        Rc[i] = ((G * M_bg[i]) / (omega[i]) ** 2.0) ** (1.0 / 3.0)
        lightradius[i] = c / omega[i]

        if Rm[i] >= (k * lightradius[i]):
            Rm[i] = k * lightradius[i]

        Ndip[i] = (
            (-2.0 / 3.0)
            * (((mu ** 2.0) * (omega[i] ** 3.0)) / (c ** 3.0))
            * ((lightradius[i] / Rm[i]) ** 3.0)
        )

        w[i] = (Rm[i] / Rc[i]) ** (3.0 / 2.0)
        n[i] = 1.0 - w[i]
        inertia[i] = 0.35 * M_bg[i] * (R ** 2.0)
        bigT[i] = 0.5 * inertia[i] * (omega[i] ** 2.0)

        modW[i] = (
            0.6
            * M_bg[i]
            * (c ** 2.0)
            * (
                ((G * M_bg[i]) / (R * (c ** 2.0)))
                / (1.0 - 0.5 * ((G * M_bg[i]) / (R * (c ** 2.0))))
            )
        )

        beta[i] = bigT[i] / modW[i]

        if beta[i] > 0.27:
            Nacc[i] = 0.0
        else:
            if Rm[i] >= R:
                Nacc[i] = n[i] * ((G * M_bg[i] * Rm[i]) ** 0.5) * Mdot[i]
                if not np.isfinite(Nacc[i]):
                    Nacc[i] = 0.0
            else:
                Nacc[i] = (
                    (1.0 - (omega[i] / (((G * M_bg[i]) / (R ** 3.0)) ** 0.5)))
                    * ((G * M_bg[i] * R) ** 0.5)
                    * Mdot[i]
                )
                if not np.isfinite(Nacc[i]):
                    Nacc[i] = 0.0

        if Rc[i] >= Rm[i]:
            Msum[i] = Mdot[i]
        else:
            Msum[i] = Msum[i - 1]

        omegadot[i] = (Ndip[i] + Nacc[i]) / inertia[i]

        Lprop[i] = (-1.0 * Nacc[i] * omega[i]) - ((G * M_bg[i] * Mdot[i]) / Rm[i])
        Ldip[i] = ((mu ** 2.0) * (omega[i] ** 4.0)) / (6.0 * (c ** 3.0))

    Lprop_bg = np.where(np.isfinite(Lprop), Lprop, 0.0)
    Lprop_bg = np.where(Lprop_bg <= 0.0, 0.0, Lprop)
    Ldip_bg = np.where(np.isfinite(Ldip), Ldip, 0.0)
    Ldip_bg = np.where(Ldip_bg <= 0.0, 0.0, Ldip)

    Ltot_bg = (propeff * Lprop_bg) + (dipeff * Ldip_bg)

    # === Plotting === #
    ax.loglog(t, Ltot_bg / 1.0e50, c="r")
    ax.loglog(t, Lprop_bg / 1.0e50, ls="--", c="r")
    ax.loglog(t, Ldip_bg / 1.0e50, ls=":", c="r")

    ax.loglog(tarr, Ltot_sg, c="k")
    ax.loglog(tarr, Lprop_sg, ls="--", c="k")
    ax.loglog(tarr, Ldip_sg, ls=":", c="k")

    ax.set_xlim(1.0e0, 1.0e6)
    ax.set_ylim(1.0e-8, 1.0e0)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.set_title(grb, fontsize=10)

    plty += 1
    if plty > 1:
        pltx += 1
        plty = 0

axes[1, 0].set_xticks([1.0e0, 1.0e2, 1.0e4, 1.0e6])
axes[1, 1].set_xticks([1.0e0, 1.0e2, 1.0e4, 1.0e6])
axes[0, 0].set_yticks([1.0e-6, 1.0e-4, 1.0e-2, 1.0e0])
axes[1, 0].set_yticks([1.0e-6, 1.0e-4, 1.0e-2, 1.0e0])

axes[1, 0].set_xlabel("Time (s)", fontsize=10)
axes[1, 1].set_xlabel("Time (s)", fontsize=10)
axes[0, 0].set_ylabel(
    "Luminosity ($10^{50}$ ${\\rm erg}$ ${\\rm s}^{-1}$)", fontsize=10
)
axes[1, 0].set_ylabel(
    "Luminosity ($10^{50}$ ${\\rm erg}$ ${\\rm s}^{-1}$)", fontsize=10
)

fig.tight_layout(h_pad=0.2, w_pad=0.1)
fig.savefig(os.path.join(ROOT, "plots/figure_5.png"))
