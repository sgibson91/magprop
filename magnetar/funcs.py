import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d


# Global constants
G = 6.674e-8  # Gravitational constant - cgs units
c = 3.0e10  # Light speed - cm/s
R = 1.0e6  # Magnetar radius - cm
Msol = 1.99e33  # Solar mass - grams
M = 1.4 * Msol  # Magnetar mass - grams
I = (4.0 / 5.0) * M * (R ** 2.0)  # Moment of inertia
GM = G * M


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
    Rm = (mu ** (4.0 / 7.0)) * (GM ** (-1.0 / 7.0)) * ((Mdisc / tvisc) ** (-2.0 / 7.0))
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


# Function that returns a model light curve
def model_lc(
    pars,
    xdata=None,
    GRBtype=None,
    dipeff=0.05,
    propeff=0.4,
    f_beam=1.0,
    n=1.0,
    alpha=0.1,
    cs7=1.0,
    k=0.9,
):
    """
Function to calculate a model gamma-ray burst, X-ray light curve based on input
parameters.

    :param pars: A list or array of input parameters in order: B, P, MdiscI,
                 RdiscI, epsilon, delta
    :param xdata: Optional array of time points for GRB data
    :param GRBtype: string indicating the GRB type
    :param dipeff: Fractional dipole energy-to-luminosity conversion efficiency
    :param propeff: Fractional propeller energy-to-luminosity conversion
                    efficiency
    :param f_beam: Beaming fraction
    :param n: Propeller "switch-on" efficiency
    :param alpha: Sound speed prescription
    :param cs7: Sound speed in disc - 10^7 cm/s
    :param k: Capping fraction

    if xdata is not None:
        :return L: array of luminosity values at given time point in xdata
                   - 10^50 erg/s
    else:
        :return: an array containing tarr, Ltot, Lprop, Ldip in units of secs
                 and 10^50 erg/s
    """
    # Select appropriate time array based on GRB type
    if (GRBtype is not None) & (GRBtype == "S"):
        tarr = np.logspace(-3.0, 6.0, num=10001, base=10.0)
    elif (GRBtype is not None) & (GRBtype == "L"):
        tarr = np.logspace(0.0, 6.0, num=10001, base=10.0)
    elif GRBtype is None:
        tarr = np.logspace(0.0, 6.0, num=10001, base=10.0)
    else:
        raise ValueError(
            "Please provide a valid value for GRBtype.\nOptions are: L, S, or None."
        )

    # Separate out parameters
    B, P, MdiscI, RdiscI, epsilon, delta = pars

    # Calculate initial conditions
    y0 = init_conds(MdiscI, P)

    # Solve the equations
    soln, info = odeint(
        odes, y0, tarr, args=(B, MdiscI, RdiscI, epsilon, delta), full_output=True
    )
    # Return a flag if the integration was not successful
    if info["message"] != "Integration successful.":
        return "flag"

    # Separate out solution
    Mdisc = np.array(soln[:, 0])
    omega = np.array(soln[:, 1])

    # Convert constants
    Rdisc = RdiscI * 1.0e5  # Disc radius - cm
    tvisc = Rdisc / (alpha * cs7 * 1.0e7)  # Viscous timescale - secs
    mu = 1.0e15 * B * (R ** 3.0)  # Magnetic dipole moment
    M0 = delta * MdiscI * Msol  # Fallback mass budget - grams
    tfb = epsilon * tvisc  # Fallback timescale - secs

    # Radii - Alfven, Corotation, Light Cylinder
    Rm = (mu ** (4.0 / 7.0)) * (GM ** (-1.0 / 7.0)) * ((Mdisc / tvisc) ** (-2.0 / 7.0))
    Rc = (GM / (omega ** 2.0)) ** (1.0 / 3.0)
    Rlc = c / omega
    # Cap Alfven radius to Light Cylinder radius
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

    # Dipole torque
    Ndip = (-1.0 * (mu ** 2.0) * (omega ** 3.0)) / (6.0 * (c ** 3.0))

    # Efficiencies and mass flow rates
    eta2 = 0.5 * (1.0 + np.tanh(n * (w - 1.0)))
    eta1 = 1.0 - eta2
    Mdotprop = eta2 * (Mdisc / tvisc)
    Mdotacc = eta1 * (Mdisc / tvisc)

    # Accretion torque
    Nacc = np.zeros_like(Mdisc)
    for i in range(len(Nacc)):
        if rot_param[i] > 0.0:
            Nacc[i] = 0.0
        else:
            if Rm[i] >= R:
                Nacc[i] = (GM * Rm[i]) ** 0.5 * (Mdotacc[i] - Mdotprop[i])
            else:
                Nacc[i] = (GM * R) ** 0.5 * (Mdotacc[i] - Mdotprop[i])

    # Luminosities - Dipole, Propeller and Total
    Ldip = dipeff * (-1.0 * Ndip * omega)
    Ldip = np.where((Ldip <= 0.0), 0.0, Ldip)
    Ldip = np.where(np.isfinite(Ldip), Ldip, 0.0)

    Lprop = propeff * (-1.0 * Nacc * omega)
    Lprop = np.where((Lprop <= 0.0), 0.0, Lprop)
    Lprop = np.where(np.isfinite(Lprop), Lprop, 0.0)

    Ltot = f_beam * (Ldip + Lprop)

    # Return values based on xdata
    if xdata is not None:
        lum_func = interp1d(tarr, Ltot)
        L = lum_func(xdata)

        return L / 1.0e50

    else:
        return np.array([tarr, Ltot / 1.0e50, Lprop / 1.0e50, Ldip / 1.0e50])


def main(args):
    import matplotlib.pyplot as plt

    # Construct parameter list
    pars = [args.B, args.P, args.M, args.R, args.eps, args.delt]

    # Calculate model
    t, Ltot, Lprop, Ldip = model_lc(pars)

    # Plot the model
    plt.loglog(t, Ltot, c="k")
    plt.loglog(t, Lprop, c="k", ls="--")
    plt.loglog(t, Ldip, c="k", ls=":")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Script to generate a model X-ray light curve for a gamma-ray burst based on input parameters."
    )

    # Required arguments
    parser.add_argument(
        "-B",
        required=True,
        type=float,
        choices=range(1.0 - 3, 10.001),
        help="Magnetic field strength - 10^15 G",
    )
    parser.add_argument(
        "-P",
        required=True,
        type=float,
        choices=range(1.0e-3, 10.001),
        help="Initial spin period - milliseconds",
    )
    parser.add_argument(
        "-M",
        required=True,
        type=float,
        choices=range(1.0e-5, 0.10001),
        help="Initial disc mass - solar masses",
    )
    parser.add_argument(
        "-R",
        required=True,
        type=float,
        choices=range(50.0, 1500.1),
        help="Disc radius - km",
    )
    parser.add_argument(
        "-eps",
        required=True,
        type=float,
        choices=range(0.1, 100.1),
        help="Fallback timescale ratio (epsilon)",
    )
    parser.add_argument(
        "-delt",
        required=True,
        type=float,
        choices=range(1.0e-3, 1000.001),
        help="Mass ratio (delta)",
    )

    # TODO: Only work on this if you can think of a sensible way to combine
    # parameters without an endless nest of conditional statements
    # 2019-06-21: argparse groups may be the answer

    # Optional arguments
    # parser.add_argument("--dipeff", type=float,
    #                     help="Fractional dipole efficiency")
    # parser.add_argument("--propeff", type=float,
    #                     help="Fractional propeller efficiency")
    # parser.add_argument("--f-beam", type=float, help="Beaming fraction")

    # REALLY optional arguments
    # parser.add_argument("--n", type=float, help="Propeller 'switch-on'")
    # parser.add_argument("--alpha", type=float,
    #                     help="Sound speed prescription")
    # parser.add_argument("--cs7", type=float, help="Sound speed - 10^7 cm/s")
    # parser.add_argument("--k", type=float, help="Capping fraction")

    args = parser.parse_args()

    main(args)
