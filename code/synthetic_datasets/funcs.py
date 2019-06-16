import os
import sys
import warnings
import contextlib
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')

# Global constants
G = 6.674e-8           # Gravitational constant (cgs)
c = 3.e10              # Speed of light (cm/s)
R = 1.e6               # Magnetar radius (cm)
Msol = 1.99e33         # Solar mass (cgs)
M = 1.4 * Msol         # Magnetar mass
I = 0.35 * M * R**2.0  # Moment of Inertia
GM = G * M
tarr = np.logspace(0.0, 6.0, num=10001, base=10.0)


# Suppress lsoda warnings
def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    if stdout is None:
        stdout = sys.stdout
    stdout_fd = fileno(stdout)
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()
        try:
            os.dup2(fileno(to), stdout_fd)
        except ValueError:
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)
        try:
            yield stdout
        finally:
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)


# Calculate initial conditions to pass to odeint
def init_conds(MdiscI, P_i):
    """
Function to convert a n initial spin value from milliseconds into seconds,
then into an angular frequency. Also converts an initial disc mass from solar
masses into grams.

Usage >>> init_conds(arr)
arr : Array object
arr[0] = Initial spin in milliseconds
arr[1] = Initial disc mass in Msol

Returns y0 (array object)
y0[0] = Initial disc mass in grams
y0[1] = Initial angular frequency (s^-1)
    """

    # ODEint initial conditions
    Mdisc0 = MdiscI * Msol                   # Disc mass
    omega0 = (2.0 * np.pi) / (1.0e-3 * P_i)  # Angular frequency

    return Mdisc0, omega0


# Model to be passed to odeint to calculate Mdisc and omega
def ODEs(y, t, B, MdiscI, RdiscI, epsilon, delta, n, alpha, cs7, k):
    """
This is the magnetar model to be integrated by ODEINT, solving for disc mass
and angular frequency over a time range.

Usage >>> odeint(ODEs, y, t, args=(B, RdiscI, epsilon, delta, n, alpha, cs7, k))
      y : initial conditions (y0 from init_conds, array)
      t : time range (either user defined or tarr from above, array)
      B : Magnetic field (x10^15 G, float)
 MdiscI : initial disc mass (solar masses, float)
 RdiscI : Disc radius (km, float)
epsilon : ratio between fallback and viscous timescales (float)
  delta : ratio between fallback and initial disc masses (float)
      n : effieicency of propeller switch-on (float, optional)
  alpha : prescription for sound speed in disc (float, optional)
    cs7 : sound speed in disc (x10^7 cm/s, float, optioanl)
      k : capping fraction (float, optional)
    """

    # Initial conditions
    Mdisc, omega = y

    # Constants
    Rdisc = RdiscI * 1.0e5                 # Disc radius
    tvisc = Rdisc / (alpha * cs7 * 1.0e7)  # Viscous timescale
    mu = 1.0e15 * B * (R ** 3.0)           # Magnetic Dipole Moment
    M0 = delta * MdiscI * Msol             # Global Mass Budget
    tfb = epsilon * tvisc                  # Fallback timescale

    # Radii - Alfven, Corotiona, Light Cylinder
    Rm = ((mu ** (4.0 / 7.0)) * (GM ** (-1.0 / 7.0)) * ((3.0 * Mdisc) / tvisc)
          ** (-2.0  /7.0))
    Rc = (GM / (omega ** 2.0))**(1.0 / 3.0)
    Rlc = c / omega
    if Rm >= (k * Rlc):
        Rm = k * Rlc

    w = (Rm / Rc) ** (3.0 / 2.0)     # Fastness Parameter
    bigT = 0.5 * I * (omega ** 2.0)  # Rotational energy
    modW = (0.6 * M * (c ** 2.0) * ((GM / (R * (c ** 2.0))) / (1.0 - 0.5 * (GM /
            (R * (c ** 2.0))))))     # Binding energy
    rot_param = bigT / modW          # Rotation parameter

    # Classical dipole torque
    Ndip = (-1.0 * (mu ** 2.0) * (omega ** 3.0)) / (6.0 * (c ** 3.0))

    # Efficiencies
    eta2 = 0.5 * (1.0 + np.tanh(n * (w - 1.0)))
    eta1 = 1.0 - eta2

    # Mass flow rates
    Mdotprop = eta2 * (Mdisc / tvisc)
    Mdotacc = eta1 * (Mdisc / tvisc)
    Mdotfb = (M0 / tfb) * ((t + tfb) / tfb) ** (-5.0 / 3.0)
    Mdotdisc = Mdotfb - Mdotacc - Mdotprop

    if rot_param > 0.27:
        Nacc = 0.0  # Prevents magnetar break-up
    else:
        # Accretion torque
        if Rm >= R:
            Nacc = ((GM * Rm) ** 0.5) * (Mdotacc - Mdotprop)
        else:
            Nacc = ((GM * R) ** 0.5) * (Mdotacc - Mdotprop)

    omegadot = (Nacc + Ndip) / I  # Angular frequency time derivative

    return Mdotdisc, omegadot


# Function that returns model light curve
def model_lum(pars, xdata=None, n=10.0, alpha=0.1, cs7=1.0, k=0.9, dipeff=1.0,
              propeff=1.0, f_beam=1.0):
    """
Function to return a light curve for the magnetar propeller model.

Usage >>> model_lum(pars, x)
   pars : List/1D Array of parameters [B, P, MdiscI, RdiscI, epsilon, delta]
  xdata : Array of time points to solve for (if == '0', tarr defined above)
      n : effieicency of propeller switch-on (float, optional)
 f_beam : beaming fraction (float, optional)
  alpha : prescription for sound speed in disc (float, optional)
    cs7 : sound speed in disc (x10^7 cm/s, float, optioanl)
      k : capping fraction (float, optional)
 dipeff : Dipole efficiency (float, optional)
propeff : Propeller efficiency (float, optional)
    """
    # Separate parameters
    B, P, MdiscI, RdiscI, epsilon, delta = pars

    y0 = init_conds(MdiscI, P)  # Inital conditions

    # Solve for Mdisc and omega
    with stdout_redirected():
        soln, info = odeint(ODEs, y0, tarr, args=(B, MdiscI, RdiscI, epsilon,
                            delta, n, alpha, cs7, k), full_output=True)
    # Catch parameters that break ODEINT
    if info['message'] != 'Integration successful.':
        return 'flag'

    Mdisc = soln[:,0]
    omega = soln[:,1]

    # Constants
    Rdisc = RdiscI * 1.0e5
    tvisc = Rdisc / (alpha * cs7 * 1.0e7)
    mu = 1.0e15 * B * (R ** 3.0)
    M0 = delta * MdiscI * Msol
    tfb = epsilon * tvisc

    # Radii - Alfven, Corotation, Light Cylinder
    Rm = ((mu ** (4.0 / 7.0)) * (GM ** (-1.0 / 7.0)) * ((3.0 * Mdisc) / tvisc)
          ** (-2.0 / 7.0))
    Rc = (GM / (omega ** 2.0)) ** (1.0 / 3.0)
    Rlc = c / omega
    inRm = Rm >= (k * Rlc)
    Rm = np.where(inRm, (k * Rlc), Rm)

    w = (Rm / Rc) ** (3.0 / 2.0)
    bigT = 0.5 * I * (omega ** 2.0)
    modW = (0.6 * M * (c ** 2.0) * ((GM / (R * (c ** 2.0))) / (1.0 - 0.5 * (GM /
            (R * (c ** 2.0))))))
    rot_param = bigT / modW

    eta2 = 0.5 * (1.0 + np.tanh(n * (w - 1.0)))
    eta1 = 1.0 - eta2
    Mdotprop = eta2 * (Mdisc / tvisc)
    Mdotacc = eta1 * (Mdisc / tvisc)

    Nacc = np.zeros_like(Mdisc)
    for i in range(len(Nacc)):
        if rot_param[i] > 0.27:
            Nacc[i] = 0.0
        else:
            if Rm[i] >= R:
                Nacc[i] = ((GM * Rm[i]) ** 0.5) * (Mdotacc[i] - Mdotprop[i])
            else:
                Nacc[i] = ((GM * R) ** 0.5) * (Mdotacc[i] - Mdotprop[i])

    # Dipole Luminosity
    Ldip = dipeff * (((mu ** 2.0) * (omega ** 4.0)) / (6.0 * (c ** 3.0)))
    inLd1 = Ldip <= 0.0
    inLd2 = np.isfinite(Ldip)
    Ldip = np.where(inLd1, 0.0, Ldip)
    Ldip = np.where(inLd2, Ldip, 0.0)

    # Propeller Luminosity
    Lprop = (propeff * ((-1.0 * Nacc * omega) - ((GM / Rm) * eta2 * (Mdisc /
             tvisc))))
    inLp1 = Lprop <= 0.0
    inLd2 = np.isfinite(Lprop)
    Lprop = np.where(inLp1, 0.0, Lprop)
    Lprop = np.where(inLd2, Lprop, 0.0)

    Ltot = f_beam * (Ldip + Lprop)  # Total (beamed) luminosity
    if xdata is None:
        return np.array([tarr, Ltot / 1.0e50, Lprop / 1.0e50, Ldip / 1.0e50])

    lum_func = interp1d(tarr, Ltot)
    L = lum_func(xdata)

    return L / 1.0e50
