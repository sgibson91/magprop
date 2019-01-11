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
def init_conds(arr):
    # ODEint initial conditions
    Mdisc0 = arr[1] * Msol                     # Disc mass
    omega0 = (2.0 * np.pi) / (1.e-3 * arr[0])  # Angular frequency
    y0 = [Mdisc0, omega0]
    return y0


# Model to be passed to odeint to calculate Mdisc and omega
def ODEs(y, t, B, MdiscI, RdiscI, epsilon, delta, n, alpha, cs7, k):

    # Initial conditions
    Mdisc = y[0]
    omega = y[1]

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

    return Mdotdisc, omegadot

B = 1.0
P = 1.0
Md = 0.1
Rd = 100.0
eps = 1.0
delt = 1.0e-6
n = [1.0, 10.0, 50.0]
lines = [':', '--', '-']
alpha = 0.1
cs7 = 1.0
k = 0.9

y0 = init_conds([P, Md])

mu = 1.e15 * B * (R ** 3.0)
Rdisc = Rd * 1.0e5
tvisc = Rdisc / (alpha * cs7 * 1.0e7)

for z in range(len(n)):
    pars = [B, P, Md, Rd, eps, delt, n[z]]
    soln = odeint(ODEs, y0, tarr, args=(B, Md, Rd, eps, delt, n[z], alpha, cs7,
                  k))

    Mdisc = soln[:,0]
    omega = soln[:,1]

    Rm = ((mu ** (4.0 / 7.0)) * (GM ** (-1.0 / 7.0)) * (((3.0 * Mdisc) / tvisc)
          ** (-2.0 / 7.0)))
    Rc = (GM / (omega ** 2.0)) ** (1.0 / 3.0)
    Rlc = c / omega
    inRm = Rm >= (0.9 * Rlc)
    Rm = np.where(inRm, (0.9 * Rlc), Rm)

    w = np.sort((Rm / Rc) ** (3.0 / 2.0))
    eta2 = 0.5 * (1.0 + np.tanh(n[z] * (w - 1.0)))

    plt.plot(w, eta2, c='k', ls=lines[z])

plt.xlim(0.8, 1.2)
plt.ylim(-0.1, 1.1)
plt.xticks([0.8, 0.9, 1.0, 1.1, 1.2])
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('Fastness Parameter, $\Omega$', fontsize=12)
plt.ylabel('Propeller Efficiency, $\eta_2$', fontsize=12)
plt.tight_layout()
plt.show()
