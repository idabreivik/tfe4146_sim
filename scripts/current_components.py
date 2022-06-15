import pandas as pd
import sesame
import numpy as np
import scipy.constants as cts
from scipy.interpolate import interp1d
import json
import matplotlib.pyplot as plt
from shlib import to_path, mkdir
import matplotlib as mpl
from cycler import cycler
matplotcolors = ['k', 'r', 'g', 'b', 'c', 'm', 'y']
mpl.rcParams['axes.prop_cycle'] = cycler(color=matplotcolors)

root = to_path(__file__).parent
gzip_dir = to_path(root, 'gzip')
mkdir(gzip_dir)

# OPTIONS
# Set this variable to plot hole and electron drift/diffusion current components
# Will be plotted in separate figures
PLOT_CURRENT_COMPONENTS = True

# Doping
# The simulations are performed on abrupt, homogenous p-n junctions with constant doping concentration on either side of the junction. Modify the variables nA and nD to change the doping concentration.
nA = 1e14 # [cm^-3]
nD = 1e14 # [cm^-3]

# Bias voltage
# This is the bias voltage across the p-n junction
bias_voltage = 0.0

# CONSTANTS
T = 300 # Temp [K]
kB_si = cts.k
q = cts.e

# Create grid and initialize system
L = 10e-4 # length of the system in the x-direction [cm]
junction = L/2 # extent of the junction from the left contact [cm]
n_points = 10000 # Number of grid points. Increasing the number of points give better accuracy, but will increase simulation time
x = np.linspace(0, L, n_points)


# Coorindates of the edges of the diode
p1 = (0, 0)
p2 = (L, 0)

# Initialize the system
sys = sesame.Builder(x, T=T)

# Add material properties
# The parameters of Silicon is placed in this json file
with open('materials/si.json', 'r') as f:
        si = json.load(f)
sys.add_material(si)

# Add dopants
n_region = lambda pos: (pos >= junction)
p_region = lambda pos: (pos < junction)


# Add the donors
sys.add_donor(nD, n_region)
# Add the acceptors
sys.add_acceptor(nA, p_region)


# Define Neutral contacts
sys.contact_type('Neutral', 'Neutral')

# Define the surface recombination velocities for electrons and holes [cm/s]
Sn_left, Sp_left, Sn_right, Sp_right =  1e7, 1e7, 1e7, 1e7  # cm/s
# Sn_left, Sp_left, Sn_right, Sp_right =  1e9,1e9,1e9,1e9
sys.contact_S(Sn_left, Sp_left, Sn_right, Sp_right)

voltages = np.array([bias_voltage])
j = sesame.IVcurve(sys, voltages, str(to_path(gzip_dir, 'j_eq')))
j = j * sys.scaling.current

# Load results
sys, result = sesame.load_sim(
    to_path(gzip_dir, 'j_eq_0.gzip'))  # load data file
az = sesame.Analyzer(sys,result)                   # get Sesame analyzer object


# Calculations
v = az.get_electrostatic_potential((p1, p2))
E = az.get_e_field((p1, p2))

# Calculate depletion region
E0 = np.max(np.abs(E))
epsilon = sys.epsilon[0] * 8.85e-14
xp0 = E0 * epsilon/(q*nA)
xn0 = E0 * epsilon/(q*nD)

xp0_pos = junction - xp0
xn0_pos = junction + xn0

# Electron and hole density
n = az.electron_density((p1, p2))
n = n * sys.scaling.density
p = az.hole_density((p1, p2))
p = p * sys.scaling.density
Delta_n = interp1d(x, n)(xp0_pos)
Delta_p = interp1d(x, p)(xn0_pos)


# Calculate current components
mu_n = sys.mu_e*sys.scaling.mobility
mu_p = sys.mu_h*sys.scaling.mobility
Dn = kB_si*T/q*mu_n # Einstein rel
Dp = kB_si*T/q*mu_p # Einstein rel
tau_n = np.mean(sys.tau_e*sys.scaling.time)
tau_p = np.mean(sys.tau_h*sys.scaling.time)
L_n = np.sqrt(Dn*tau_n)
L_p = np.sqrt(Dp*tau_p)

# Calculate electron current
dx = (p2[0] - p1[0])/len(v)
Jn_drift = q*mu_n*n*E
Jn_diff = q*Dn*np.gradient(n, dx)
Jn = Jn_drift + Jn_diff
# Hole current
Jp_drift = q*mu_p*p*E
Jp_diff = -q*Dp*np.gradient(p, dx)
Jp = Jp_drift + Jp_diff



if PLOT_CURRENT_COMPONENTS:
    scale = 1e6
    plt.figure()
    plt.plot(x*1e4, Jn_drift*scale, label='electron drift current')
    plt.plot(x*1e4, Jn_diff*scale, label='electron diffusion current')
    plt.plot(x*1e4, Jn*scale, label='total electron current')
    plt.plot(x*1e4, (Jp+Jn)*scale, label='total current')
    plt.xlabel(r'Position [$\mu$m]')
    plt.ylabel('Current density [uA/cm2]')
    plt.axvline(x=xn0_pos*1e4, ls='--', label=r'$x_{n0}$', c='grey')
    plt.axvline(x=xp0_pos*1e4, ls='--', label=r'$x_{p0}$', c='grey')
    plt.legend()

    plt.figure()
    plt.plot(x*1e4, Jp_drift*scale, label='hole drift current')
    plt.plot(x*1e4, Jp_diff*scale, label='hole diffusion current')
    plt.plot(x*1e4, Jp*scale, label='total hole current')
    plt.plot(x*1e4, (Jp+Jn)*scale, label='total current')
    plt.axvline(x=xn0_pos*1e4, ls='--', label=r'$x_{n0}$', c='grey')
    plt.axvline(x=xp0_pos*1e4, ls='--', label=r'$x_{p0}$', c='grey')
    plt.xlabel(r'Position [$\mu$m]')
    plt.ylabel('Current density [uA/cm2]')
    plt.legend()





# Diffusion currents at the edge of the depletion region
# Use interpolation to get current at exact pos
Jp0 = interp1d(x, Jp_diff)(xn0_pos)
Jn0 = interp1d(x, Jn_diff)(xp0_pos)

data = {
    'Jp0': f'{Jp0*1e3} mA/cm2',
    'Jp0_hat': f'{np.mean(q*Dp/L_p * Delta_p)*1e3} mA/cm2',
    'Jn0': f'{Jn0*1e3} mA/cm2',
    'Jn0_hat': f'{np.mean(q*Dn/L_n * Delta_n)*1e3} mA/cm2',
    # 'Dn': np.mean(Dn),
    # 'Dp': np.mean(Dp),
    # 'tau_n': np.mean(tau_n),
    # 'tau_p': np.mean(tau_p),
    # 'L_n': np.mean(L_n),
    # 'L_p': np.mean(L_p),
}
df = pd.Series(data=data)
print(df)

plt.show()
