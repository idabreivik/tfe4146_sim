import sesame
import numpy as np
import scipy.constants as cts
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
# This example support simulation on abrupt p-n junctions and contain code for plotting band diagram, electrostatic potential, electric field, space charge density, current components and electron density.
# The plots enabled in this dictionary will be created with shared x-axis
PLOT = {
    'BAND_DIAGRAM' : True,
    'ELECTROSTATIC_POTENTIAL' : True,
    'ELECTRIC_FIELD' : True,
    'SPACE_CHARGE_DENSITY' : True,
    'MINORITY_CARRIER_DENSITY' : True,
    'NORMALIZED_CURRENT_COMPONENTS' : True,
}
# Use this option to indicate the depletion region, when the width of the depletion region is calculated from the junction potential V0
INDICATE_DEPLETION_REGION = True


# Doping
# The simulations are performed on abrupt, homogenous p-n junctions with constant doping concentration on either side of the junction. Modify the variables nA and nD to change the doping concentration.
nA = 1e15 # [cm^-3]
nD = 1e15 # [cm^-3]

# Bias voltage
# This is the bias voltage across the p-n junction
bias_voltage = 0.0

# CONSTANTS
T = 300 # Temp [K]
kB_si = cts.k
q = cts.e
epsilon_0 = cts.epsilon_0 / 1e2 # [F/cm]

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
buf = L/10 * 0 # Use this parameter to specify a doping free zone around the junction
n_region = lambda pos: (pos >= junction + buf)
p_region = lambda pos: (pos < junction - buf)



doping_profile = np.array([nA if p < junction else -nD for p in x])
sys.set_doping_profile(doping_profile)
# Define Neutral contacts
sys.contact_type('Neutral', 'Neutral')

# Define the surface recombination velocities for electrons and holes [cm/s]
Sn_left, Sp_left, Sn_right, Sp_right =  1e7, 1e7, 1e7, 1e7  # cm/s
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
n = az.electron_density((p1, p2))
p = az.hole_density((p1, p2))
rho = az.get_space_charge_density((p1, p2))

# Calculate depletion region
E0 = np.max(np.abs(E))
epsilon = sys.epsilon[0] * epsilon_0
xp0 = E0 * epsilon/(q*nA)
xn0 = E0 * epsilon/(q*nD)

xp0_pos = junction - xp0
xn0_pos = junction + xn0

# Electron and hole density
n = n * sys.scaling.density
p = p * sys.scaling.density

# Calculate current components
mu_n = sys.mu_e
mu_p = sys.mu_h
Dn = kB_si*T/q*mu_n # Einstein rel
Dp = kB_si*T/q*mu_p # Einstein rel

# Calculate electron current
dx = (p2[0] - p1[0])/len(v)
Jn_drift = q*mu_n*n*E
Jn_diff = q*Dn*np.gradient(n, dx)
Jn = Jn_drift + Jn_diff
# Hole current
Jp_drift = q*mu_p*p*E
Jp_diff = -q*Dp*np.gradient(p, dx)
Jp = Jp_drift + Jp_diff


ni = np.mean(sys.ni) * sys.scaling.density
V0_hat = kB_si*T / q * np.log(nA*nD/(ni**2))
print(f'ni: {ni/1e10}e10')
print(f'V0_hat: {V0_hat}')
print(f'V0: {np.max(v) - np.min(v)}')




# PLOTTING
num_plots = sum(PLOT.values())

if num_plots > 0:
    fig, axs = plt.subplots(nrows=num_plots, sharex=True)
    if num_plots == 1:
        axs = [axs]
    plt_index = num_plots - 1



    if PLOT['SPACE_CHARGE_DENSITY']:
        ax = axs[plt_index]
        plt_index -= 1
        ax.plot(x*1e4, rho, label=r'$\rho(x)$')
        ax.set_ylabel(r'Space charge [C/cm^3]')
        if INDICATE_DEPLETION_REGION == True:
            ax.axvline(x=xn0_pos*1e4, ls='--', c='grey')
            ax.axvline(x=xp0_pos*1e4, ls='--', c='grey')
        ax.legend(loc='right')

    # E field
    if PLOT['ELECTRIC_FIELD']:
        ax = axs[plt_index]
        plt_index -= 1
        ax.plot(x*1e4, E, label=r'$\mathcal{E}(x)$')
        ax.set_ylabel('Electric field [V/cm]')
        if INDICATE_DEPLETION_REGION == True:
            ax.axvline(x=xn0_pos*1e4, ls='--', c='grey')
            ax.axvline(x=xp0_pos*1e4, ls='--', c='grey')
        ax.legend()

    if PLOT['ELECTROSTATIC_POTENTIAL']:
        ax = axs[plt_index]
        plt_index -= 1
        ax.set_ylabel('Potential [V]')
        ax.plot(x*1e4, v, label='V(x)')
        if INDICATE_DEPLETION_REGION == True:
            ax.axvline(x=xn0_pos*1e4, ls='--', c='grey')
            ax.axvline(x=xp0_pos*1e4, ls='--', c='grey')
        ax.legend()

    if PLOT['MINORITY_CARRIER_DENSITY']:
        ax = axs[plt_index]
        plt_index -= 1
        # Plot n_p
        ax.plot(x[x<xp0_pos]*1e4, n[x<xp0_pos], label=r'$n(x)$')
        # Plot p_n
        ax.plot(x[x>xn0_pos]*1e4, p[x>xn0_pos], label=r'$p(x)$')
        ax.set_ylabel(r'Carrier concentration [cm$^{-3}$]')
        if INDICATE_DEPLETION_REGION == True:
            ax.axvline(x=xn0_pos*1e4, ls='--', c='grey')
            ax.axvline(x=xp0_pos*1e4, ls='--', c='grey')
        ax.legend()


    if PLOT['NORMALIZED_CURRENT_COMPONENTS'] and bias_voltage > 0:
        ax = axs[plt_index]
        plt_index -= 1
        ax.plot(x*1e4, Jn/(Jp+Jn), label='Electron current')
        ax.plot(x*1e4, Jp/(Jp+Jn), label='Hole current')
        ax.set_ylabel('Normalized current')
        if INDICATE_DEPLETION_REGION == True:
            ax.axvline(x=xn0_pos*1e4, ls='--', c='grey')
            ax.axvline(x=xp0_pos*1e4, ls='--', c='grey')
        ax.legend()


    if PLOT['BAND_DIAGRAM']:
        ax = axs[plt_index]
        if INDICATE_DEPLETION_REGION == True:
            ax.axvline(x=xn0_pos*1e4, ls='--', c='grey')
            ax.axvline(x=xp0_pos*1e4, ls='--', c='grey')
        plt_index -= 1
        az.band_diagram((p1, p2), ax=ax)


    axs[-1].set_xlabel(r'Position [$\mathregular{\mu m}$]')
    fig.suptitle(r'$V_{bias}=$' + f'{bias_voltage}V')
    plt.show()

