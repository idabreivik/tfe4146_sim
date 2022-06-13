import sesame
import numpy as np
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
    'MINORITY_CARRIER_DENSITY' : False,
    'NORMALIZED_CURRENT_COMPONENTS' : False,
}
# Set this variable to plot hole and electron drift/diffusion current components
# Will be plotted in separate figures
PLOT_CURRENT_COMPONENTS = False

# Doping
# The simulations are performed on abrupt, homogenous p-n junctions with constant doping concentration on either side of the junction. Modify the variables nA and nD to change the doping concentration.
nA = 1e15 # [cm^-3]
nD = 1.4e15 # [cm^-3]

# Bias voltage
# This is the bias voltage across the p-n junction
bias_voltage = 0

# CONSTANTS
T = 300 # Temp [K]
kB = 8.62e-5 # Boltzmann [eV/K]
q = 1.6e-19 # Electron charge [C]

# Create grid and initialize system
L = 8e-4 # length of the system in the x-direction [cm]
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
rho = az.get_space_charge_density((p1, p2))

# Calculate current components
mu_n = sys.mu_e
mu_p = sys.mu_h
Dn = kB*T*mu_n # Einstein rel
Dp = kB*T*mu_p # Einstein rel

# Calculate electron current
dx = (p2[0] - p1[0])/len(v)
Jn_drift = q*mu_n*n*E
Jn_diff = q*Dn*np.gradient(n, dx)
Jn = Jn_drift + Jn_diff
# Hole current
Jp_drift = q*mu_p*p*E
Jp_diff = -q*Dp*np.gradient(p, dx)
Jp = Jp_drift + Jp_diff






# PLOTTING
num_plots = sum(PLOT.values())
fig, axs = plt.subplots(nrows=num_plots, sharex=True)
plt_index = num_plots - 1



if PLOT['SPACE_CHARGE_DENSITY']:
    ax = axs[plt_index]
    plt_index -= 1
    ax.plot(x*1e4, rho, label=r'$\rho(x)$')
    ax.set_ylabel(r'Space charge [C/cm^3]')
    ax.legend()

# E field
if PLOT['ELECTRIC_FIELD']:
    ax = axs[plt_index]
    plt_index -= 1
    ax.plot(x*1e4, E, label=r'$\mathcal{E}(x)$')
    ax.set_ylabel('Electric field [V/cm]')
    ax.legend()

if PLOT['ELECTROSTATIC_POTENTIAL']:
    ax = axs[plt_index]
    plt_index -= 1
    ax.set_ylabel('Potential [V]')
    ax.plot(x*1e4, v, label='V(x)')
    ax.legend()

if PLOT['MINORITY_CARRIER_DENSITY']:
    ax = axs[plt_index]
    plt_index -= 1
    # Plot n_p
    ax.plot(x[x<xp0_pos]*1e4, n[x<xp0_pos], label=r'$n(x_p)$')
    # Plot p_n
    ax.plot(x[x>xn0_pos]*1e4, p[x>xn0_pos], label=r'$p(x_n)$')
    ax.set_ylabel(r'Carrier concentration [cm$^{-3}$]')
    ax.legend()


if PLOT['NORMALIZED_CURRENT_COMPONENTS'] and bias_voltage > 0:
    ax = axs[plt_index]
    plt_index -= 1
    ax.plot(x*1e4, Jn/(Jp+Jn), label='Electron current')
    ax.plot(x*1e4, Jp/(Jp+Jn), label='Hole current')
    ax.set_ylabel('Normalized current')
    ax.legend()


if PLOT['BAND_DIAGRAM']:
    ax = axs[plt_index]
    plt_index -= 1
    az.band_diagram((p1, p2), ax=ax)



if PLOT_CURRENT_COMPONENTS:
    scale = 1e6
    plt.figure()
    plt.plot(x*1e4, Jn_drift*scale, label='electron drift current')
    plt.plot(x*1e4, Jn_diff*scale, label='electron diffusion current')
    plt.plot(x*1e4, Jn*scale, label='total electron current')
    plt.plot(x*1e4, (Jp+Jn)*scale, label='total current')
    plt.xlabel(r'Position [$\mu$m]')
    plt.ylabel('Current [uA]')
    plt.legend()

    plt.figure()
    plt.plot(x*1e4, Jp_drift*scale, label='hole drift current')
    plt.plot(x*1e4, Jp_diff*scale, label='hole diffusion current')
    plt.plot(x*1e4, Jp*scale, label='total hole current')
    plt.plot(x*1e4, (Jp+Jn)*scale, label='total current')
    plt.xlabel(r'Position [$\mu$m]')
    plt.ylabel('Current [uA]')
    plt.legend()



# # jn = az.electron_current(location=(p1, p2))

# # # plt.plot(n1d[len(n1d)//2:len(n1d)//2+20])
# # plt.plot(jn)
axs[-1].set_xlabel(r'Position [$\mathregular{\mu m}$]')
fig.suptitle(r'$V_{bias}=$' + f'{bias_voltage}V')
plt.show()

