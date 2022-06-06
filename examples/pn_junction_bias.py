from cProfile import label
import sesame
import numpy as np
import json
import matplotlib.pyplot as plt
from shlib import to_path, mkdir
root = to_path(__file__).parent
gzip_dir = to_path(root, 'gzip')
mkdir(gzip_dir)

# OPTIONS
# This example support simulation on abrupt p-n junctions and contain code for plotting band diagram, electrostatic potential, electric field, space charge density, current components and electron density. Use the boolean variables to enable/disable plots:
PLOT_BAND_DIAGRAM = True
PLOT_ELECTROSTATIC_POTENTIAL = True
PLOT_ELECTRON_DENSITY = False
PLOT_ELECTRIC_FIELD = True
PLOT_SPACE_CHARGE_DENSITY = True
PLOT_CURRENT_COMPONENTS = True


# Doping
# The simulations are performed on abrupt, homogenous p-n junctions with constant doping concentration on either side of the junction. Modify the variables nA and nD to change the doping concentration.
nA = 1e15 # [cm^-3]
nD = 1e14 # [cm^-3]

# Bias voltage
# This is the bias voltage across the p-n junction
bias_voltage = -0.5

# CONSTANTS
T = 300 # Temp [K]
kB = 8.62e-5 # Boltzmann [eV/K]
q = 1.6e-19 # Electron charge [C]

# Create grid and initialize system
L = 20e-4 # length of the system in the x-direction [cm]
junction = L/2 # extent of the junction from the left contact [cm]
n_points = 10000 # Number of grid points. Increasing the number of points give better accuracy, but will increase simulation time
x = np.linspace(0, L, n_points)




# Coorindates of the edges of the diode
p1 = (0, 0)
p2 = (L, 0)

# Initialize the system
sys = sesame.Builder(x, T=T)

# Add material properties
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

# Load and plot
sys, result = sesame.load_sim(
    to_path(gzip_dir, 'j_eq_0.gzip'))  # load data file
az = sesame.Analyzer(sys,result)                   # get Sesame analyzer object

# ec = az.get_band_edge((p1, p2))


if PLOT_BAND_DIAGRAM:
    fig = plt.figure()
    az.band_diagram((p1, p2), fig=fig)


v = az.get_electrostatic_potential((p1, p2))
if PLOT_ELECTROSTATIC_POTENTIAL:
    plt.figure()
    plt.title('Electrostatic potential')
    plt.ylabel(r'Volt [V]')
    plt.xlabel(r'Position [$\mu$m]')
    plt.plot(x*1e4, v)

# Electron and hole density
n = az.electron_density((p1, p2))
n = n * sys.scaling.density
p = az.hole_density((p1, p2))
p = p * sys.scaling.density

if PLOT_ELECTRON_DENSITY:
    fig = plt.figure()
    plt.plot(x*1e4, n)
    plt.title(r'Electron density')
    plt.ylabel(r'[cm$^{-3}$]')
    plt.xlabel(r'Position [$\mu$m]')


# E field
E = az.get_e_field((p1, p2))
if PLOT_ELECTRIC_FIELD:
    fig = plt.figure()
    plt.plot(x*1e4, E)
    plt.title(r'Electric field')
    plt.ylabel(r'$\mathcal{E}(x)$ [V/cm]')
    plt.xlabel(r'Position [$\mu$m]')


# E field
rho = az.get_space_charge_density((p1, p2))
if PLOT_SPACE_CHARGE_DENSITY:
    fig = plt.figure()
    plt.plot(x*1e4, rho)
    plt.title(r'Space charge density')
    plt.ylabel(r'$\rho(x)$ [C/cm^2]')
    plt.xlabel(r'Position [$\mu$m]')


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

plt.show()

