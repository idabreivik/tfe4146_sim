from cProfile import label
import sesame
import numpy as np
import json
import matplotlib.pyplot as plt
from shlib import to_path, mkdir
root = to_path(__file__).parent
gzip_dir = to_path(root, 'gzip')
mkdir(gzip_dir)


# Doping
nA = 1e15 # [cm^-3]
nD = 1e15 # [cm^-3]

# Bias voltage
voltages = np.linspace(-0.6, 0.6, 100)

# CONSTANTS
T = 300 # Temp [K]
kB = 8.62e-5 # Boltzmann [eV/K]
q = 1.6e-19 # Electron charge [C]


# Create grid and initialize system
L = 10e-4 # length of the system in the x-direction [cm]
junction = L/2 # extent of the junction from the left contact [cm]
x = np.linspace(0, L, 1000)
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
sys.contact_type('Ohmic', 'Ohmic')

# Define the surface recombination velocities for electrons and holes [cm/s]
Sn_left, Sp_left, Sn_right, Sp_right =  1e7, 1e7, 1e7, 1e7  # cm/s
# Sn_left, Sp_left, Sn_right, Sp_right =  1e9,1e9,1e9,1e9
sys.contact_S(Sn_left, Sp_left, Sn_right, Sp_right)

j = sesame.IVcurve(sys, voltages, str(to_path(gzip_dir, 'j_eq')))
j = j * sys.scaling.current

plt.plot(voltages, j,'-o')        # plot j-v curve
plt.xlabel('Voltage [V]')
plt.ylabel('Current [A/cm^2]')
plt.grid()                       # show grid lines
plt.show()
