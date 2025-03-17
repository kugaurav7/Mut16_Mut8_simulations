# Import all the essential libraries
import MDAnalysis as mda
import mdtraj as md
import numpy as np
import gsd.hoomd
import argparse
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser(description="Please enter your file paths")
parser.add_argument('gsd_file', help="GSD file containing topology")
parser.add_argument('dcd_file', help="DCD file containing trajectory")
parser.add_argument('output_file', help="File to save density profiles")
args = parser.parse_args()

# Convert gsd to pdb
# Try opening in 'r' mode instead of 'rb'
with gsd.hoomd.open(gsd_file, 'r') as traj:
    snap = traj[0]  # Extract first snapshot

# Convert to MDAnalysis format and save as PDB
u = mda.Universe.empty(n_atoms=snap.particles.N, trajectory=True)
u.atoms.positions = snap.particles.position  # Assign atom positions
u.atoms.write("pdb_output.pdb")  # Save as PDB

# Load universe
u = mda.Universe("pdb_output.pdb", args.dcd_file)
box = u.trajectory[-1].dimensions[:]
temperature = 275

# Define an estimated mass per bead (110 Da per residue)
average_bead_mass = 110.0  
mut16.masses = np.full(len(mut16), average_bead_mass)  # Assign mass to each bead
mut8.masses = np.full(len(mut8), average_bead_mass)

# Select the proteins
mut16 = u.select_atoms("index 0:62399")
mut8 = u.select_atoms("index 62400:64949")

# Histogram parameters
dz = 0.5
histoVolume = box[0] * box[1] * dz
histoCount = int(box[2] / dz) + 1
histoBins = np.linspace(0, box[2], num=histoCount)  # Start from 0, not negative

# Density computation
density_mut16 = []
density_mut8 = []

# Initialize lists to store density profiles
density_mut16 = []
density_mut8 = []

for ts in u.trajectory:

    # Compute mut16 COM
    mut16_COM = mut16.center_of_mass()

    # Shift all atoms to recenter the protein
    shift_vector = np.array([box[0] / 2, box[1] / 2, box[2] / 2]) - mut16_COM
    mut16.positions += shift_vector
    mut8.positions += shift_vector

    # Apply periodic boundary conditions (PBC)
    mut16.atoms.wrap(compound='group', box=box)  
    mut8.atoms.wrap(compound='group', box=box)

    # Extract z-coordinates after shifting and wrapping
    z_mut16 = mut16.positions[:, 2]
    z_mut8 = mut8.positions[:, 2]

    # Compute histogram for density
    density_mut16_frame, _ = np.histogram(z_mut16, bins=histoBins, density=False)
    density_mut8_frame, _ = np.histogram(z_mut8, bins=histoBins, density=False)

    # Normalize by volume
    density_mut16.append(density_mut16_frame / histoVolume)  
    density_mut8.append(density_mut8_frame / histoVolume)  

# Convert to NumPy array **AFTER** the loop
density_mut16 = np.array(density_mut16)
density_mut8 = np.array(density_mut8)

# Compute mean and standard error of mean (SEM)
density_mut16_mean = np.mean(density_mut16, axis=0)
density_mut8_mean = np.mean(density_mut8, axis=0)
density_mut16_sem = np.std(density_mut16, axis=0) / np.sqrt(density_mut16.shape[0])
density_mut8_sem = np.std(density_mut8, axis=0) / np.sqrt(density_mut8.shape[0])


# Define the 'tanh' function
def tanh_function(z, rhol, rhov, width, z0):
    return 0.5 * (rhol + rhov) + 0.5 * (rhov - rhol) * np.tanh((z - z0) / width)

# Function to estimate `z0`
def guess_z0(bins, hist, shift=5):
    """
    Estimate the transition point (z0) by finding the max density gradient.
    """
    diff = hist[:-shift] - hist[shift:]
    guess_ind = np.argmax(diff) - shift
    return bins[guess_ind]

# Function to extract a single-line protein sequence from a file
def extract_protein_sequence(file_name):
    """
    Reads and extracts the protein sequence from a single-line file.

    Parameters:
        file_name (str): The path to the .dat file containing the sequence.

    Returns:
        str: The extracted protein sequence.
    """
    with open(file_name, 'r') as file:
        sequence = file.readline().strip()  # Read the first (only) line and remove extra spaces

    return sequence

mut8_seq = extract_protein_sequence("mut8_mutant.dat")

# Function to fit density profile
def density_fit(bins, hist):
    """
    Fit a tanh function to the histogram data to determine liquid and vapor densities.
    """
    initial_guess = (hist[:10].mean(), hist[-10:].mean(), 1.0, guess_z0(bins, hist))
    param_bounds = ((0., 0., 0., 0.), (np.inf, np.inf, np.inf, bins.max()))

    # Fit the tanh function
    params, cov = curve_fit(tanh_function, bins[:-1], hist, p0=initial_guess, bounds=param_bounds, maxfev=10000)

    # Extract parameters and errors
    errors = np.sqrt(np.diag(cov))

    return params, errors

# Apply the density fit to the data
params, errors = density_fit(histoBins, density_mut8_mean)

# Extract fitted values
rhol_fit, rhov_fit, width_fit, z0_fit = params
rhol_fit_err, rhov_fit_err, width_fit_err, z0_fit_err = errors

# Save density profile and fit results
np.savetxt("Output_table.dat", np.column_stack((histoBins[:-1], density_mut8_mean)), fmt='%1.4e')

# Calculate delta G of transfer from the rhol and rhov values
deltaG = -np.log(rhol_fit / rhov_fit)*0.0083*temperature

# Save phase densities (liquid and vapor) with errors
with open("Output_table.dat", 'w') as f:
    f.write("Sequence\tLiquid Density\tVapor Density\t Delta G\n")
    f.write(f"{mut8_seq}\t{rhol_fit:.5f}\t{rhov_fit:.5f}\t{deltaG:.5f}\n")

# Print results
print(f"Sequence: {mut8_seq}")
print(f"Fitted Liquid Density: {rhol_fit:.5f} ± {rhol_fit_err:.5f}")
print(f"Fitted Vapor Density: {rhov_fit:.5f} ± {rhov_fit_err:.5f}")
print(f"deltaG: {deltaG:.5f} (KJ/mol)")
