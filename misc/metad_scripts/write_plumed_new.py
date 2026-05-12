import sys

from ase.io import read, write
import numpy as np
import random

plumed_fname = 'plumed.dat'
selected_type = 8        # example, use the actual type for O in your system
zmin, zmax = 30, 80

# Load the LAMMPS data file

#atoms = read(sys.argv[1], format="lammps-data")  # ASE Atoms object
atoms = read(sys.argv[1], format="gromacs")  # ASE Atoms object

positions = atoms.get_positions()
types = atoms.get_atomic_numbers()
print(types[:6])
# Find all unique atom types
unique_types = np.unique(types)

# Collect indices by type (1-based for PLUMED)
atom_indices_by_type = {t: np.where(types == t)[0] + 1 for t in unique_types}
unique_types = [8,1,74] #[6,7,8,1] #drop virtual site
selected_atoms = [
    i + 1 for i, (t, pos) in enumerate(zip(types, positions))
    if t == selected_type and zmin <= pos[2] <= zmax
]
#random.shuffle(selected_atoms) 
print('selected 30/', len(selected_atoms))
#selected_atoms = selected_atoms[:60]
z = positions[:, 2]
#print("z min, max (ASE units):", z.min(), z.max())
#print(positions[21,2])
import matplotlib.pyplot as plt
plt.scatter(positions[:,1], positions[:,2])
plt.show()
# ---------------------------
# Write plumed.dat
# ---------------------------
with open(plumed_fname, "w") as f:
    f.write("# Automatically generated plumed.dat\n\n")

    # Example SOAP block
    f.write(
        "soap_selected: METATOMIC ...\n"
        "      MODEL=model_soap.pt\n"
        "      EXTENSIONS_DIRECTORY=extensions\n"
    )

    species_mapping = {t: i + 1 for i, t in enumerate(unique_types)}
    # Assign SPECIES dynamically
    for t in unique_types:
        f.write(f"      SPECIES{species_mapping[t]}={','.join(str(idx) for idx in atom_indices_by_type[t])}\n")
    f.write(f"      SPECIES_TO_TYPES={','.join(str(t) for t in unique_types)}\n")
    if selected_atoms:
        f.write(f"      SELECTED_ATOMS={','.join(str(idx) for idx in selected_atoms)}\n")
    f.write(f"...\n")

    # Printing directives
    f.write(
        "PRINT ARG=soap_selected FILE=soap_selected_data STRIDE=1 FMT=%8.4f\n"
        "CV: SUM ARG=soap_selected PERIODIC=NO\n"
        "PRINT ARG=CV FILE=COLVAR STRIDE=10\n"
        "FLUSH STRIDE=100\n"
    )

print(f"plumed.dat written with {len(selected_atoms)} selected atoms")
