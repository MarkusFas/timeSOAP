import ase.io
from ase.visualize.plot import plot_atoms
import matplotlib.pyplot as plt
import numpy as np
import random
import pathlib

def read_trj(file, index=':'):
    trj = ase.io.read(file, index=index)
    print(f'Read trajectory with {len(trj)} frames from {file}')
    for structure in trj:
        structure.wrap()
    #fig, ax = plt.subplots()
    #plot_atoms(trj[0],
    #       ax=ax,
    #       radii=0.3,
    #       rotation=(('90x,0y,0z')),  # view direction
    #       show_unit_cell=1)         # draw full box

    # Save figure
    #plt.show()
    #plt.savefig("atoms.png", dpi=300, bbox_inches="tight")
    #plt.close()
    #ids_atoms = [atom.index for atom in trj[0] if atom.symbol == 'Te']
    #ids_atoms = [atom.index for atom in trj[0]]
    return trj


def get_molecule_indices(o_idx):
        return [o_idx, o_idx+1, o_idx+2]


def tamper_water_trj(traj):
    " select 25 water molecules"

    atoms = traj[0]
    cell = atoms.get_cell()
    zmax = cell[2, 2]

    # Get oxygens
    oxygen_indices = [i for i, a in enumerate(atoms) if a.symbol == 'O']
    oz = atoms.positions[oxygen_indices, 2]

    # Classify oxygens by region
    ice_mask = (oz >= 25.0) & (oz <= 30.0)
    water_mask = (oz < 12.0) & (oz >= 7.0)

    ice_oxygens = [oxygen_indices[i] for i in np.where(ice_mask)[0]]
    liquid_oxygens = [oxygen_indices[i] for i in np.where(water_mask)[0]]

    print(f"Found {len(ice_oxygens)} ice molecules and {len(liquid_oxygens)} liquid molecules.")

    # Randomly pick 20 from each
    ice_sel = random.sample(ids_atoms, 20)
    liquid_sel = random.sample(liquid_oxygens, 20)

    print("Selected ice oxygens:", ice_sel)
    print("Selected liquid oxygens:", liquid_sel)

    # If each O is followed by its H’s (OHH ordering)

    ice_molecules = [get_molecule_indices(i) for i in ice_sel]
    liquid_molecules = [get_molecule_indices(i) for i in liquid_sel]

    # Flatten lists
    ice_molecules = [idx for mol in ice_molecules for idx in mol]
    liquid_molecules = [idx for mol in liquid_molecules for idx in mol]


    new_traj = []
    for atoms in traj:
        new_traj.append(atoms[ice_molecules])

    for atoms in traj:
        new_traj.append(atoms[liquid_molecules])

    return new_traj

if __name__=='__main__':
    print('Nothing to do here')
