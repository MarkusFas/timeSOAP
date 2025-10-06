from src.setup.read_data import read_trj, tamper_water_trj
from src.old_scripts.descriptors import SOAP_mean, SOAP_full
from src.transformations.PCAtransform import pcatransform, PCA_obj
from src.timeaverages import timeaverage
from src.old_scripts.fourier import fourier_trafo
from src.visualize import plot_compare_atoms, plot_compare_timeave
import numpy as np
from tqdm import tqdm
from deeptime_modules import run_analysis
from pathlib import Path
import os
data = '/Users/markusfasching/EPFL/Work/project-SOAP/scripts/SOAP-time-code/data/gete/ramp_up.pos_0.extxyz'
data2 = '/Users/markusfasching/EPFL/Work/project-SOAP/scripts/SOAP-time-code/data/gete/ramp_down.pos_0.extxyz'
data = '/Users/markusfasching/EPFL/Work/project-SOAP/scripts/SOAP-time-code/data/water-ice/traj_waterice.xyz'
data = '/Users/markusfasching/EPFL/Work/project-SOAP/scripts/SOAP-time-code/data/positions.xyz'
SOAP_cutoff = 5
SOAP_max_angular = 5
SOAP_max_radial = 6

centers = [8] # center on Te
neighbors  = [1,8]

HYPER_PARAMETERS = {
    "cutoff": {
        "radius": SOAP_cutoff, #4 #5 #6
        "smoothing": {"type": "ShiftedCosine", "width": 0.5},
    },
    "density": {
        "type": "Gaussian",
        "width": 0.25, #changed from 0.3
    },
    "basis": {
        "type": "TensorProduct",
        "max_angular": SOAP_max_angular, #8
        "radial": {"type": "Gto", "max_radial": SOAP_max_radial}, #6
    },
}


if __name__=='__main__':

    trj1, ids_atoms = read_trj(data)
    #trj2, ids_atoms2 = read_trj(data2)
    #trj = trj1 + trj2
    trj = trj1
    #trj = tamper_water_trj(trj)
    #trj = trj[:2]

    oxygen_atoms = [idx for idx, number in enumerate(trj[0].get_atomic_numbers()) if number==8] # only oxygen atoms

    dir = f'results/waterfullbox/visual/CUTOFF/SOAP_deeptime_single'
    Path(dir).mkdir(parents=True, exist_ok=True)

    label = f'SOAP_deeptime_single_water_r{SOAP_cutoff}_maxang{SOAP_max_angular}_maxrad{SOAP_max_radial}'
    label = os.path.join(dir, label)

    ids_atoms_train = [atom.index for atom in trj[0] if atom.number == centers[0]][:10]
    test_intervals = [1, 25, 100, 1000]
    X_values = []
    for interval in test_intervals:
        X, properties = SOAP_full(trj, interval, ids_atoms_train, HYPER_PARAMETERS, centers, neighbors)
        X_values.append(X[0]) # first center type TxNxD
    SOAP_idx = np.random.randint(0, X_values[0].shape[-1], 6)

    print('done with calculation')
    plot_compare_timeave(X_values, SOAP_idx, label, properties.values.numpy(), test_intervals)
    plot_compare_atoms(X_values, SOAP_idx, label, properties.values.numpy(), test_intervals)