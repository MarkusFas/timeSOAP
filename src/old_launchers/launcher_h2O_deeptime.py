from src.read_data import read_trj, tamper_water_trj
from src.descriptors import SOAP_mean
from src.PCAtransform import pcatransform, PCA_obj
from src.timeaverages import timeaverage
from src.fourier import fourier_trafo
from src.visualize import plot_pca, plot_pca_merge, plot_pca_height, plot_2pca, plot_fourier
import numpy as np
from tqdm import tqdm
from deeptime_modules import run_analysis
from pathlib import Path
import os
data = '/Users/markusfasching/EPFL/Work/project-SOAP/scripts/SOAP-time-code/data/gete/ramp_up.pos_0.extxyz'
data2 = '/Users/markusfasching/EPFL/Work/project-SOAP/scripts/SOAP-time-code/data/gete/ramp_down.pos_0.extxyz'
data = '/Users/markusfasching/EPFL/Work/project-SOAP/scripts/SOAP-time-code/data/water-ice/traj_waterice.xyz'

SOAP_cutoff = 6
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
    trj = tamper_water_trj(trj)
    #trj = trj[::2]
    interval = 10
    #trj = trj[::4][500:1000]
    oxygen_atoms = [idx for idx, number in enumerate(trj[0].get_atomic_numbers()) if number==8] # only oxygen atoms
    for interval in [1, 10, 100]:
        dir = f'results/water_ice/test_set/SOAP_deeptime_single/interval{interval}'
        Path(dir).mkdir(parents=True, exist_ok=True)
        for lag in tqdm([10, 25, 50], 'Processing lag times'):
            label = f'SOAP_deeptime_single_lag{lag}_water_interval{interval}_r6_maxang5_maxrad6'
            label = os.path.join(dir, label)
            color = np.array([atoms.positions[:,2] for atoms in trj]).T
            
            ids_atoms_train = [atom.index for atom in trj[0] if atom.number == centers[0]][:1]
            ids_atoms_train__ = [atom.index for atom in trj[0] if atom.number == centers[0]][1:2]
            X_train = SOAP_mean(trj, interval, ids_atoms_train, HYPER_PARAMETERS, centers, neighbors)[0] # first center type TxD
            X_train2 = SOAP_mean(trj, 1000, ids_atoms_train, HYPER_PARAMETERS, centers, neighbors)[0] # first center type TxD
            
            X_train__ = SOAP_mean(trj, interval, ids_atoms_train__, HYPER_PARAMETERS, centers, neighbors)[0] # first center type TxD
            X_train2__ = SOAP_mean(trj, 1000, ids_atoms_train__, HYPER_PARAMETERS, centers, neighbors)[0] # first center type TxD
            from sklearn.decomposition import PCA
            pca1 = PCA(n_components=4)
            pca1.fit(X_train)
            pca2 = PCA(n_components=4)
            pca2.fit(X_train2)
            idx = np.argmax(np.abs(pca1.components_[0,:]))
            idx2 = np.argmax(np.abs(pca2.components_[0,:]))
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()

            for i in [idx, idx2]:
                ax.plot(X_train[:,i], color='C0', alpha=0.5)
                ax.plot(X_train2[:,i], color='C1', alpha=0.5)
                ax.plot(X_train__[:,i], color='C0', alpha=0.5, linestyle='--')
                ax.plot(X_train2__[:,i], color='C1', alpha=0.5, linestyle='--')
            plt.show()

            """
            for i in [200,300,400,500,350]:
                ax.plot(X_train[:,i], color='C0', alpha=0.5)
                ax.plot(X_train2[:,i], color='C1', alpha=0.5)
                ax.plot(X_train__[:,i], color='C0', alpha=0.5, linestyle='--')
                ax.plot(X_train2__[:,i], color='C1', alpha=0.5, linestyle='--')"""
            plt.show()
            # single atom and no TA for test data
            ids_atoms_test = [atom.index for atom in trj[0] if atom.number == centers[0]][-1:]
            X_test = SOAP_mean(trj, 1, ids_atoms_test, HYPER_PARAMETERS, centers, neighbors)[0] # first center type TxD

            # DEEPTIME ANALYSIS
            run_analysis(X_train, X_test, lag, label)
