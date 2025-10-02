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
from random import shuffle
data = '/Users/markusfasching/EPFL/Work/project-SOAP/scripts/SOAP-time-code/data/gete/ramp_up.pos_0.extxyz'
data2 = '/Users/markusfasching/EPFL/Work/project-SOAP/scripts/SOAP-time-code/data/gete/ramp_down.pos_0.extxyz'
#data = '/Users/markusfasching/EPFL/Work/project-SOAP/scripts/SOAP-time-code/data/water-ice/traj_waterice.xyz'

SOAP_cutoff = 6
SOAP_max_angular = 5
SOAP_max_radial = 6

centers = [52] # center on Te
neighbors  = [32, 52]

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
    trj2, ids_atoms2 = read_trj(data2)
    trj = trj1 + trj2

    ids_atoms = [atom.index for atom in trj[0] if atom.number == centers[0]] 
    shuffle(ids_atoms)
    train_atoms = ids_atoms[:20] # take only 20 random atoms
    #trj = trj1
    #trj = tamper_water_trj(trj)
    #trj = trj[::2]
    interval = 200
    #trj = trj[::4][500:1000]
    #dir = f'results/GeTe/NEW/SOAP_PCA/temporalCOV_PCA'
    dir = f'results/GeTe/NEW/deeptime/mean'
    #dir = f'results/GeTe/NEW/SOAP_PCA/fullCOV_PCA'
    Path(dir).mkdir(parents=True, exist_ok=True)
    for lag in tqdm([10, 20, 50, 100], 'Processing lag times'):
        label = dir + f'mean_lag{lag}_gete_updown_interval{interval}_r6_maxang5_maxrad6'
        
        soap_features[0] = SOAP_mean(trj, interval, train_atoms, HYPER_PARAMETERS, centers, neighbors)[0] # first center type TxD

        n_frames, n_features = soap_features.shape

        # Option 1: treat each atom as a sample and flatten
        X = soap_features #.reshape(n_frames, n_atoms * n_features)

        # single atom and no TA for test data
        ids_atoms_test = [atom.index for atom in trj[0] if atom.number == centers[0]][-1:]
        X_test = SOAP_mean(trj, 1, ids_atoms_test, HYPER_PARAMETERS, centers, neighbors)[0] # first center type TxD

        # DEEPTIME ANALYSIS
        run_analysis(X_train, X_test, lag, label)