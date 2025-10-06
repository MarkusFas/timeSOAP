from src.read_data import read_trj, tamper_water_trj
from src.descriptors import eval_SOAP, SOAP_COV, SOAP_COV_repair, SOAP_PCA, SOAP_COV_test, SOAP_COV_directPCAtest
from src.PCAtransform import pcatransform, PCA_obj
from src.timeaverages import timeaverage
from src.fourier import fourier_trafo
from src.visualize import plot_pca, plot_pca_merge, plot_pca_height, plot_2pca, plot_compare_timeave_PCA, plot_compare_atoms_PCA
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
from random import shuffle
data = '/Users/markusfasching/EPFL/Work/project-SOAP/scripts/SOAP-time-code/data/gete/ramp_up.pos_0.extxyz'
data2 = '/Users/markusfasching/EPFL/Work/project-SOAP/scripts/SOAP-time-code/data/gete/ramp_down.pos_0.extxyz'

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


interval = 1
if __name__=='__main__':

    trj1, ids_atoms = read_trj(data)
    trj2, ids_atoms2 = read_trj(data2)
    trj = trj1 + trj2
    #trj = trj1
    #trj = tamper_water_trj(trj)
    trj = trj[:10]
    color = np.array([atoms.positions[:,2] for atoms in trj]).T
    oxygen_atoms = [idx for idx, number in enumerate(trj[0].get_atomic_numbers()) if number==52] # only oxygen atoms
    shuffle(oxygen_atoms)
    train_atoms = oxygen_atoms[:1] # take only 20 random atoms
    test_atoms = oxygen_atoms[30:50] # take only 20 random atoms
    
    dir = f'results/GeTe/NEW/SOAP_PCA/temporalCOV_PCA'
    #dir = f'results/GeTe/TESTS/SOAP_PCA/temporalGEV'
    #dir = f'results/GeTe/NEW/SOAP_PCA/fullCOV_PCA'
    Path(dir).mkdir(parents=True, exist_ok=True)
    X_values = []
    test_intervals = [1] #[1, 25, 200, 1000]
    for interval in tqdm(test_intervals, 'Processing intervals'):
        label = f'SOAP_GeTe_interval_{interval}_r{SOAP_cutoff}_maxang{SOAP_max_angular}_maxrad{SOAP_max_radial}'
        label = os.path.join(dir, label)

        X = []
        heights = []
        #for oxygen in tqdm(oxygen_atoms[:1], 'Processing oxygen atoms'):
        ids_atoms = oxygen_atoms
        #height = np.array([atoms.positions[oxygen_atoms,2] for atoms in trj])
        #means, cov, atomsel_element = SOAP_COV_repair(trj, interval, ids_atoms, HYPER_PARAMETERS, centers, neighbors)
        means, cov_i, cov_t, X_train, atomsel_element = SOAP_COV_directPCAtest(trj, interval, train_atoms, HYPER_PARAMETERS, centers, neighbors)
        for i, cov in enumerate(cov_t):
            #label = f'SOAP_PCA_repair_GeTe_{centers[i]}_updown_less_run_interval_{interval}_r{SOAP_cutoff}_maxang{SOAP_max_angular}_maxrad{SOAP_max_radial}'
            #label = f'SOAP_PCA_singleatoms_water_{centers[i]}_interval_{interval}_r{SOAP_cutoff}_maxang{SOAP_max_angular}_maxrad{SOAP_max_radial}'
            pca_obj = PCA_obj(4, label)
            # GEV 
            #pca_obj.compute_eigen_NEW(means[i], cov_t[i], cov_i[i])
            # temporal covariance only
            pca_obj.compute_eigen(means[i], cov_t[i] + cov_i[i])
            # full covariance PCA
            #pca_obj.compute_eigen(means[i], cov[i])
            #pca_obj.load()
            pca_obj.save()

            # direct PCA like deeptime:
            from sklearn.decomposition import PCA
            from scipy.linalg import eigh
            pca = PCA(n_components=1)
            pca.fit(X_train[i])
            pca_projection = pca.transform(X_train[i][:-1,:])
            bufferpca_projection = pca_obj.trafo(X_train[i])

            scat = np.einsum('ij,ik->jk', X_train[i][:-1,:], X_train[i][:-1,:]) / X_train[i][:-1,:].shape[0]
            COV = scat - np.einsum('i,j->ij', np.mean(X_train[i][:-1,:], axis=0), np.mean(X_train[i][:-1,:], axis=0))
            eigva, eigvec  = eigh(0.5*(COV + COV.T + 1e-8*np.eye(COV.shape[0])))
            print('Eigenvector from sklearn PCA:', pca.components_[0,:10])
            print('Eigenvector from eigenvalue decomposition:', eigvec[:,::-1][:10, 0])
            print('Eigevector from buffer:', pca_obj.eigvecs[:10,0])