from src.read_data import read_trj, tamper_water_trj
from src.descriptors import eval_SOAP, SOAP_COV, SOAP_COV_repair, SOAP_PCA, SOAP_COV_test
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

def run_simulation(name):
    trj1, ids_atoms = read_trj(data)
    trj2, ids_atoms2 = read_trj(data2)
    trj = trj1 + trj2
    #trj = trj1
    #trj = tamper_water_trj(trj)
    #trj = trj[:10]
    color = np.array([atoms.positions[:,2] for atoms in trj]).T
    oxygen_atoms = [idx for idx, number in enumerate(trj[0].get_atomic_numbers()) if number==52] # only oxygen atoms
    shuffle(oxygen_atoms)
    train_atoms = oxygen_atoms[:20] # take only 20 random atoms
    test_atoms = oxygen_atoms[30:50] # take only 20 random atoms
    
    
    X_values = []
    test_intervals = [1, 50, 200, 1000]
    for interval in tqdm(test_intervals, 'Processing intervals'):
        label = f'SOAP_GeTe_interval_{interval}_r{SOAP_cutoff}_maxang{SOAP_max_angular}_maxrad{SOAP_max_radial}'
        label = os.path.join(dir, label)

        X = []
        heights = []
        #for oxygen in tqdm(oxygen_atoms[:1], 'Processing oxygen atoms'):
        ids_atoms = oxygen_atoms
        #height = np.array([atoms.positions[oxygen_atoms,2] for atoms in trj])
        if name=='temporalCOV_PCA' or name=='temporalGEV':
            means, cov_i, cov_t, atomsel_element = SOAP_COV_test(trj, interval, train_atoms, HYPER_PARAMETERS, centers, neighbors)
        elif name=='atomsCOV_PCA':
            means, cov_i, cov_t, atomsel_element = SOAP_COV_test(trj, interval, train_atoms, HYPER_PARAMETERS, centers, neighbors)
        elif name=='fullCOV_PCA_old'   
            means, cov_t, atomsel_element = SOAP_COV_repair(trj, interval, ids_atoms, HYPER_PARAMETERS, centers, neighbors)
        
        dir = f'results/GeTe/CORRECT/SOAP_PCA/{name}'
        #dir = f'results/GeTe/NEW/SOAP_PCA/temporalGEV'
        #dir = f'results/GeTe/NEW/SOAP_PCA/fullCOV_PCA'
        Path(dir).mkdir(parents=True, exist_ok=True)


        for i, cov in enumerate(cov_t):
            #label = f'SOAP_PCA_repair_GeTe_{centers[i]}_updown_less_run_interval_{interval}_r{SOAP_cutoff}_maxang{SOAP_max_angular}_maxrad{SOAP_max_radial}'
            #label = f'SOAP_PCA_singleatoms_water_{centers[i]}_interval_{interval}_r{SOAP_cutoff}_maxang{SOAP_max_angular}_maxrad{SOAP_max_radial}'
            pca_obj = PCA_obj(4, label)
            
            if name=='temporalGEV':
                # GEV 
                pca_obj.compute_eigen_NEW(means[i], cov_t[i], cov_i[i])
            elif name=='temporalCOV_PCA':
                # temporal covariance only
                pca_obj.compute_eigen(means[i], cov_t[i])
            elif name=='atomsCOV_PCA':
                # full covariance PCA  ['temporalCOV_PCA', 'temporalGEV', 'fullCOV_PCA']
                pca_obj.compute_eigen(means[i], cov_i[i])
            elif name=='fullCOV_PCA_old':
                pca_obj.compute_eigen(means[i], cov_t[i]) # old approach
            #pca_obj.load()
            pca_obj.save()

            # test model on unseen atoms
            projected = SOAP_PCA(trj, test_atoms, HYPER_PARAMETERS, centers, neighbors, pca_obj)
            #fourier = fourier_trafo(projected.transpose(1,0,2)).transpose(1,0,2)
            #plot_fourier(fourier, color[atomsel_element[i]], label)
            X=projected
            heights=np.array([atoms.positions[test_atoms,2] for atoms in trj]).T
            #X = np.concatenate(X, axis=0)
            plot_pca_height(X, heights, label)
            plot_pca_merge(X, heights, label)
            plot_pca(X, heights, label)
            plot_2pca(X, heights, label)
            X_values.append(X.transpose(1,0,2))
            print('done normal plots')
    plot_compare_timeave_PCA(X_values, [0,1], label, test_intervals) # need to transpose to T,N,P
    print('done time ave plots')
    plot_compare_atoms_PCA(X_values, [0,1], label, test_intervals) # need to transpose to T,N,P

if __name__=='__main__':
    methods = ['temporalCOV_PCA', 'temporalGEV', 'atomsCOV_PCA', 'fullCOV_PCA_old']
    for name in tqdm(methods, 'Running different simulations'):
        run_simulation(name)