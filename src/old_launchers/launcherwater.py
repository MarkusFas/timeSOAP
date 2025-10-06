from src.setup.read_data import read_trj, tamper_water_trj
from src.old_scripts.descriptors import eval_SOAP, SOAP_COV, SOAP_COV_repair, SOAP_PCA
from src.transformations.PCAtransform import pcatransform, PCA_obj
from src.timeaverages import timeaverage
from src.old_scripts.fourier import fourier_trafo
from src.visualize import plot_pca, plot_pca_merge, plot_pca_height, plot_2pca, plot_fourier
import numpy as np
from tqdm import tqdm
data = '/Users/markusfasching/EPFL/Work/project-SOAP/scripts/SOAP-time-code/data/water-ice/traj_waterice.xyz'

SOAP_cutoff = 6
SOAP_max_angular = 5
SOAP_max_radial = 6

#centers = [32, 52] # center on Te
#neighbors  = [32, 52]
centers = [8]
neighbors = [1,8]
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

SIGMA = 10 # unit get from trj??
interval = 1
if __name__=='__main__':

    trj1, ids_atoms = read_trj(data)
    #trj2, ids_atoms2 = read_trj(data2)
    #trj = trj1 + trj2
    trj = trj1
    trj = tamper_water_trj(trj)
    
    color = np.array([atoms.positions[:,2] for atoms in trj]).T
    oxygen_atoms = [idx for idx, number in enumerate(trj[0].get_atomic_numbers()) if number==8] # only oxygen atoms

    X = []
    heights = []
    for oxygen in tqdm(oxygen_atoms[:20], 'Processing oxygen atoms'):
        ids_atoms = [oxygen]
        height = np.array([atoms.positions[oxygen,2] for atoms in trj])
        means, covs, atomsel_element = SOAP_COV_repair(trj, interval, ids_atoms, HYPER_PARAMETERS, centers, neighbors)
        for i, cov in enumerate(covs):
            #label = f'SOAP_PCA_repair_GeTe_{centers[i]}_updown_less_run_interval_{interval}_r{SOAP_cutoff}_maxang{SOAP_max_angular}_maxrad{SOAP_max_radial}'
            label = f'SOAP_PCA_singleatoms_water_{centers[i]}_interval_{interval}_r{SOAP_cutoff}_maxang{SOAP_max_angular}_maxrad{SOAP_max_radial}'
            pca_obj = PCA_obj(4, label)
            pca_obj.compute_eigen(means[i], cov)
            #pca_obj.load()
            pca_obj.save()
            projected = SOAP_PCA(trj, atomsel_element[i], HYPER_PARAMETERS, centers, neighbors, pca_obj)
            #fourier = fourier_trafo(projected.transpose(1,0,2)).transpose(1,0,2)
            #plot_fourier(fourier, color[atomsel_element[i]], label)
            X.append(projected)
            heights.append(height)
    X = np.concatenate(X, axis=0)
    plot_pca_height(X, heights, label)
    plot_pca_merge(X, heights, label)
    plot_pca(X, heights, label)
    plot_2pca(X, heights, label)
