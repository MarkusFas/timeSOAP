from src.read_data import read_trj
from src.descriptors import eval_SOAP, SOAP_COV, SOAP_PCA
from src.PCAtransform import pcatransform, PCA_obj
from src.timeaverages import timeaverage
from src.fourier import fourier_trafo
from src.visualize import plot_pca, plot_pca_merge, plot_pca_height, plot_2pca, plot_fourier
import numpy as np
from tqdm import tqdm
data = '/Users/markusfasching/EPFL/Work/project-SOAP/scripts/SOAP-time-code/data/water-ice/traj_waterice.xyz'

SOAP_cutoff = 8
SOAP_max_angular = 6
SOAP_max_radial = 6

centers = [8]
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

    trj, ids_atoms = read_trj(data)
    trj = trj
    color = np.array([atoms.positions[:,2] for atoms in trj]).T
    for interval in [1,25, 50, 100]:
        label = f'less_run_interval_{interval}_r{SOAP_cutoff}_maxang{SOAP_max_angular}_maxrad{SOAP_max_radial}'
        #COV = SOAP_COV(trj, interval, ids_atoms, HYPER_PARAMETERS, centers, neighbors)
        #COV_oxygens = COV[0]
        pca_obj = PCA_obj(4, label)
        #pca_obj.compute_eigen(COV_oxygens)
        pca_obj.load()
        #pca_obj.save()
        projected = SOAP_PCA(trj, ids_atoms, HYPER_PARAMETERS, centers, neighbors, pca_obj)
        fourier = fourier_trafo(projected.transpose(1,0,2)).transpose(1,0,2)
        
        plot_fourier(fourier, color[ids_atoms], label)
        plot_pca_height(projected, color[ids_atoms], label)
        plot_pca_merge(projected, color[ids_atoms], label)
        plot_pca(projected, color[ids_atoms], label)
        plot_2pca(projected, color[ids_atoms], label)
