from src.setup.read_data import read_trj, tamper_water_trj
from src.old_scripts.descriptors import  SOAP_COV_test
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
    ids_atoms = oxygen_atoms[:10]
    
    means, inframecov, outframecov, atomsel_element = SOAP_COV_test(trj, interval, ids_atoms, HYPER_PARAMETERS, centers, neighbors)
    
    print(np.trace(inframecov[0]), np.trace(outframecov[0]))