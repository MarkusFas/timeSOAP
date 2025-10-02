from src.read_data import read_trj
from src.descriptors import eval_SOAP, SOAP_COV, SOAP_COV_repair, SOAP_PCA
from src.PCAtransform import pcatransform, PCA_obj
from src.timeaverages import timeaverage
from src.fourier import fourier_trafo
from src.visualize import plot_pca, plot_pca_merge, plot_pca_height, plot_2pca, plot_fourier
import numpy as np
from tqdm import tqdm
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


if __name__=='__main__':

    trj1, ids_atoms = read_trj(data)
    trj2, ids_atoms2 = read_trj(data2)
    trj = trj1 + trj2
    
    ids_atoms = ids_atoms[:] #selecting only one atom for now
    color = np.array([atoms.positions[:,2] for atoms in trj]).T
    atomsel_element=[[atom.index for atom in trj[0] if atom.number == atom_type] for atom_type in centers] #one entry for each SOAP center
    atomsel_element = [ids_atoms]
    #TODO check for ids_atoms when defining atomsel_elementss
    for interval in [1,10,25,50,75,100,200]:
        means, covs, atomsel_element = SOAP_COV_repair(trj, interval, ids_atoms, HYPER_PARAMETERS, centers, neighbors)
        for i, cov in enumerate(covs):
            label = f'SOAP_PCA_avgatoms_GeTe_{centers[i]}_interval_{interval}_r{SOAP_cutoff}_maxang{SOAP_max_angular}_maxrad{SOAP_max_radial}'
            pca_obj = PCA_obj(4, label)
            pca_obj.compute_eigen(means[i], cov)

            # load a previous model
            #pca_obj.load()
            pca_obj.save()
            projected = SOAP_PCA(trj, ids_atoms, HYPER_PARAMETERS, centers, neighbors, pca_obj)

            plot_pca_height(projected, color[atomsel_element[i]], label)
            plot_pca_merge(projected, color[atomsel_element[i]], label)
            plot_pca(projected, color[atomsel_element[i]], label)
            plot_2pca(projected, color[atomsel_element[i]], label)
