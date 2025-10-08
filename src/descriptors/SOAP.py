import torch 
import metatensor.torch as mts
from metatomic.torch import System, ModelEvaluationOptions, ModelOutput, systems_to_torch, load_atomistic_model
from metatensor.torch import Labels, TensorBlock, mean_over_samples
from featomic.torch import SoapPowerSpectrum
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import ase.neighborlist
from vesin import ase_neighbor_list
from memory_profiler import profile


class SOAP_descriptor():
    def __init__(self, HYPERS, centers, neighbors, selected_atoms=None):
        self.calculator = SoapPowerSpectrum(**HYPERS)
        self.centers = centers
        self.neighbors = neighbors
        self.sel_keys = Labels(
            names=["center_type", "neighbor_1_type", "neighbor_2_type"],
            values=torch.tensor([[i,j,k] for i in centers for j in neighbors for k in neighbors if j <=
                k], dtype=torch.int32),
        )

        self.sel_samples = None
        #TODO default to all atoms in the trajectory

        if selected_atoms is not None:
            self.sel_samples = Labels(
                names=["atom"],
                values=torch.tensor(selected_atoms, dtype=torch.int64).unsqueeze(-1),
            )

    
    def set_samples(self, selected_atoms):
        self.sel_samples = Labels(
            names=["atom"],
            values=torch.tensor(selected_atoms, dtype=torch.int64).unsqueeze(-1),
        )

    def calculate(self, systems):
        
        soap = self.calculator(
            systems,
            selected_samples=self.sel_samples,
            selected_keys=self.sel_keys,
        )
        
        soap = soap.keys_to_samples("center_type")
        soap = soap.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])
        soap_block = soap.block()
        return soap_block #TODO: return numpy