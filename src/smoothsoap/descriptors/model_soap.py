import os
from typing import Dict, List, Optional

import torch
import numpy as np
from scipy.stats import moment
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    System,
    systems_to_torch,
)
from featomic.torch import SoapPowerSpectrum

class SOAP_CV(torch.nn.Module):
    def __init__(self, cutoff, max_angular, max_radial, centers, neighbors, projection_matrix=None):
        super().__init__()
        HYPER_PARAMETERS = {
            "cutoff": {
                "radius": cutoff, #4 #5 #6
                "smoothing": {"type": "ShiftedCosine", "width": 0.5},
            },
            "density": {
                "type": "Gaussian",
                "width": 0.25, #changed from 0.3
            },
            "basis": {
                "type": "TensorProduct",
                "max_angular": max_angular, #8
                "radial": {"type": "Gto", "max_radial": max_radial}, #6
            },
        }
        self.calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)
        self.centers = centers
        self.neighbors = neighbors
        self.selected_keys = Labels(
            names=["center_type", "neighbor_1_type", "neighbor_2_type"],
            values=torch.tensor([[i,j,k] for i in centers for j in neighbors for k in neighbors if j <=
                k], dtype=torch.int32),
        )

        self.id = f"SOAP_{cutoff}{max_angular}{max_radial}_{centers}"
        
        if projection_matrix !=None:
            self.register_buffer("projection_matrix", torch.tensor(trans_matrix.copy()).T)#[0].T)
        else:
            self.projection_matrix=None

        self.hypers={}

    def calculate(self, systems, selected_samples=None):
        if selected_samples is None:
            selected_samples = self.selected_samples

        soap = self.calculator(
            systems,
            selected_samples=selected_samples,
            selected_keys=self.selected_keys,
        )
        
        soap = soap.keys_to_samples("center_type")
        soap = soap.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])
        self.soap_block = soap.block()
        return self.soap_block.values
    
    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        if "features" not in outputs:
            return {}

        if outputs["features"].per_atom:
            raise ValueError("per_atom=True is not supported directly, output will be in features/per_atom")

        if len(systems[0]) == 0:
            # PLUMED is trying to determine the size of the output
            projected = torch.zeros((0,len(self.proj_dims)), dtype=torch.float64)
            projected_mean = torch.zeros((0,len(self.proj_dims)), dtype=torch.float64)
            samples = Labels(["system"], torch.zeros((0, 1), dtype=torch.int32))
            samples_per_atom = Labels(["system", "atom"], torch.zeros((0,2), dtype=torch.int32))
        else:
            soap = self.calculator(systems, selected_samples=selected_atoms, selected_keys=self.selected_keys)
            soap = soap.keys_to_samples("center_type")
            soap = soap.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])#self.neighbor_type_pairs)

            soap_block = soap.block()
    
            projected = torch.einsum('ij,jk->ik',(soap_block.values - self.mu), self.projection_matrix[:,self.proj_dims])#, dtype=torch.float64)

            samples_per_atom = soap_block.samples.remove("center_type")
            samples = Labels(["system"], torch.zeros((1, 1), dtype=torch.int32))
            
            projected_mean = torch.mean(projected, dim=0)
            projected_mean = projected_mean.unsqueeze(0)

        block_per_atom = TensorBlock(
            values=projected,
            samples=samples_per_atom,
            components=[],
            properties=Labels("soap_pca", torch.tensor(self.proj_dims, dtype=torch.int).unsqueeze(-1)),
        )
        cv_per_atom = TensorMap(
            keys=Labels("_", torch.tensor([[0]])),
            blocks=[block_per_atom],
        )

        block = TensorBlock(
            values=projected_mean,
            samples=samples,
            components=[],
            properties=Labels("soap_pca", torch.tensor(self.proj_dims, dtype=torch.int).unsqueeze(-1)),
        )
        cv = TensorMap(
            keys=Labels("_", torch.tensor([[0]])),
            blocks=[block],
        )
        return {"features": cv, "features/per_atom": cv_per_atom}#, "soaps": soap }

    def set_samples(self, selected_atoms):
        self.selected_samples = Labels(
            names=["atom"],
            values=torch.tensor(selected_atoms, dtype=torch.int32).unsqueeze(-1),
        )

    def set_atom_types(self, trj):
        types=[i.number for j in trj for w in j for i in w]
        self.atomic_types = sorted(set(types), key=types.index) #[torch.tensor([i for i in centers]+[j for j in neighbors if j not in centers ], dtype=torch.int32)]

    def set_projection_dims(self, dims):
        self.proj_dims = dims

    def set_projection_mu(self, mu):
        self.mu = torch.tensor(mu, dtype=torch.float64)

    def update_hypers(self, hypers): #hypers has to be dict
        self.hypers.update({key: str(val) for key, val in hypers.items()})

    def set_projection_matrix(self,matrix):
        self.projection_matrix=torch.tensor(matrix.copy())

    def save_model(self, path='.', name='soap_model'):
        capabilities = ModelCapabilities(
            outputs={"features": ModelOutput(per_atom=False),
                "features/per_atom": ModelOutput(per_atom=True),
            },
            interaction_range=10.0,
            supported_devices=["cpu"],
            length_unit="A",
            atomic_types=self.atomic_types,
            dtype="float64",
        )
        
        metadata = ModelMetadata(name="SOAP based CV", authors=['SmoothSOAP'], description='Hyperparameters in extra', extra=self.hypers)
        model = AtomisticModel(self, metadata, capabilities)
        model.save("{}/{}.pt".format(path,name), collect_extensions=f"{path}/extensions")
        print(f'model saved at {path}/{name}.pt')


    def compute_cumulants(self, X, n_cumulants):
        """
        Compute cumulants for each feature and concatenate them horizontally.
        
        Parameters
        ----------
        X : np.ndarray, shape (N, P)
            Data matrix with N samples and P features.
        n_cumulants : int
            Number of cumulants to compute per feature.
        
        Returns
        -------
        X_cumulant : np.ndarray, shape (N, P * n_cumulants)
            New feature matrix where cumulants of each original feature 
            are concatenated along the feature axis.
        """
        X = np.asarray(X)
        N, P = X.shape
        
        cumulant_matrix = []
        for j in range(P):
            x = X[:, j]
            m = np.mean(x)
            centered = x - m

            # Compute central moments up to n_cumulants
            mu = np.array([moment(centered, moment=i) for i in range(1, n_cumulants + 1)])
            c = np.zeros(n_cumulants)
            
            # First cumulants (mean, variance, skewness, kurtosis, ...)
            c[0] = m
            if n_cumulants > 1:
                c[1] = mu[1]                 # variance
            if n_cumulants > 2:
                c[2] = mu[2]                 # 3rd central moment
            if n_cumulants > 3:
                c[3] = mu[3] - 3 * mu[1]**2  # 4th cumulant (kurtosis-related)
            # higher orders could follow recursion, but are rarely stable
            if n_cumulants > 4:
                c[4] = mu[4] - 10 * mu[1] * mu[2]
            # Broadcast cumulant values to N samples
            cumulant_matrix.append(np.tile(c, (N, 1)))
  
        # Concatenate all cumulant blocks for each feature
        X_cumulant = np.hstack(cumulant_matrix)
        return X_cumulant


class SOAP_CV(torch.nn.Module):

    cutoff: float
    max_angular: int
    max_radial: int
    centers: List[int]
    neighbors: List[int]

    def __init__(self, cutoff, max_angular, max_radial, centers, neighbors, projection_matrix=None):
        super().__init__()

        # TorchScript-survibable init params
        self.cutoff = cutoff
        self.max_angular = max_angular
        self.max_radial = max_radial
        self.centers = centers
        self.neighbors = neighbors

        self._init_params = {
            'cutoff': cutoff,
            'max_angular': max_angular,
            'max_radial': max_radial,
            'centers': centers,
            'neighbors': neighbors,
        }

        HYPER_PARAMETERS = {
            "cutoff": {
                "radius": cutoff,
                "smoothing": {"type": "ShiftedCosine", "width": 0.5},
            },
            "density": {
                "type": "Gaussian",
                "width": 0.25,
            },
            "basis": {
                "type": "TensorProduct",
                "max_angular": max_angular,
                "radial": {"type": "Gto", "max_radial": max_radial},
            },
        }
        self.calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)
        self.selected_keys = Labels(
            names=["center_type", "neighbor_1_type", "neighbor_2_type"],
            values=torch.tensor(
                [[i, j, k] for i in centers for j in neighbors for k in neighbors if j <= k],
                dtype=torch.int32
            ),
        )
        self.id = f"SOAP_{cutoff}{max_angular}{max_radial}_{centers}"
        self.hypers = {}

        # register all buffers upfront — None until set via setters
        self.register_buffer("projection_matrix", None)
        self.register_buffer("mu", None)
        self.register_buffer("proj_dims", None)
        self.register_buffer("atomic_types", None)

        if projection_matrix is not None:
            self.register_buffer("projection_matrix", torch.tensor(projection_matrix.copy()).T)


    def calculate(self, systems, selected_samples=None):
        if selected_samples is None:
            selected_samples = self.selected_samples

        soap = self.calculator(
            systems,
            selected_samples=selected_samples,
            selected_keys=self.selected_keys,
        )
        soap = soap.keys_to_samples("center_type")
        soap = soap.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])
        self.soap_block = soap.block()
        return self.soap_block.values

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        if "features" not in outputs:
            return {}

        if outputs["features"].per_atom:
            raise ValueError("per_atom=True is not supported directly, output will be in features/per_atom")

        if len(systems[0]) == 0:
            # PLUMED is trying to determine the size of the output
            projected = torch.zeros((0, len(self.proj_dims)), dtype=torch.float64)
            projected_mean = torch.zeros((0, len(self.proj_dims)), dtype=torch.float64)
            samples = Labels(["system"], torch.zeros((0, 1), dtype=torch.int32))
            samples_per_atom = Labels(["system", "atom"], torch.zeros((0, 2), dtype=torch.int32))
        else:
            soap = self.calculator(systems, selected_samples=selected_atoms, selected_keys=self.selected_keys)
            soap = soap.keys_to_samples("center_type")
            soap = soap.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])

            soap_block = soap.block()

            projected = torch.einsum(
                'ij,jk->ik',
                (soap_block.values - self.mu),
                self.projection_matrix[:, self.proj_dims]
            )

            samples_per_atom = soap_block.samples.remove("center_type")
            samples = Labels(["system"], torch.zeros((1, 1), dtype=torch.int32))

            projected_mean = torch.mean(projected, dim=0).unsqueeze(0)

        block_per_atom = TensorBlock(
            values=projected,
            samples=samples_per_atom,
            components=[],
            properties=Labels("soap_pca", self.proj_dims.to(torch.int32).unsqueeze(-1)),
        )
        cv_per_atom = TensorMap(
            keys=Labels("_", torch.tensor([[0]])),
            blocks=[block_per_atom],
        )

        block = TensorBlock(
            values=projected_mean,
            samples=samples,
            components=[],
            properties=Labels("soap_pca", self.proj_dims.to(torch.int32).unsqueeze(-1)),
        )
        cv = TensorMap(
            keys=Labels("_", torch.tensor([[0]])),
            blocks=[block],
        )
        return {"features": cv, "features/per_atom": cv_per_atom}

    def set_samples(self, selected_atoms):
        self.selected_samples = Labels(
            names=["atom"],
            values=torch.tensor(selected_atoms, dtype=torch.int32).unsqueeze(-1),
        )

    def set_atom_types(self, trj):
        types = [i.number for j in trj for w in j for i in w]
        atomic_types = sorted(set(types), key=types.index)
        self.register_buffer("atomic_types", torch.tensor(atomic_types, dtype=torch.int32))

    def set_projection_dims(self, dims):
        self.register_buffer("proj_dims", torch.tensor(dims, dtype=torch.int64))

    def set_projection_mu(self, mu):
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float64))

    def set_projection_matrix(self, matrix):
        self.register_buffer("projection_matrix", torch.tensor(matrix.copy()))

    def update_hypers(self, hypers):
        self.hypers.update({key: str(val) for key, val in hypers.items()})

    def save_checkpoint(self, path='.', name='soap_model'):
        os.makedirs(path, exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'init_params': self._init_params,
            'hypers': self.hypers,
        }, f"{path}/{name}.ckpt")
        print(f'Checkpoint saved at {path}/{name}.ckpt')

    @classmethod
    def load_checkpoint(cls, path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model = cls(**ckpt['init_params'])
        model.load_state_dict(ckpt['state_dict'])
        model.hypers = ckpt['hypers']
        return model

    def save_model(self, path='.', name='soap_model'):
        capabilities = ModelCapabilities(
            outputs={
                "features": ModelOutput(per_atom=False),
                "features/per_atom": ModelOutput(per_atom=True),
            },
            interaction_range=10.0,
            supported_devices=["cpu"],
            length_unit="A",
            atomic_types=self.atomic_types.tolist(),
            dtype="float64",
        )
        metadata = ModelMetadata(
            name="SOAP based CV",
            authors=['SmoothSOAP'],
            description='Hyperparameters in extra',
            extra=self.hypers
        )
        model = AtomisticModel(self, metadata, capabilities)
        model.save(f"{path}/{name}.pt", collect_extensions=f"{path}/extensions")
        print(f'Model saved at {path}/{name}.pt')