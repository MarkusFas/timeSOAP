
#from sklearn.linear_model import Ridge
#from sklearn.linear_model import LinearRegression
import torch 
import numpy as np
from typing import Dict, List, Optional
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
        return self.soap_block.values.numpy()

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
        
        metadata = ModelMetadata(name="Projection to ICA", authors=['SmoothSOAP'], description='Hyperparameters in extra', extra=self.hypers)
        #print(metadata)
        model = AtomisticModel(self, metadata, capabilities)
        print(f'model saved at {path}/{name}.pt')
        model.save("{}/{}.pt".format(path,name), collect_extensions=f"{path}/extensions")


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


class CumulantSOAP_CV(torch.nn.Module):
    def __init__(self, cutoff, max_angular, max_radial, centers, neighbors, n_cumulants, projection_matrix=None):
        super().__init__()
        HYPER_PARAMETERS = {
            "cutoff": {
                "radius": cutoff, #4 #5 #6
                "smoothing": {"type": "ShiftedCosine", "width": 0.5},
            },
            "density": {
                "type": "Gaussian",
                "width": 1, #changed from 0.3 0.25
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
        self.n_cumulants = n_cumulants
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
        return self.soap_block.values.numpy()

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        if "features" not in outputs:
            return {}

        if outputs["features"].per_atom:
            raise ValueError("per_atom=True is not supported")

        if len(systems[0]) == 0:
            # PLUMED is trying to determine the size of the output
            projected = torch.zeros((0,1), dtype=torch.float64)
            samples = Labels(["system", "atom"], torch.zeros((0, 2), dtype=torch.int32))
            properties = Labels("soap_pca", torch.tensor(self.proj_dims, dtype=torch.int).unsqueeze(-1))
            
        else:
            soap = self.calculator(systems, selected_samples=selected_atoms, selected_keys=self.selected_keys)
            soap = soap.keys_to_samples("center_type")
            soap = soap.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])#self.neighbor_type_pairs)

            soap_block = soap.block()
            
            X = self.compute_cumulants_fwd(soap_block.values, self.n_cumulants) 
            projected = torch.einsum('ij,jk->ik',(X - self.mu), self.projection_matrix[:,self.proj_dims])#, dtype=torch.float64)

            #samples = soap_block.samples.remove("center_type")
            samples = Labels(["system", "atom"], torch.zeros((1, 2), dtype=torch.int32))
            properties = Labels("soap_pca", torch.tensor(self.proj_dims, dtype=torch.int).unsqueeze(-1))
            #properties = Labels("soap_pca", torch.tensor(self.proj_dims, dtype=torch.int).unsqueeze(-1))


        block = TensorBlock(
            values=projected,
            samples=samples,
            components=[],
            #properties=Labels("soap_pca", torch.tensor([[0]])),
            #properties=Labels("soap_pca", torch.tensor([[0], [1]])),
            properties=properties,
        )
        cv = TensorMap(
            keys=Labels("_", torch.tensor([[0]])),
            blocks=[block],
        )
        return {"features": cv}#, "soaps": soap }
    
    def set_samples(self, selected_atoms):
        self.selected_samples = Labels(
            names=["atom"],
            values=torch.tensor(selected_atoms, dtype=torch.int32).unsqueeze(-1),
        )

    def set_atom_types(self, trj):
        types=[i.number for j in trj for w in j for i in w]
        self.atomic_types= sorted(set(types), key=types.index) #[torch.tensor([i for i in centers]+[j for j in neighbors if j not in centers ], dtype=torch.int32)]

    def set_projection_dims(self, dims):
        self.proj_dims = torch.tensor(dims)

    def set_projection_mu(self, mu):
        self.mu = torch.tensor(mu, dtype=torch.float64)

    def set_projection_matrix(self,matrix):
        self.projection_matrix=torch.tensor(matrix.copy())

    def save_model(self, path='.', name='soap_model'):
        capabilities = ModelCapabilities(
            outputs={"features": ModelOutput(per_atom=False)},
            interaction_range=10.0,
            supported_devices=["cpu"],
            length_unit="A",
            atomic_types=self.atomic_types,
            dtype="float32",
        )
        
        metadata = ModelMetadata(name="Collective Variable test")
        self.eval()
        model = AtomisticModel(self, metadata, capabilities)
        model.save("{}/{}.pt".format(path,name), collect_extensions=f"{path}/extensions")


    def compute_cumulants_fwd(self, X: torch.Tensor, n_cumulants: int) -> torch.Tensor:
        """
        TorchScript-friendly computation of cumulants.

        X: (N, P) tensor
        n_cumulants: number of cumulants per feature
        returns: (N, P * n_cumulants) tensor
        """
        # ensure float
        X = X.float()
        N, P = X.shape  # Python ints

        # Preallocate output
        out = torch.empty((1, P * n_cumulants), dtype=X.dtype, device=X.device)

        # Temporary tensors reused per feature
        moments = torch.empty((n_cumulants,), dtype=X.dtype, device=X.device)
        c = torch.empty((n_cumulants,), dtype=X.dtype, device=X.device)

        jbase = 0
        for j in range(P):
            x = X[:, j]

            # mean
            m = torch.mean(x)
            centered = x - m

            # compute central moments μ_k = mean((x - m)^k) for k=1..n_cumulants
            k = 1
            while k <= n_cumulants:
                moments[k - 1] = torch.mean(centered ** k)
                k += 1

            # fill cumulant vector c
            # 1st cumulant = mean
            c[0] = m

            # 2nd cumulant = variance (μ2)
            if n_cumulants > 1:
                c[1] = moments[1 - 1]  # μ2

            # 3rd cumulant = μ3
            if n_cumulants > 2:
                c[2] = moments[2 - 1]  # μ3

            # 4th cumulant = μ4 − 3 μ2²
            if n_cumulants > 3:
                mu2 = moments[1]
                mu4 = moments[3 - 1]
                c[3] = mu4 - 3.0 * (mu2 * mu2)

            # 5th cumulant = μ5 − 10 μ2 μ3
            if n_cumulants > 4:
                mu2 = moments[1]
                mu3 = moments[2]
                mu5 = moments[5 - 1]
                c[4] = mu5 - 10.0 * mu2 * mu3

            # broadcast c to N rows without extra Python list
            # c_row: (1, n_cumulants) then DONT expanded to (N, n_cumulants)
            c_row = c.unsqueeze(0)  # .expand(N, n_cumulants)

            # write into output slice
            out[:, jbase:jbase + n_cumulants] = c_row

            jbase += n_cumulants

        return out


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
            # still tiling in order to get the right shape, for saving memory we can just set N=1
            # as we output per structure and anyway average over the values later..
            cumulant_matrix.append(np.tile(c, (N, 1)))
        
        # Concatenate all cumulant blocks for each feature
        X_cumulant = np.hstack(cumulant_matrix)
        return X_cumulant
