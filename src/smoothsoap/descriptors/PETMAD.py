import torch
import metatensor.torch as mts
from typing import Dict, List, Optional
from metatomic.torch import (
    System, 
    ModelEvaluationOptions, 
    ModelOutput, 
    systems_to_torch, 
    load_atomistic_model,
    ModelCapabilities,
    ModelMetadata,
    AtomisticModel,
)
from metatensor.torch import Labels, TensorBlock, TensorMap
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy.stats import moment
from vesin.metatomic import compute_requested_neighbors


class PETMAD_descriptor():
    """
    This function has been replaced with model_soap 
    """
    def __init__(self, cutoff, max_angular, max_radial, centers, neighbors, selected_atoms=None):
        self.centers = centers
        self.neighbors = neighbors
        self.id = f"PETMAD"
        self.model = load_atomistic_model('data/PET-MAD.pt')
        self.nl_options = self.model.requested_neighbor_lists()[0]
        self.selected_samples = None
        #TODO default to all atoms in the trajectory
        self.output = ModelOutput(
            quantity='features', # mtt::aux::energy_last_layer_features
            unit='',
            per_atom=True,
            explicit_gradients=[],
        )
        self.hypers={}

    def calculate(self, systems, selected_samples=None):
        systems = systems
        if selected_samples is None:
            selected_samples = self.selected_samples
        self.options = ModelEvaluationOptions(
            length_unit='angstrom',
            outputs={
                'mtt::aux::energy_last_layer_features': self.output, 
            }, # check features, check 'mtt::aux::energy_last_layer_features'
            selected_atoms=selected_samples,
        )
        #systems = systems_to_torch(structures, dtype=torch.float32)
        if len(systems[0].known_neighbor_lists())==0:
            compute_requested_neighbors(
                systems=systems,
                system_length_unit='angstrom',
                model=self.model,
                model_length_unit='angstrom',
            )

        out = self.model(systems,
            options=self.options,
            check_consistency=True,
        )
        self.soap_block = out['mtt::aux::energy_last_layer_features'].block()
        features = self.soap_block.values.numpy()
        return features

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
            projected = torch.zeros((0,len(self.proj_dims)), dtype=torch.float32)
            projected_mean = torch.zeros((0,len(self.proj_dims)), dtype=torch.float32)
            samples = Labels(["system"], torch.zeros((0, 1), dtype=torch.int32))
            samples_per_atom = Labels(["system", "atom"], torch.zeros((0,2), dtype=torch.int32))
        else:
            features = self.calculate(systems, selected_samples=selected_atoms, selected_keys=self.selected_keys)
    
            projected = torch.einsum('ij,jk->ik',(features - self.mu), self.projection_matrix[:,self.proj_dims])#, dtype=torch.float64)

            samples_per_atom = Labels(["system", "atom"], torch.stack([torch.zeros_like(self.selected_samples), self.selected_samples], dim=1))
                                      
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
            names=["system", "atom"],
            values=torch.tensor([[0, i] for i in selected_atoms], dtype=torch.int64),
        )

        self.options = ModelEvaluationOptions(
            length_unit='angstrom', 
            outputs={
                'mtt::aux::energy_last_layer_features': self.output, 
            }, # check features, check 'mtt::aux::energy_last_layer_features'
            selected_atoms=self.selected_samples,
        )

    def set_atom_types(self, trj):
        types=[i.number for j in trj for w in j for i in w]
        self.atomic_types= sorted(set(types), key=types.index) #[torch.tensor([i for i in centers]+[j for j in neighbors if j not in centers ], dtype=torch.int32)]

    def set_projection_dims(self, dims):
        self.proj_dims = torch.tensor(dims)

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
            supported_devices=["cpu", "cuda"],
            length_unit="A",
            atomic_types=self.atomic_types,
            dtype="float32",
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



"""
in case sel atoms still doesnt work
new_map = mts.split(
            features,
            axis="samples",
            selections=[
                atomsel,
            ],
        )
    #feat = mean_over_samples(new_map[0], sample_names=["atom"]) 
    
    return new_map[0].block().values.numpy()
"""