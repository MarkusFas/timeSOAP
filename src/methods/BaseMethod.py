import torch 
from abc import ABC, abstractmethod
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

from src.transformations.PCAtransform import PCA_obj


class FullMethodBase(ABC):
    """
    Base class for full descriptor-based slow mode methods.

    Defines a unified interface for training and projecting models
    based on descriptor covariance matrices. Subclasses must implement
    the descriptor-specific covariance computation in `compute_COV()`.
    """

    def __init__(self, descriptor, interval, lag, label):
        self.interval = interval
        self.lag = lag
        self.label = label
        self.descriptor = descriptor
        self.transformations = None

    # ------------------------------------------------------------------
    # Shared methods
    # ------------------------------------------------------------------
    def train(self, traj, selected_atoms):
        """
        Train the method using a molecular dynamics trajectory.

        Parameters
        ----------
        traj : list[ase.Atoms]
            The atomic configurations to compute the new representation for.
        selected_atoms : list[int]
            Indices of atoms to be included in the training.
        """
        self.selected_atoms = selected_atoms
        self.descriptor.set_samples(selected_atoms)
        mean, cov1, cov2 = self.compute_COV(traj)

        # Example: use PCA-based transformation
        self.transformations = [PCA_obj(n_components=4, label=self.label) for n in range(cov1.shape[0])]

        for i, trafo in enumerate(self.transformations):
            trafo.solve_GEV(mean, cov1[i], cov2[i])

    def predict(self, traj, selected_atoms):
        """
        Project new trajectory frames into the trained collective variable (CV) space.

        Parameters
        ----------
        traj : list[ase.Atoms]
            Trajectory to project.
        selected_atoms : list[int]
            Indices of atoms to project.

        Returns
        -------
        np.ndarray, shape (n_atoms, n_frames, n_components)
            Projected low-dimensional representation.
        """
        if self.transformations is None:
            raise RuntimeError("Call train() before predict().")

        self.selected_atoms = selected_atoms
        self.descriptor.set_samples(selected_atoms)
        systems = systems_to_torch(traj, dtype=torch.float64)
       
        projected_per_type = []
        for trafo in self.transformations:
            projected = []
            for system in systems:
                descriptor = self.descriptor.calculate([system]).values.numpy()
                projected.append(trafo.project(descriptor))

            projected_per_type.append(np.stack(projected, axis=0).transpose(1, 0, 2))

        return projected_per_type  # shape: (#centers ,N_atoms, T, latent_dim)

    # ------------------------------------------------------------------
    # Abstract — subclasses must implement this
    # ------------------------------------------------------------------
    @abstractmethod
    def compute_COV(self, traj):
        """
        Compute descriptor covariance matrices for the trajectory.

        Must be implemented by subclasses. Compute time-averaged SOAP covariance 
        matrices for each atomic species.

        This method computes the temporal and ensemble covariance of SOAP 
        descriptors for different atomic species over a molecular dynamics 
        trajectory. It uses a Gaussian kernel to smooth SOAP vectors in time 
        and separates intra-atomic (within-atom) and inter-atomic (between-atoms)
        covariance contributions. Should compute the covariance or time correlation
        used to solve the Generalized EV problem (so 2 Covariance like matrixes should 
        be returned).
        Also for proper prediction, the correct mean used for computing the covariance(s) 
        has to be carried over.
        
        Returns
        -------
        mean_mu_t, mean_cov_t, cov_mu_t : np.ndarray
        """
        pass