import os 
import numpy as np
from vesin import NeighborList
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
from pathlib import Path

from src.transformations.PCAtransform import PCA_obj


class FullMethodBase(ABC):
    """
    Base class for full descriptor-based slow mode methods.

    Defines a unified interface for training and projecting models
    based on descriptor covariance matrices. Subclasses must implement
    the descriptor-specific covariance computation in `compute_COV()`.
    """

    def __init__(self, descriptor, interval, lag, sigma, root, method=None):
        self.interval = interval
        self.lag = lag
        self.sigma = sigma
        self.root = os.path.join(root, method)
        self.descriptor = descriptor
        self.transformations = None
        label = os.path.join(
            self.root,
            f'interval_{self.interval}',
            f'lag_{self.lag}',
            f'sigma_{self.sigma}',
        )
        Path(label).mkdir(parents=True, exist_ok=True)
        self.label = os.path.join(label, self.descriptor.id)
    # ------------------------------------------------------------------
    # Shared methods
    # ------------------------------------------------------------------
    def train(self, trajs, selected_atoms):
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

        traj_means = []
        traj_cov1 = []
        traj_cov2 = []
        traj_N = []
        for traj in trajs:
            mean, cov1, cov2 = self.compute_COV(traj)
            traj_means.append(mean)
            traj_cov1.append(cov1)
            traj_cov2.append(cov2)
            traj_N.append(len(traj))
        
        #combine trajectories:
        total_N = np.sum(traj_N)
        self.mean = np.mean(traj_means, axis=0)
        
        # Compute within-class covariance (average of per-trajectory covariances)
        class_cov1 = np.mean(traj_cov1, axis=0)
        class_cov2 = np.mean(traj_cov2, axis=0)

        # Compute between-class covariance (mean shifts)
        between_cov1 = sum(
            traj_N[i] * np.einsum('ci,cj->cij',class_mean - self.mean, class_mean - self.mean)
            for i, class_mean in enumerate(traj_means)
        ) / total_N

        between_cov2 = sum(
            traj_N[i] * np.einsum('ci,cj->cij',class_mean - self.mean, class_mean - self.mean)
            for i, class_mean in enumerate(traj_means)
        ) / total_N

        # Combine both parts
        self.cov1 = class_cov1 + between_cov1
        self.cov2 = class_cov2 + between_cov2

        # Example: use PCA-based transformation
        self.transformations = [PCA_obj(n_components=4, label=self.label) for n in range(cov1.shape[0])]

        for i, trafo in enumerate(self.transformations):
            trafo.solve_GEV(self.mean[i], self.cov1[i], self.cov2[i])


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


    def spatial_averaging(self, system, features, sigma):

        # get neighborlist (but not full, only for selected atoms)
        #TODO: dont know what happens if centers and selectedatoms dont align here, then it 
        # selects the wrong atom position here. Check SOAP samples & centers and order of SOAP feature
        # and compare to order of positions
        points = system.positions[self.selected_atoms]
        calculator = NeighborList(cutoff=5.0, full_list=True)
        i, j, S, d = calculator.compute(
            points=points,
            box=system.cell,
            periodic=True,
            quantities="ijSd"
        )
        i = i.astype(np.int64)
        j = j.astype(np.int64)
        d = d.astype(np.float64)
        # get distance 
        w = np.exp(-d**2 / (2*sigma**2), dtype=np.float64)     # pairwise weights
        # add self-weight term separately later
        self_weight = 1.0
        h = np.zeros_like(features)             # (N_atoms, n_features)
        np.add.at(h, i, w[:, None] * features[j])

        h += self_weight * features

        # get calculate the gaussian weight

        # normalize vs all 

        # return the averaged features 
        return h, cov
        


    def full_spatial_averaging(self, system, features, sigma):

        # get neighborlist (but not full, only for selected atoms)
        #TODO: dont know what happens if centers and selectedatoms dont align here, then it 
        # selects the wrong atom position here. Check SOAP samples & centers and order of SOAP feature
        # and compare to order of positions
        points = system.positions[self.selected_atoms]
        calculator = NeighborList(cutoff=5.0, full_list=True)
        i, j, S, d = calculator.compute(
            points=points,
            box=system.cell,
            periodic=True,
            quantities="ijSd"
        )
        i = i.astype(np.int64)
        j = j.astype(np.int64)
        d = d.astype(np.float64)
        # get distance 
        w = np.exp(-d**2 / (2*sigma**2), dtype=np.float64)     # pairwise weights
        #
        N, D = features.shape
        n_pairs = len(w)

        # Per-cluster sums
        sum_w  = np.zeros(N)
        sum_x  = np.zeros((N, D))
        sum_xx = np.zeros((N, D, D))

        # Weighted sums
        np.add.at(sum_w, i, w) #sums of the weights
        np.add.at(sum_x, i, w[:, None] * features[j])
        
        for k in range(n_pairs):
            sum_xx[i[k]] += w[k] * np.outer(features[j[k]], features[j[k]])
        
        # Add self-term
        self_weight = 1.0
        sum_w += self_weight
        sum_x += self_weight * features
        sum_xx += self_weight * np.einsum('ni,nj->nij', features, features)
       
        # Compute per-cluster means
        means = sum_x / sum_w[:, None]

        # Compute per-cluster covariances
        covs = sum_xx / sum_w[:, None, None] - np.einsum('ni,nj->nij', means, means)

        #COMPUTE COVARIANCES
        Wtot = sum_w.sum()
        mu_global = (sum_w[:, None] * means).sum(axis=0) / Wtot

        # within-cluster weighted covariance
        S_within = (sum_w[:, None, None] * covs).sum(axis=0) / Wtot

        # between-cluster covariance
        diff = means - mu_global
        S_between = (sum_w[:, None, None] * np.einsum('ni,nj->nij', diff, diff)).sum(axis=0) / Wtot

        return S_within, S_between
        



    def grid_spatial_averaging(self, system, features, sigma):

        # get neighborlist (but not full, only for selected atoms)
        #TODO: dont know what happens if centers and selectedatoms dont align here, then it 
        # selects the wrong atom position here. Check SOAP samples & centers and order of SOAP feature
        # and compare to order of positions
        points = system.positions[self.selected_atoms]
        calculator = NeighborList(cutoff=5.0, full_list=True)
        i, j, S, d = calculator.compute(
            points=points,
            box=system.cell,
            periodic=True,
            quantities="ijSd"
        )
        i = i.astype(np.int64)
        j = j.astype(np.int64)
        d = d.astype(np.float64)
        # get distance 
        w = np.exp(-d**2 / (2*sigma**2), dtype=np.float64)     # pairwise weights
        # add self-weight term separately later
        self_weight = 1.0
        h = np.zeros_like(features)             # (N_atoms, n_features)
        np.add.at(h, i, w[:, None] * features[j])

        h += self_weight * features
        
        # get calculate the gaussian weight

        # normalize vs all 

        # return the averaged features 
        return h
        
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
        mean_mu_t, cov_mu_t, mean_cov_t : np.ndarray
        """
        pass

    # ------------------------------------------------------------------
    # Abstract — subclasses must implement this
    # ------------------------------------------------------------------
    @abstractmethod
    def log_metrics(self):
        """
        Log metrics from the run, including the covariances.

        
        Returns
        -------
        empty
        """
        pass

