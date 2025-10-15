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
from src.methods.BaseMethod import FullMethodBase


class LDA(FullMethodBase):

    def __init__(self, descriptor, interval, root):
        super().__init__(descriptor, interval, lag=0, root=root)


    def compute_COV(self, traj, labels):
        """
        Compute within-class and between-class covariance matrices.

        Parameters
        ----------
        traj : list[ase.Atoms]
        labels : list[int] or np.ndarray

        Returns
        -------
        mean_per_type : np.ndarray
            Mean descriptor vector per atomic species.
        Sw : np.ndarray
            Within-class covariance matrices.
        Sb : np.ndarray
            Between-class covariance matrices.
        """
        systems = systems_to_torch(traj)
        descriptors = np.stack(
            [self.descriptor.calculate([sys]).values.numpy().mean(axis=0)
             for sys in systems]
        )  # shape: (n_samples, descriptor_dim)

        labels = np.array(labels)
        unique_labels = np.unique(labels)

        overall_mean = descriptors.mean(axis=0)

        Sw = np.zeros((1, descriptors.shape[1], descriptors.shape[1]))
        Sb = np.zeros((1, descriptors.shape[1], descriptors.shape[1]))

        for lbl in unique_labels:
            class_desc = descriptors[labels == lbl]
            class_mean = class_desc.mean(axis=0)
            n_c = len(class_desc)

            # Within-class scatter
            diff = class_desc - class_mean
            Sw[0] += diff.T @ diff

            # Between-class scatter
            mean_diff = (class_mean - overall_mean).reshape(-1, 1)
            Sb[0] += n_c * (mean_diff @ mean_diff.T)

        # Normalize (optional, improves numerical stability)
        Sw[0] /= len(descriptors)
        Sb[0] /= len(descriptors)

        mean_per_type = np.array([overall_mean])
        return mean_per_type, Sw, Sb

    def log_metrics(self):
        """
        Log metrics from the run, including the covariances.


        Returns
        -------
        empty
        """
        metrics = np.array([[np.trace(tot_cov)] for tot_cov in self.cov_tot])
        header = ["spatialCov", "tempCov"]

        # Make metrics a 2D row vector: shape (1, 2)
        
        np.savetxt(
            self.label + "_.csv",
            metrics,
            fmt="%.6f",
            delimiter="\t",
            header="\t".join(header),
            comments=""
        )