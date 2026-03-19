import os

import torch 
from abc import ABC, abstractmethod
import metatensor.torch as mts
from metatomic.torch import System, ModelEvaluationOptions, ModelOutput, systems_to_torch, load_atomistic_model
from metatensor.torch import Labels, TensorBlock, mean_over_samples
from featomic.torch import SoapPowerSpectrum
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from itertools import chain
from vesin import NeighborList
from memory_profiler import profile
from sklearn.decomposition import PCA as skPCA
from sklearn.linear_model import Ridge 

from smoothsoap.transformations.PCAtransform import PCA_obj
from smoothsoap.methods.BaseMethod import FullMethodBase



class DistinctPCA(FullMethodBase):

    def __init__(self, descriptor, interval, ridge_alpha, root):
        self.name = 'PCA_distinct'
        super().__init__(descriptor, interval, lag=0, root=root, sigma=0, ridge_alpha=ridge_alpha, method=self.name)

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
        print('before compute cov')
        for traj in trajs:
            mean, cov1, cov2 = self.compute_COV(traj)
            traj_means.append(mean)
            traj_cov1.append(cov1)
            traj_cov2.append(cov2)
            traj_N.append(len(traj))
        
        #combine trajectories:
        total_N = np.sum(traj_N)
        self.mean = np.mean(traj_means, axis=0)
        #self.cov1 = np.mean(traj_cov1, axis=0)
        #self.cov2 = np.mean(traj_cov2, axis=0)
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

        # Example: use PCA-based transformation for each center
        #self.transformations = [PCA_obj(n_components=4, label=self.label) for n in range(self.cov1.shape[0])]
        #self.atomsel_atom
        self.transformations = {i:PCA_obj(n_components=4, label=self.label) for i in self.atomsel_atom}
        for i, trafo in enumerate(self.transformations.values()):
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
        systems = systems_to_torch(traj, dtype=torch.float32)

        projected_per_type = []

        #for trafo in self.transformations:
        
        projected = []
        for system in systems:
            descriptor = self.descriptor.calculate([system]) # shape (atoms, soap)
            descriptor_proj = []
            for idx, index in enumerate(selected_atoms):
                trafo = self.transformations[index]
                descriptor_proj.append(trafo.project(descriptor[[idx]]).squeeze(0)) #select the correct atom
            projected.append(np.stack(descriptor_proj, axis=0))
            # TODO:
            #self.ridge.fit(descriptor, descriptor_proj)
        projected_per_type.append(np.stack(projected, axis=0).transpose(1, 0, 2))

        return projected_per_type # only works for one center type for now  # shape: (#centers ,N_atoms, T, latent_dim)
  
   
    def predict_ridge(self, traj, selected_atoms):
        self.selected_atoms = selected_atoms
        self.descriptor.set_samples(selected_atoms)
        systems = systems_to_torch(traj, dtype=torch.float32)
       
        projected_per_type = []

        projected = []
        for system in systems:
            descriptor = self.descriptor.calculate([system])
            pred = []
            for idx, index in enumerate(self.selected_atoms):
                ridge_pred = self.ridge[index].predict(descriptor[idx][None, :])
                pred.append(ridge_pred.squeeze(0))
            projected.append(np.stack(pred, axis=0))
        projected_per_type.append(np.stack(projected, axis=0).transpose(1, 0, 2))

        return projected_per_type
  

    def fit_ridge_nonincremental(self, traj):
        systems = systems_to_torch(traj, dtype=torch.float32)
        soap_block = self.descriptor.calculate(systems[:1], selected_samples=self.descriptor.selected_samples)
        print(soap_block.shape)
        first_soap = soap_block  
        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2)) # cutoff at 3 sigma, leaves 0.1%
        kernel /= kernel.sum() #kernel = delta
        self.ridge = {}

        
        trafo = self.transformations[self.selected_atoms[0]]
        # maybe eed to add all trafos 
        avg_soap_proj = trafo.project(first_soap)
        print('avg_soap_proj',avg_soap_proj.shape)


        soap_values=np.zeros((first_soap.shape[0],len(systems)-self.interval, first_soap.shape[1]))
        avg_soaps_projs=np.zeros((first_soap.shape[0],len(systems)-self.interval, avg_soap_proj.shape[-1]))
        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Fit Ridge"):
            new_soap_values = self.descriptor.calculate([system], selected_samples=self.descriptor.selected_samples)
            if fidx >= self.interval:
                roll_kernel = np.roll(kernel, fidx%self.interval)
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                avg_soap = np.einsum("t,ita->ia", roll_kernel, buffer) #smoothen
                avg_soap_proj_single = []
                for idx, index in enumerate(self.selected_atoms):
                    trafo = self.transformations[index]
                    avg_soap_proj_single.append(trafo.project(avg_soap[[idx]]).squeeze(0)) #select only the selected atom
                avg_soap_proj = np.stack(avg_soap_proj_single, axis=0) # shape (atoms, SOAP)
                #print('projshape', avg_soap_proj.shape)
                #print('nonprog.shape',new_soap_values.shape)
                soap_values[:,fidx-self.interval,:] = new_soap_values
                avg_soaps_projs[:,fidx-self.interval,:]=  avg_soap_proj
            buffer[:,fidx%self.interval,:] = new_soap_values
        print('soapvals',soap_values.dtype)
        print('soapvals',soap_values.shape)
        print('avg_soaps_proj',avg_soaps_projs.shape)
        #soap_values=soap_values.reshape((soap_values.shape[0]*soap_values.shape[1],soap_values.shape[2]))
        #avg_soaps_projs=avg_soaps_projs.reshape((avg_soaps_projs[0]*avg_soaps_projs[1],avg_soaps_projs.shape[2]))
        #soap_values=soap_values.reshape(soap_values.shape[0]*soap_values.shape[1],soap_values.shape[2])
        #avg_soaps_projs=avg_soaps_projs.reshape(avg_soaps_projs.shape[0]*avg_soaps_projs.shape[1],avg_soaps_projs.shape[2])
        #np.reshape(avg_soaps_projs.soap_values,(avg_soaps_projs.shape[0]*avg_soaps_projs.shape[1],avg_soaps_projs.shape[2]))
        #p.reshape(avg_soaps_projs,(avg_soaps_projs[0]*avg_soaps_projs[1],avg_soaps_projs.shape[2]))
        #print('soapvals',soap_values.shape)
        #print('avg_soaps_proj',avg_soaps_projs.shape)
        for idx, index in enumerate(self.selected_atoms):
            self.ridge[index] = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
            self.ridge[index].fit(soap_values[idx], avg_soaps_projs[idx])
   

    def compute_COV(self, traj):
        """
        Compute time-averaged SOAP covariance matrices for each atomic species.

        This method computes the temporal and ensemble covariance of SOAP 
        descriptors for different atomic species over a molecular dynamics 
        trajectory. It uses a Gaussian kernel to smooth SOAP vectors in time 
        and separates intra-atomic (within-atom) and inter-atomic (between-atoms)
        covariance contributions.

        Parameters
        ----------
        traj : ase.io.Trajectory or list of ase.Atoms
            Molecular dynamics trajectory containing atomic configurations 
            for which the SOAP descriptors are computed.

        Returns
        -------
        mean_mu_t : np.ndarray, shape (n_species, n_features)
            Time-averaged mean SOAP vector for each atomic species.
        mean_cov_t : np.ndarray, shape (n_species, n_features, n_features)
            Mean covariance of SOAP descriptors across all timesteps and atoms 
            of a given species.
        cov_mu_t : np.ndarray, shape (n_species, n_features, n_features)
            Temporal covariance of SOAP descriptor means (fluctuations in time).
        """
        systems = systems_to_torch(traj, dtype=torch.float32)
        soap_block = self.descriptor.calculate(systems[:1])
        first_soap = soap_block  
        # TODO: only implemented for one center type, could extend if necessary
        self.atomsel_atom = [label[1] for idx, label in enumerate(self.descriptor.soap_block.samples.values.numpy()) if label[2] == self.descriptor.centers[0]] # label[1] is the middle entry (index)
        self.atomsel_element = [[idx for idx, label in enumerate(self.descriptor.soap_block.samples.values.numpy()) if label[2] == atom_type] for atom_type in self.descriptor.centers]
        
        #if soap_block.shape[0] == 1:
        #    self.atomsel_atom = [[0] for atom_type in self.descriptor.centers]
        
        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        cov_t = np.zeros((len(self.atomsel_atom), first_soap.shape[1], first_soap.shape[1],))
        sum_soaps = np.zeros((len(self.atomsel_atom), first_soap.shape[1],))
        nsmp = np.zeros(len(self.atomsel_atom))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2)) # cutoff at 3 sigma, leaves 0.1%
        kernel /= kernel.sum()
        ntimesteps = np.zeros(len(self.atomsel_atom), dtype=int)

        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate([system])
            if fidx >= self.interval:
                roll_kernel = np.roll(kernel, fidx%self.interval)
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                avg_soap = np.einsum("t,ita->ia", roll_kernel, buffer) #smoothen
                
                sum_soaps += avg_soap
                cov_t += np.einsum("ia,ib->iab", avg_soap, avg_soap) #sum over all same atoms (have already summed over all times before) 
                nsmp += 1 #1np.ones(len(self.atomsel_atom))
                ntimesteps += 1 #np.ones(len(self.atomsel_atom))

            buffer[:,fidx%self.interval,:] = new_soap_values

        #mu = np.zeros((len(self.atomsel_atom), new_soap_values.shape[0], new_soap_values.shape[1]))
        #cov = np.zeros((len(self.atomsel_atom), new_soap_values.shape[0], new_soap_values.shape[1], new_soap_values.shape[1]))
        
        # autocorrelation matrix - remove mean
       
        mu = sum_soaps/nsmp[:,None]
        # COV = 1/N ExxT - mumuT
        cov = cov_t/nsmp[:,None,None] - np.einsum('ni,nj->nij', mu, mu)
        self.cov_tot = cov
        return mu, cov, [np.eye(c.shape[0]) for c in cov] #hard coded only for first center!
    


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
        
        for index in self.selected_atoms: 
            trafo = self.transformations[index]
            torch.save(
                torch.tensor(trafo.eigvals.copy()),
                self.label + f"_atom{index}" + f"_eigvals.pt",
            )

            torch.save(
                torch.tensor(trafo.eigvecs.copy()),
                self.label + f"_atom{index}" + f"_eigvecs.pt",
            )
