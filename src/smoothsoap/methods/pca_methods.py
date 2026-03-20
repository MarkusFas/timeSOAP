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
import ase.neighborlist
from itertools import chain
from vesin import NeighborList
from memory_profiler import profile
from sklearn.decomposition import PCA as skPCA
from sklearn.linear_model import Ridge 

from smoothsoap.transformations.PCAtransform import PCA_obj
from smoothsoap.methods.BaseMethod import FullMethodBase



class ScikitPCA(FullMethodBase):

    def __init__(self, descriptor, interval, ridge_alpha, root):
        self.name = 'ScikitPCA'
        super().__init__(descriptor, interval, lag=0, root=root, sigma=0, ridge_alpha=ridge_alpha, method=self.name)
        self.pca = None

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
        traj = list(chain(*trajs))
        self.pca = self.compute_COV(trajs[0])


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
        if self.pca is None:
            raise RuntimeError("Call train() before predict().")

        self.selected_atoms = selected_atoms
        self.descriptor.set_samples(selected_atoms)
        systems = systems_to_torch(traj, dtype=torch.float32)

        projected_per_type = []
        self.ridge = Ridge(alpha=1e-6)
        projected = []
        for system in systems:
            descriptor = self.descriptor.calculate([system])
            descriptor_proj = self.pca.transform(descriptor)
            projected.append(descriptor_proj)
            # TODO:
            self.ridge.fit(descriptor, descriptor_proj)
        projected_per_type.append(np.stack(projected, axis=0).transpose(1, 0, 2))

        return projected_per_type  # shape: (#centers ,N_atoms, T, latent_dim)
 
    def predict_ridge(self, traj, selected_atoms):
        self.selected_atoms = selected_atoms
        self.descriptor.set_samples(selected_atoms)
        systems = systems_to_torch(traj, dtype=torch.float32)
       
        projected_per_type = []

        
        projected = []
        for system in systems:
            descriptor = self.descriptor.calculate([system])
            ridge_pred = self.ridge.predict(descriptor)
            projected.append(ridge_pred)
            
        projected_per_type.append(np.stack(projected, axis=0).transpose(1, 0, 2))

        return projected_per_type
        

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
        self.atomsel_element = [[idx for idx, label in enumerate(self.descriptor.soap_block.samples.values.numpy()) if label[2] == atom_type] for atom_type in self.descriptor.centers]
        if soap_block.shape[0] == 1:
            self.atomsel_element = [[0] for atom_type in self.descriptor.centers]
        
        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        cov_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        sum_soaps = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        nsmp = np.zeros(len(self.atomsel_element))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2)) # cutoff at 3 sigma, leaves 0.1%
        kernel /= kernel.sum()
        ntimesteps = np.zeros(len(self.atomsel_element), dtype=int)
        avg_soap = np.zeros((len(traj)-self.interval, first_soap.shape[0], first_soap.shape[1]))
        i = 0
        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate([system])
            if fidx >= self.interval:
                
                roll_kernel = np.roll(kernel, fidx%self.interval)
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                avg_soap[i,:,:] = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                i += 1 
            buffer[:,fidx%self.interval,:] = new_soap_values

        pca = skPCA(n_components=4)
        pca.fit(avg_soap.reshape(-1, avg_soap.shape[2]))
        #free memory
        buffer = None
        new_soap_values = None
        return pca


    def log_metrics(self):
        """
        Log metrics from the run, including the covariances.

        
        Returns
        -------
        empty
        """
        pass



class PCA(FullMethodBase): 

    def __init__(self, descriptor, interval, ridge_alpha, root):
        self.name = 'PCA'
        super().__init__(descriptor, interval, lag=0, root=root, sigma=0, ridge_alpha=ridge_alpha, method=self.name)
        

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
        #soap_block = self.descriptor.calculate(systems[:1])
        soap_block = self.descriptor.calculate(traj[:1]) #TODO change back
        first_soap = soap_block  
        #self.atomsel_element = [[idx for idx, label in enumerate(self.descriptor.soap_block.samples.values.numpy()) if label[2] == atom_type] for atom_type in self.descriptor.centers]
        self.atomsel_element = [np.arange(0,len(self.selected_atoms))]
        if soap_block.shape[0] == 1:
            self.atomsel_element = [[0] for atom_type in self.descriptor.centers]
        
        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        cov_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        sum_soaps = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        nsmp = np.zeros(len(self.atomsel_element))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2)) # cutoff at 3 sigma, leaves 0.1%
        kernel /= kernel.sum()
        ntimesteps = np.zeros(len(self.atomsel_element), dtype=int)

        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate(traj[fidx:fidx+1]) #TODO
            if fidx >= self.interval:
                roll_kernel = np.roll(kernel, fidx%self.interval)
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                for atom_type_idx, atom_type in enumerate(self.atomsel_element):
                    sum_soaps[atom_type_idx] += avg_soap[atom_type].sum(axis=0)
                    cov_t[atom_type_idx] += np.einsum("ia,ib->ab", avg_soap[atom_type], avg_soap[atom_type]) #sum over all same atoms (have already summed over all times before) 
                    nsmp[atom_type_idx] += len(atom_type)
                    ntimesteps[atom_type_idx] += 1

            buffer[:,fidx%self.interval,:] = new_soap_values

        mu = np.zeros((len(self.atomsel_element), new_soap_values.shape[1]))
        cov = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        
        # autocorrelation matrix - remove mean
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            mu[atom_type_idx] = sum_soaps[atom_type_idx]/nsmp[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            cov[atom_type_idx] = cov_t[atom_type_idx]/nsmp[atom_type_idx] - np.einsum('i,j->ij', mu[atom_type_idx], mu[atom_type_idx])
        self.cov_tot = cov
        buffer = None
        new_soap_values = None
        return mu, cov, [np.eye(c.shape[0]) for c in cov]


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
        
        for trafo in self.transformations:
            first_soap = self.descriptor.calculate([systems[0]])
            first_soap = trafo.project(first_soap)
            projected = np.zeros((len(systems), first_soap[0], first_soap[1]))
            for i, system in enumerate(systems):
                descriptor = self.descriptor.calculate([system])
                descriptor = trafo.project(descriptor)
                projected[i] = descriptor
                # TODO:
                #self.ridge.fit(descriptor, descriptor_proj)
            projected_per_type.append(projected.transpose(1, 0, 2))

        return projected_per_type  # shape: (#centers ,N_atoms, T, latent_dim)
 

    def fit_ridge_nonincremental(self, traj):
        systems = systems_to_torch(traj, dtype=torch.float32)
        soap_block = self.descriptor.calculate(traj[:1], selected_samples=self.descriptor.selected_samples)
        print(soap_block.shape)
        first_soap = soap_block  
        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2)) # cutoff at 3 sigma, leaves 0.1%
        kernel /= kernel.sum() #kernel = delta
        self.ridge = {}


        for idx, trafo in enumerate(self.transformations):
            self.ridge[idx] = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
            avg_soap_proj = trafo.project(first_soap) 
            print('avg_soap_proj',avg_soap_proj.shape)
            soap_values=np.zeros((first_soap.shape[0],len(systems)-self.interval, first_soap.shape[1]))
            avg_soaps_projs=np.zeros((first_soap.shape[0],len(systems)-self.interval, avg_soap_proj.shape[-1]))
            for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Fit Ridge"):
                new_soap_values = self.descriptor.calculate([traj[fidx]], selected_samples=self.descriptor.selected_samples)
                if fidx >= self.interval:
                    roll_kernel = np.roll(kernel, fidx%self.interval)
                    # computes a contribution to the correlation function
                    # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                    avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                    avg_soap_proj = trafo.project(avg_soap)
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
            soap_values=soap_values.reshape(soap_values.shape[0]*soap_values.shape[1],soap_values.shape[2])
            avg_soaps_projs=avg_soaps_projs.reshape(avg_soaps_projs.shape[0]*avg_soaps_projs.shape[1],avg_soaps_projs.shape[2])
            #np.reshape(avg_soaps_projs.soap_values,(avg_soaps_projs.shape[0]*avg_soaps_projs.shape[1],avg_soaps_projs.shape[2]))
            #p.reshape(avg_soaps_projs,(avg_soaps_projs[0]*avg_soaps_projs[1],avg_soaps_projs.shape[2]))
            print('soapvals',soap_values.shape)
            print('avg_soaps_proj',avg_soaps_projs.shape)
            self.ridge[idx].fit(soap_values, avg_soaps_projs)

  
    def predict_ridge(self, traj, selected_atoms):
        self.selected_atoms = selected_atoms
        self.descriptor.set_samples(selected_atoms)
        systems = systems_to_torch(traj, dtype=torch.float32)
       
        projected_per_type = []

        for idx, trafo in enumerate(self.transformations):
            projected = []
            for fidx,system in enumerate(systems):
                descriptor = self.descriptor.calculate([traj[fidx]])
                ridge_pred = self.ridge[idx].predict(descriptor)
                projected.append(ridge_pred)
               
            projected_per_type.append(np.stack(projected, axis=0).transpose(1, 0, 2))

        return projected_per_type
        

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
        for i, trafo in enumerate(self.transformations):
            torch.save(
                torch.tensor(trafo.eigvals.copy()),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_eigvals.pt",
            )

            torch.save(
                torch.tensor(trafo.eigvecs.copy()),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_eigvecs.pt",
            ) 

class PCAtest(FullMethodBase):

    def __init__(self, descriptor, interval, ridge_alpha, root):
        self.name = 'PCAtest'
        super().__init__(descriptor, interval, lag=0, root=root, sigma=0, ridge_alpha=ridge_alpha, method=self.name)
        
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
        first_soap =  soap_block  
        self.atomsel_element = [[idx for idx, label in enumerate(self.descriptor.soap_block.samples.values.numpy()) if label[2] == atom_type] for atom_type in self.descriptor.centers]
        if soap_block.shape[0] == 1:
            self.atomsel_element = [[0] for atom_type in self.descriptor.centers]
        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        cov_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        sum_soaps = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        nsmp = np.zeros(len(self.atomsel_element))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2*3)) # cutoff at 3 sigma, leaves 0.1%
        kernel = delta
        ntimesteps = np.zeros(len(self.atomsel_element), dtype=int)

        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate([system])
            if fidx >= self.interval:
                roll_kernel = np.roll(kernel, fidx%self.interval)
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                for atom_type_idx, atom_type in enumerate(self.atomsel_element):
                    sum_soaps[atom_type_idx] += avg_soap[atom_type].sum(axis=0)
                    cov_t[atom_type_idx] += np.einsum("ia,ib->ab", avg_soap[atom_type], avg_soap[atom_type]) #sum over all same atoms (have already summed over all times before) 
                    nsmp[atom_type_idx] += len(atom_type)
            buffer[:,fidx%self.interval,:] = new_soap_values

        mu = np.zeros((len(self.atomsel_element), new_soap_values.shape[1]))
        cov = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        
        # autocorrelation matrix - remove mean
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            mu[atom_type_idx] = sum_soaps[atom_type_idx]/nsmp[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            cov[atom_type_idx] = cov_t[atom_type_idx]/nsmp[atom_type_idx] - np.einsum('i,j->ij', mu[atom_type_idx], mu[atom_type_idx])
        
        return mu, cov, [np.eye(c.shape[0]) for c in cov]


    def log_metrics(self):
        """
        Log metrics from the run, including the covariances.

        
        Returns
        -------
        empty
        """
        metrics = np.array([[np.trace(mean_cov), np.trace(cov_mu)] 
                    for mean_cov, cov_mu in zip(self.mean_cov_t, self.cov_mu_t)])
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


class PCAfull(FullMethodBase):

    def __init__(self, descriptor, interval, ridge_alpha, root):
        self.name = 'PCAfull'
        super().__init__(descriptor, interval, lag=0, root=root, sigma=0, ridge_alpha=ridge_alpha, method=self.name)


    def predict_avg(self, traj, selected_atoms):
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

        for trafo in self.transformations:
            descriptor = self.descriptor.calculate([systems[0]])
            buffer = np.zeros((descriptor.shape[0], self.interval, descriptor.shape[1]))
            transformed = trafo.project(descriptor)
            projected = np.zeros((len(systems) - self.interval, transformed.shape[0], transformed.shape[1]))
            delta=np.zeros(self.interval)
            delta[self.interval//2]=1
            kernel=gaussian_filter(delta,sigma=(self.interval-1))#gaussian_filter(delta,sigma=(self.interval-1)//(2*3)) # cutoff at 3 sigma, leaves 0.1%
            kernel /= np.sum(kernel)
            for i, system in enumerate(systems):
                buffer[:,i%self.interval,:] = descriptor
                if i >= self.interval:
                    roll_kernel = np.roll(kernel, i%self.interval)
                    avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer)
                    avg_soap = trafo.project(avg_soap)
                    projected[i-self.interval] = avg_soap
                descriptor = self.descriptor.calculate([system])
                
            projected_per_type.append(projected.transpose(1, 0, 2))
        self.get_explained_variance(traj, selected_atoms)
        return projected_per_type  # shape: (#centers ,N_atoms, T, latent_dim)

    def compute_(self, soap, scatter_mut, sum_mu_t, cov_t, nsmp, ntimesteps):  
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            mu_t = soap[atom_type].mean(axis=0)
            scatter_mut[atom_type_idx] += np.einsum(
                "a,b->ab",
                mu_t,
                mu_t,
            )

            sum_mu_t[atom_type_idx] += mu_t #sum over all same atoms

            cov_t[atom_type_idx] += np.einsum("ia,ib->ab", soap[atom_type] - mu_t, soap[atom_type] - mu_t)/len(atom_type) #sum over all same atoms (have already summed over all times before) 
            nsmp[atom_type_idx] += len(atom_type)
            ntimesteps[atom_type_idx] += 1

    def compute_COV(self, systems):
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

        soap_block = self.descriptor.calculate(systems[:1])
        first_soap = soap_block  
        if self.descriptor.id[:4] == "SOAP":
            self.atomsel_element = [[idx for idx, label in enumerate(self.descriptor.soap_block.samples.values) if label[2] == atom_type] for atom_type in self.descriptor.centers]
        else:
            self.atomsel_element = [[idx for idx in range(len(self.descriptor.selected_samples))] for atom_type in self.descriptor.centers]
        if soap_block.shape[0] == 1:
            self.atomsel_element = [[0] for atom_type in self.descriptor.centers]
        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        cov_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        sum_mu_t = np.zeros((len(self.atomsel_element), first_soap.shape[1],))
        scatter_mut = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        nsmp = np.zeros(len(self.atomsel_element))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1))#gaussian_filter(delta,sigma=(self.interval-1)//(2*3)) # cutoff at 3 sigma, leaves 0.1%
        kernel /= np.sum(kernel)
        ntimesteps = np.zeros(len(self.atomsel_element), dtype=int)

        #plt.plot(kernel)
        #plt.savefig(self.label + '_kernel.png')
        #plt.close()
        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate([system])
            if fidx >= self.interval:
                roll_kernel = np.roll(kernel, fidx%self.interval)
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                #fig, ax = plt.subplots(1,1, figsize=(5,4))
                #ax.plot(buffer[0,:,0])
                #ax2 = ax.twinx()
                #ax2.plot(roll_kernel)
                #ax.scatter([fidx%self.interval + self.interval//2],[avg_soap[0,0]], color='red', s=50)
                #plt.savefig(self.label + f'_buffer_{fidx}.png')
                #plt.close()
                self.compute_(avg_soap, scatter_mut, sum_mu_t, cov_t, nsmp, ntimesteps)
                
            buffer[:,fidx%self.interval,:] = new_soap_values
        #exit()
        if len(systems) == 1:
            self.compute_(first_soap, scatter_mut, sum_mu_t, cov_t, nsmp, ntimesteps)

        mean_cov_t = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        cov_mu_t = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        mean_mu_t = np.zeros((len(self.atomsel_element), first_soap.shape[1],))

        # autocorrelation matrix - remove mean
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            mean_cov_t[atom_type_idx] = cov_t[atom_type_idx]/ntimesteps[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            mean_mu_t[atom_type_idx] = sum_mu_t[atom_type_idx]/ntimesteps[atom_type_idx]
            # add temporal covariance
            cov_mu_t[atom_type_idx] = scatter_mut[atom_type_idx]/ntimesteps[atom_type_idx] - np.einsum('i,j->ij', mean_mu_t[atom_type_idx], mean_mu_t[atom_type_idx])

        #all_soap_values = eval_SOAP(systems, calculator, sel, atomsel).values.numpy()
        #C_np = np.cov(all_soap_values, rowvar=False, bias=True)   # population covariance
        #print(np.allclose(C_np, avgcc[0], atol=1e-8))

        self.mean_mu_t = mean_mu_t
        self.mean_cov_t = mean_cov_t
        self.cov_mu_t = cov_mu_t
        
        total_cov = mean_cov_t + cov_mu_t
        return mean_mu_t, total_cov, [np.eye(cov.shape[0]) for cov in total_cov]
        #return mean_mu_t, mean_cov_t, cov_mu_t

    def log_metrics(self):
        """
        Log metrics from the run, including the covariances.

        
        Returns
        -------
        empty
        """
        metrics = np.array([[np.trace(mean_cov), np.trace(cov_mu)] 
                    for mean_cov, cov_mu in zip(self.mean_cov_t, self.cov_mu_t)])
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

        for i, trafo in enumerate(self.transformations):
            
            np.savetxt(
                self.label + f"_center{self.descriptor.centers[i]}" + "_cov_mu_t.csv",
                self.cov_mu_t[0],
            )

            np.savetxt(
                self.label + f"_center{self.descriptor.centers[i]}" + f"_mean_cov_t.csv",
                self.mean_cov_t[0],
            )

            torch.save(
                torch.tensor(trafo.eigvals.copy()),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_eigvals.pt",
            )

            torch.save(
                torch.tensor(trafo.eigvecs.copy()),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_eigvecs.pt",
            )

            torch.save(
                self.descriptor.soap_block.properties.values,
                self.label + f"_center{self.descriptor.centers[i]}" + f"_properties.pt",
            )


class SpatialPCA(FullMethodBase):

    def __init__(self, descriptor, interval, sigma, cutoff, ridge_alpha, root):
        self.name = 'SpatialPCA'
        super().__init__(descriptor, interval, lag=0, root=root, sigma=sigma, ridge_alpha=ridge_alpha, method=self.name)
        self.descriptor_spatial = descriptor
        self.spatial_cutoff = cutoff
        self.label = self.label + f'cut_{self.spatial_cutoff}'
    

    def fit_ridge_nonincremental(self, systems):
        soap_block = self.descriptor.calculate(systems[:1], selected_samples=self.descriptor.selected_samples)
        first_soap = soap_block  
        first_soap = self.spatial_average_with_nl(systems[0], first_soap)
        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2)) # cutoff at 3 sigma, leaves 0.1%
        kernel /= kernel.sum() #kernel = delta
        self.ridge = {}
        if self.interval > 1:
            raise ValueError("Not  possible to time average SpatailPCA")
        for idx, trafo in enumerate(self.transformations):
            self.ridge[idx] = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
            avg_soap_proj = trafo.project(first_soap) 
            print('avg_soap_proj',avg_soap_proj.shape)
            soap_values=np.zeros((first_soap.shape[0],len(systems)-self.interval, first_soap.shape[1]))
            avg_soaps_projs=np.zeros((first_soap.shape[0],len(systems)-self.interval, avg_soap_proj.shape[-1]))
            
            for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Fit Ridge"):
                new_soap_values = self.descriptor.calculate([system], selected_samples=self.descriptor.selected_samples)
                soap_spatial_avg = self.spatial_average_with_nl(system, new_soap_values)

                avg_soap_proj = trafo.project(soap_spatial_avg)

                soap_values[:,fidx-self.interval,:] = new_soap_values
                avg_soaps_projs[:,fidx-self.interval,:] = avg_soap_proj
                
            #soap_values=soap_values.reshape((soap_values.shape[0]*soap_values.shape[1],soap_values.shape[2]))
            #avg_soaps_projs=avg_soaps_projs.reshape((avg_soaps_projs[0]*avg_soaps_projs[1],avg_soaps_projs.shape[2]))
            if len(systems) == 1:
                soap_values = first_soap[:,None,:]
                avg_soaps_projs = trafo.project(first_soap)[:,None,:]

            soap_values=soap_values.reshape(soap_values.shape[0]*soap_values.shape[1],soap_values.shape[2])
            avg_soaps_projs=avg_soaps_projs.reshape(avg_soaps_projs.shape[0]*avg_soaps_projs.shape[1],avg_soaps_projs.shape[2])
            #np.reshape(avg_soaps_projs.soap_values,(avg_soaps_projs.shape[0]*avg_soaps_projs.shape[1],avg_soaps_projs.shape[2]))
            #p.reshape(avg_soaps_projs,(avg_soaps_projs[0]*avg_soaps_projs[1],avg_soaps_projs.shape[2]))
            print('soapvals',soap_values.shape)
            print('avg_soaps_proj',avg_soaps_projs.shape)
            self.ridge[idx].fit(soap_values, avg_soaps_projs)


    def predict_avg(self, systems, selected_atoms):
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

        projected_per_type = []

        for trafo in self.transformations:
            first_soap = self.descriptor.calculate([systems[0]])
            first_soap = self.spatial_average_with_nl(systems[0], first_soap)
            first_soap = trafo.project(first_soap)
            projected = np.zeros((len(systems), first_soap.shape[0], first_soap.shape[1]))
            for i, system in enumerate(systems):
                descriptor = self.descriptor.calculate([system])
                descriptor = self.spatial_average_with_nl(system, descriptor)
                descriptor = trafo.project(descriptor)
                projected[i] = descriptor
            projected_per_type.append(projected.transpose(1, 0, 2))
        self.get_explained_variance(systems, selected_atoms)
        return projected_per_type  # shape: (#centers ,N_atoms, T, latent_dim)

 
    def compute_(self, soap, sum_soaps, cov_t, nsmp, ntimesteps):
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            sum_soaps[atom_type_idx] += soap[atom_type].sum(axis=0)
            cov_t[atom_type_idx] += np.einsum("ia,ib->ab", soap[atom_type], soap[atom_type]) #sum over all same atoms (have already summed over all times before) 
            nsmp[atom_type_idx] += len(atom_type)
            ntimesteps[atom_type_idx] += 1

    def compute_COV(self, systems):
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
        soap_block = self.descriptor.calculate(systems[:1])
        first_soap =  soap_block  
        self.atomsel_element = [[idx for idx, label in enumerate(self.descriptor.soap_block.samples.values.numpy()) if label[2] == atom_type] for atom_type in self.descriptor.centers]
        if soap_block.shape[0] == 1:
            self.atomsel_element = [[0] for atom_type in self.descriptor.centers]
        #initialize neighbor list for spatial avg
        self.make_neighborlist(self.spatial_cutoff)

        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        cov_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        sum_soaps = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        sum_mu_t = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        scatter_mut = np.zeros((len(self.atomsel_element),first_soap.shape[1], first_soap.shape[1],))
        nsmp = np.zeros(len(self.atomsel_element))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2)) # cutoff at 3 sigma, leaves 0.1%
        kernel /= np.sum(kernel)
        ntimesteps = np.zeros(len(self.atomsel_element), dtype=int)
        logger = np.zeros((len(self.atomsel_element), len(systems), first_soap.shape[1],))
 
        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate([system])
            avg_soap_values = self.spatial_average_with_nl(system, new_soap_values)
            if fidx >= self.interval:
                roll_kernel = np.roll(kernel, fidx%self.interval)
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                #avg_soap = self.spatial_averaging(system, avg_soap, self.sigma)
                self.compute_(avg_soap, sum_soaps, cov_t, nsmp, ntimesteps)

            buffer[:,fidx%self.interval,:] = avg_soap_values

        if len(systems) == 1:
            self.compute_(first_soap, sum_soaps, cov_t, nsmp, ntimesteps)

        mu = np.zeros((len(self.atomsel_element), avg_soap_values.shape[1]))
        cov = np.zeros((len(self.atomsel_element), avg_soap_values.shape[1], avg_soap_values.shape[1]))
        
        # autocorrelation matrix - remove mean
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            mu[atom_type_idx] = sum_soaps[atom_type_idx]/nsmp[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            cov[atom_type_idx] = cov_t[atom_type_idx]/nsmp[atom_type_idx] - np.einsum('i,j->ij', mu[atom_type_idx], mu[atom_type_idx])
        self.cov = cov
        return mu, cov, [np.eye(c.shape[0]) for c in cov]

    """            buffer[:,fidx%self.interval,:] = avg_soap_values
        np.savetxt(self.label + '_mu_t.csv', logger[0])
        mean_cov_t = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        mean_cov_intra_t = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        cov_mu_t = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        mean_mu_t = np.zeros((len(self.atomsel_element), first_soap.shape[1],))

        # autocorrelation matrix - remove mean
        for atom_type_idx, atom_type in enumerate(self.atomsel_element): 
            mean_cov_t[atom_type_idx] = cov_t[atom_type_idx]/ntimesteps[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            mean_mu_t[atom_type_idx] = sum_mu_t[atom_type_idx]/ntimesteps[atom_type_idx]
            # add temporal covariance
            cov_mu_t[atom_type_idx] = scatter_mut[atom_type_idx]/ntimesteps[atom_type_idx] - np.einsum('i,j->ij', mean_mu_t[atom_type_idx], mean_mu_t[atom_type_idx])

        #all_soap_values = eval_SOAP(systems, calculator, sel, atomsel).values.numpy()
        #C_np = np.cov(all_soap_values, rowvar=False, bias=True)   # population covariance
        #print(np.allclose(C_np, avgcc[0], atol=1e-8))

        self.mean_mu_t = mean_mu_t
        self.mean_cov_t = mean_cov_t
        self.cov_mu_t = cov_mu_t
        
        # temp vs inter
        return mean_mu_t, cov_mu_t + mean_cov_t, [np.eye(c.shape[0]) for c in cov_mu_t] #mean_cov_t  #+ cov_mu_t"""


    def make_neighborlist(self, cutoff):
        self.nlist = NeighborList(cutoff=cutoff, full_list=True)


    def spatial_average_with_nl(self, system, features):
        positions = system.positions  # (N,3) numpy array
        box = system.cell             # 3x3 array or None for non-periodic

        i, j, d, D = self.nlist.compute(
            points=positions,
            box=box,
            periodic=True,
            quantities="ijdD"
        )

        # filter pairs to only the centers you care about:
        mask = np.isin(i, self.selected_atoms)
        atom_types = system.types.numpy()
        i_sel, j_sel, d_sel, D_sel = i[mask], j[mask], d[mask], D[mask]
        averaged_features = np.zeros_like(features)
        self_weight = 1.0

        for idx, atom_idx in enumerate(self.selected_atoms):
            mask_i = i_sel == atom_idx 
            j_neighbors = j_sel[mask_i]
            d_neighbors = d_sel[mask_i]
            D_neighbors = D_sel[mask_i]

            # --- Deduplicate: keep only closest image per base atom
            if len(j_neighbors) > 0:
                unique_js = np.unique(j_neighbors)
                best_idx = np.array([
                    np.argmin(d_neighbors[j_neighbors == j]) 
                    for j in unique_js
                ])
                j_neighbors = unique_js
                d_neighbors = d_neighbors[best_idx]
                D_neighbors = D_neighbors[best_idx]

            mask_j = atom_types[j_neighbors] == system.types[atom_idx]
            if len(j_neighbors) == 0:
                averaged_features[idx] = features[idx]
                continue
            w = np.exp(-d_neighbors[mask_j]**2 / (2*self.sigma**2))
            indices = np.searchsorted(self.selected_atoms, j_neighbors[mask_j])

            #sel_samples = Labels(
            #    names=["atom"],
            #    values=torch.tensor(j_neighbors, dtype=torch.int64).unsqueeze(-1),
            #)
            #feats = self.descriptor_spatial.calculate([system], sel_samples)
            
            weighted_sum = (w[:, None] * features[indices]).sum(axis=0)
            h_i = (weighted_sum + self_weight * features[idx]) / (self_weight + w.sum())
            averaged_features[idx] = h_i

        return averaged_features

    def spatial_average_with_nl(self, system, features):
        positions = system.positions  # (N,3) numpy array
        box = system.cell             # 3x3 array or None for non-periodic

        i, j, d, D = self.nlist.compute(
            points=positions,
            box=box,
            periodic=True,
            quantities="ijdD"
        )
        sort_idx = np.lexsort((d, j, i))  # primary i, secondary j, tertiary d
        i_sorted = i[sort_idx]
        j_sorted = j[sort_idx]
        d_sorted = d[sort_idx]
        S_sorted = D[sort_idx]
        mask = np.ones(len(i_sorted), dtype=bool)
        mask[1:] = (i_sorted[1:] != i_sorted[:-1]) | (j_sorted[1:] != j_sorted[:-1])
        i = i_sorted[mask]
        j = j_sorted[mask]
        d = d_sorted[mask]
        S = S_sorted[mask]
        feats = np.zeros_like(features)
        self_weight = 1.0
        atom_types = system.types.numpy()
        #for idx, atom_idx in enumerate(np.unique(i)):
        for idx, atom_idx in enumerate(self.selected_atoms):
            mask_i = i == atom_idx 
            j_neighbors = j[mask_i]
            d_neighbors = d[mask_i]
            S_neighbors = D[mask_i]
            mask_j = atom_types[j_neighbors] == system.types[atom_idx]
            w = np.exp(-d_neighbors[mask_j]**2 / (2*self.sigma**2))
            indices = np.searchsorted(self.selected_atoms, j_neighbors[mask_j])
            #w = np.exp(-d_neighbors**2 / (2*self.sigma**2))
            #indices = np.searchsorted(self.selected_atoms, j_neighbors)
            #sel_samples = Labels(
            #    names=["atom"],
            #    values=torch.tensor(j_neighbors, dtype=torch.int64).unsqueeze(-1),
            #)
            #feats = self.descriptor_spatial.calculate([system], sel_samples)
            
            weighted_sum = (w[:, None] * features[indices]).sum(axis=0)
            h_i = (weighted_sum + self_weight * features[idx]) / (self_weight + w.sum())
        
            feats[idx] = h_i
        return feats

    """
        self.nlist.update(system.positions)
        averaged_features = {}
        self_weight = 1.0
        for i in self.selected_atoms:
            j, d = self.nlist.get_neighbors(i, return_distances=True)
            if len(j) == 0:
                averaged_features[i] = features[i]
                continue

            w = np.exp(-d**2 / (2*self.sigma**2))
            self.descriptor_spatial.set_samples(j)
            feats = self.descriptor_spatial.calculate([system])
            weighted_sum = (w[:, None] * feats).sum(axis=0)
            h_i = (weighted_sum + self_weight * features[i]) / (self_weight + w.sum())
            averaged_features[i] = h_i

        return averaged_features

    """


    def log_metrics(self):
        """
        Log metrics from the run, including the covariances.


        Returns
        -------
        empty
        """
        metrics = np.array([[np.trace(mean_cov)] 
                    for  mean_cov in self.cov])
        header = ["spataverageCov"]

        # Make metrics a 2D row vector: shape (1, 2)
        np.savetxt(
            self.label + "_.csv",
            metrics,
            fmt="%.6f",
            delimiter="\t",
            header="\t".join(header),
            comments=""
        )
        for i, trafo in enumerate(self.transformations):
            """np.savetxt(
                self.label + f"_center{self.descriptor.centers[i]}" + "_cov_mu_t.csv",
                self.cov_mu_t[0],
            )

            np.savetxt(
                self.label + f"_center{self.descriptor.centers[i]}" + f"_mean_cov_t.csv",
                self.mean_cov_t[0],
            )"""

            torch.save(
                torch.tensor(trafo.eigvals.copy()),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_eigvals.pt",
            )

            torch.save(
                torch.tensor(trafo.eigvecs.copy()),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_eigvecs.pt",
            )


    def soap_avg_aml(self, soap, frame, spat_cutoff, sigma):
        """
        :param soap: soap vectors
        :param frame: ase.Atoms object
        :param spat_cutoff: spatial IVAC cutoff 
        :param sigma: Gaussian width for averaging
        
        
        :returns: the averaged soap vectors
        """
        pos = frame.positions #frame_iron.positions[:core_idx,:]
        cell = frame.cell
        nlist = vesin.NeighborList(cutoff=spat_cutoff, full_list=True)
        i, j, S, d = nlist.compute(
            points=pos,
            box=cell, 
            periodic=True,
            quantities="ijSd"
        )
        sort_idx = np.lexsort((d, j, i))  # primary i, secondary j, tertiary d
        i_sorted = i[sort_idx]
        j_sorted = j[sort_idx]
        d_sorted = d[sort_idx]
        S_sorted = S[sort_idx]
        mask = np.ones(len(i_sorted), dtype=bool)
        mask[1:] = (i_sorted[1:] != i_sorted[:-1]) | (j_sorted[1:] != j_sorted[:-1])
        i = i_sorted[mask]
        j = j_sorted[mask]
        d = d_sorted[mask]
        S = S_sorted[mask]
        
        feats = np.zeros_like(soap)
        self_weight = 1.0

        for idx, atom_idx in enumerate(np.unique(i)):
            mask_i = i == atom_idx 
            j_neighbors = j[mask_i]
            d_neighbors = d[mask_i]
            S_neighbors = S[mask_i]
            
            w = np.exp(-d_neighbors**2 / (2*sigma**2))

            #sel_samples = Labels(
            #    names=["atom"],
            #    values=torch.tensor(j_neighbors, dtype=torch.int64).unsqueeze(-1),
            #)
            #feats = self.descriptor_spatial.calculate([system], sel_samples)
            
            weighted_sum = (w[:, None] * soap[j_neighbors]).sum(axis=0)
            h_i = (weighted_sum + self_weight * soap[idx]) / (self_weight + w.sum())
            feats[idx] = h_i

        return feats



class SpatialTempPCA(FullMethodBase):

    def __init__(self, descriptor, interval, sigma, cutoff, ridge_alpha, root):
        self.name = 'SpatialTempPCA'
        super().__init__(descriptor, interval, lag=0, root=root, sigma=sigma, ridge_alpha=ridge_alpha, method=self.name)
        self.descriptor_spatial = descriptor
        self.spatial_cutoff = cutoff
        self.label = self.label + f'cut_{self.spatial_cutoff}'
    
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
        first_soap =  soap_block  
        self.atomsel_element = [[idx for idx, label in enumerate(self.descriptor.soap_block.samples.values.numpy()) if label[2] == atom_type] for atom_type in self.descriptor.centers]
        if soap_block.shape[0] == 1:
            self.atomsel_element = [[0] for atom_type in self.descriptor.centers]
        #initialize neighbor list for spatial avg
        self.make_neighborlist(self.spatial_cutoff)

        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        cov_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        cov_intra_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        sum_mu_t = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        scatter_mut = np.zeros((len(self.atomsel_element),first_soap.shape[1], first_soap.shape[1],))
        nsmp = np.zeros(len(self.atomsel_element))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2)) # cutoff at 3 sigma, leaves 0.1%
        kernel /= np.sum(kernel)
        ntimesteps = np.zeros(len(self.atomsel_element), dtype=int)
        logger = np.zeros((len(self.atomsel_element), len(systems), first_soap.shape[1],))
        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate([system])
            avg_soap_values = self.spatial_average_with_nl(system, new_soap_values)
            if fidx >= self.interval:
                roll_kernel = np.roll(kernel, fidx%self.interval)
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                avg_soap = self.spatial_averaging(system, avg_soap, self.sigma)
                for atom_type_idx, atom_type in enumerate(self.atomsel_element):
                    mu_t = avg_soap[atom_type].mean(axis=0)
                    scatter_mut[atom_type_idx] += np.einsum(
                        "a,b->ab",
                        mu_t,
                        mu_t,
                    )

                    sum_mu_t[atom_type_idx] += mu_t #sum over all same atoms
                    logger[atom_type_idx][fidx] = mu_t
                    cov_t[atom_type_idx] += np.einsum("ia,ib->ab", avg_soap[atom_type] - mu_t, avg_soap[atom_type] - mu_t)/len(atom_type) #sum over all same atoms (have already summed over all times before)
                    nsmp[atom_type_idx] += len(atom_type)
                    ntimesteps[atom_type_idx] += 1

            buffer[:,fidx%self.interval,:] = avg_soap_values
        np.savetxt(self.label + '_mu_t.csv', logger[0])
        mean_cov_t = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        mean_cov_intra_t = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        cov_mu_t = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        mean_mu_t = np.zeros((len(self.atomsel_element), first_soap.shape[1],))

        # autocorrelation matrix - remove mean
        for atom_type_idx, atom_type in enumerate(self.atomsel_element): 
            mean_cov_t[atom_type_idx] = cov_t[atom_type_idx]/ntimesteps[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            mean_mu_t[atom_type_idx] = sum_mu_t[atom_type_idx]/ntimesteps[atom_type_idx]
            # add temporal covariance
            cov_mu_t[atom_type_idx] = scatter_mut[atom_type_idx]/ntimesteps[atom_type_idx] - np.einsum('i,j->ij', mean_mu_t[atom_type_idx], mean_mu_t[atom_type_idx])

        #all_soap_values = eval_SOAP(systems, calculator, sel, atomsel).values.numpy()
        #C_np = np.cov(all_soap_values, rowvar=False, bias=True)   # population covariance
        #print(np.allclose(C_np, avgcc[0], atol=1e-8))

        self.mean_mu_t = mean_mu_t
        self.mean_cov_t = mean_cov_t
        self.cov_mu_t = cov_mu_t
        
        # temp vs inter
        return mean_mu_t, cov_mu_t, mean_cov_t #[np.eye(c.shape[0]) for c in cov_mu_t] #mean_cov_t  #+ cov_mu_t



    def make_neighborlist(self, cutoff):
        self.nlist = NeighborList(cutoff=cutoff, full_list=False)


    def spatial_average_with_nl(self, system, features):
        positions = system.positions  # (N,3) numpy array
        box = system.cell             # 3x3 array or None for non-periodic
        i, j, d, D = self.nlist.compute(
            points=positions,
            box=box,
            periodic=True,
            quantities="ijdD"
        )
        # filter pairs to only the centers you care about:
        mask = np.isin(i, self.selected_atoms)
        atom_types = system.types.numpy()
        i_sel, j_sel, d_sel, D_sel = i[mask], j[mask], d[mask], D[mask]
        averaged_features = np.zeros_like(features)
        self_weight = 1.0
    


        for idx, atom_idx in enumerate(self.selected_atoms):
            mask_i = i_sel == atom_idx 
            j_neighbors = j_sel[mask_i]
            d_neighbors = d_sel[mask_i]
            mask_j = atom_types[j_neighbors] == 8
            if len(j_neighbors) == 0:
                averaged_features[idx] = features[idx]
                continue

            w = np.exp(-d_neighbors[mask_j]**2 / (2*self.sigma**2))
            sel_samples = Labels(
                names=["atom"],
                values=torch.tensor(j_neighbors, dtype=torch.int64).unsqueeze(-1),
            )
            feats = self.descriptor_spatial.calculate([system], sel_samples)
            weighted_sum = (w[:, None] * feats).sum(axis=0)
            h_i = (weighted_sum + self_weight * features[idx]) / (self_weight + w.sum())
            averaged_features[idx] = h_i

        return averaged_features

    def spatial_average_with_nl(self, system, features):
        positions = system.positions  # (N,3) numpy array
        box = system.cell             # 3x3 array or None for non-periodic
        i, j, d, D = self.nlist.compute(
            points=positions,
            box=box,
            periodic=True,
            quantities="ijdD"
        )

        # filter pairs to only the centers you care about:
        mask = np.isin(i, self.selected_atoms)
        atom_types = system.types.numpy()
        i_sel, j_sel, d_sel, D_sel = i[mask], j[mask], d[mask], D[mask]
        averaged_features = np.zeros_like(features)
        self_weight = 1.0

        for idx, atom_idx in enumerate(self.selected_atoms):
            mask_i = i_sel == atom_idx 
            j_neighbors = j_sel[mask_i]
            d_neighbors = d_sel[mask_i]
            D_neighbors = D_sel[mask_i]

            # --- Deduplicate: keep only closest image per base atom
            if len(j_neighbors) > 0:
                unique_js = np.unique(j_neighbors)
                best_idx = np.array([
                    np.argmin(d_neighbors[j_neighbors == j]) 
                    for j in unique_js
                ])
                j_neighbors = unique_js
                d_neighbors = d_neighbors[best_idx]
                D_neighbors = D_neighbors[best_idx]

            mask_j = atom_types[j_neighbors] == 8
            if len(j_neighbors) == 0:
                averaged_features[idx] = features[idx]
                continue

            w = np.exp(-d_neighbors[mask_j]**2 / (2*self.sigma**2))
            sel_samples = Labels(
                names=["atom"],
                values=torch.tensor(j_neighbors, dtype=torch.int64).unsqueeze(-1),
            )
            feats = self.descriptor_spatial.calculate([system], sel_samples)
            weighted_sum = (w[:, None] * feats).sum(axis=0)
            h_i = (weighted_sum + self_weight * features[idx]) / (self_weight + w.sum())
            averaged_features[idx] = h_i

        return averaged_features





    """
        self.nlist.update(system.positions)
        averaged_features = {}
        self_weight = 1.0
        for i in self.selected_atoms:
            j, d = self.nlist.get_neighbors(i, return_distances=True)
            if len(j) == 0:
                averaged_features[i] = features[i]
                continue

            w = np.exp(-d**2 / (2*self.sigma**2))
            self.descriptor_spatial.set_samples(j)
            feats = self.descriptor_spatial.calculate([system])
            weighted_sum = (w[:, None] * feats).sum(axis=0)
            h_i = (weighted_sum + self_weight * features[i]) / (self_weight + w.sum())
            averaged_features[i] = h_i

        return averaged_features

    """


    def log_metrics(self):
        """
        Log metrics from the run, including the covariances.


        Returns
        -------
        empty
        """
        metrics = np.array([[np.trace(mean_cov), np.trace(cov_mu)] 
                    for  mean_cov, cov_mu in zip (self.mean_cov_t, self.cov_mu_t)])
        header = ["spatialCov_inter", "tempCov"]

        # Make metrics a 2D row vector: shape (1, 2)
        np.savetxt(
            self.label + "_.csv",
            metrics,
            fmt="%.6f",
            delimiter="\t",
            header="\t".join(header),
            comments=""
        )
        for i, trafo in enumerate(self.transformations):
            """            np.savetxt(
                self.label + f"_center{self.descriptor.centers[i]}" + "_cov_mu_t.csv",
                self.cov_mu_t[0],
            )

            np.savetxt(
                self.label + f"_center{self.descriptor.centers[i]}" + f"_mean_cov_t.csv",
                self.mean_cov_t[0],
            )"""

            torch.save(
                torch.tensor(trafo.eigvals.copy()),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_eigvals.pt",
            )

            torch.save(
                torch.tensor(trafo.eigvecs.copy()),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_eigvecs.pt",
            )

 



class CumulantPCA(FullMethodBase):

    def __init__(self, descriptor, interval, ridge_alpha, n_cumulants, root):
        self.name = 'CumulantPCA'
        self.n_cumulants = n_cumulants
        super().__init__(descriptor, interval, lag=0, root=root, sigma=0, ridge_alpha=ridge_alpha, method=self.name)
        

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

        for trafo in self.transformations:
            projected = []
            for system in systems:
                descriptor = self.descriptor.calculate([system])
                cum_descriptor = self.descriptor.compute_cumulants(descriptor, self.n_cumulants)
                descriptor_proj = trafo.project(cum_descriptor)
                projected.append(descriptor_proj)
                # TODO:
                #self.ridge.fit(descriptor, descriptor_proj)
            projected_per_type.append(np.stack(projected, axis=0).transpose(1, 0, 2))

        return projected_per_type  # shape: (#centers ,N_atoms, T, latent_dim)
    
    
    def predict_ridge(self, traj, selected_atoms):
        self.selected_atoms = selected_atoms
        self.descriptor.set_samples(selected_atoms)
        systems = systems_to_torch(traj, dtype=torch.float32)
       
        projected_per_type = []

        for idx, trafo in enumerate(self.transformations):
            projected = []
            for system in systems:
                descriptor = self.descriptor.calculate([system])
                cum_descriptor = self.descriptor.compute_cumulants(descriptor, self.n_cumulants)
            
                ridge_pred = self.ridge[idx].predict(cum_descriptor)
                projected.append(ridge_pred)
               
            projected_per_type.append(np.stack(projected, axis=0).transpose(1, 0, 2))

        return projected_per_type
        

    def compute_(self, avg_soap, sum_soaps, cov_t, nsmp, ntimesteps):   
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            sum_soaps[atom_type_idx] += avg_soap[atom_type].sum(axis=0)
            cov_t[atom_type_idx] += np.einsum("ia,ib->ab", avg_soap[atom_type], avg_soap[atom_type]) #sum over all same atoms (have already summed over all times before) 
            nsmp[atom_type_idx] += len(atom_type)
            ntimesteps[atom_type_idx] += 1


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
        
        first_soap = self.descriptor.compute_cumulants(soap_block, self.n_cumulants)
        
        self.atomsel_element = [[0] for atom_type in self.descriptor.centers]
        
        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        cov_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        sum_soaps = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        nsmp = np.zeros(len(self.atomsel_element))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2)) # cutoff at 3 sigma, leaves 0.1%
        kernel /= kernel.sum()
        ntimesteps = np.zeros(len(self.atomsel_element), dtype=int)

        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate([system])
            cum_soap_values = self.descriptor.compute_cumulants(new_soap_values, self.n_cumulants)
            new_soap_values = None
            if fidx >= self.interval:
                roll_kernel = np.roll(kernel, fidx%self.interval)
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                for atom_type_idx, atom_type in enumerate(self.atomsel_element):
                    self.compute_(avg_soap, sum_soaps, cov_t, nsmp, ntimesteps)


            buffer[:,fidx%self.interval,:] = cum_soap_values

        if len(systems) == 1:
            self.compute_(first_soap, sum_soaps, cov_t, nsmp, ntimesteps)

        mu = np.zeros((len(self.atomsel_element), cum_soap_values.shape[1]))
        cov = np.zeros((len(self.atomsel_element), cum_soap_values.shape[1], cum_soap_values.shape[1]))
        
        # autocorrelation matrix - remove mean
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            mu[atom_type_idx] = sum_soaps[atom_type_idx]/nsmp[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            cov[atom_type_idx] = cov_t[atom_type_idx]/nsmp[atom_type_idx] - np.einsum('i,j->ij', mu[atom_type_idx], mu[atom_type_idx])
        self.cov_tot = cov
        return mu, cov, [np.eye(c.shape[0]) for c in cov]


    def fit_ridge_nonincremental(self, traj):
        systems = systems_to_torch(traj, dtype=torch.float32)
        soap_block = self.descriptor.calculate(systems[:1], selected_samples=self.descriptor.selected_samples)
        print(soap_block.shape)
        first_soap = self.descriptor.compute_cumulants(soap_block, self.n_cumulants)
        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2)) # cutoff at 3 sigma, leaves 0.1%
        kernel /= kernel.sum() #kernel = delta
        self.ridge = {}


        for idx, trafo in enumerate(self.transformations):
            self.ridge[idx] = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
            avg_soap_proj = trafo.project(first_soap) 
            print('avg_soap_proj',avg_soap_proj.shape)
            soap_values=np.zeros((first_soap.shape[0],len(systems)-self.interval, first_soap.shape[1]))
            avg_soaps_projs=np.zeros((first_soap.shape[0],len(systems)-self.interval, avg_soap_proj.shape[-1]))
            for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Fit Ridge"):
                new_soap_values = self.descriptor.calculate([system], selected_samples=self.descriptor.selected_samples)
                cum_soap_values = self.descriptor.compute_cumulants(new_soap_values, self.n_cumulants)
                new_soap_values = None
                if fidx >= self.interval:
                    roll_kernel = np.roll(kernel, fidx%self.interval)
                    roll_delta = np.roll(delta, fidx%self.interval)
                    # computes a contribution to the correlation function
                    # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                    avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                    avg_soap_proj = trafo.project(avg_soap)
                    #print('projshape', avg_soap_proj.shape)
                    #print('nonprog.shape',new_soap_values.shape)
                    soap_values[:,fidx-self.interval,:] = np.einsum("j,ija->ia", roll_delta, buffer)
                    avg_soaps_projs[:,fidx-self.interval,:] = avg_soap_proj
                buffer[:,fidx%self.interval,:] = cum_soap_values

            if len(systems) == 1:
                soap_values = first_soap[:,None,:]
                avg_soaps_projs = trafo.project(first_soap)[:,None,:]

            #soap_values=soap_values.reshape((soap_values.shape[0]*soap_values.shape[1],soap_values.shape[2]))
            #avg_soaps_projs=avg_soaps_projs.reshape((avg_soaps_projs[0]*avg_soaps_projs[1],avg_soaps_projs.shape[2]))
            soap_values=soap_values.reshape(soap_values.shape[0]*soap_values.shape[1],soap_values.shape[2])
            avg_soaps_projs=avg_soaps_projs.reshape(avg_soaps_projs.shape[0]*avg_soaps_projs.shape[1],avg_soaps_projs.shape[2])
            #np.reshape(avg_soaps_projs.soap_values,(avg_soaps_projs.shape[0]*avg_soaps_projs.shape[1],avg_soaps_projs.shape[2]))
            #p.reshape(avg_soaps_projs,(avg_soaps_projs[0]*avg_soaps_projs[1],avg_soaps_projs.shape[2]))
            print('soapvals',soap_values.shape)
            print('avg_soaps_proj',avg_soaps_projs.shape)
            self.ridge[idx].fit(soap_values, avg_soaps_projs)
      

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
        for i, trafo in enumerate(self.transformations):
            torch.save(
                torch.tensor(trafo.eigvals.copy()),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_eigvals.pt",
            )

            torch.save(
                torch.tensor(trafo.eigvecs.copy()),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_eigvecs.pt",
            ) 

            torch.save(
                torch.tensor(self.cov1[i]),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_cov1.pt",
            ) 

            torch.save(
                self.descriptor.soap_block.properties.values,
                self.label + f"_center{self.descriptor.centers[i]}" + f"_properties.pt",
            )




class PCAnorm(FullMethodBase):

    def __init__(self, descriptor, interval, ridge_alpha, root):
        self.name = 'PCAnorm'
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
        self.transformations = [PCA_obj(n_components=4, label=self.label) for n in range(self.cov1.shape[0])]

        self.std = np.zeros((len(self.transformations), self.cov1.shape[1]))
        for i, trafo in enumerate(self.transformations):
            self.std[i] = np.sqrt(np.diag(self.cov1[i]))
            R = self.cov1[i] / np.outer(self.std[i], self.std[i])
            trafo.solve_GEV(self.mean[i], R, self.cov2[i])



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

        for i, trafo in enumerate(self.transformations):
            projected = []
            for system in systems:
                descriptor = self.descriptor.calculate([system])
                descriptor /= self.std[i]
                descriptor_proj = trafo.project(descriptor)
                projected.append(descriptor_proj)
                # TODO:
                #self.ridge.fit(descriptor, descriptor_proj)
            projected_per_type.append(np.stack(projected, axis=0).transpose(1, 0, 2))
        self.get_explained_variance(traj, selected_atoms)
        return projected_per_type  # shape: (#centers ,N_atoms, T, latent_dim)

   
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


        for idx, trafo in enumerate(self.transformations):
            self.ridge[idx] = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
            avg_soap_proj = trafo.project(first_soap) 
            print('avg_soap_proj',avg_soap_proj.shape)
            soap_values=np.zeros((first_soap.shape[0],len(systems)-self.interval, first_soap.shape[1]))
            avg_soaps_projs=np.zeros((first_soap.shape[0],len(systems)-self.interval, avg_soap_proj.shape[-1]))
            for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Fit Ridge"):
                new_soap_values = self.descriptor.calculate([system], selected_samples=self.descriptor.selected_samples)
                new_soap_values /= self.std[idx]
                if fidx >= self.interval:
                    roll_kernel = np.roll(kernel, fidx%self.interval)
                    # computes a contribution to the correlation function
                    # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                    avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                    avg_soap_proj = trafo.project(avg_soap)
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
            soap_values=soap_values.reshape(soap_values.shape[0]*soap_values.shape[1],soap_values.shape[2])
            avg_soaps_projs=avg_soaps_projs.reshape(avg_soaps_projs.shape[0]*avg_soaps_projs.shape[1],avg_soaps_projs.shape[2])
            #np.reshape(avg_soaps_projs.soap_values,(avg_soaps_projs.shape[0]*avg_soaps_projs.shape[1],avg_soaps_projs.shape[2]))
            #p.reshape(avg_soaps_projs,(avg_soaps_projs[0]*avg_soaps_projs[1],avg_soaps_projs.shape[2]))
            print('soapvals',soap_values.shape)
            print('avg_soaps_proj',avg_soaps_projs.shape)
            self.ridge[idx].fit(soap_values, avg_soaps_projs)
      


    def fit_ridge(self, traj):
        systems = systems_to_torch(traj, dtype=torch.float32)
        soap_block = self.descriptor.calculate(systems[:1])
        first_soap = soap_block  
        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2)) # cutoff at 3 sigma, leaves 0.1%
        kernel /= kernel.sum() #kernel = delta
        self.ridge = {}
        for idx, trafo in enumerate(self.transformations):
            #self.ridge[idx] = Ridge(alpha=ridge_alpha, fit_intercept=False)
            base = SGDRegressor(penalty="l2", alpha=self.ridge_alpha, fit_intercept=False)
            self.ridge[idx] = MultiOutputRegressor(base)

            for epoch in range(100):
                print(epoch)

                for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Fit Ridge"):
                    new_soap_values = self.descriptor.calculate([system])
                    new_soap_values /= self.std[idx]
                    if fidx >= self.interval:
                        roll_kernel = np.roll(kernel, fidx%self.interval)
                        # computes a contribution to the correlation function
                        # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                        avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                        avg_soap_proj = trafo.project(avg_soap) 
                        self.ridge[idx].partial_fit(new_soap_values, avg_soap_proj)
                    buffer[:,fidx%self.interval,:] = new_soap_values


        """for trafo in self.transformations:
            projected = []
            for system in systems:
                descriptor = self.descriptor.calculate([system])
                descriptor_proj = trafo.project(descriptor)
                projected.append(descriptor_proj)
                # TODO:
                self.ridge.fit(descriptor, descriptor_proj)"""
    
    def predict_ridge(self, traj, selected_atoms):
        self.selected_atoms = selected_atoms
        self.descriptor.set_samples(selected_atoms)
        systems = systems_to_torch(traj, dtype=torch.float32)
       
        projected_per_type = []

        for idx, trafo in enumerate(self.transformations):
            projected = []
            for system in systems:
                descriptor = self.descriptor.calculate([system])
                descriptor /= self.std[idx]
                ridge_pred = self.ridge[idx].predict(descriptor)
                projected.append(ridge_pred)
               
            projected_per_type.append(np.stack(projected, axis=0).transpose(1, 0, 2))
        self.get_explained_variance_ridge(traj, selected_atoms)
        return projected_per_type


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
        first_soap =  soap_block  
        self.atomsel_element = [[idx for idx, label in enumerate(self.descriptor.soap_block.samples.values.numpy()) if label[2] == atom_type] for atom_type in self.descriptor.centers]
        if soap_block.shape[0] == 1:
            self.atomsel_element = [[0] for atom_type in self.descriptor.centers]
        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        cov_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        sum_mu_t = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        scatter_mut = np.zeros((len(self.atomsel_element),first_soap.shape[1], first_soap.shape[1],))
        nsmp = np.zeros(len(self.atomsel_element))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1))#gaussian_filter(delta,sigma=(self.interval-1)//(2*3)) # cutoff at 3 sigma, leaves 0.1%
        kernel /= np.sum(kernel)
        ntimesteps = np.zeros(len(self.atomsel_element), dtype=int)
        import matplotlib.pyplot as plt
        #plt.plot(kernel)
        #plt.savefig(self.label + '_kernel.png')
        #plt.close()
        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate([system])
            if fidx >= self.interval:
                roll_kernel = np.roll(kernel, fidx%self.interval)
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                #fig, ax = plt.subplots(1,1, figsize=(5,4))
                #ax.plot(buffer[0,:,0])
                #ax2 = ax.twinx()
                #ax2.plot(roll_kernel)
                #ax.scatter([fidx%self.interval + self.interval//2],[avg_soap[0,0]], color='red', s=50)
                #plt.savefig(self.label + f'_buffer_{fidx}.png')
                #plt.close()
               
                for atom_type_idx, atom_type in enumerate(self.atomsel_element):
                    mu_t = avg_soap[atom_type].mean(axis=0)
                    scatter_mut[atom_type_idx] += np.einsum(
                        "a,b->ab",
                        mu_t,
                        mu_t,
                    )

                    sum_mu_t[atom_type_idx] += mu_t #sum over all same atoms

                    cov_t[atom_type_idx] += np.einsum("ia,ib->ab", avg_soap[atom_type] - mu_t, avg_soap[atom_type] - mu_t)/len(atom_type) #sum over all same atoms (have already summed over all times before) 
                    nsmp[atom_type_idx] += len(atom_type)
                    ntimesteps[atom_type_idx] += 1

            buffer[:,fidx%self.interval,:] = new_soap_values
        #exit()
        mean_cov_t = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        cov_mu_t = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        mean_mu_t = np.zeros((len(self.atomsel_element), first_soap.shape[1],))

        # autocorrelation matrix - remove mean
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            
            mean_cov_t[atom_type_idx] = cov_t[atom_type_idx]/ntimesteps[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            mean_mu_t[atom_type_idx] = sum_mu_t[atom_type_idx]/ntimesteps[atom_type_idx]
            # add temporal covariance
            cov_mu_t[atom_type_idx] = scatter_mut[atom_type_idx]/ntimesteps[atom_type_idx] - np.einsum('i,j->ij', mean_mu_t[atom_type_idx], mean_mu_t[atom_type_idx])

        #all_soap_values = eval_SOAP(systems, calculator, sel, atomsel).values.numpy()
        #C_np = np.cov(all_soap_values, rowvar=False, bias=True)   # population covariance
        #print(np.allclose(C_np, avgcc[0], atol=1e-8))

        self.mean_mu_t = mean_mu_t
        self.mean_cov_t = mean_cov_t
        self.cov_mu_t = cov_mu_t
        
        total_cov = mean_cov_t + cov_mu_t
        return mean_mu_t, total_cov, [np.eye(cov.shape[0]) for cov in total_cov]
        #return mean_mu_t, mean_cov_t, cov_mu_t

    def log_metrics(self):
        """
        Log metrics from the run, including the covariances.

        
        Returns
        -------
        empty
        """
        metrics = np.array([[np.trace(mean_cov), np.trace(cov_mu)] 
                    for mean_cov, cov_mu in zip(self.mean_cov_t, self.cov_mu_t)])
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

        for i, trafo in enumerate(self.transformations):
            
            np.savetxt(
                self.label + f"_center{self.descriptor.centers[i]}" + "_cov_mu_t.csv",
                self.cov_mu_t[0],
            )

            np.savetxt(
                self.label + f"_center{self.descriptor.centers[i]}" + f"_mean_cov_t.csv",
                self.mean_cov_t[0],
            )

            torch.save(
                torch.tensor(trafo.eigvals.copy()),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_eigvals.pt",
            )

            torch.save(
                torch.tensor(trafo.eigvecs.copy()),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_eigvecs.pt",
            )

            torch.save(
                self.descriptor.soap_block.properties.values,
                self.label + f"_center{self.descriptor.centers[i]}" + f"_properties.pt",
            )

       
    

class SpatialPCA_deprecated(FullMethodBase):

    def __init__(self, descriptor, interval, sigma, ridge_alpha, root):
        self.name = 'SpatialPCA'
        super().__init__(descriptor, interval, lag=0, root=root, sigma=sigma, ridge_alpha=ridge_alpha, method=self.name)


    def compute_(self, soap, scatter_mut, sum_mu_t, cov_t, nsmp, ntimesteps):
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            #COV_intra_cluster, COV_inter_cluster = self.full_spatial_averaging(system, avg_soap, self.sigma)
            mu_t = soap[atom_type].mean(axis=0)
            scatter_mut[atom_type_idx] += np.einsum(
                "a,b->ab",
                mu_t,
                mu_t,
            )

            sum_mu_t[atom_type_idx] += mu_t #sum over all same atoms

            #cov_t[atom_type_idx] += np.einsum("ia,ib->ab", avg_soap[atom_type] - mu_t, avg_soap[atom_type] - mu_t)/len(atom_type) #sum over all same atoms (have already summed over all times before) 
            cov_t[atom_type_idx] += COV_inter_cluster
            cov_intra_t[atom_type_idx] += COV_intra_cluster
            nsmp[atom_type_idx] += len(atom_type)
            ntimesteps[atom_type_idx] += 1

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
        first_soap =  soap_block  
        self.atomsel_element = [[idx for idx, label in enumerate(self.descriptor.soap_block.samples.values.numpy()) if label[2] == atom_type] for atom_type in self.descriptor.centers]
        if soap_block.shape[0] == 1:
            self.atomsel_element = [[0] for atom_type in self.descriptor.centers]
        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        cov_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        cov_intra_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        sum_mu_t = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        scatter_mut = np.zeros((len(self.atomsel_element),first_soap.shape[1], first_soap.shape[1],))
        nsmp = np.zeros(len(self.atomsel_element))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2*3)) # cutoff at 3 sigma, leaves 0.1%
        ntimesteps = np.zeros(len(self.atomsel_element), dtype=int)

        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate([system])
            avg_soap_values = self.spatial_average_with_nl([system], new_soap_values)
            if fidx >= self.interval:
                roll_kernel = np.roll(kernel, fidx%self.interval)
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                self.compute_()
                for atom_type_idx, atom_type in enumerate(self.atomsel_element):
                    
                    #COV_intra_cluster, COV_inter_cluster = self.full_spatial_averaging(system, avg_soap, self.sigma)
                    soapsum = avg_soap[atom_type].mean(axis=0)
                    scatter_mut[atom_type_idx] += np.einsum(
                        "a,b->ab",
                        mu_t,
                        mu_t,
                    )

                    sum_mu_t[atom_type_idx] += mu_t #sum over all same atoms

                    #cov_t[atom_type_idx] += np.einsum("ia,ib->ab", avg_soap[atom_type] - mu_t, avg_soap[atom_type] - mu_t)/len(atom_type) #sum over all same atoms (have already summed over all times before) 
                    cov_t[atom_type_idx] += COV_inter_cluster
                    cov_intra_t[atom_type_idx] += COV_intra_cluster
                    nsmp[atom_type_idx] += len(atom_type)
                    ntimesteps[atom_type_idx] += 1

            buffer[:,fidx%self.interval,:] = avg_soap_values

        mean_cov_t = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        mean_cov_intra_t = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        cov_mu_t = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        mean_mu_t = np.zeros((len(self.atomsel_element), first_soap.shape[1],))

        # autocorrelation matrix - remove mean
        for atom_type_idx, atom_type in enumerate(self.atomsel_element): 
            mean_cov_t[atom_type_idx] = cov_t[atom_type_idx]/ntimesteps[atom_type_idx]
            mean_cov_intra_t[atom_type_idx] = cov_intra_t[atom_type_idx]/ntimesteps[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            mean_mu_t[atom_type_idx] = sum_mu_t[atom_type_idx]/ntimesteps[atom_type_idx]
            # add temporal covariance
            cov_mu_t[atom_type_idx] = scatter_mut[atom_type_idx]/ntimesteps[atom_type_idx] - np.einsum('i,j->ij', mean_mu_t[atom_type_idx], mean_mu_t[atom_type_idx])

        #all_soap_values = eval_SOAP(systems, calculator, sel, atomsel).values.numpy()
        #C_np = np.cov(all_soap_values, rowvar=False, bias=True)   # population covariance
        #print(np.allclose(C_np, avgcc[0], atol=1e-8))

        self.mean_mu_t = mean_mu_t
        self.mean_cov_t = mean_cov_t
        self.mean_cov_intra_t = mean_cov_intra_t
        self.cov_mu_t = cov_mu_t

        # inter vs intra
        #return mean_mu_t, mean_cov_t, mean_cov_intra_t  #+ cov_mu_t
        # intra vs inter
        return mean_mu_t, mean_cov_intra_t, mean_cov_t  #+ cov_mu_t

        # FULL
        #return mean_mu_t, mean_cov_t + mean_cov_intra_t + cov_mu_t, [np.eye(c.shape[0]) for c in cov_mu_t]







    def log_metrics(self):
        """
        Log metrics from the run, including the covariances.


        Returns
        -------
        empty
        """
        metrics = np.array([[np.trace(mean_cov_intra), np.trace(mean_cov), np.trace(cov_mu)] 
                    for mean_cov_intra, mean_cov, cov_mu in zip(self.mean_cov_intra_t, self.mean_cov_t, self.cov_mu_t)])
        header = ["spatialCov_intra", "spatialCov_inter", "tempCov"]

        # Make metrics a 2D row vector: shape (1, 2)
        np.savetxt(
            self.label + "_.csv",
            metrics,
            fmt="%.6f",
            delimiter="\t",
            header="\t".join(header),
            comments=""
        )




class PCA_time_norm(FullMethodBase):

    def __init__(self, descriptor, interval, ridge_alpha, root):
        self.name = 'PCA_time_norm'
        super().__init__(descriptor, interval, lag=0, root=root, sigma=0, ridge_alpha=ridge_alpha, method=self.name)


    def predict_avg(self, traj, selected_atoms):
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

        for trafo in self.transformations:
            descriptor = self.descriptor.calculate([systems[0]])
            buffer = np.zeros((descriptor.shape[0], self.interval, descriptor.shape[1]))
            transformed = trafo.project(descriptor)
            projected = np.zeros((len(systems) - self.interval, transformed.shape[0], transformed.shape[1]))
            delta=np.zeros(self.interval)
            delta[self.interval//2]=1
            kernel=gaussian_filter(delta,sigma=(self.interval-1))#gaussian_filter(delta,sigma=(self.interval-1)//(2*3)) # cutoff at 3 sigma, leaves 0.1%
            kernel /= np.sum(kernel)
            for i, system in enumerate(systems):
                buffer[:,i%self.interval,:] = descriptor
                if i >= self.interval:
                    roll_kernel = np.roll(kernel, i%self.interval)
                    avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer)
                    avg_soap = trafo.project(avg_soap)
                    projected[i-self.interval] = avg_soap
                descriptor = self.descriptor.calculate([system])
            projected_per_type.append(projected.transpose(1, 0, 2))
        self.get_explained_variance(traj, selected_atoms)
        return projected_per_type  # shape: (#centers ,N_atoms, T, latent_dim)


    def compute_(self, soap, scatter_mut, sum_mu_t, cov_t, nsmp, ntimesteps):   
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            mu_t = soap[atom_type].mean(axis=0)
            scatter_mut[atom_type_idx] += np.einsum(
                "a,b->ab",
                mu_t,
                mu_t,
            )

            sum_mu_t[atom_type_idx] += mu_t #sum over all same atoms

            cov_t[atom_type_idx] += np.einsum("ia,ib->ab", soap[atom_type] - mu_t, soap[atom_type] - mu_t)/len(atom_type) #sum over all same atoms (have already summed over all times before) 
            nsmp[atom_type_idx] += len(atom_type)
            ntimesteps[atom_type_idx] += 1


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
        self.atomsel_element = [[idx for idx, label in enumerate(self.descriptor.soap_block.samples.values.numpy()) if label[2] == atom_type] for atom_type in self.descriptor.centers]
        if soap_block.shape[0] == 1:
            self.atomsel_element = [[0] for atom_type in self.descriptor.centers]
        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        cov_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        sum_mu_t = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        scatter_mut = np.zeros((len(self.atomsel_element),first_soap.shape[1], first_soap.shape[1],))
        nsmp = np.zeros(len(self.atomsel_element))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1))#gaussian_filter(delta,sigma=(self.interval-1)//(2*3)) # cutoff at 3 sigma, leaves 0.1%
        kernel /= np.sum(kernel)
        ntimesteps = np.zeros(len(self.atomsel_element), dtype=int)

        cov_t_inst = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        sum_mu_t_inst = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        scatter_mut_inst = np.zeros((len(self.atomsel_element),first_soap.shape[1], first_soap.shape[1],))
        nsmp_inst = np.zeros(len(self.atomsel_element))
        ntimesteps_inst = np.zeros(len(self.atomsel_element), dtype=int)
        #plt.plot(kernel)
        #plt.savefig(self.label + '_kernel.png')
        #plt.close()
        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate([system])
            if fidx >= self.interval:
                roll_kernel = np.roll(kernel, fidx%self.interval)
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                #fig, ax = plt.subplots(1,1, figsize=(5,4))
                #ax.plot(buffer[0,:,0])
                #ax2 = ax.twinx()
                #ax2.plot(roll_kernel)
                #ax.scatter([fidx%self.interval + self.interval//2],[avg_soap[0,0]], color='red', s=50)
                #plt.savefig(self.label + f'_buffer_{fidx}.png')
                #plt.close()
                self.compute_(avg_soap, scatter_mut, sum_mu_t, cov_t, nsmp, ntimesteps)
                self.compute_(new_soap_values, scatter_mut_inst, sum_mu_t_inst, cov_t_inst, nsmp_inst, ntimesteps_inst)
            buffer[:,fidx%self.interval,:] = new_soap_values

        #exit()
        if len(systems) == 1:
            self.compute_(first_soap, scatter_mut, sum_mu_t, cov_t, nsmp, ntimesteps)

        mean_cov_t = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        cov_mu_t = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        mean_mu_t = np.zeros((len(self.atomsel_element), first_soap.shape[1],))

        mean_cov_t_inst = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        cov_mu_t_inst = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        mean_mu_t_inst = np.zeros((len(self.atomsel_element), first_soap.shape[1],))

        # autocorrelation matrix - remove mean
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            mean_cov_t[atom_type_idx] = cov_t[atom_type_idx]/ntimesteps[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            mean_mu_t[atom_type_idx] = sum_mu_t[atom_type_idx]/ntimesteps[atom_type_idx]
            # add temporal covariance
            cov_mu_t[atom_type_idx] = scatter_mut[atom_type_idx]/ntimesteps[atom_type_idx] - np.einsum('i,j->ij', mean_mu_t[atom_type_idx], mean_mu_t[atom_type_idx])

            mean_cov_t_inst[atom_type_idx] = cov_t[atom_type_idx]/ntimesteps[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            mean_mu_t_inst[atom_type_idx] = sum_mu_t[atom_type_idx]/ntimesteps[atom_type_idx]
            # add temporal covariance
            cov_mu_t_inst[atom_type_idx] = scatter_mut[atom_type_idx]/ntimesteps[atom_type_idx] - np.einsum('i,j->ij', mean_mu_t[atom_type_idx], mean_mu_t[atom_type_idx])

        #all_soap_values = eval_SOAP(systems, calculator, sel, atomsel).values.numpy()
        #C_np = np.cov(all_soap_values, rowvar=False, bias=True)   # population covariance
        #print(np.allclose(C_np, avgcc[0], atol=1e-8))

        self.mean_mu_t = mean_mu_t
        self.mean_cov_t = mean_cov_t
        self.cov_mu_t = cov_mu_t
        
        total_cov = mean_cov_t + cov_mu_t
        
        return mean_mu_t, total_cov, mean_cov_t_inst + cov_mu_t_inst
        #return mean_mu_t, mean_cov_t, cov_mu_t

    def log_metrics(self):
        """
        Log metrics from the run, including the covariances.

        
        Returns
        -------
        empty
        """
        metrics = np.array([[np.trace(mean_cov), np.trace(cov_mu)] 
                    for mean_cov, cov_mu in zip(self.mean_cov_t, self.cov_mu_t)])
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

        for i, trafo in enumerate(self.transformations):
            
            np.savetxt(
                self.label + f"_center{self.descriptor.centers[i]}" + "_cov_mu_t.csv",
                self.cov_mu_t[0],
            )

            np.savetxt(
                self.label + f"_center{self.descriptor.centers[i]}" + f"_mean_cov_t.csv",
                self.mean_cov_t[0],
            )

            torch.save(
                torch.tensor(trafo.eigvals.copy()),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_eigvals.pt",
            )

            torch.save(
                torch.tensor(trafo.eigvecs.copy()),
                self.label + f"_center{self.descriptor.centers[i]}" + f"_eigvecs.pt",
            )

            torch.save(
                self.descriptor.soap_block.properties.values,
                self.label + f"_center{self.descriptor.centers[i]}" + f"_properties.pt",
            )

