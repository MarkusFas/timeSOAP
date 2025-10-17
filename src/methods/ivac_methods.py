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

#TODO : add lag to self.label
class TICA(FullMethodBase):

    def __init__(self, descriptor, interval, lag, root):
        self.name = 'TICA'
        super().__init__(descriptor, interval, lag=lag, root=root, method=self.name)

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
        systems = systems_to_torch(traj, dtype=torch.float64)
        soap_block = self.descriptor.calculate(systems[:1])
        first_soap = soap_block.values.numpy()  
        self.atomsel_element = [[idx for idx, label in enumerate(soap_block.samples.values.numpy()) if label[2] == atom_type] for atom_type in self.descriptor.centers]

        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        buffer_t = np.zeros((first_soap.shape[0], self.lag+1, first_soap.shape[1]))
        cov_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        corr_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        sum_soaps = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        sum_soaps_corr = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        nsmp = np.zeros(len(self.atomsel_element))
        nsmp_corr = np.zeros(len(self.atomsel_element))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2*3)) # cutoff at 3 sigma, leaves 0.1%
        ntimesteps = np.zeros(len(self.atomsel_element), dtype=int)
        ntimesteps_corr = np.zeros(len(self.atomsel_element), dtype=int)

        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate([system]).values.numpy()
            if fidx >= self.interval:
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                roll_kernel = np.roll(kernel, fidx%self.interval)
                avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                for atom_type_idx, atom_type in enumerate(self.atomsel_element):
                    sum_soaps[atom_type_idx] += avg_soap[atom_type].sum(axis=0)
                    cov_t[atom_type_idx] += np.einsum("ia,ib->ab", avg_soap[atom_type], avg_soap[atom_type]) #sum over all same atoms (have already summed over all times before) 
                    nsmp[atom_type_idx] += len(atom_type)
                    ntimesteps[atom_type_idx] += 1

            if fidx >= self.interval + self.lag:
                roll_kernel = np.roll(kernel, fidx%self.lag)
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                soap_0 = buffer_t[:,fidx%(self.lag+1),:]
                soap_lag = buffer_t[:,(fidx-1)%(self.lag+1),:]
                for atom_type_idx, atom_type in enumerate(self.atomsel_element):
                    sum_soaps_corr[atom_type_idx] += soap_0[atom_type].sum(axis=0)
                    corr_t[atom_type_idx] += np.einsum("ia,ib->ab", soap_0[atom_type], soap_lag[atom_type]) #sum over all same atoms (have already summed over all times before) 
                    nsmp_corr[atom_type_idx] += len(atom_type)
                    ntimesteps_corr[atom_type_idx] += 1

            buffer[:,fidx%self.interval,:] = new_soap_values
            if fidx >= self.interval:
                buffer_t[:,fidx%(self.lag+1),:] = avg_soap
                
        mu = np.zeros((len(self.atomsel_element), new_soap_values.shape[1]))
        cov = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        mu_corr = np.zeros((len(self.atomsel_element), new_soap_values.shape[1]))
        corr = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        
        # autocorrelation matrix - remove mean
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            mu[atom_type_idx] = sum_soaps[atom_type_idx]/nsmp[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            cov[atom_type_idx] = cov_t[atom_type_idx]/nsmp[atom_type_idx] - np.einsum('i,j->ij', mu[atom_type_idx], mu[atom_type_idx])
            
            mu_corr[atom_type_idx] = sum_soaps_corr[atom_type_idx]/nsmp_corr[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            corr[atom_type_idx] = corr_t[atom_type_idx]/nsmp_corr[atom_type_idx] - np.einsum('i,j->ij', mu[atom_type_idx], mu[atom_type_idx])
        
        self.cov = cov
        self.corr = corr
        self.mu = mu
        self.mu_corr = mu_corr
        return mu, corr, cov

    def log_metrics(self):
        """
        Log metrics from the run, including the covariances.

        Returns
        -------
        empty
        """
        metrics = np.array([self.mu[0], self.mu_corr[0], np.diag(self.cov[0]), np.diag(self.corr[0])])
        header = ["mu", "mu_corr", "cov", "corr"]

        # Make metrics a 2D row vector: shape (1, 2)
        np.savetxt(
            self.label + "_.csv",
            metrics,
            fmt="%.6f",
            delimiter="\t",
            header="\t".join(header),
            comments=""
        )


class IVAC(FullMethodBase):

    def __init__(self, descriptor, interval, max_lag, min_lag, lag_step, root):
        self.name = 'IVAC'
        self.max_lag = max_lag
        self.min_lag = min_lag
        self.lag_step = lag_step
        super().__init__(descriptor, interval, lag='ivac', root=root, method=self.name)

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
        systems = systems_to_torch(traj, dtype=torch.float64)
        soap_block = self.descriptor.calculate(systems[:1])
        first_soap = soap_block.values.numpy()  
        self.atomsel_element = [[idx for idx, label in enumerate(soap_block.samples.values.numpy()) if label[2] == atom_type] for atom_type in self.descriptor.centers]

        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        buffer_t = np.zeros((first_soap.shape[0], self.max_lag+1, first_soap.shape[1]))
        cov_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        corr_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        sum_soaps = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        sum_soaps_corr = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        nsmp = np.zeros(len(self.atomsel_element))
        nsmp_corr = np.zeros(len(self.atomsel_element))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2*3)) # cutoff at 3 sigma, leaves 0.1%
        ntimesteps = np.zeros(len(self.atomsel_element), dtype=int)
        ntimesteps_corr = np.zeros(len(self.atomsel_element), dtype=int)
        #IVAC specific:
        lags = np.arange(self.min_lag, self.max_lag + self.lag_step, self.lag_step)
        delta_soap_lag = np.zeros((len(lags), first_soap.shape[0], first_soap.shape[1]))
        soap_0_mu = np.zeros((len(self.atomsel_element), first_soap.shape[1],))
        soap_lag_mu = np.zeros((len(self.atomsel_element), len(lags), first_soap.shape[1],))
        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate([system]).values.numpy()
            if fidx >= self.interval:
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                roll_kernel = np.roll(kernel, fidx%self.interval)
                avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                for atom_type_idx, atom_type in enumerate(self.atomsel_element):
                    sum_soaps[atom_type_idx] += avg_soap[atom_type].sum(axis=0)
                    cov_t[atom_type_idx] += np.einsum("ia,ib->ab", avg_soap[atom_type], avg_soap[atom_type]) #sum over all same atoms (have already summed over all times before) 
                    nsmp[atom_type_idx] += len(atom_type)
                    ntimesteps[atom_type_idx] += 1

            if fidx >= self.interval + self.max_lag + 1:
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                soap_0 = buffer_t[:,fidx%self.max_lag,:]
                soap_lags = [buffer_t[:,(fidx+lag)%self.max_lag,:] for lag in lags]
                for atom_type_idx, atom_type in enumerate(self.atomsel_element):
                    sum_soaps_corr[atom_type_idx] += soap_0[atom_type].sum(axis=0)
                    delta_soap_0 = soap_0_mu[atom_type_idx] - soap_0[atom_type]  
                    #TODO: think about mean here, change if statement
                    if fidx == self.interval + self.max_lag + 1:
                        soap_0_mu[atom_type_idx] += delta_soap_0.mean(axis=0)
                    else:
                        soap_0_mu[atom_type_idx] += delta_soap_0.mean(axis=0) / ntimesteps_corr[atom_type_idx]
                    for i, soap_lag in enumerate(soap_lags):
                        delta_soap_lag[i] = soap_lag_mu[atom_type_idx, i] - soap_lag[atom_type]
                        #TODO: think about mean here
                        if fidx == self.interval + self.max_lag + 1:
                            soap_lag_mu[atom_type_idx, i] += delta_soap_lag[i].mean(axis=0)
                        else:
                            soap_lag_mu[atom_type_idx, i] += delta_soap_lag[i].mean(axis=0) / ntimesteps_corr[atom_type_idx]
                        corr_t[atom_type_idx] += np.einsum("ia,ib->ab", delta_soap_0, delta_soap_lag[i]) #sum over all same atoms (have already summed over all times before) 
                    nsmp_corr[atom_type_idx] += len(atom_type)
                    ntimesteps_corr[atom_type_idx] += 1

            buffer[:,fidx%self.interval,:] = new_soap_values
            if fidx >= self.interval:
                buffer_t[:,fidx%(self.max_lag+1),:] = avg_soap
                
        mu = np.zeros((len(self.atomsel_element), new_soap_values.shape[1]))
        cov = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        mu_corr = np.zeros((len(self.atomsel_element), new_soap_values.shape[1]))
        corr = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        
        # autocorrelation matrix - remove mean
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            mu[atom_type_idx] = sum_soaps[atom_type_idx]/nsmp[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            cov[atom_type_idx] = cov_t[atom_type_idx]/nsmp[atom_type_idx] - np.einsum('i,j->ij', mu[atom_type_idx], mu[atom_type_idx])
            
            mu_corr[atom_type_idx] = sum_soaps_corr[atom_type_idx]/nsmp_corr[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            corr[atom_type_idx] = corr_t[atom_type_idx]/(nsmp_corr[atom_type_idx]*len(lags)) #- np.einsum('i,j->ij', mu[atom_type_idx], mu[atom_type_idx])
        
        self.cov = cov
        self.corr = corr
        self.mu = mu
        self.mu_corr = mu_corr
        return mu, corr, cov

    def log_metrics(self):
            """
            Log metrics from the run, including the covariances.

            
            Returns
            -------
            empty
            """
            pass