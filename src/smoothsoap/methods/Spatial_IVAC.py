import torch
from abc import ABC, abstractmethod
import metatensor.torch as mts
from metatomic.torch import System, ModelEvaluationOptions, ModelOutput, systems_to_torch, load_atomistic_model
from metatensor.torch import Labels, TensorBlock, mean_over_samples
from featomic.torch import SoapPowerSpectrum
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from vesin.torch import NeighborList
from memory_profiler import profile
from sklearn.linear_model import Ridge

from smoothsoap.transformations.PCAtransform import PCA_obj
from smoothsoap.methods.BaseMethod import FullMethodBase


class SpatialIVAC(FullMethodBase):

    def __init__(self, descriptor, interval, ridge_alpha, spatial_cutoff, root):
        self.name = 'SpatialIVAC'
        self.spatial_cutoff = spatial_cutoff
        super().__init__(descriptor, interval, lag=0, root=root, sigma=0, ridge_alpha=ridge_alpha, method=self.name)
        self.label = self.label + f'cut_{self.spatial_cutoff}'

    def compute_(self, soap, sum_soaps, cov_t, cov_t_cum, nsmp, ntimesteps):   
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            sum_soaps[atom_type_idx] += soap[atom_type].mean(axis=0)
            cov_t[atom_type_idx] += np.einsum("ia,ib->ab", soap[atom_type], soap[atom_type]) #sum over all same atoms (have already summed over all times before) 
            cov_t_cum[atom_type_idx] += np.einsum("a,b->ab", sum_soaps[atom_type_idx], sum_soaps[atom_type_idx]) #sum over all same atoms (have already summed over all times before) 
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
        first_soap_cum = self.descriptor.compute_cumulants(soap_block, 1)

        self.atomsel_element = [[idx for idx, label in enumerate(self.descriptor.soap_block.samples.values.numpy()) if label[2] == atom_type] for atom_type in self.descriptor.centers]
        if soap_block.shape[0] == 1:
            self.atomsel_element = [[0] for _ in self.descriptor.centers]
        
        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        buffer_cum = np.zeros((first_soap.shape[0], self.interval, first_soap_cum.shape[1]))
        cov_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        cov_t_cum = np.zeros((len(self.atomsel_element), first_soap_cum.shape[1], first_soap_cum.shape[1],))
        sum_soaps = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        sum_soaps_cum = np.zeros((len(self.atomsel_element),first_soap_cum.shape[1],))
        nsmp = np.zeros(len(self.atomsel_element))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2)) # cutoff at 3 sigma, leaves 0.1%
        kernel /= kernel.sum()
        ntimesteps = np.zeros(len(self.atomsel_element), dtype=int)

        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate([system])

            if fidx >= self.interval:
                roll_kernel = np.roll(kernel, fidx%self.interval)
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                #avg_soap_cum = np.einsum("j,ija->ia", roll_kernel, buffer_cum) #smoothen
                for atom_type_idx, atom_type in enumerate(self.atomsel_element):
                    self.compute_(avg_soap, sum_soaps, cov_t, cov_t_cum, nsmp, ntimesteps)

            buffer[:,fidx%self.interval,:] = new_soap_values
            #buffer_cum[:,fidx%self.interval,:] = cum_soap_values

        if len(systems) == 1:
            self.compute_(first_soap, sum_soaps, cov_t, cov_t_cum, nsmp, ntimesteps)

        mu = np.zeros((len(self.atomsel_element), new_soap_values.shape[1]))
        cov = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        mu_cum = np.zeros((len(self.atomsel_element), new_soap_values.shape[1]))
        cov_cum = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))

        # autocorrelation matrix - remove mean
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            mu[atom_type_idx] = sum_soaps[atom_type_idx]/ntimesteps[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            cov[atom_type_idx] = cov_t[atom_type_idx]/nsmp[atom_type_idx] - np.einsum('i,j->ij', mu[atom_type_idx], mu[atom_type_idx])
            # COV = 1/N ExxT - mumuT
            cov_cum[atom_type_idx] = cov_t_cum[atom_type_idx]/(ntimesteps[atom_type_idx]) - np.einsum('i,j->ij', mu[atom_type_idx], mu[atom_type_idx])
        
        self.cov_tot = cov
        return mu, cov_cum, cov
  

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



class SpatialIVACnorm(FullMethodBase):

    def __init__(self, descriptor, interval, ridge_alpha, spatial_cutoff, root, eps2factor):
        self.name = 'SpatialIVACnorm'
        self.spatial_cutoff = spatial_cutoff
        super().__init__(descriptor, interval, lag=0, root=root, sigma=0, ridge_alpha=ridge_alpha, method=self.name, eps2factor=eps2factor)
        self.label = self.label + f'cut_{self.spatial_cutoff}'
        self.make_neighborlist(cutoff=self.spatial_cutoff)

    def make_neighborlist(self, cutoff):
        self.nlist = NeighborList(cutoff=cutoff, full_list=True) # TODO: debatable if full lsit or not

    def spatial_correlate(self, system, sel_atoms):
        pos = system.positions[sel_atoms]
        cell = system.cell
        i, j, S, d = self.nlist.compute(
            points=pos,
            box=cell, 
            periodic=True,
            quantities="ijSd"
        )
        i = i.cpu().numpy()
        j = j.cpu().numpy()
        d = d.cpu().numpy()
        S = S.cpu().numpy()
        sort_idx = np.lexsort((d, j, i))  # primary i, secondary j, tertiary d

        i_sorted = i[sort_idx]
        j_sorted = j[sort_idx]
        d_sorted = d[sort_idx]
        S_sorted = S[sort_idx]

        mask = np.ones(len(i_sorted), dtype=bool)
        mask[1:] = (i_sorted[1:] != i_sorted[:-1]) | (j_sorted[1:] != j_sorted[:-1])

        i_final = i_sorted[mask]
        j_final = j_sorted[mask]
        d_final = d_sorted[mask]
        S_final = S_sorted[mask]

        return i_final, j_final, S_final, d_final


    def compute_(self, soap, system, sum_soaps, sum_soaps_dist, sum_soaps_spat, cov_t, cov_t_cum, nsmp, ntimesteps):   
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            sum_soaps[atom_type_idx] += soap[atom_type].sum(axis=0)
            cov_t[atom_type_idx] += np.einsum("ia,ib->ab", soap[atom_type], soap[atom_type]) #sum over all same atoms (have already summed over all times before) 
            a,b,S,d = self.spatial_correlate(system, atom_type) # return nl indexes of center, neighbor and dist
            dist = 1/(4.0*np.pi*d**2)
            sum_soaps_dist[atom_type_idx] += np.sum(dist)
            sum_soaps_spat[atom_type_idx] += np.einsum('m,mk->k', dist, soap[a,:] + soap[b,:]) / 2  # weighted mean over all pairs
            cov_t_cum[atom_type_idx] += np.einsum("n, ni,nj->ij", dist, soap[a,:], soap[b,:]) #sum over all same atoms (have already summed over all times before) 
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
        first_soap_cum = self.descriptor.compute_cumulants(soap_block, 1)

        self.atomsel_element = [[idx for idx, label in enumerate(self.descriptor.soap_block.samples.values.numpy()) if label[2] == atom_type] for atom_type in self.descriptor.centers]
        if soap_block.shape[0] == 1:
            self.atomsel_element = [[0] for atom_type in self.descriptor.centers]

        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        buffer_cum = np.zeros((first_soap.shape[0], self.interval, first_soap_cum.shape[1]))
        cov_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        cov_t_cum = np.zeros((len(self.atomsel_element), first_soap_cum.shape[1], first_soap_cum.shape[1],))
        sum_soaps = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        sum_soaps_spat = np.zeros((len(self.atomsel_element),first_soap_cum.shape[1],))
        sum_soaps_dist = np.zeros((len(self.atomsel_element),first_soap_cum.shape[1],))
        nsmp = np.zeros(len(self.atomsel_element))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2)) # cutoff at 3 sigma, leaves 0.1%
        kernel /= kernel.sum()
        ntimesteps = np.zeros(len(self.atomsel_element), dtype=int)

        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate([system]) #N, S
            if fidx >= self.interval:
                roll_kernel = np.roll(kernel, fidx%self.interval)
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                self.compute_(avg_soap, system, sum_soaps, sum_soaps_dist, sum_soaps_spat, cov_t, cov_t_cum, nsmp, ntimesteps)

            buffer[:,fidx%self.interval,:] = new_soap_values

        if len(systems) == 1:
            self.compute_(first_soap, systems[0], sum_soaps, sum_soaps_dist, sum_soaps_spat, cov_t, cov_t_cum, nsmp, ntimesteps)

        mu = np.zeros((len(self.atomsel_element), new_soap_values.shape[1]))
        cov = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        mu_spat = np.zeros((len(self.atomsel_element), new_soap_values.shape[1]))
        cov_spat = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))

        # autocorrelation matrix - remove mean
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            mu[atom_type_idx] = sum_soaps[atom_type_idx]/nsmp[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            cov[atom_type_idx] = cov_t[atom_type_idx]/nsmp[atom_type_idx] - np.einsum('i,j->ij', mu[atom_type_idx], mu[atom_type_idx])
            # COV = 1/N ExxT - mumuT
            mu_spat[atom_type_idx] = sum_soaps_spat[atom_type_idx] / sum_soaps_dist[atom_type_idx]
            cov_spat[atom_type_idx] = cov_t_cum[atom_type_idx]/sum_soaps_dist[atom_type_idx] - np.einsum('i,j->ij', mu_spat[atom_type_idx], mu_spat[atom_type_idx])

        self.cov_tot = cov
        return mu, cov_spat, cov



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


class SpatialIVAC(FullMethodBase):

    def __init__(self, descriptor, interval, ridge_alpha, spatial_cutoff, spatial_sigma, root):
        self.name = 'SpatialIVACnew'
        self.spatial_cutoff = spatial_cutoff
        super().__init__(descriptor, interval, lag=0, root=root, sigma=spatial_sigma, ridge_alpha=ridge_alpha, method=self.name)



    def make_neighborlist(self, cutoff):
        self.nlist = NeighborList(cutoff=cutoff, full_list=True) # TODO: debatable if full lsit or not

    def spatial_correlate(self, system, sel_atoms):
        pos = system.positions[sel_atoms]
        cell = system.cell
        self.make_neighborlist(cutoff=self.spatial_cutoff) # example cutoff
        i, j, S, d = self.nlist.compute(
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

        i_final = i_sorted[mask]
        j_final = j_sorted[mask]
        d_final = d_sorted[mask]
        S_final = S_sorted[mask]

        return i_final, j_final, S_final, d_final


    def compute_(self, soap, system, sum_soaps, sum_soaps_dist, sum_soaps_spat, cov_t, cov_t_cum, nsmp, ntimesteps):   
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            soap_avg = np.zeros(soap.shape)
            norm = np.zeros(soap.shape[0])
            a,b,S,d = self.spatial_correlate(system, atom_type) # return nl indexes of center, neighbor and dist
            w = np.exp(-(d**2)/(2*self.sigma**2))
            np.add.at(soap_avg, a, w[:, None] * soap[b,:])
            np.add.at(soap_avg, a, soap[a,:])
            np.add.at(norm, a, w)
            np.add.at(norm, atom_type, np.ones(len(atom_type)))
            soap_avg /= norm[:, None] + 1e-12
            sum_soaps[atom_type_idx] += soap_avg[atom_type].sum(axis=0)
            cov_t[atom_type_idx] += np.einsum("ia,ib->ab", soap_avg[atom_type], soap_avg[atom_type]) #sum over all same atoms (have already summed over all times before) 
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
        first_soap_cum = self.descriptor.compute_cumulants(soap_block, 1)

        self.atomsel_element = [[idx for idx, label in enumerate(self.descriptor.soap_block.samples.values.numpy()) if label[2] == atom_type] for atom_type in self.descriptor.centers]
        if soap_block.shape[0] == 1:
            self.atomsel_element = [[0] for atom_type in self.descriptor.centers]

        buffer = np.zeros((first_soap.shape[0], self.interval, first_soap.shape[1]))
        buffer_cum = np.zeros((first_soap.shape[0], self.interval, first_soap_cum.shape[1]))
        cov_t = np.zeros((len(self.atomsel_element), first_soap.shape[1], first_soap.shape[1],))
        cov_t_cum = np.zeros((len(self.atomsel_element), first_soap_cum.shape[1], first_soap_cum.shape[1],))
        sum_soaps = np.zeros((len(self.atomsel_element),first_soap.shape[1],))
        sum_soaps_spat = np.zeros((len(self.atomsel_element),first_soap_cum.shape[1],))
        sum_soaps_dist = np.zeros((len(self.atomsel_element),first_soap_cum.shape[1],))
        nsmp = np.zeros(len(self.atomsel_element))
        delta=np.zeros(self.interval)
        delta[self.interval//2]=1
        kernel=gaussian_filter(delta,sigma=(self.interval-1)//(2)) # cutoff at 3 sigma, leaves 0.1%
        kernel /= kernel.sum()
        ntimesteps = np.zeros(len(self.atomsel_element), dtype=int)

        for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Computing SOAPs"):
            new_soap_values = self.descriptor.calculate([system]) #N, S
            if fidx >= self.interval:
                roll_kernel = np.roll(kernel, fidx%self.interval)
                # computes a contribution to the correlation function
                # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                self.compute_(avg_soap, system, sum_soaps, sum_soaps_dist, sum_soaps_spat, cov_t, cov_t_cum, nsmp, ntimesteps)

            buffer[:,fidx%self.interval,:] = new_soap_values

        if len(systems) == 1:
            self.compute_(first_soap, systems[0], sum_soaps, sum_soaps_dist, sum_soaps_spat, cov_t, cov_t_cum, nsmp, ntimesteps)

        mu = np.zeros((len(self.atomsel_element), new_soap_values.shape[1]))
        cov = np.zeros((len(self.atomsel_element), new_soap_values.shape[1], new_soap_values.shape[1]))
        
        # autocorrelation matrix - remove mean
        for atom_type_idx, atom_type in enumerate(self.atomsel_element):
            mu[atom_type_idx] = sum_soaps[atom_type_idx]/nsmp[atom_type_idx]
            # COV = 1/N ExxT - mumuT
            cov[atom_type_idx] = cov_t[atom_type_idx]/nsmp[atom_type_idx] - np.einsum('i,j->ij', mu[atom_type_idx], mu[atom_type_idx])
            # COV = 1/N ExxT - mumuT
            
        self.cov_tot = cov
        return mu, cov, [np.eye(c.shape[0]) for c in cov]


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

