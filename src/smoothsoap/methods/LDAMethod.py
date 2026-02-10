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

from smoothsoap.transformations.PCAtransform import PCA_obj
from smoothsoap.methods.BaseMethod import FullMethodBase


class LDA(FullMethodBase):

    def __init__(self, descriptor, interval, ridge_alpha, root):
        self.name = 'LDA'
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
        self.cov1 = 0.5*(class_cov1 + between_cov1.transpose(0,2,1))
        eps1 = [1E-8 * np.trace(COV_1) / COV_1.shape[0] for COV_1 in self.cov1]
        COV_1_reg = [0.5*(COV_1 + COV_1.T) + eps1[i]*np.eye(COV_1.shape[0]) for i, COV_1 in enumerate(self.cov1)]
        self.cov2 = 0.5*(class_cov2 + between_cov2.transpose(0,2,1))
        eps2 = [1E-8 * np.trace(COV_1) / COV_1.shape[0] for COV_1 in self.cov1]
        COV_2_reg = [0.5*(COV_2 + COV_2.T) + eps2[i]*np.eye(COV_2.shape[0]) for i, COV_2 in enumerate(self.cov2)]
        
        #N1 = len()
        # arithmetic mean of covariances
        #self.inclass = (class_cov1*traj_N[0] + class_cov2*traj_N[1]) / total_N
        # harmonic mean of covariances
        self.inclass = np.array([2 * np.linalg.inv(np.linalg.inv(cov1) + np.linalg.inv(cov2)) for cov1, cov2 in zip(COV_1_reg, COV_2_reg)])
        diff = (traj_means[0]*traj_N[0] - traj_means[1]*traj_N[1]) / total_N 
        self.class_diff = np.einsum('ci,cj->cij', diff, diff)
        # Example: use PCA-based transformation for each center
        self.transformations = [PCA_obj(n_components=4, label=self.label) for n in range(self.cov1.shape[0])]
        for i, trafo in enumerate(self.transformations):
            trafo.solve_GEV(self.mean[i], self.class_diff[i], self.inclass[i])

    def train(self, trajs, selected_atoms):
        self.selected_atoms = selected_atoms
        self.descriptor.set_samples(selected_atoms)
        self.compute_COV(trajs[0])

        class_means = []
        class_covs = []
        for traj in trajs:
            mean, cov1, cov2 = self.compute_COV(traj)
            class_means.append(mean)
            class_covs.append(cov1 + cov2)
        
        # total mean
        self.mean = np.mean(class_means, axis=0)

        # between-class covariance
        self.class_diff = np.einsum('nyi,nyj->yij', class_means - self.mean, class_means - self.mean) / len(trajs)

        # within-class covariance
        within_cov = np.mean(class_covs, axis=0)
        eps = 1E-8 * np.array([np.trace(cov) / cov.shape[0] for cov in within_cov])
        self.inclass = np.array([0.5*(cov + cov.T) + eps[i]*np.eye(cov.shape[0]) for i, cov in enumerate(within_cov)])

        # solve generalized eigenvalue problem
        self.transformations = [PCA_obj(n_components=4, label=self.label) for _ in range(self.class_diff.shape[0])]
        for i, trafo in enumerate(self.transformations):
            trafo.solve_GEV(self.mean[i], self.class_diff[i], self.inclass[i])

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

    def log_metrics(self):
        """
        Log metrics from the run, including the covariances.

        
        Returns
        -------
        empty
        """
        metrics = np.array([[np.trace(class_diff), np.trace(inclass)] 
                    for class_diff, inclass in zip(self.class_diff, self.inclass)])
        header = ["classdiff", "inclass"]

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