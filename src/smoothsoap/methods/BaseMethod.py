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
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from smoothsoap.transformations.PCAtransform import PCA_obj


class FullMethodBase(ABC):
    """
    Base class for full descriptor-based slow mode methods.

    Defines a unified interface for training and projecting models
    based on descriptor covariance matrices. Subclasses must implement
    the descriptor-specific covariance computation in `compute_COV()`.
    """

    def __init__(self, descriptor, interval, lag, sigma, ridge_alpha, root, method=None, eps2factor=None):
        self.interval = interval
        self.lag = lag
        self.sigma = sigma
        self.root = os.path.join(root, method)
        self.descriptor = descriptor
        self.transformations = None
        self.ridge_alpha = ridge_alpha
        self.eps2factor = eps2factor
        dir = (
            root
            / method
            / f'interval_{self.interval}'
            / f'lag_{self.lag}'
            / f'sigma_{self.sigma}'
            / f'ridge_a{ridge_alpha}'
        )
        dir.mkdir(parents=True, exist_ok=True)
        self.label = str((dir / self.descriptor.id))
        
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
        self.transformations = [PCA_obj(n_components=4, label=self.label, eps2factor=self.eps2factor) for n in range(self.cov1.shape[0])]

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
        systems = systems_to_torch(traj, dtype=torch.float32)

        projected_per_type = []

        for trafo in self.transformations:
            first_soap = self.descriptor.calculate([systems[0]])
            first_soap = trafo.project(first_soap)
            projected = np.zeros((len(systems), first_soap.shape[0], first_soap.shape[1]))
            for i, system in enumerate(systems):
                descriptor = self.descriptor.calculate([system])
                descriptor = trafo.project(descriptor)
                projected[i] = descriptor
            projected_per_type.append(projected.transpose(1, 0, 2))
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
            old_soap_values = first_soap
            soap_values=np.zeros((first_soap.shape[0],len(systems)-self.interval, first_soap.shape[1]))
            avg_soaps_projs=np.zeros((first_soap.shape[0],len(systems)-self.interval, avg_soap_proj.shape[-1]))
            for fidx, system in tqdm(enumerate(systems), total=len(systems), desc="Fit Ridge"):
                new_soap_values = self.descriptor.calculate([system], selected_samples=self.descriptor.selected_samples)
                #TODO: make more efficient, need buffer?
                if fidx >= self.interval:
                    roll_kernel = np.roll(kernel, fidx%self.interval)
                    roll_delta = np.roll(delta, fidx%self.interval)
                    # computes a contribution to the correlation function
                    # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
                    avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer) #smoothen
                    inst_soap = np.einsum("j,ija->ia", roll_delta, buffer) # instantaneous
                    #avg_soap = old_soap_values
                    avg_soap_proj = trafo.project(avg_soap)
                    #print('projshape', avg_soap_proj.shape)
                    #print('nonprog.shape',new_soap_values.shape)
                    soap_values[:,fidx-self.interval,:] = inst_soap
                    avg_soaps_projs[:,fidx-self.interval,:] = avg_soap_proj
                buffer[:,fidx%self.interval,:] = new_soap_values
                #old_soap_values = new_soap_values
            #soap_values=soap_values.reshape((soap_values.shape[0]*soap_values.shape[1],soap_values.shape[2]))
            #avg_soaps_projs=avg_soaps_projs.reshape((avg_soaps_projs[0]*avg_soaps_projs[1],avg_soaps_projs.shape[2]))
            if len(systems) == 1:
                soap_values = first_soap[:,None,:]
                avg_soaps_projs = trafo.project(first_soap)[:,None,:]

            soap_values=soap_values.reshape(soap_values.shape[0]*soap_values.shape[1],soap_values.shape[2])
            avg_soaps_projs=avg_soaps_projs.reshape(avg_soaps_projs.shape[0]*avg_soaps_projs.shape[1],avg_soaps_projs.shape[2])
            #np.reshape(avg_soaps_projs.soap_values,(avg_soaps_projs.shape[0]*avg_soaps_projs.shape[1],avg_soaps_projs.shape[2]))
            #p.reshape(avg_soaps_projs,(avg_soaps_projs[0]*avg_soaps_projs[1],avg_soaps_projs.shape[2]))
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
                ridge_pred = self.ridge[idx].predict(descriptor)
                projected.append(ridge_pred)
               
            projected_per_type.append(np.stack(projected, axis=0).transpose(1, 0, 2))
        self.get_explained_variance_ridge(traj, selected_atoms)
        return projected_per_type
        

    def make_neighborlist(atoms, cutoff):
        self.nlist = NeighborList(cutoff=cutoff, pbc=True)


    def spatial_average_with_nl(selected_atoms, positions, features, compute_feature_fn, sigma):
        self.nlist.update(positions)
        averaged_features = {}
        self_weight = 1.0
        for i in selected_atoms:
            j, d = self.nlist.get_neighbors(i, return_distances=True)
            if len(j) == 0:
                averaged_features[i] = features[i]
                continue

            w = np.exp(-d**2 / (2*sigma**2))
            self.descriptor_spatial.set_samples(self, selected_atoms)
            feats = np.stack([
                features[selected_atoms[jj]] if jj in selected_atoms else self.descriptor.calculate([system])
                for jj in j
            ])

            weighted_sum = (w[:, None] * feats).sum(axis=0)
            h_i = (weighted_sum + self_weight * features[i]) / (self_weight + w.sum())
            averaged_features[i] = h_i

        return averaged_features

    def get_explained_variance(self, traj, selected_atoms):
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
        for i, trafo in enumerate(self.transformations):
            proj_sum = np.zeros(trafo.n_components)
            proj_scatter = np.zeros(trafo.n_components)
            x_scatter = np.zeros(trafo.eigvecs.shape[0])
            x_sum = np.zeros(trafo.eigvecs.shape[0])
            x_var = 0
            n = 0
            n_proj = 0
            for system in systems:
                descriptor = self.descriptor.calculate([system])
                descriptor_proj = trafo.project(descriptor)
                # for temporal only
                descriptor = np.sum(descriptor, axis=0)
                descriptor_proj = np.sum(descriptor_proj, axis=0)

                #x_sum += np.sum(descriptor, axis=0)
                x_sum += descriptor
                #x_scatter += np.einsum('ij,ij->j', descriptor, descriptor)
                x_scatter += descriptor*descriptor
                # proj_sum += descriptor_proj.sum(axis=0)
                proj_sum += descriptor_proj
                #proj_scatter += np.einsum('ij,ij->j', descriptor_proj, descriptor_proj)
                proj_scatter += descriptor_proj*descriptor_proj
                #n_proj += descriptor_proj.shape[0]
                n_proj += 1
                #n += descriptor.shape[0]
                n += 1

            proj_mean = proj_sum/n_proj
            proj_var = proj_scatter/n_proj - proj_mean*proj_mean
            x_mean = x_sum/n
            x_var = x_scatter/n - x_mean*x_mean
            #x_var = np.sum(x_var)
            #x_var = proj_scatter/n - proj_mean*proj_mean
            EVR = proj_var / np.sum(x_var)
            torch.save(
                torch.tensor(EVR),
                self.label + f"_center{self.descriptor.centers[i]}" + f"EVR.pt",
            )
            print('Explained variance ratio: ', EVR)


    def get_explained_variance_ridge(self, traj, selected_atoms):
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
        for i, trafo in enumerate(self.transformations):
            proj_sum = np.zeros(trafo.n_components)
            proj_scatter = np.zeros(trafo.n_components)
            x_scatter = np.zeros(trafo.eigvecs.shape[0])
            x_sum = np.zeros(trafo.eigvecs.shape[0])
            x_var = 0
            n = 0
            n_proj = 0
            for system in systems:
                descriptor = self.descriptor.calculate([system])
                descriptor_proj = self.ridge[i].predict(descriptor)
                descriptor -= trafo.mu
                # for temporal only
                descriptor = np.sum(descriptor, axis=0)
                descriptor_proj = np.sum(descriptor_proj, axis=0)

                #x_sum += np.sum(descriptor, axis=0)
                x_sum += descriptor
                #x_scatter += np.einsum('ij,ij->j', descriptor, descriptor)
                x_scatter += descriptor*descriptor
                # proj_sum += descriptor_proj.sum(axis=0)
                proj_sum += descriptor_proj
                #proj_scatter += np.einsum('ij,ij->j', descriptor_proj, descriptor_proj)
                proj_scatter += descriptor_proj*descriptor_proj
                #n_proj += descriptor_proj.shape[0]
                n_proj += 1
                #n += descriptor.shape[0]
                n += 1

            proj_mean = proj_sum/n_proj
            proj_var = proj_scatter/n_proj - proj_mean*proj_mean
            x_mean = x_sum/n
            x_var = x_scatter/n - x_mean*x_mean
            #x_var = np.sum(x_var)
            #x_var = proj_scatter/n - proj_mean*proj_mean
            EVR = proj_var / np.sum(x_var)
            torch.save(
                torch.tensor(EVR),
                self.label + f"_center{self.descriptor.centers[i]}" + f"EVR_ridge.pt",
            )
            print('Explained ridge variance ratio: ', EVR)



    def spatial_averaging(self, system, features, sigma):
        averaged_features = {}
        selected_atoms = self.selected_atoms
        points = system.positions #[self.selected_atoms]
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

        positions = system.positions
        cutoff = 5.0
        for i in selected_atoms:
            # Find neighbors within cutoff
            ri = positions[i]
            dvec = positions - ri
            dist = np.linalg.norm(dvec, axis=1)
            mask = (dist < cutoff) & (dist > 0.0)
            neighbors = np.where(mask)[0]

            # Compute Gaussian weights
            w = np.exp(-dist[neighbors]**2 / (2 * sigma**2))

            # Collect neighbor features
            neighbor_features = []
            for j in neighbors:
                if j in features:
                    f_j = features[j]
                else:
                    f_j = compute_feature_fn(j)
                    features[j] = f_j  # cache for later reuse
                neighbor_features.append(f_j)

            neighbor_features = np.array(neighbor_features)

            # Compute weighted average
            self_weight = 1.0
            weighted_sum = (w[:, None] * neighbor_features).sum(axis=0)
            h_i = (weighted_sum + self_weight * features[i]) / (self_weight + w.sum())

            averaged_features[i] = h_i

        return averaged_features




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

        """# TODO: check which of the atoms in the cutoff have already soaps
        neighbors = {}
        neighbor_soaps = []
        for atom_idx in self.selected_atoms:
            
            sel_samples = Labels(
                names=["atom"],
                values=torch.tensor(j[i == atom_idx], dtype=torch.int64).unsqueeze(-1),
            )
       
            for neigh_idx in j[i == atom_idx]:
                if neigh_idx in self.selected_atoms: # as for those we already have a precomputed SOAP
                    neighbor_soaps.append(features[np.where(self.selected_atoms == neigh_idx)[0][0]])
                else:
                    neighbor_soaps.append(self.descriptor.calculate([system], sel_samples)) #TODO"""
        # compute for those without
        # get distance 
        w = np.exp(-d**2 / (2*sigma**2), dtype=np.float64)     # pairwise weights
        # add self-weight term separately later
        self_weight = 1.0
        h = np.zeros_like(features)             # (N_atoms, n_features)
        np.add.at(h, i, w[:, None] * features[j])

        h += self_weight * features
        h /= (self_weight + w.sum())
        # get calculate the gaussian weight

        # normalize vs all 

        # return the averaged features 
        return h
        

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

    def get_label(self, systems, sel_atoms, switch_index):
        # This assumes that all frames have the same number of atoms, could be extended
        labels = np.zeros((len(systems), len(sel_atoms)), dtype=np.int64)
        labels[switch_index:, :] = 1
        return labels

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
