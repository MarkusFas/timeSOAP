import numpy as np
from scipy.linalg import eigh
import torch 

def pcatransform(X, NCOMPONENTS=4):
    
    N, T, S = X.shape
    eps = 1e-10

    # 1. Per-atom mean: shape (N, S)
    mu = X.mean(axis=1)

    # 3. Centered data
    X_centered = X - mu[:, None, :]  # (N, 2T, S)

    # 4. Per-atom covariance (over tine) matrices using einsum
    covariances = np.einsum('nts,ntk->nks', X_centered, X_centered) / (2*T - 1)

    # 5. Average over atoms
    #avg_cov = covariances.mean(axis=0)

    # 6. Eigenvalue decomposition of covariance matrix for each atom
    eigvals = [None for _ in range(covariances.shape[0])]
    eigvecs = [None for _ in range(covariances.shape[0])]
    for n, avg_cov in enumerate(covariances):
        eigvals[n], eigvecs[n] = eigh(avg_cov + eps * np.eye(S))  # Regularize

    # Sort eigenvectors by descending eigenvalue
    eigvals = np.array(eigvals)[:, ::-1]
    eigvecs = np.array(eigvecs)[:, :, ::-1]

    # 7. Project centered test data: center using mean of train data
    mu_global = mu.mean(axis=0)  # (S,)
    #X_test_centered = X_test - mu_global[None, None, :]
    projected = np.einsum('nts,nsk->ntk', X_centered, eigvecs[:, :, :NCOMPONENTS])
    return eigvals, eigvecs, projected, avg_cov

class PCA_obj:
    def __init__(self, n_components, label):
        COV = None
        self.eigvecs = None
        self.eigvals = None
        self.n_components = n_components
        self.run_label = label

    def compute_eigen(self, mu, COV):
        eps = 1e-10
        self.eigvals, self.eigvecs = eigh(COV + eps * np.eye(COV.shape[0]))
        # reorder so that largest EV is first
        self.mu = mu
        self.eigvals = self.eigvals[::-1]
        self.eigvecs = self.eigvecs[:,::-1]
    
    def solve_GEV(self, mu, COV_1, COV_2):
        eps_factors = [1e-10, 1e-9, 1e-8, 1e-7]
        eps1 = 1E-8 * np.trace(COV_1) / COV_1.shape[0]
        COV_1_reg = 0.5*(COV_1 + COV_1.T) + eps1*np.eye(COV_1.shape[0])
        
        factor = 1E-10
        while(factor <= 1E-7):
            eps2 = factor * np.trace(COV_2) / COV_2.shape[0]
            COV_2_reg = 0.5*(COV_2 + COV_2.T) + eps2*np.eye(COV_2.shape[0])
            try:
                self.eigvals, self.eigvecs = eigh(COV_1_reg, COV_2_reg)
            except np.linalg.LinAlgError:
                factor *= 10

        print(f'used a factor of {factor} for regularization')
        # reorder so that largest EV is first
        self.mu = mu
        self.eigvals = self.eigvals[::-1]
        self.eigvecs = self.eigvecs[:,::-1]

    def compute_eigen_NEW(self, mu, COV_1, COV_2):
        eps = 1e-10
        eps1 = 1e-8 * np.trace(COV_1) / COV_1.shape[0]
        COV_1_reg = 0.5*(COV_1 + COV_1.T) + eps1*np.eye(COV_1.shape[0])
            
        eps2 = 1e-8 * np.trace(COV_2) / COV_2.shape[0]
        COV_2_reg = 0.5*(COV_2 + COV_2.T) + eps2*np.eye(COV_2.shape[0])
            
        self.eigvals, self.eigvecs = eigh(COV_1_reg, COV_2_reg)
        # reorder so that largest EV is first
        self.mu = mu
        self.eigvals = self.eigvals[::-1]
        self.eigvecs = self.eigvecs[:,::-1]

    def trafo(self, X):
        # X could be matrix or array, for both:
        Y = np.einsum('ij,jk->ik',(X - self.mu), self.eigvecs[:,:self.n_components])
        return Y

    def project(self, X):
        # X could be matrix or array, for both:
        Y = np.einsum('ij,jk->ik',(X - self.mu), self.eigvecs[:,:self.n_components])
        return Y
    
    def save(self):
        torch.save(torch.from_numpy(self.eigvecs.copy()),f'{self.run_label}_pca_eigvecs_timeavg.pt')
        torch.save(torch.from_numpy(self.eigvals.copy()),f'{self.run_label}_pca_eigvals_timeavg.pt')
        torch.save(torch.from_numpy(self.mu.copy()),f'{self.run_label}_pca_mu_timeavg.pt')

    def load(self):
        self.eigvecs = torch.load(f'{self.run_label}_pca_eigvecs_timeavg.pt').numpy()
        self.eigvals = torch.load(f'{self.run_label}_pca_eigvals_timeavg.pt').numpy()
        self.mu = torch.load(f'{self.run_label}_pca_mu_timeavg.pt').numpy()


class TICA_obj:
    def __init__(self, n_components, label):
        COV = None
        self.eigvecs = None
        self.eigvals = None
        self.n_components = n_components
        self.run_label = label

    def compute_eigen(self, mu, COV, norm):
        eps = 1e-10
        self.eigvals, self.eigvecs = eigh(COV + eps * np.eye(COV.shape[0]), norm)
        # reorder so that largest EV is first
        self.mu = mu
        self.eigvals = self.eigvals[::-1]
        self.eigvecs = self.eigvecs[::-1]

    def trafo(self, X):
        # X could be matrix or array, for both:
        Y = self.eigvecs[:self.n_components] @ (X - self.mu).T 
        return Y.T
    
    def save(self):
        torch.save(torch.from_numpy(self.eigvecs.copy()),f'pca_eigvecs_timeavg_{self.run_label}.pt')
        torch.save(torch.from_numpy(self.eigvals.copy()),f'pca_eigvals_timeavg_{self.run_label}.pt')
        torch.save(torch.from_numpy(self.mu.copy()),f'pca_mu_timeavg_{self.run_label}.pt')

    def load(self):
        self.eigvecs = torch.load(f'pca_eigvecs_timeavg_{self.run_label}.pt').numpy()
        self.eigvals = torch.load(f'pca_eigvals_timeavg_{self.run_label}.pt').numpy()
        self.mu = torch.load(f'pca_mu_timeavg_{self.run_label}.pt').numpy()
        

if __name__=='__main__':
    print('Nothing to do here')

    A = np.array([1,2,3])
    B = np.array([[1,0,0],[0,1,0],[0,0,1]])
    C =B[:2] @ A
    D = np.einsum('ij,j->i', B[:2],A)
    print(C)
    print(D)
    print(np.eye(B))


   