import matplotlib.pyplot as plt
import mpltex
import numpy as np

@mpltex.acs_decorator
def plot_soap_atoms(X_values, SOAP_idx, label, properties, intervals):
    # X is T,N,D
    # plot different models for time averageing
    for n, X in enumerate(X_values):
        fig, ax = plt.subplots(1,len(SOAP_idx),figsize=( 4*len(SOAP_idx), 4), squeeze=False)
        # plot different atoms
        
        for j, soap_i in enumerate(SOAP_idx):
            ax[0,j].set_title(fr'$l$ {properties[soap_i,2]}, $n_1$ {properties[soap_i,3]}, $n_2$ {properties[soap_i,4]}')
            ax[0,j].set_xlabel('steps')
            mean = np.mean(X[:,:,soap_i], axis=1)
            std = np.std(X[:,:,soap_i], axis=1)
            ax[0,j].plot(mean, alpha=1.0, color='C1')
            ax[0,j].fill_between(np.arange(len(mean)), mean-std, mean+std, alpha=0.1, color='C0')
            for i in range(X.shape[1]):
                ax[0,j].plot(X[:,i, soap_i], label=f'atom {i}', alpha=0.2, color='C0')
            ax[0,j].legend()
        plt.tight_layout()
        plt.savefig(label+f'_SOAP_compare_atoms_interval_{intervals[n]}.png', dpi=200)
        plt.close()

@mpltex.acs_decorator
def plot_projection_atoms(X_values, PCA_idx, label, intervals):
    print('plot_compare_atoms_PCA')
    # X is T,N,D
    # plot different models for time averageing
    for n, X in enumerate(X_values):
        print(n)
        fig, ax = plt.subplots(1,len(PCA_idx),figsize=( 4*len(PCA_idx), 4), squeeze=False)
        # plot different atoms
        
        for j, soap_i in enumerate(PCA_idx):
            ax[0,j].set_title(fr'PCA component {soap_i}')
            ax[0,j].set_xlabel('steps')
            mean = np.mean(X[:,:,soap_i], axis=1)
            std = np.std(X[:,:,soap_i], axis=1)
            ax[0,j].plot(mean, alpha=1.0, color='C1')
            ax[0,j].fill_between(np.arange(len(mean)), mean-std, mean+std, alpha=0.25, color='C0')
            for i in range(X.shape[1]):
                ax[0,j].plot(X[:,i, soap_i], label=f'atom {i}', alpha=0.05, color='C0')
            
        plt.tight_layout()
        plt.savefig(label+f'_PCA_compare_atoms_interval_{intervals[n]}.png', dpi=200)
        plt.close()

@mpltex.acs_decorator
def plot_projection_atoms_models(X_values, PCA_idx, label, intervals):
    print('plot_compare_timeave_PCA')
    # X is T,N,D
    fig, ax = plt.subplots(X_values[0].shape[1],len(PCA_idx),figsize=( 4*len(PCA_idx), 4*X_values[0].shape[1]), squeeze=False)
    # plot different models for time averageing
    for n, X in enumerate(X_values):
        # plot different atoms
        for i in range(X.shape[1]):
            for j, soap_i in enumerate(PCA_idx):
                if j==0:
                    ax[i,j].set_ylabel(f'Atom {i}')
                if i==0:
                    ax[i,j].set_title(f'PCA component {soap_i}')
                elif i==X.shape[1]-1:
                    ax[i,j].set_xlabel('steps')
                ax[i,j].plot(X[:,i, soap_i], label=f'lag {intervals[n]}', alpha=0.3)
                ax[0,j].legend()
    plt.tight_layout()
    plt.savefig(label+'_PCAvectors.png', dpi=200)
    plt.close()

