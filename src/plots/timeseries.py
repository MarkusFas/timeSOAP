import matplotlib.pyplot as plt
import mpltex
import numpy as np


@mpltex.acs_decorator
def plot_projection_atoms(X_values, PCA_idx, label, intervals):
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

