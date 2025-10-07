import matplotlib.pyplot as plt
import mpltex
import numpy as np


@mpltex.acs_decorator
def plot_2pca(X, label):
    fig, ax = plt.subplots(1,1, figsize=(12,6))
    for i, trj in enumerate(X.transpose(1,0,2)):
        color = np.arange(len(trj[:,0]))
        sc = ax.scatter(
            trj[:,0], 
            trj[:,1], #first PCA component
            c=color, 
            cmap='RdYlBu', 
            vmin=0, #np.min(color), 
            vmax=len(trj[:,0]), #np.max(color),
            alpha=0.2,
            s=2.5,
        ) 
        
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('z')
    plt.savefig(label + f'_2dhistogram.png', dpi=200)
