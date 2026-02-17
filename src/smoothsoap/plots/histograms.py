import matplotlib.pyplot as plt
import mpltex
import numpy as np
from matplotlib.animation import FuncAnimation

@mpltex.acs_decorator
def plot_2pca(X, label): #T,N,P
    fig, ax = plt.subplots(1,1, figsize=(12,6))
    for i, trj in enumerate(X.transpose(1,0,2)): #N, T, P
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
    plt.savefig(label + f'_projection.png', dpi=200)
    plt.close()

@mpltex.acs_decorator
def plot_2pca_atoms(X, label, sel_atoms):
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    for i, trj in enumerate(X.transpose(1,0,2)): # shape N, T, P
        sc = ax.scatter(
            trj[:,0], 
            trj[:,1], #first PCA component
            color='black',
            alpha=0.1,
            s=0.5,
        )
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
    plt.tight_layout()  
    plt.savefig(label + f'_projection_atoms.png', dpi=200)
    plt.close()

@mpltex.acs_decorator
def plot_2pca_height(X, label, sel_atoms, traj):
    fig, ax = plt.subplots(1,1, figsize=(12,6))
    positions = np.stack([atoms.positions[sel_atoms, :] for atoms in traj])
    for i, trj in enumerate(X.transpose(1,0,2)): # shape N, T, P
        #color = 'green' if sel_atoms[i] > 256*3//2 else 'orange'
        color = np.arange(len(trj[:,0]))
        ax.scatter(
            trj[:,0], 
            positions[:,i, 2], #first PCA component
            c=color, 
            cmap='RdYlBu', 
            vmin=0, #np.min(color), 
            vmax=len(trj[:,0]), #np.max(color),
            alpha=0.2,
            s=2.5,
        )
    plt.savefig(label + f'_projection_height.png', dpi=200)
    plt.close()


def plot_2pca_spatial(X, label, sel_atoms, traj):
    fig, ax = plt.subplots(1,1, figsize=(12,6))
    pca = X[...,0] # first PC
    color = pca[0]
    positions = np.stack([atoms.positions[sel_atoms, :] for atoms in traj])
    ax.scatter(
        positions[0,:,0], 
        positions[0,:,1], 
        c=color, 
        cmap='RdYlBu', 
        vmin=np.min(color), 
        vmax=np.max(color),
        alpha=0.2,
        s=2.5,
    )
    plt.savefig(label + f'_projection_height.png', dpi=200)
    plt.close()

def plot_2pca_spatial_movie(X, label, sel_atoms, traj, interval=300):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # PCA values: shape assumed (n_frames, n_atoms, n_pcs) OR similar
    pca = X[..., 0]   # first PC over time

    # Atom positions over trajectory
    positions = np.stack([atoms.positions[sel_atoms, :] for atoms in traj])
    # positions.shape -> (n_frames, n_atoms, 3)

    n_frames = 100
    traj_idx = np.linspace(0, pca.shape[0] - 1, n_frames, dtype=int)
    movie_duration = 10.0  # seconds
    interval = movie_duration / n_frames * 1000  # ms

    # Initial frame
    color0 = pca[0]

    scat = ax.scatter(
        positions[traj_idx[0], :, 0],
        positions[traj_idx[0], :, 1],
        c=color0,
        cmap="RdYlBu",
        vmin=pca.min(),
        vmax=pca.max(),
        alpha=0.2,
        s=2.5,
    )

    cbar = plt.colorbar(scat, ax=ax)
    ax.set_title("PCA frame 0")

    if pca.shape[0] == 1:
        plt.savefig(label + "_spatial.png")
        plt.close()
        return 0
    
    def update(k):
        i = traj_idx[k]

        # update positions
        scat.set_offsets(positions[i, :, :2])

        # update colors
        scat.set_array(pca[i])

        ax.set_title(f"Frame {i}")
        return scat,

    ani = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=interval,
        blit=False
    )

    ani.save(label + "_projection_height.mp4", writer="ffmpeg", dpi=200)
    plt.close(fig)




@mpltex.acs_decorator
def plot_histogram(X, label, sel_atoms, traj , i_pca=0):
    fig, ax = plt.subplots(1,1, figsize=(12,6))
    positions = np.array([atoms.positions[sel_atoms, 2] for atoms in traj])
    T, N, P = X.shape
    data = X.reshape(-1, P)
    ax.hist(data[:,i_pca], bins=100, alpha=0.7)
    plt.savefig(label + f'_histogram.png', dpi=200)
    #plt.savefig(label + f'_histogram.png', dpi=200)
    plt.close()
