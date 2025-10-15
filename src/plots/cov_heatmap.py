import matplotlib.pyplot as plt
import numpy as np
import mpltex

@mpltex.acs_decorator
def plot_heatmap(cov1, cov2, label):
    diff = np.abs(cov1 - cov2)
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    cm = ax.imshow(diff, cmap='hot', interpolation='nearest')
    plt.colorbar(cm)
    plt.savefig(label + f'_cov_heatmap.png', dpi=200)