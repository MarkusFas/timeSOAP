import numpy as np
import chemiscope

from smoothsoap.plots.onion import plot_onion, plot_snapshot
from smoothsoap.plots.timeseries import plot_projection_atoms, plot_projection_atoms_models, plot_projection_CV
from smoothsoap.plots.histograms import plot_2pca, plot_2pca_atoms, plot_2pca_height, plot_histogram, plot_2pca_spatial, plot_2pca_spatial_movie

def post_processing(X, trj_predict, test_atoms, methodname, label, inter,**kwargs):
    print('methodname',methodname, label)
#    print('trj_predict', len(trj_predict), trj_predict)
    plots = kwargs.get("plots", [])
    i_pca = kwargs.get("i_pca")
    if "onion" in plots:
        for i, proj in enumerate(X):
            plot_onion(proj, trj_predict[0], test_atoms, label + f'_{i}', i_pca=i_pca) 
        print(f'Plotted ONION histogram of {methodname} first component')

    if "snapshot" in plots:
        for i, proj in enumerate(X):
            plot_snapshot(proj, trj_predict[0], test_atoms, label + f'_{i}', i_pca=i_pca) 

    if "projection" in plots:
        plot_projection_atoms(X, [0,1,2,3], label, [inter]) # need to transpose to T,N,P
        print(f'Plotted projected timeseries for {methodname}')

    if "pca" in plots:
        for i, proj in enumerate(X):
            plot_2pca(proj, label + f'_{i}')
        print(f'Plotted scatterplot of {methodname}')
    
    if "pca_spatial" in plots:
        for i, proj in enumerate(X):
            plot_2pca_spatial_movie(proj, label + f'_{i}', test_atoms, trj_predict)
        print(f'Plotted spatial of {methodname}')

    if "pca_atoms" in plots:
        for i, proj in enumerate(X):
            plot_2pca_atoms(proj, label + f'_{i}', test_atoms)
        print(f'Plotted scatterplot of {methodname} atoms labels')
    
    if "CV" in plots:
        for i, proj in enumerate(X):
            plot_projection_CV(proj, label + f'_{i}', i_pca=i_pca)
        print(f'Plotted projected CV timeseries for {methodname}')
    
    if "pca_height" in plots:
        for i, proj in enumerate(X):
            plot_2pca_height(proj, label + f'_{i}', test_atoms, trj_predict)
        print(f'Plotted scatterplot of {methodname} height labels')

    if "histogram" in plots:
        for i, proj in enumerate(X):
            plot_histogram(proj, label + f'_{i}', test_atoms, trj_predict, i_pca=i_pca)
        print(f'Plotted histogram of {methodname} first component')

    print('Plots saved at ' + label)

    if "cs" in plots:
        cs = chemiscope.show(trj[0],
            properties={
                "IC[0]": {"target": "atom", "values": X[0][...,0].flatten()},
                "IC[1]": {"target": "atom", "values": X[0][...,1].flatten()},
                "time": {"target": "atom", "values": np.repeat(np.arange(X[0].shape[0]), X[0].shape[1])},
            },
            environments = [[i,j,4] for i in range(X[0].shape[0]) for j in test_atoms], # maybe range(X[0].shape[1])
            settings=chemiscope.quick_settings(periodic=True, trajectory=True, target="atom", map_settings={"joinPoints": False})
        )
        cs.save(label + '_cs.json')
        print("saved chemiscope")
