import os
import random

import numpy as np
from tqdm import tqdm
from pathlib import Path
from itertools import chain

from src.plots.cov_heatmap import plot_heatmap
from src.plots.timeseries import plot_projection_atoms, plot_projection_atoms_models
from src.plots.histograms import plot_2pca


def run_simulation(trj, methods_intervals, **kwargs):
    print(np.shape(trj))
    if len(np.shape(trj)) == 2:
        trj = [trj]
    print(np.shape(trj))
    for i, methods in tqdm(enumerate(methods_intervals), desc="looping through intervals"):
        for j, method in tqdm(enumerate(methods), desc="looping through methods"):
            random.seed(7)
            # create labels and directories for results
            
        
            N_train = kwargs.get('train_selected_atoms')
            N_test = kwargs.get('test_selected_atoms')
            is_shuffled = False
            if isinstance(N_train , int):
                selected_atoms = [idx for idx, number in enumerate(trj[0][0].get_atomic_numbers()) if number==method.descriptor.centers[0]]
                random.shuffle(selected_atoms) 
                train_atoms = selected_atoms[:N_train]
                is_shuffled = True
            else:
                train_atoms = N_train
            if isinstance(N_test , int):
                selected_atoms = [idx for idx, number in enumerate(trj[0][0].get_atomic_numbers()) if number==method.descriptor.centers[0]]
                if not is_shuffled:
                    random.shuffle(selected_atoms)
                    test_atoms = selected_atoms[:N_test]
                else:
                    test_atoms = selected_atoms[10+N_train: 10+N_train + N_test]
                    test_atoms = selected_atoms[:N_test] # single atom case
            else:
                test_atoms = N_test
        

            # train our method by specifying the selected atoms
            method.train(trj, train_atoms)

            method.log_metrics()
            
                
            # get predictions with the new representation
            # for prediction we can use the concatenated trajectories

            trj_predict = list(chain(*trj))
            X = method.predict(trj_predict, test_atoms) ##centers T,N,P
            X = [proj.transpose(1,0,2) for proj in X]
            
            #4 Post processing
            plots = kwargs.get("plots", [])

            if "projection" in plots:
                plot_projection_atoms(X, [0,1,2,3], method.label, [method.interval]) # need to transpose to T,N,P
                #plot_projection_atoms_models(X, [0,1,2,3], label, [method.interval])
                print('Plotted projected timeseries for test atoms')

            if "pca" in plots:
                for i, proj in enumerate(X):
                    plot_2pca(proj, method.label + f'_{i}')
                print('Plotted scatterplot of PCA')

            print('Plots saved at ' + method.label)

    if "heatmap" in plots and len(methods_intervals) >= 2:
        interval_0 = methods_intervals[0]
        interval_1 = methods_intervals[1]
        cov1_int0 = interval_0[0].cov_mu_t
        cov2_int0 = interval_0[0].mean_cov_t
        cov1_int1 = interval_1[0].cov_mu_t
        cov2_int1 = interval_1[0].mean_cov_t
        for i, center in enumerate(interval_0[0].descriptor.centers):
            plot_heatmap(cov1_int0[i], cov1_int1[i], method.root + f'_temporal_interval{interval_0[0].interval}{interval_1[0].interval}_center{center}' + f'_{i}')
            plot_heatmap(cov2_int0[i], cov2_int1[i], method.root + f'_spatial_interval{interval_0[0].interval}{interval_1[0].interval}_center{center}' + f'_{i}')
        print('Plotted heatmap')
 

if __name__ == '__main__':
    print('Nothing to do here')