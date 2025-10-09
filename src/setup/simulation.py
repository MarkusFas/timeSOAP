import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
import random

from src.plots.timeseries import plot_projection_atoms, plot_projection_atoms_models
from src.plots.histograms import plot_2pca

random.seed(7)

def run_simulation(trj, used_methods, run_ids, run_dirs, **kwargs):

    for i, method in tqdm(enumerate(used_methods)):
        random.seed(1)
        # create labels and directories for results 
        label = run_ids[i]
        run_dir = run_dirs[i]
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        label = os.path.join(run_dir, label)
        N_train = kwargs.get('train_selected_atoms')
        N_test = kwargs.get('test_selected_atoms')
        selected_atoms = [idx for idx, number in enumerate(trj[0].get_atomic_numbers()) if number==method.descriptor.centers[0]]
        random.shuffle(selected_atoms) 
        train_atoms = selected_atoms[:N_train]
        test_atoms = selected_atoms[N_train: N_train + N_test]

        # train our method by specifying the selected atoms
        method.train(trj, train_atoms)

        metrics = np.array([np.trace(method.mean_cov_t[0]),
                    np.trace(method.cov_mu_t[0])])
        header = ["spatialCov", "tempCov"]

        # Make metrics a 2D row vector: shape (1, 2)
        np.savetxt(
            label + "_.csv",
            metrics.reshape(1, -1),
            fmt="%.6f",
            delimiter="\t",
            header="\t".join(header),
            comments=""
        )
            
        # get predictions with the new representation
        X = method.predict(trj, test_atoms) ##centers T,N,P
        X = [proj.transpose(1,0,2) for proj in X]
        
        #4 Post processing
        plots = kwargs.get("plots", [])

        if "projection" in plots:
            plot_projection_atoms(X, [0,1,2,3], label, [method.interval]) # need to transpose to T,N,P
            #plot_projection_atoms_models(X, [0,1,2,3], label, [method.interval])
            print('Plotted projected timeseries for test atoms')

        if "histogram" in plots:
            for i, proj in enumerate(X):
                plot_2pca(proj, label + f'_{i}')
            print('Plotted histogram')

        print('Plots saved at ' + label)

if __name__ == '__main__':
    print('Nothing to do here')