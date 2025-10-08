import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
from random import shuffle

from src.plots.timeseries import plot_projection_atoms, plot_projection_atoms_models
from src.plots.histograms import plot_2pca

def run_simulation_test(trj, used_methods, run_ids, run_dirs, **kwargs):

    pca=[]
    for i, method in tqdm(enumerate(used_methods)):
        # create labels and directories for results 
        label = run_ids[i]
        run_dir = run_dirs[i]
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        label = os.path.join(run_dir, label)

        selected_atoms = [idx for idx, number in enumerate(trj[0].get_atomic_numbers()) if number==method.descriptor.centers[0]]
        shuffle(selected_atoms) 
        N_train = kwargs.get('train_selected_atoms')
        N_test = kwargs.get('test_selected_atoms')
        train_atoms = selected_atoms[:N_train]
        test_atoms = selected_atoms[N_train: N_train + N_test]

        # train our method by specifying the selected atoms
        method.train(trj, train_atoms)

        # get predictions with the new representation
        X = method.predict(trj, test_atoms) ##centers T,N,P
        X = [proj.transpose(1,0,2) for proj in X]
        
        pca.append(method.transformations[0].eigvecs)
        #4 Post processing
    
    print(pca[0] - pca[1])

if __name__ == '__main__':
    print('Nothing to do here')