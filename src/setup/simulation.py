import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
from random import shuffle

from src.old_scripts.descriptors import SOAP_mean, SOAP_full
from src.transformations.PCAtransform import pcatransform, PCA_obj
from src.old_scripts.fourier import fourier_trafo
from src.visualize import plot_compare_atoms, plot_compare_timeave
from src.plots.timeseries import plot_projection_atoms


def run_simulation(trj, used_methods, run_ids, run_dirs, **kwargs):

    for i, method in tqdm(enumerate(used_methods)):
        label = run_ids[i]
        run_dir = run_dirs[i]
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        label = os.path.join(run_dir, label)

        selected_atoms = [idx for idx, number in enumerate(trj[0].get_atomic_numbers()) if number==method.descriptor.centers[0]]
        shuffle(selected_atoms) 
        N_train = kwargs.get('train_selected_atoms')
        N_test = kwargs.get('test_selected_atoms')
        train_atoms = selected_atoms[:N_train]
        test_atoms = selected_atoms[N_train:N_test]

        # train our method by specifying the selected atoms
        method.train(trj, train_atoms)

        # get predictions with the new representation
        X = method.predict(trj, test_atoms) ##centers N,T,P

        #4 Post processing
        plots = kwargs.get("plots", [])

        if "projection" in plots:
            plot_projection_atoms(X, [0,1,2,3], label, [method.interval]) # need to transpose to T,N,P
            print('Plotted projected timeseries for test atoms')

        if "histogram" in plots:
            raise NotImplementedError('histogram plotting will be implemnted soon')
        