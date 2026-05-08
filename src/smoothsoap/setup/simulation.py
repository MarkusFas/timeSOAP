import os
import random
import gc 
import metatomic.torch as mta
import numpy as np
from tqdm import tqdm
from pathlib import Path
from itertools import chain
import chemiscope

from smoothsoap.plots.cov_heatmap import plot_heatmap
from smoothsoap.plots.post_processing import post_processing
from smoothsoap.classifier.Logreg import run_logistic_regression
import torch
from metatomic.torch import systems_to_torch, ModelEvaluationOptions, ModelOutput, load_atomistic_model
from metatensor.torch import Labels
import datetime
from vesin.metatomic import compute_requested_neighbors


def split_train_test(trj, trj_test, kwargs, is_shared, randomseed=7):
    random.seed(randomseed)
    # create labels and directories for results

    N_train = kwargs.get('train_selected_atoms')
    N_test = kwargs.get('test_selected_atoms')
    descriptor_centers=kwargs['SOAP_params']['centers']
    is_shuffled = False
    if isinstance(N_train , int):
        selected_atoms = [idx for idx, number in enumerate(trj[0][0].get_atomic_numbers()) if number==descriptor_centers]
        random.shuffle(selected_atoms) 
        train_atoms = selected_atoms[:N_train]
        is_shuffled = True
    else:
        train_atoms = N_train
    if isinstance(N_test , int):
        if is_shared:
            if not is_shuffled:
                selected_atoms = [idx for idx, number in enumerate(trj_test[0][0].get_atomic_numbers()) if number==descriptor_centers]
                random.shuffle(selected_atoms)
            test_atoms = selected_atoms[-N_test:]
        else:
            print('test from shuffled')
            selected_atoms = [idx for idx, number in enumerate(trj_test[0][0].get_atomic_numbers()) if number==descriptor_centers]
            random.shuffle(selected_atoms)
            test_atoms = selected_atoms[-N_test:] # single atom case
    else:
        test_atoms = N_test
    if test_atoms is None:
        test_atoms = train_atoms
    train_atoms = sorted(train_atoms)
    test_atoms = sorted(test_atoms)
    #print("WARNING SAME TEST AND TRAIN ATOMS")
    #test_atoms = train_atoms
    # train our method by specifying the selected atoms
    return train_atoms, test_atoms

def do_ridge_fit(method, systems, systems_predict, test_atoms):
    print('Starting to fit the Ridge ...')
    #print('fit ridge 2')
    systems_ridge = list(chain(*systems))
    method.fit_ridge_nonincremental(systems_ridge)
    del systems_ridge
    print('Finished the Ridge fit')
    #X_ridge = method.predict_ridge(trj[0], train_atoms)
    print('Starting Ridge prediction ...')
    X_ridge = method.predict_ridge(systems_predict, test_atoms)
    X_ridge = [proj.transpose(1,0,2) for proj in X_ridge]
    print('Finished Ridge prediction')
    return X_ridge

def get_nl(systems_list, method):
    for systems in systems_list:
        compute_requested_neighbors(
            systems=systems, 
            system_length_unit='angstrom',
            model=method.descriptor.model,
            model_length_unit='angstrom',
            check_consistency=True)

def run_simulation(trj, trj_test, methods_intervals, **kwargs):
    is_shared = False
    if trj_test is None:
        is_shared = True
        trj_test = trj

    if not isinstance(trj[0], list):
        trj = [trj]

    if not isinstance(trj_test[0], list):
        trj_test = [trj_test]

    if kwargs['model_load']==None:
        for i, methods in tqdm(enumerate(methods_intervals), desc="looping through intervals"):
            train_systems = [systems_to_torch(run, dtype=torch.float64) for run in trj]
            test_systems = [systems_to_torch(run, dtype=torch.float64) for run in trj_test]
            for j, method in tqdm(enumerate(methods), desc="looping through methods"):
                train_atoms, test_atoms = split_train_test(trj, trj_test, kwargs, is_shared, randomseed=7)
                
                
                if method.descriptor.id == "PETMAD":
                    get_nl(train_systems, method)
                    print("Computed nl for training")
                    get_nl(test_systems, method)
                    print("Computed nl for testing")
                
                method.train(train_systems, train_atoms)
                # for saving eigvecs, eigvals, mu etc for analysis
                if kwargs['log']:
                    method.log_metrics()
         
                # get predictions with the new representation
                # for prediction we can use the concatenated trajectories
                trj_predict = list(chain(*trj_test))
                systems_predict = list(chain(*test_systems))
                if kwargs["ridge"]:
                    #method.fit_ridge(trj_predict)
                    X_ridge=do_ridge_fit(method, train_systems, systems_predict, test_atoms)
                
                print('Starting to predict ...')
                X = method.predict(systems_predict, test_atoms) ##centers N,T,P
                print('Finished the prediction')
                X = [proj.transpose(1,0,2) for proj in X]#centers T,N,P
       
                # label the trajectories:
                if kwargs['classify']['request']:
                    if kwargs['classify']['switch_index'] is not None:
                        y = method.get_label(trj[0], test_atoms, kwargs['classify']['switch_index'])
                    elif len(trj) > 1:
                        y = np.concatenate([np.full((len(t),len(test_atoms)), i) for i, t in enumerate(trj)])
                    else:
                        ValueError("No labels provided for the trajectory. Please provide 'switch_index' or multiple trajectories for classification.")
        
                    # Classification
                    run_logistic_regression(
                        X[0], y, 
                        outfile_prefix=method.label + '_logreg',
                        random_state=42,
                        solver='lbfgs',
                        max_iter=500
                    )
    
                if kwargs["model_save"]:
                    for i, trans in enumerate(method.transformations):
                        print('saving model now for {}'.format(method.label))
                        method.descriptor.set_atom_types(trj)
    
                        keys_to_save = ['system', 'version', 'specifier',
                                        'train_selected_atoms', 'test_selected_atoms', 'input_params', 
                                        'output_params', 'descriptor', 'SOAP_params', 'ridge', 
                                        'ridge_save', 'model_proj_dims', 'i_pca', 'classify', 'base_path']
                        run_specific={'method':method.name, 'interval':method.interval,
                                      'descriptor': method.descriptor, 'label':method.label, 
                                      'time':datetime.datetime.now()}
                        savekwargs={key: kwargs[key] for key in keys_to_save}
                        method.descriptor.update_hypers(savekwargs)
                        method.descriptor.update_hypers(run_specific)
    
                        if hasattr(method, 'spatial_cutoff'):
                            method.descriptor.update_hypers({'spatial_cutoff':method.spatial_cutoff})
                        if hasattr(method, 'sigma'):
                            method.descriptor.update_hypers({'sigma':method.sigma})
                        if hasattr(method, 'n_cumulants'):
                            method.descriptor.update_hypers({'n_cumulants':method.cumulants})
                        if kwargs['lag']!=0:
                            method.descriptor.update_hypers({'lag':method.lag, 'max_lag':kwargs['max_lag'], 'min_lag':kwargs['min_lag'],'lag_step':kwargs['lag_step']})
                        if kwargs['ridge']==True:
                            method.descriptor.update_hypers({'ridge_alpha':method.ridge_alpha})
    
                        if kwargs["ridge"] and kwargs["ridge_save"]:
                            method.descriptor.set_projection_matrix(method.ridge[i].coef_.T)
                        else:
                            method.descriptor.set_projection_matrix(trans.eigvecs)
                        
                        #for CV
                        method.descriptor.set_projection_dims(dims=kwargs['model_proj_dims'])
                        method.descriptor.set_projection_mu(mu=trans.mu)
                        method.descriptor.eval()   
                        method.descriptor.save_model(path=method.label, name='model_soap') 
                        
                        #for reloading
                        method.descriptor.set_projection_dims(dims=list(range(4))) # 4 is number of saved n_components for pca
                        method.descriptor.update_hypers({'model_proj_dims':np.arange(4)})
                        method.descriptor.set_projection_mu(mu=trans.mu)
                        method.descriptor.eval()
                        method.descriptor.save_model(path=method.label, name='model_soap_alldim')
    
                        post_processing(X, trj_predict, test_atoms, method.name, method.label, method.interval, **kwargs)
                        if kwargs["ridge"]:
                            post_processing(X_ridge, trj_predict, test_atoms, method.name, method.label + f'_ridge', method.interval, **kwargs)
                        if kwargs["predict_avg"] and (method.name == "SpatialPCA" or method.name == "PCAfull"):
                            X_fromavg = method.predict_avg(systems_predict, test_atoms) ##centers N,T,P
                            X_fromavg = [proj.transpose(1,0,2) for proj in X_fromavg]
                            print('Finished the prediction for averaged')
                            post_processing(X_fromavg, trj_predict, test_atoms, method.name, method.label + f'_fromavg', method.interval, **kwargs)
                        if kwargs["output_per_structure"]:
                            X = [np.mean(x, axis=1)[:, np.newaxis, :] for x in X]
                            newlabel = method.label + f"_per_structure"
                            post_processing(X, trj_predict, test_atoms, method.name, newlabel, method.interval, **kwargs)
                            if kwargs["ridge"]:
                                X_ridge = [np.mean(x, axis=1)[:, np.newaxis, :] for x in X_ridge]
                                post_processing(X_ridge, trj_predict, test_atoms, method.name, newlabel+ f'_ridge', method.interval, **kwargs)
                            if kwargs["predict_avg"] and (method.name == "SpatialPCA" or method.name == "PCAfull"):
                                X_fromavg = [np.mean(x, axis=1)[:, np.newaxis, :] for x in X_fromavg]
                                post_processing(X_fromavg, trj_predict, test_atoms, method.name, newlabel + f'_fromavg', method.interval, **kwargs)

                #savedX=X.copy()
                del X
                if kwargs["ridge"]:
                    del X_ridge
                if kwargs["predict_avg"] and (method.name == "SpatialPCA" or method.name == "PCAfull"):
                    del X_fromavg
                gc.collect()
    else: # load_model=True
        print('LOADING OF MODELS HAS BEEN REQUESTED, MANY KEYWORDS IN THE INPUT WILL BE IGNORED')
        models = []  
        for model_path in kwargs['model_load']:

            print(f'-------------- Trying to loading model from {model_path} --------------')
            model=load_atomistic_model(model_path)
            models.append(model) 

        for model in models:
            print(f'-------------- Using model from {model_path} --------------')
            loadedargs=model.metadata().extra
            print('Loaded model was computed with the following keywords:',loadedargs)         

            train_atoms, test_atoms = split_train_test(trj,trj_test, kwargs, is_shared,randomseed=7)

            trj_predict = list(chain(*trj_test))
            systems=systems_to_torch(trj_predict, dtype=torch.float64)
            loadedargs=model.metadata().extra
            atomtypes=eval(loadedargs['SOAP_params'])['centers']
            selected_atoms = Labels(
                ["system", "atom"], 
                torch.tensor(
                    [
                        [j, i.index]
                        for i in trj_test[0][0]
                        if i.number in atomtypes
                        for j in range(len(systems))
                    ]
                ),
            )

            
            opts = ModelEvaluationOptions(
                    length_unit="A",
                    outputs={"features": ModelOutput(quantity="", per_atom=False), "features/per_atom":ModelOutput(quantity="", per_atom=True)}, #True
                    selected_atoms=selected_atoms,
                )
            
            X=[]
            projected=[]
            for system in systems:
                cv=model.forward([system], options=opts, check_consistency=False)
                Xs=cv['features/per_atom'][0].values
                projected.append(Xs)
            X.append(np.stack(projected, axis=0))


            trj_predict = list(chain(*trj_test))


            if loadedargs['ridge']:
                X_ridge=X.copy()
                del X
            elif not loadedargs['ridge'] and kwargs['ridge']:
                print('--- postprocessing with ridge fit for a model without ridge is not implemented ---')
                
            if loadedargs["ridge"]:
                post_processing(X_ridge, trj_predict, test_atoms, loadedargs['method'],  loadedargs['label'] + f'_ridge', loadedargs['interval'] , **kwargs)

                if kwargs["output_per_structure"]:
                    X_ridge = [np.mean(x, axis=1)[:, np.newaxis, :] for x in X_ridge]
                    newlabel = loadedargs['method'] + f"_per_structure"
                    post_processing(X_ridge, trj_predict, test_atoms, loadedargs['method'], newlabel+ f'_ridge', loadedargs['interval'], **kwargs)

            else:
                post_processing(X, trj_predict, test_atoms, loadedargs['method'], loadedargs['label'], loadedargs['interval'], **kwargs)

                if kwargs["output_per_structure"]:
                    X = [np.mean(x, axis=1)[:, np.newaxis, :] for x in X]
                    newlabel = loadedargs['method'] + f"_per_structure"
                    post_processing(X, trj_predict, test_atoms, loadedargs['method'], newlabel, loadedargs['interval'], **kwargs)

                if kwargs["predict_avg"] and (method.name == "SpatialPCA" or method.name == "PCAfull"):
                    X_fromavg = [np.mean(x, axis=1)[:, np.newaxis, :] for x in X_fromavg]
                    post_processing(X_fromavg, trj_predict, test_atoms, loadedargs['method'], newlabel + f'_fromavg', loadedargs['interval'], **kwargs)


if __name__ == '__main__':
    print('Nothing to do here')
