import os 

from ase.io.trajectory import Trajectory
from itertools import chain
import warnings
from pathlib import Path

from smoothsoap.descriptors.PETMAD import PETMAD_descriptor
from smoothsoap.descriptors.SOAP import SOAP_descriptor_special
from smoothsoap.descriptors.model_soap import SOAP_CV, CumulantSOAP_CV
from smoothsoap.descriptors.model_soap_distinct import SOAP_CV_distinct
from smoothsoap.methods import (
    PCA, IVAC, TICA, TILDA, TempPCA, 
    PCAfull, PCAtest, LDA, SpatialPCA, 
    SpatialTempPCA, ScikitPCA, CumulantPCA, 
    CumulantIVAC, DistinctPCA, PCAnorm, 
    SpatialIVAC, SpatialIVACnorm, PCA_time_norm,
)
from smoothsoap.setup.simulation import run_simulation
from smoothsoap.setup.simulation_test import run_simulation_test
from smoothsoap.setup.read_data import read_trj


def check_file_input(**kwargs):
    fnames = kwargs["fname"]
    indices = kwargs["indices"]
    if isinstance(fnames, str):
        fnames = [fnames]
    elif isinstance(fnames, list):
        if not all(isinstance(name, str) for name in fnames):
            raise TypeError(f"All elements of '{fnames}' must be strings.")
    else:
        raise TypeError(f"'{fnames}' must be a str or a list of str, got {type(fnames).__name__}")

    if isinstance(indices, str):
        indices = [indices for _ in fnames]
    elif isinstance(indices, list):
        if not all(isinstance(v, str) for v in indices):
            raise TypeError(f"All elements of '{indices}' must be str.")
        pass
    else:
        raise TypeError(f"'{indices}' must be a str or a list of str, got {type(indices).__name__}")

    return fnames, indices


def check_analysis_inputs(trajs, test_trajs, **kwargs):
    intervals = kwargs["interval"]
    if isinstance(intervals, int):
        #TODO if intervals > len(trajs)
        kwargs['interval'] = [intervals]
    elif isinstance(intervals, list):
        if not all(isinstance(i, int) for i in intervals):
            raise TypeError("all elements of 'interval' list must be integers")
    else:
        raise TypeError("interval must be an integer or list of integers")

    lags = kwargs["lag"]
    if isinstance(lags, int):
        kwargs['lag'] = [lags]
    elif isinstance(lags, list):
        if not all(isinstance(i, int) for i in lags):
            raise TypeError("all elements of 'lag' list must be integers")
    else:
        raise TypeError("lag must be an integer or list of integers")

    sigmas = kwargs["sigma"]
    if isinstance(sigmas, float) or isinstance(sigmas, int):
        kwargs['sigma'] = [sigmas]
    elif isinstance(sigmas, list):
        if not all(isinstance(i, float) or isinstance(i, int) for i in sigmas):
            raise TypeError("all elements of 'sigma' list must be integers or floats")
    else:
        raise TypeError("sigma must be an integer, float or list of integers")

    n_cumulants = kwargs["n_cumulants"]
    if isinstance(n_cumulants, int):
        kwargs['n_cumulants'] = [n_cumulants]
    elif isinstance(n_cumulants, list):
        if not all(isinstance(n, int) for n in n_cumulants):
            raise TypeError("all elements of 'n_cumulants' list must be integers")
    else:
        raise TypeError("n_cumulants must be an integer or list of integers")


    spatial_cutoff = kwargs["spatial_cutoff"]
    if isinstance(spatial_cutoff, float) or isinstance(spatial_cutoff, int):
        kwargs['spatial_cutoff'] = [spatial_cutoff]
    elif isinstance(spatial_cutoff, list):
        if not all(isinstance(i, float) or isinstance(i, int) for i in spatial_cutoff):
            raise TypeError("all elements of 'spatial_cutoff' list must be integers or floats")
    else:
        raise TypeError("spatial_cutoff must be an integer, float or list thereof")

    sigma2factor = kwargs["sigma2factor"]
    if isinstance(sigma2factor, str):
        try:
            sigma2factor = float(sigma2factor)
        except ValueError:
            raise TypeError("sigma2factor must be a float")
    kwargs["sigma2factor"] = sigma2factor

    if not isinstance(kwargs['train_selected_atoms'], list):
        if not isinstance(kwargs['train_selected_atoms'], int):
            raise TypeError("train_selected_atoms must be integer or list of integers")
    else:
        if not all(isinstance(x, int) for x in kwargs['train_selected_atoms']):
            raise TypeError("All elements of train_selected_atoms must be integers")
        if not all(atoms_idx < len(traj[0]) for atoms_idx in kwargs['train_selected_atoms'] for traj in trajs):
            raise ValueError(" Some of the selected atoms are not in the traj")

    if not isinstance(kwargs['test_selected_atoms'], list):
        if not isinstance(kwargs['test_selected_atoms'], int):
            if kwargs['test_selected_atoms'] is not None:
                raise TypeError("test_selected_atoms must be integer or list of integers")
    else:
        if not all(isinstance(x, int) for x in kwargs['test_selected_atoms']):
            raise TypeError("All elements of test_selected_atoms must be integers")
        if not all(atoms_idx < len(traj[0]) for atoms_idx in kwargs['test_selected_atoms'] for traj in test_trajs):
            raise ValueError(" Some of the selected atoms are not in the traj")

    if isinstance(kwargs['train_selected_atoms'], list) and isinstance(kwargs['test_selected_atoms'], list):
        if set(kwargs['train_selected_atoms']) & set(kwargs['test_selected_atoms']):
            warnings.warn("train selected atoms and test atoms shouldn't contain shared atoms")

    if not isinstance(kwargs['methods'], list):
        if not isinstance(kwargs['methods'], str):
            raise TypeError('methods need to be a str or List of str')
        else:
            kwargs['methods'] = [kwargs['methods']]
    
    if not isinstance(kwargs['model_proj_dims'], list):
        if not isinstance(kwargs['model_proj_dims'], int):
            raise TypeError("test_selected_atoms must be integer or list of integers")
    else:
        if not all(isinstance(x, int) for x in kwargs['model_proj_dims']):
            raise TypeError("All elements of test_selected_atoms must be integers")

    ridge_alphas = kwargs["ridge_alpha"]
    ridge_alphas_out = []
    if isinstance(ridge_alphas, str):
        try:
            ridge_alphas = float(ridge_alphas)
        except ValueError:
            raise TypeError("ridge_alpha must be a float")
        
    if isinstance(ridge_alphas, float) or isinstance(ridge_alphas, int):
        ridge_alphas_out = [ridge_alphas]
    elif isinstance(ridge_alphas, list):
        for alpha in ridge_alphas:
            try:
                ridge_alphas_out.append(float(alpha))
            except ValueError:
                raise TypeError("all elements of 'ridge_alpha' list must be floats")
        if not all(isinstance(i, float) or isinstance(i, int) for i in ridge_alphas_out):
            raise TypeError("all elements of 'ridge_alpha' list must be integers or floats")
    else:
        raise TypeError("ridge_alpha must be an integer, float or list of integers")
    kwargs['ridge_alpha'] = ridge_alphas_out

    return kwargs


def check_SOAP_inputs(trajs, **kwargs):
    required = ["centers", "neighbors", "cutoff", "max_angular", "max_radial"]
    for key in required:
        if key not in kwargs:
            raise ValueError(f"Missing SOAP parameter: {key}")

    # type/value checks
    if not isinstance(kwargs["cutoff"], (int, float)) or kwargs["cutoff"] <= 0:
        raise ValueError("SOAP_cutoff must be a positive number")
    if not isinstance(kwargs["max_angular"], int) or kwargs["max_angular"] <= 0:
        raise ValueError("max_angular must be a positive integer")
    if not isinstance(kwargs["max_radial"], int) or kwargs["max_radial"] <= 0:
        raise ValueError("max_radial must be a positive integer")
    for center in kwargs["centers"]:
        if not all(center in traj[0].get_atomic_numbers() for traj in trajs):
            raise ValueError(f"Center {center} is not in the atomic types of the trajectory.")
    for neighbor in kwargs["neighbors"]:
        if not all(center in traj[0].get_atomic_numbers() for traj in trajs):
            raise ValueError(f"Neighbor {neighbor} is not in the atomic types of the trajectory.")
        
    return kwargs


def setup_simulation(**kwargs):

    #1 check trajectory
    fnames, indices = check_file_input(**kwargs["input_params"])
    trajs = [read_trj(fname, indices[i]) for i, fname in enumerate(fnames)]
    if kwargs["input_params"].get('concatenate'):
        trajs = [list(chain(*trajs))]
    elif not kwargs["input_params"].get('concatenate'):
        pass
    else:
        raise TypeError('concatenate, needs to be either true or false')

    # check the test data
    if kwargs["output_params"]["fname"] is None:
        kwargs["output_params"]["fname"] = kwargs["input_params"]["fname"]
        kwargs["output_params"]["indices"] = kwargs["input_params"]["indices"]
        kwargs["output_params"]["concatenate"] = kwargs["input_params"]["concatenate"]
        test_trajs = trajs #[read_trj(fname, indices) for fname, indices in zip(fnames, indices)]
        print('Testing on input trajectories')
    else:
        fnames_test, indices_test = check_file_input(**kwargs["output_params"])
        test_trajs = [read_trj(fname, indices) for fname, indices in zip(fnames_test, indices_test)]
        if kwargs["output_params"].get('concatenate'):
            test_trajs = [list(chain(*test_trajs))]
        elif not kwargs["output_params"].get('concatenate'):
            pass
        else:
            raise TypeError('concatenate, needs to be either true or false')

    #2 check descriptor
    descriptor_name = kwargs['descriptor']
    if descriptor_name == 'SOAP':
        SOAP_kwargs = check_SOAP_inputs(trajs, **kwargs["SOAP_params"])
        centers = SOAP_kwargs.get('centers')
        neighbors = SOAP_kwargs.get('neighbors')
        SOAP_cutoff = SOAP_kwargs.get('cutoff')
        SOAP_max_angular = SOAP_kwargs.get('max_angular')
        SOAP_max_radial = SOAP_kwargs.get('max_radial')
        descriptor_id = f"{SOAP_cutoff}{SOAP_max_angular}{SOAP_max_radial}"
        descriptor = SOAP_CV(SOAP_cutoff, SOAP_max_angular, SOAP_max_radial, centers, neighbors)
        
    elif descriptor_name == 'SOAP_atom':
        SOAP_kwargs = check_SOAP_inputs(trajs, **kwargs["SOAP_params"])
        centers = SOAP_kwargs.get('centers')
        neighbors = SOAP_kwargs.get('neighbors')
        SOAP_cutoff = SOAP_kwargs.get('cutoff')
        SOAP_max_angular = SOAP_kwargs.get('max_angular')
        SOAP_max_radial = SOAP_kwargs.get('max_radial')
        descriptor_id = f"{SOAP_cutoff}{SOAP_max_angular}{SOAP_max_radial}"
        
        descriptor = SOAP_descriptor_special(SOAP_cutoff, SOAP_max_angular, SOAP_max_radial, centers, neighbors)
    
    elif descriptor_name == 'PETMAD':
        SOAP_kwargs = check_SOAP_inputs(trajs, **kwargs["SOAP_params"])
        centers = SOAP_kwargs.get('centers')
        neighbors = SOAP_kwargs.get('neighbors')
        SOAP_cutoff = SOAP_kwargs.get('cutoff')
        SOAP_max_angular = SOAP_kwargs.get('max_angular')
        SOAP_max_radial = SOAP_kwargs.get('max_radial')
        descriptor_id = f"{SOAP_cutoff}{SOAP_max_angular}{SOAP_max_radial}"
        descriptor = PETMAD_descriptor(SOAP_cutoff, SOAP_max_angular, SOAP_max_radial, centers, neighbors)
   
    elif descriptor_name == 'SOAP_distinct':
        SOAP_kwargs = check_SOAP_inputs(trajs, **kwargs["SOAP_params"])
        centers = SOAP_kwargs.get('centers')
        neighbors = SOAP_kwargs.get('neighbors')
        SOAP_cutoff = SOAP_kwargs.get('cutoff')
        SOAP_max_angular = SOAP_kwargs.get('max_angular')
        SOAP_max_radial = SOAP_kwargs.get('max_radial')
        descriptor_id = f"{SOAP_cutoff}{SOAP_max_angular}{SOAP_max_radial}"
        descriptor = SOAP_CV_distinct(SOAP_cutoff, SOAP_max_angular, SOAP_max_radial, centers, neighbors)
    
    else:
        raise NotImplementedError(f"{descriptor} has not been implemented yet.")
    
    #3 Check Analysis
    kwargs = check_analysis_inputs(trajs, test_trajs, **kwargs)
    
    opt_methods = kwargs.get('methods')  # list of methods
    implemented_opt = ['PCA', 'PCAfull', 'TICA','IVAC', 'TEMPPCA', 'PCAtest', "LDA", "SpatialPCA", "CumulantPCA", "DistinctPCA"]

    system = kwargs["system"]
    version = kwargs["version"]
    specifier = kwargs["specifier"]

    print("Parameters checked. Simulation is initialized...")
    methods_intervals = []  # nested list: intervals x methods

    if kwargs['model_load']!=None:
        model_paths = kwargs["model_load"]
        if isinstance(model_paths, str):
            model_paths = [model_paths]
        elif isinstance(model_paths, list):
            if not all(isinstance(name, str) for name in model_paths):
                raise TypeError(f"All elements of '{model_paths}' must be strings.")
        else:
            raise TypeError(f"'{model_paths}' must be a str or a list of str, got {type(model_paths).__name__}")
        for model_path in model_paths:
            if isinstance(model_path, (str, bytes, Path)):
                if not os.path.exists(model_path):
                    raise TypeError(f"given model path '{model_path}' does not exist")
    else: # model_load==None
        model_paths = []
        lag = kwargs.get("lag")
        for interval in kwargs.get('interval'):
            used_methods = []
            for lag in kwargs.get('lag'):
                for sigma in kwargs.get('sigma'):
                    for spatial_cutoff in kwargs.get('spatial_cutoff'):
                        for ridge_alpha in kwargs.get('ridge_alpha'):
                            for n_cumulants in kwargs.get('n_cumulants'):
                                for method in opt_methods:
                                    base: Path = kwargs.get("base_path")
                                    run_dir = (
                                        base
                                        / "results"
                                        / system
                                        / version
                                        / kwargs.get("descriptor")
                                        / descriptor_id
                                        / specifier
                                        
                                    )
                                    run_dir.mkdir(parents=True, exist_ok=True)
                                    # Instantiate method
                                    method_obj = None
                                    if method.upper() == 'PCA':
                                        method_obj = PCA(descriptor, interval, ridge_alpha, run_dir)
                                    elif method.upper() == 'IVAC':
                                        #TODO: input checks for the lag parameters
                                        max_lag = kwargs.get("max_lag")
                                        min_lag = kwargs.get("min_lag")
                                        lag_step = kwargs.get("lag_step")
                                        sigma2factor = kwargs.get("sigma2factor")
                                        method_obj = IVAC(descriptor, interval, max_lag, min_lag, lag_step, ridge_alpha, run_dir, sigma2factor)
                                    elif method.upper() == 'TEMPPCA':
                                        method_obj = TempPCA(descriptor, interval, ridge_alpha, run_dir)
                                    elif method.upper() == 'PCAFULL':
                                        method_obj = PCAfull(descriptor, interval, ridge_alpha, run_dir)
                                    elif method.upper() == 'PCATEST':
                                        method_obj = PCAtest(descriptor, interval, ridge_alpha, run_dir)
                                    elif method.upper() == 'SPATIALPCA':
                                        #TODO add input check
                                        method_obj = SpatialPCA(descriptor, interval, sigma, spatial_cutoff, ridge_alpha, run_dir)
                                    elif method.upper() == 'DISTINCTPCA':
                                        method_obj = DistinctPCA(descriptor, interval, ridge_alpha, run_dir)
                                    elif method.upper() == 'SPATIALTEMPPCA':
                                        #TODO add input check
                                        method_obj = SpatialTempPCA(descriptor, interval, sigma, spatial_cutoff, ridge_alpha, run_dir)
                                    elif method.upper() == 'LDA':
                                        method_obj = LDA(descriptor, interval, ridge_alpha, run_dir)
                                    elif method.upper() == 'TICA':
                                        method_obj = TICA(descriptor, interval, lag, sigma, ridge_alpha, run_dir)
                                    elif method.upper() == 'TILDA':
                                        method_obj = TILDA(descriptor, interval, lag, sigma, ridge_alpha, run_dir)
                                    elif method.upper() == 'SCIKITPCA':
                                        method_obj = ScikitPCA(descriptor, interval, ridge_alpha, run_dir)
                                    elif method.upper() == 'PCANORM':
                                        method_obj = PCAnorm(descriptor, interval, ridge_alpha, run_dir)
                                    elif method.upper() == 'CUMULANTPCA':
                                        descriptor = CumulantSOAP_CV(SOAP_cutoff, SOAP_max_angular, SOAP_max_radial, centers, neighbors, n_cumulants)
                                        method_obj = CumulantPCA(descriptor, interval, ridge_alpha, n_cumulants, run_dir)
                                    elif method.upper() == 'CUMULANTIVAC':
                                        max_lag = kwargs.get("max_lag")
                                        min_lag = kwargs.get("min_lag")
                                        lag_step = kwargs.get("lag_step")
                                        descriptor = CumulantSOAP_CV(SOAP_cutoff, SOAP_max_angular, SOAP_max_radial, centers, neighbors, n_cumulants)
                                        method_obj = CumulantIVAC(descriptor, interval, max_lag, min_lag, lag_step, ridge_alpha, n_cumulants, run_dir)
                                    elif method.upper() == 'SPATIALIVAC':
                                        n_cumulants = 1
                                        method_obj = SpatialIVAC(descriptor, interval, ridge_alpha, spatial_cutoff, sigma, run_dir)
                                    elif method.upper() == 'SPATIALIVACNORM':
                                        n_cumulants = 1
                                        sigma2factor = kwargs.get("sigma2factor")
                                        method_obj = SpatialIVACnorm(descriptor, interval, ridge_alpha, spatial_cutoff, run_dir, sigma2factor)
                                    else:
                                        raise NotImplementedError(f"Method must be one of {implemented_opt}, got {method}")
    
                                    used_methods.append(method_obj)
            methods_intervals.append(used_methods)
        # TODO: check requested plots
        # Pass nested lists to run_simulation
        print("Simulation is started...")
    run_simulation(trajs, test_trajs, methods_intervals, **kwargs)
