import os 

from ase.io.trajectory import Trajectory
from itertools import chain

from src.methods import PCA, IVAC, TICA, TempPCA, PCAfull, PCAtest, LDA, SpatialPCA
from src.descriptors.SOAP import SOAP_descriptor
from src.setup.simulation import run_simulation
from src.setup.simulation_test import run_simulation_test
from src.setup.read_data import read_trj


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


def check_analysis_inputs(trajs, **kwargs):
    if not isinstance(kwargs['lag'], int):
        raise TypeError("lag must be an integer")

    intervals = kwargs["interval"]

    if isinstance(intervals, int):
        #TODO if intervals > len(trajs)
        kwargs['interval'] = [intervals]
    elif isinstance(intervals, list):
        if not all(isinstance(i, int) for i in intervals):
            raise TypeError("all elements of 'interval' list must be integers")
    else:
        raise TypeError("interval must be an integer or list of integers")

    
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
            raise TypeError("test_selected_atoms must be integer or list of integers")
    else:
        if not all(isinstance(x, int) for x in kwargs['test_selected_atoms']):
            raise TypeError("All elements of test_selected_atoms must be integers")
        if not all(atoms_idx < len(traj[0]) for atoms_idx in kwargs['test_selected_atoms'] for traj in trajs):
            raise ValueError(" Some of the selected atoms are not in the traj")

    if isinstance(kwargs['train_selected_atoms'], list) and isinstance(kwargs['test_selected_atoms'], list):
        if set(kwargs['train_selected_atoms']) & set(kwargs['test_selected_atoms']):
            raise ValueError("train selected atoms and test atoms shouldn't contain shared atoms")
    
    if not isinstance(kwargs['methods'], list):
        if not isinstance(kwargs['methods'], str):
            raise TypeError('methods need to be a str or List of str')
        else:
            kwargs['methods'] = [kwargs['methods']]
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
    positive_keys = ["true", "yes"]
    negative_keys = ["false", "no"]
    if kwargs["input_params"].get('concatenate'):
        trajs = [list(chain(*trajs))]
    elif not kwargs["input_params"].get('concatenate'):
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
        
        descriptor = SOAP_descriptor(SOAP_cutoff, SOAP_max_angular, SOAP_max_radial, centers, neighbors)
    else:
        raise NotImplementedError(f"{descriptor} has not been implemented yet.")
    
    #3 Check Analysis
    kwargs = check_analysis_inputs(trajs, **kwargs)
    
    opt_methods = kwargs.get('methods')  # list of methods
    implemented_opt = ['PCA', 'PCAfull', 'TICA', 'TEMPPCA', 'PCAtest', "LDA", "SpatialPCA"]

    system = kwargs["system"]
    version = kwargs["version"]
    specifier = kwargs["specifier"]

    methods_intervals = []  # nested list: intervals x methods
    lag = kwargs.get("lag")
    for interval in kwargs.get('interval'):
        used_methods = []
        for method in opt_methods:
            run_dir = f'results/{system}/{version}/{kwargs.get("descriptor")}/{descriptor_id}/{specifier}/'
            
            # Instantiate method
            method_obj = None
            if method.upper() == 'PCA':
                method_obj = PCA(descriptor, interval, run_dir)
            elif method.upper() == 'IVAC':
                #TODO: input checks for the lag parameters
                max_lag = kwargs.get("max_lag")
                min_lag = kwargs.get("min_lag")
                lag_step = kwargs.get("lag_step")
                method_obj = IVAC(descriptor, interval, max_lag, min_lag, lag_step, run_dir)
            elif method.upper() == 'TEMPPCA':
                method_obj = TempPCA(descriptor, interval, run_dir)
            elif method.upper() == 'PCAFULL':
                method_obj = PCAfull(descriptor, interval, run_dir)
            elif method.upper() == 'PCATEST':
                method_obj = PCAtest(descriptor, interval, run_dir)
            elif method.upper() == 'SPATIALPCA':
                #TODO add input check
                sigma = kwargs.get('sigma')
                method_obj = SpatialPCA(descriptor, interval, sigma, run_dir)
            elif method.upper() == 'LDA':
                method_obj = LDA(descriptor, interval, run_dir)
            elif method.upper() == 'TICA':
                method_obj = TICA(descriptor, interval, lag, run_dir)
            
            else:
                raise NotImplementedError(f"Method must be one of {implemented_opt}, got {method}")

            used_methods.append(method_obj)
        methods_intervals.append(used_methods)

    # TODO: check requested plots
    
    # Pass nested lists to run_simulation
    run_simulation(trajs, methods_intervals, **kwargs)
