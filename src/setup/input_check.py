from ase.io.trajectory import Trajectory
from itertools import chain

from src.methods import PCA, IVAC, TempPCA, PCAfull
from src.descriptors.SOAP import SOAP_descriptor
from src.setup.simulation import run_simulation
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

def check_analysis_inputs(traj, **kwargs):
    if not isinstance(kwargs['lag'], int):
        raise TypeError("lag must be an integer")

    if not isinstance(kwargs['interval'], int):
        raise TypeError("interval must be an integer")

    if kwargs['lag'] > kwargs['interval']:
        raise ValueError("lag cannot be larger than interval length")


    if not isinstance(kwargs['train_selected_atoms'], list):
        if not isinstance(kwargs['train_selected_atoms'], int):
            raise TypeError("train_selected_atoms must be integer or list of integers")
    else:
        if not all(isinstance(x, int) for x in kwargs['train_selected_atoms']):
            raise TypeError("All elements of train_selected_atoms must be integers")
        if not all(atoms_idx in traj[0].index for atoms_idx in kwargs['train_selected_atoms']):
            raise ValueError(" Some of the selected atoms are not in the traj")
        
    if not isinstance(kwargs['test_selected_atoms'], list):
        if not isinstance(kwargs['test_selected_atoms'], int):
            raise TypeError("test_selected_atoms must be integer or list of integers")
    else:
        if not all(isinstance(x, int) for x in kwargs['test_selected_atoms']):
            raise TypeError("All elements of test_selected_atoms must be integers")
        if not all(atoms_idx in traj[0].index for atoms_idx in kwargs['test_selected_atoms']):
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


def check_SOAP_inputs(traj, **kwargs):
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
        if not center in traj[0].get_atomic_numbers():
            raise ValueError(f"Center {center} is not in the atomic types of the trajectory.")
    for neighbor in kwargs["neighbors"]:
        if not center in traj[0].get_atomic_numbers():
            raise ValueError(f"Neighbor {neighbor} is not in the atomic types of the trajectory.")
        
    return kwargs


def setup_simulation(**kwargs):

    #1 check trajectory
    fnames, indices = check_file_input(**kwargs["input_params"])
    trajs = [read_trj(fname, indices[i]) for i, fname in enumerate(fnames)]
    traj = list(chain(*trajs))

    #2 check descriptor
    descriptor_name = kwargs['descriptor']
    if descriptor_name == 'SOAP':
        SOAP_kwargs = check_SOAP_inputs(traj, **kwargs["SOAP_params"])
        centers = SOAP_kwargs.get('centers')
        neighbors = SOAP_kwargs.get('neighbors')
        SOAP_cutoff = SOAP_kwargs.get('cutoff')
        SOAP_max_angular = SOAP_kwargs.get('max_angular')
        SOAP_max_radial = SOAP_kwargs.get('max_radial')
        descriptor_id = f"SOAP_{SOAP_cutoff}{SOAP_max_angular}{SOAP_max_radial}_{centers}"
        HYPER_PARAMETERS = {
            "cutoff": {
                "radius": SOAP_cutoff, #4 #5 #6
                "smoothing": {"type": "ShiftedCosine", "width": 0.5},
            },
            "density": {
                "type": "Gaussian",
                "width": 0.25, #changed from 0.3
            },
            "basis": {
                "type": "TensorProduct",
                "max_angular": SOAP_max_angular, #8
                "radial": {"type": "Gto", "max_radial": SOAP_max_radial}, #6
            },
        }
        descriptor = SOAP_descriptor(HYPER_PARAMETERS, centers, neighbors)
    else:
        raise NotImplementedError(f"{descriptor} has not been implemented yet.")
    

    #3 Check Analysis
    kwargs = check_analysis_inputs(traj, **kwargs)
    
    interval = kwargs.get('interval')
    lag = kwargs.get('lag')
    opt_methods = kwargs.get('methods')
    opt_id = f'interval_{interval}_lag{lag}'
    implemented_opt = ['PCA', 'IVAC', 'TEMPPCA']

    system = kwargs["system"]
    version = kwargs["version"]
    specifier = kwargs["specifier"]
    run_dirs = [f'results/{system}/{version}/{descriptor_name}/{method}/{specifier}' for method in opt_methods]
    run_ids = [method + '_' + opt_id + '_' + descriptor_id for method in opt_methods]
    run_labels = [dir + id for dir, id in zip(run_dirs, run_ids)]
    used_methods = []

    for i, method in enumerate(opt_methods):
        if method == 'PCA':
            used_methods.append(PCA(descriptor, interval, run_labels[i]))
        elif method == 'IVAC':
            raise NotImplementedError('Ivac implemteation coming soon')
        elif method == 'TEMPPCA':
            used_methods.append(TempPCA(descriptor, interval, run_labels[i]))
        elif method == 'PCAfull':
            used_methods.append(PCAfull(descriptor, interval, run_labels[i]))
        else:
            raise NotImplementedError(f"method must be one of {implemented_opt}, got {opt_methods}")

    
    #4 Check Post Processing
    # TODO check in kwargs which output plots etc are requested and should be computed
    
    # Start simulation with the set inputs

    #TODO right strategy for passing kwargs downstream, e.g. methods
    run_simulation(traj, used_methods, run_ids, run_dirs, **kwargs)