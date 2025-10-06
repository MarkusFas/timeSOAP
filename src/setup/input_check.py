from ase.io.trajectory import Trajectory
from itertools import chain

from src.methods.BaseMethod import allPCA, IVAC, tempPCA
from src.descriptors.SOAP import SOAP_descriptor
from src.setup.launcher import run_simulation
from src.setup.input_check import read_data

def check_analysis_inputs(traj, **kwargs):
    if not isinstance(kwargs['lag'], int):
        raise TypeError("lag must be an integer")

    if not isinstance(kwargs['interval'], int):
        raise TypeError("interval must be an integer")

    if kwargs['lag'] > kwargs['interval']:
        raise ValueError("lag cannot be larger than interval length")

    if not isinstance(kwargs['selected_atoms'], list):
        raise TypeError("selected_atoms must be a list")
    
    if not all(isinstance(x, int) for x in kwargs['selected_atoms']):
        raise TypeError("All elements of selected_atoms must be integers")
    
    if not all(atoms_idx in traj[0].index for atoms_idx in kwargs['selected_atoms']):
        raise ValueError(" Some of the selected atoms are not in the traj")
    
    return kwargs


def check_SOAP_inputs(traj, **kwargs):
    required = ["centers", "neighbors", "SOAP_cutoff", "SOAP_max_angular", "SOAP_max_radial"]
    for key in required:
        if key not in kwargs:
            raise ValueError(f"Missing SOAP parameter: {key}")

    # type/value checks
    if not isinstance(kwargs["SOAP_cutoff"], (int, float)) or kwargs["SOAP_cutoff"] <= 0:
        raise ValueError("SOAP_cutoff must be a positive number")
    if not isinstance(kwargs["SOAP_max_angular"], int) or kwargs["SOAP_max_angular"] <= 0:
        raise ValueError("SOAP_max_angular must be a positive integer")
    if not isinstance(kwargs["SOAP_max_radial"], int) or kwargs["SOAP_max_radial"] <= 0:
        raise ValueError("SOAP_max_radial must be a positive integer")
    for center in kwargs["centers"]:
        if not center in traj[0].get_atomic_numbers():
            raise ValueError(f"Center {center} is not in the atomic types of the trajectory.")
    for neighbor in kwargs["neighbors"]:
        if not center in traj[0].get_atomic_numbers():
            raise ValueError(f"Neighbor {neighbor} is not in the atomic types of the trajectory.")
        
    return kwargs


def setup_simulation(**kwargs):

    #1 check trajectory
    trajs = [read_data(fname, kwargs["indices"][i]) for i, fname in enumerate(kwargs["fname"])]
    traj = list(chain(*trajs))

    #2 check descriptor
    descriptor = kwargs['descriptor']
    if descriptor == 'SOAP'
        kwargs = check_SOAP_inputs(traj, **kwargs)
        centers = kwargs.pop('centers')
        neighbors = kwargs.pop('neighbors')
        SOAP_cutoff = kwargs.pop('SOAP_cutoff')
        SOAP_max_angular = kwargs.pop('SOAP_max_angular')
        SOAP_max_radial = kwargs.pop('SOAP_max_radial')
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
        descriptor = SOAP_descriptor(HYPER_PARAMETERS, selected_atoms, centers, neighbors)
    else:
        raise NotImplementedError(f"{descriptor} has not been implemented yet.")
    

    #3 Check Analysis
    check_analysis_inputs(traj, **kwargs)
    selected_atoms = kwargs.pop('selected_atoms')
    interval = kwargs.pop('interval')
    lag = kwargs.pop('lag')
    opt_methods = kwargs.pop('method')
    implemented_opt = ['PCA', 'IVAC', 'TEMPPCA']
    used_methods = []
    for method in opt_methods:
        if method == 'PCA':
            used_methods.append(allPCA(descriptor, selected_atoms, interval, lag))
        elif method == 'IVAC':
            used_methods.append(IVAC())
        elif method == 'TEMPPCA':
            used_methods.append(tempPCA())
    else:
       raise ValueError(f"method must be one of {implemented_opt}, got {opt_methods}")

    
    #4 Check Post Processing

    label = descriptor_id
    run_simulation(traj, used_methods, **kwargs)