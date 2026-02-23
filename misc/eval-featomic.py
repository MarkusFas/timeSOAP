import chemiscope
import torch 
import numpy as np
import os
import sys
from typing import Dict, List, Optional
from scipy.stats import moment
import argparse 
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    System,
    systems_to_torch,
    load_atomistic_model
)
from featomic.torch import SoapPowerSpectrum
import vesin
from ase.io import read
import matplotlib.pyplot as plt
from featomic.torch import SoapPowerSpectrum
import time 
import metatensor.torch as mts

def get_systems(structures):
    systems = systems_to_torch(structures, dtype=torch.float64)
    systems_new = []
    for i, system in enumerate(systems): 
        #atoms = structures[i]
        nlistoptions = CVmodel.requested_neighbor_lists()[0]
        nlist = vesin.NeighborList(cutoff=nlistoptions.cutoff, full_list=nlistoptions.full_list) 
        i, j, S, D = nlist.compute(
            points=system.positions,
            box=system.cell, 
            periodic=True,
            quantities="ijSD"
        )
        #i, j, S, D = ase_neighbor_list(quantities="ijSD", a=atoms, cutoff=4.5)
        i = torch.from_numpy(i.astype(int))
        j = torch.from_numpy(j.astype(int))
        neighbor_indices = torch.stack([i, j], dim=1)
        neighbor_shifts = torch.from_numpy(S.astype(int))

        sample_values = torch.hstack([neighbor_indices, neighbor_shifts])
        samples = Labels(
            names=[
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            values=sample_values,
        )

        neighbors = TensorBlock(
            values=torch.from_numpy(D).reshape(-1, 3, 1),
            samples=samples,
            components=[Labels.range("xyz", 3)],
            properties=Labels.range("distance", 1),
        )
        system.add_neighbor_list(nlistoptions, neighbors)
        systems_new.append(system)
    return systems_new

if __name__ == "__main__":
    trj_name = sys.argv[1]
    structures = read(trj_name, index=':')
    systems = systems_to_torch(structures, dtype=torch.float64)
    #systems = get_systems(structures)

    HYPER_PARAMETERS = {
        "cutoff": {
            "radius": 6, #4 #5 #6
            "smoothing": {"type": "ShiftedCosine", "width": 0.5},
        },
        "density": {
            "type": "Gaussian",
            "width": 0.25, #changed from 0.3
        },
        "basis": {
            "type": "TensorProduct",
            "max_angular": 6, #8
            "radial": {"type": "Gto", "max_radial": 6}, #6
        },
    }
    calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)
    
    selected_samples_wrapper_sel = mts.load_labels("selected_samples_wrapper_sel.pt") 
    selected_keys_wrapper_sel = mts.load_labels("selected_keys_wrapper_sel.pt")
    selected_samples_sel = mts.load_labels("selected_samples_sel.pt") 
    selected_keys_sel = mts.load_labels("selected_keys_sel.pt")
    print(len(selected_samples_wrapper_sel.values))
    print(len(selected_samples_sel.values))
    print()
    for i,ele in enumerate(selected_samples_wrapper_sel.values):
        print(ele, selected_samples_sel.values[i])#wrapper selection model
    tstart = time.time()
    for system in systems:
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
        ) as prof:
            soap = calculator(systems,
                    selected_samples=selected_samples_wrapper_sel,
                    selected_keys=selected_keys_wrapper_sel,
            )
    tend = time.time()
    print("eval time with wrapper", tend - tstart)

    #direct selection model
    tstart = time.time()
    for system in systems:
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
        ) as prof:
            soap = calculator(systems,
                    selected_samples=selected_samples_sel, 
                    selected_keys=selected_keys_sel,
            )
    tend = time.time()
    print("eval time pre-selected", tend - tstart)
