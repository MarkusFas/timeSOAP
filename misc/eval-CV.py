import chemiscope
import torch 
import numpy as np
import os
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
    load_atomistic_model,
    register_autograd_neighbors,
)
from featomic.torch import SoapPowerSpectrum
import vesin
from vesin.metatomic import compute_requested_neighbors
from ase.io import read
import matplotlib.pyplot as plt

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
        #i = torch.from_numpy(i.astype(int))
        #j = torch.from_numpy(j.astype(int))
        i = i.to(torch.int64)
        j = j.to(torch.int64)
        D = D.to(torch.float64)
        neighbor_indices = torch.stack([i, j], dim=1)
        neighbor_shifts = S.to(torch.int64) #torch.from_numpy(S.astype(int))
        print("i", i.type())
        print("j", j.type())
        print("neighbor indices", neighbor_indices.type())
        print("neighbor shifts", neighbor_shifts.type())
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
            values=D.reshape(-1, 3, 1),
            samples=samples,
            components=[Labels.range("xyz", 3)],
            properties=Labels.range("distance", 1),
        )
        system.add_neighbor_list(nlistoptions, neighbors)
        systems_new.append(system)
    return systems_new

def get_systems(structures, model):
    systems_list = systems_to_torch(structures, dtype=torch.float64)
    for systems in systems_list:
        compute_requested_neighbors(
            systems=systems, 
            system_length_unit='A',
            model=model,
            model_length_unit='A',
            check_consistency=True)
    return systems_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate SOAP model on CV')
    parser.add_argument('--model', type=str, default=None, help='Path to the model to evaluate')
    parser.add_argument('--trj', type=str, default=None, help='Path to the trajectory to evaluate on')
    parser.add_argument('--nskip', type=int, default=1, help='Path to the trajectory to evaluate on')
    parser.add_argument('--out', type=str, default="./CV.json.gz", help='output filename')
    model_name = parser.parse_args().model
    trj_name = parser.parse_args().trj
    nskip = parser.parse_args().nskip
    out_path = parser.parse_args().out

    CVmodel = load_atomistic_model(model_name, extensions_directory='.')

    structures = read(trj_name, index='::{}'.format(parser.parse_args().nskip))
    systems = get_systems(structures, CVmodel)
    for system in systems:
        system.positions.requires_grad_(True)
        register_autograd_neighbors(system, system.get_neighbor_list(CVmodel.requested_neighbor_lists()[0]))

    selected_atoms = Labels(
            names=["system", "atom"],
            values=torch.tensor([[0, j] for j in np.arange(0, len(structures[0]), 3)], dtype=torch.int32),)
        
    eval_options = ModelEvaluationOptions(
                length_unit='',
                outputs={"features": ModelOutput(per_atom=False), 
                        "features/per_atom": ModelOutput(per_atom=True)
                },
                selected_atoms=selected_atoms,
            )
    CVs = []
    CVs_per_atom = []
    sel_atoms = []
    forces = []
    for system in systems:
        pos = system.positions
        cv = CVmodel(
            systems=[system],
            options=eval_options,
            check_consistency=False,
        )
        CV = cv['features'].block().values[0]
        forces.append(-torch.autograd.grad(CV, pos)[0].detach().numpy())
        CVs.append(CV.detach().numpy())
        CVs_per_atom.append(cv['features/per_atom'].block().values)
        sel_atoms.append(cv['features/per_atom'].block().samples.values)

    N = len(structures)
    CVperatom = [Cpa[:,0].detach().numpy() for Cpa in CVs_per_atom]
    if out_path[-3:] == "txt":
        np.savetxt(f'CV-{out_path}', CVs)
        np.savetxt(f'forces-{out_path}', forces)
        exit()
    elif out_path[-2:] == "pt":
        torch.save(torch.tensor(CVs), f'CV-{out_path}')
        torch.save(torch.tensor(forces), f'forces-{out_path}')
        exit()
    #torch.stack(CVs_per_atom, dim=0).squeeze().numpy()
    ins = [sel[:,1] for sel in sel_atoms]
    print("ins", ins)
    #CVperatom_full = np.zeros((N, len(structures[0])))
    #for f in range(N):
    #    CVperatom_full[f, ins[f]] = CVperatom[f]

    time=[]
    #for atoms in traj:
    #    time.append(atoms.info['timestep'])
    time = np.arange(0, len(structures))*1.0  # 1ps interval

    envs = []
    for j in range(len(structures)):
        for i in ins[j]:
            print(len(ins[j]))
            envs.append((j, i, 3.5))


    properties = {
            'time': time,
            'cv_peratom': {"values": [i for j in CVperatom for i in j], "target": "atom"},
            'cv': {"values": CVs, "target": "structure"},
    }

    settings = chemiscope.quick_settings(trajectory=True, structure_settings={
        "unitCell":True,
        'environments': {'activated': True}, 
        'map': {'color': {'property': 'cv_peratom'}}, 
        'color': {'property': 'cv_peratom', 
                'min': -0.1, 
                'max': 0.1, 
                'transform': 'linear',
                'palette': 'bwr',
            } 
        }
    )

    chemiscope.write_input(
        path=out_path,
        # dataset metadata can also be included to provide a self-contained description
        # of the data, authors, and references
        metadata={
            "name": "SOAP CV trj",
            "description": (
                "CV per atom for a water-ice trajectory"
            ),
            "authors": ["Markus Fasching, Hannah Tuerk"],
        },
        structures=structures,
        properties=properties,
        environments=envs,
        settings=settings,
    )


    exit()
import chemiscope
import torch 
import numpy as np
import os
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
    parser = argparse.ArgumentParser(description='Evaluate SOAP model on CV')
    parser.add_argument('--model', type=str, default=None, help='Path to the model to evaluate')
    parser.add_argument('--trj', type=str, default=None, help='Path to the trajectory to evaluate on')
    parser.add_argument('--nskip', type=int, default=1, help='Path to the trajectory to evaluate on')
    parser.add_argument('--out', type=str, default="./CV.json.gz", help='output filename')
    model_name = parser.parse_args().model
    trj_name = parser.parse_args().trj
    nskip = parser.parse_args().nskip
    out_path = parser.parse_args().out
    CVmodel = load_atomistic_model(model_name, extensions_directory='.')

    structures = read(trj_name, index='::{}'.format(parser.parse_args().nskip))
    systems = get_systems(structures)

    selected_atoms = Labels(
            names=["system", "atom"],
            values=torch.tensor([[0, j] for j in np.arange(0, len(structures[0]), 3)], dtype=torch.int32),)
        
    eval_options = ModelEvaluationOptions(
                length_unit='',
                outputs={"features": ModelOutput(per_atom=False), 
                        "features/per_atom": ModelOutput(per_atom=True)},
                selected_atoms=selected_atoms,
            )
    CVs = []
    CVs_per_atom = []
    sel_atoms = []
    for system in systems:
        cv = CVmodel(
            systems=[system],
            options=eval_options,
            check_consistency=True,
        )
        CVs.append(cv['features'].block().values)
        CVs_per_atom.append(cv['features/per_atom'].block().values)
    print('appended everything')
    #CVperatom = torch.stack(CVs_per_atom, dim=0).squeeze().numpy()
    CVperatom = torch.cat(CVs_per_atom, dim=0).squeeze().numpy()
    CV = torch.stack(CVs, dim=0).squeeze().numpy()

    io = structures[0].get_atomic_numbers() == 8
    ins = [np.where(np.isin(frame.symbols, ["O"]))[0] for frame in structures] #O indices
    time=[]
    #for atoms in traj:
    #    time.append(atoms.info['timestep'])
    time = np.arange(0, len(structures))*1.0  # 1ps interval

    envs = []
    for j in range(len(structures)):
        for i in ins[j]:
            envs.append((j, i, 3.5))

    properties = {
            'time': time,
            'cv_peratom': {"values": [i for j in CVperatom for i in j], "target": "atom"},
            'cv': {"values": CV, "target": "structure"},
    }

    settings = chemiscope.quick_settings(trajectory=True, structure_settings={
        "unitCell":True,
        'environments': {'activated': True}, 
        'map': {'color': {'property': 'cv_peratom'}}, 
        'color': {'property': 'cv_peratom', 
                'min': -0.1, 
                'max': 0.1, 
                'transform': 'linear',
                'palette': 'bwr',
            } 
        }
    )

    chemiscope.write_input(
        path=out_path,
        # dataset metadata can also be included to provide a self-contained description
        # of the data, authors, and references
        metadata={
            "name": "SOAP CV trj",
            "description": (
                "CV per atom for a water-ice trajectory"
            ),
            "authors": ["Markus Fasching, Hannah Tuerk"],
        },
        structures=structures,
        properties=properties,
        environments=envs,
        settings=settings,
    )
