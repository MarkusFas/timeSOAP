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

    indices = np.array([1,10,13,16,22,40,43,106,109,112,118,121,136,139,142,433,436,439,451,463,676,685,691,694,703,706,712,823,835,838,850,853,1018,1021,1024,1030,1033,1048,1051,1054,1204,1207,1210,1213,1219,1222,1231,1234,1357,1360,1366,1369,1384,1387,1390,1738,1741,1744,1753,1768,1771,1774,1882,1885,1888,1894,1897,1912,1918,1924,1939,1951,2314,2317,2320,2326,2329,2344,2347,2350,2698,2701,2704,2710,2731,3028,3031,3037,3043,3046,3061,3313,3316,3340,3343,3346,3349,3412,3556,3571,3583,3589,3610,3613,3616,3622,3625,3640,3643,3844,3856,3859,3868,3874,4129,4156,4162,4165,4168,4186,4189,4192,4198,4216,4219,4222,4243,4246,4252,4255,4258,4261,4516,4519,4522,4540,4546,4549,4810,4813,4816,4822,4825,4840,4843,4846,5050,5053,5056,5062,5080,5083,5086,5338,5341,5344,5350,5353,5371,5473,5476,5479,5482,5491,5494,5497,5506,5533,5536,5542,5560,5563,5566,5674,5677,5680,5686,5689,5704,5707,5866,5869,5872,5878,5881,5896,5899,5902,5953,5956,5983,5986,5998,6097,6100,6103,6109,6115,6130])-1
    selected_atoms = Labels(
        names=["system", "atom"],
        values=torch.tensor([[0,j] for j in indices]),
    )

    eval_options = ModelEvaluationOptions(
                length_unit='',
                outputs={"features": ModelOutput(per_atom=False), 
                        "features/per_atom": ModelOutput(per_atom=False)
                },
                selected_atoms=selected_atoms,
            )

    CVs = []
    CVs_per_atom = []
    sel_atoms = []
    for system in systems:
        print('start profiling with subselection')
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
        ) as prof:
            cv = CVmodel(
                systems=[system],
                options=eval_options,
                check_consistency=False,
            )
            CVs.append(cv['features'].block().values)
            CVs_per_atom.append(cv['features/per_atom'].block().values)
        print(prof.key_averages().table(sort_by="cpu_time_total"))
            #sel_atoms.append(cv['features/per_atom'].block().samples.values)
    


    N = len(structures)
    CVperatom = [Cpa[:,0].numpy() for Cpa in CVs_per_atom]
    print(CVs_per_atom)
    exit()
    #torch.stack(CVs_per_atom, dim=0).squeeze().numpy()
    CV = [C[:,0].numpy() for C in CVs]
    #CV = torch.stack(CVs, dim=0).squeeze().numpy()
    plt.plot(CV)
    ins = [sel[:,1] for sel in sel_atoms]
    CVperatom_full = np.zeros((N, len(structures[0])))
    for f in range(N):
        CVperatom_full[f, ins[f]] = CVperatom[f]

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
        #CVs_per_atom.append(cv['features/per_atom'].block().values)

    #CVperatom = torch.stack(CVs_per_atom, dim=0).squeeze().numpy()
    #CVperatom = torch.cat(CVs_per_atom, dim=0).squeeze().numpy()
    CV = torch.stack(CVs, dim=0).squeeze().numpy()

    
    exit()
    io = structures[0].get_atomic_numbers() == 8
    ins = [np.where(np.isin(frame.symbols, ["O"]))[0] for frame in structures] #O indices
    print([len(ins_i) for ins_i in ins])
    time=[]
    #for atoms in traj:
    #    time.append(atoms.info['timestep'])
    time = np.arange(0, len(structures))*1.0  # 1ps interval

    envs = []
    for j in range(len(structures)):
        for i in ins[j]:
            envs.append((j, i, 3.5))

    cs = chemiscope.show(
        structures, 
        properties={
            'time': time,
            #'o_cv': {"values": np.hstack(CVperatom), "target": "atom"},
            'o_cv': {"values": CVperatom, "target": "atom"},
            'cv': {"values": CV, "target": "structure"},},
        mode="structure",
        settings=chemiscope.quick_settings(trajectory=True, structure_settings={"unitCell":True,
                'environments': {'activated': True}, 'map': {'color': {'property': 'o_cv'}}, 'color': {'property': 'o_cv', 'min': -0.1,
                                'max': 0.1, 'transform': 'linear','palette': 'bwr'} }),
        environments=envs, #[(j,i,3.5) for j in range(len(structures)) for i in io]
    )


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
