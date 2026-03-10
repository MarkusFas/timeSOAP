import torch 
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import moment
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
import argparse
from pathlib import Path
from torch.profiler import record_function


class Wrapper(torch.nn.Module):
    def __init__(self, model, zmin, zmax, nlistoptions):
        super().__init__()
        self.model = model
        self.zmin = zmin
        self.zmax = zmax
        self.nlistoptions = nlistoptions
    

    def wrap(self, systems: List[System]):
        cell = systems[0].cell
        z = systems[0].positions[:, 2]
        shift = ()//(cell[2, 2] - cell[2, 2])


    def pre_selected(
            self,
            systems: List[System],
            selected_atoms: Optional[Labels]
    ) -> Labels:
        if selected_atoms is None:
            pos = systems[0].positions
            mask = (pos[:, 2] > self.zmin) & (pos[:, 2] < self.zmax)
            indices = torch.arange(len(pos))
            if len(indices[mask]) == 0:
                newselected_atoms = Labels(
                    names=["system", "atom"],
                    values=torch.tensor([[0, int(i)] for i in [0]]),
                )
                return newselected_atoms

            newselected_atoms = Labels(
                names=["system", "atom"],
                values=torch.tensor([[0, int(i)] for i in indices[mask]]),
            )
        else:
            pos = systems[0].positions[selected_atoms.values[:,1]]
            mask = (pos[:, 2] > self.zmin) & (pos[:, 2] < self.zmax)
            newselected_atoms = Labels(
                names=selected_atoms.names,
                values=selected_atoms.values[mask],
            )
        return newselected_atoms

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        
        newselected_atoms = self.pre_selected(systems, selected_atoms)
        features = self.model(systems, outputs, newselected_atoms)
        return features

    def requested_neighbor_lists(self):
        return self.nlistoptions

    def save_model(self, metadata, capabilities, path='.', name='wrapper'):
        wrapper = AtomisticModel(self, metadata, capabilities)
        print("saving to {}/{}.pt".format(path, name))
        wrapper.save("{}/{}.pt".format(path, name), collect_extensions=f"{path}/extensions")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SOAP wrapper based on z-coordinate filtering')
    parser.add_argument('model_path', type=str, help='model to wrap')
    parser.add_argument('--zmin', type=float, default=-np.inf, help='minimum z-coordinate')
    parser.add_argument('--zmax', type=float, default=np.inf, help='maximum z-coordinate')
    parser.add_argument('--per_atom', type=bool, default=False, help='whether to return per-atom features')

    model_path = Path(parser.parse_args().model_path)
    zmin = parser.parse_args().zmin
    zmax = parser.parse_args().zmax
    per_atom = parser.parse_args().per_atom

    #model_path = Path('/Users/markusfasching/EPFL/Work/project-SOAP/scripts/SOAP-time-code/results/icewaterinterface4ns_oxygen_proper_testset_1ps/v0/SOAP/866/testLDA/LDA/interval_100/lag_0/sigma_0/ridge_a1e-05/SOAP_866_[8]/model_soap.pt')
    #zmin = 25
    #zmax=31
    model = load_atomistic_model(str(model_path), extensions_directory=f"{model_path.parent / 'extensions'}")
    print(model.capabilities().outputs)
    wrapper = Wrapper(model.module, zmin, zmax, nlistoptions=model.requested_neighbor_lists())
    wrapper.eval()
    print("saving wrapper ...")
    wrapper.save_model(model.metadata(), model.capabilities(), path='.', name=f'soap_wrapper_zmin{zmin}_zmax{zmax}')
    #wrapper = load_atomistic_model(f'./soap_wrapper_zmin{zmin}_zmax{zmax}.pt', extensions_directory='./extensions')
    exit() 
    from ase.io import read, write
    #structures = read('/Users/markusfasching/EPFL/Work/project-SOAP/scripts/SOAP-time-code/data/icemeltinterface/nobias/positions.lammpstrj', index=':')
    structures = read('/Users/markusfasching/EPFL/Work/project-SOAP/scripts/SOAP-time-code/data/icemeltinterface/nobias/short.lammpstrj', index=':')
    structures = structures[:10]
    systems = systems_to_torch(structures, dtype=torch.float64)
    wrapper = load_atomistic_model(f'./soap_wrapper_zmin{zmin}_zmax{zmax}.pt', extensions_directory='./extensions')
    systems_new = []
    for i, system in enumerate(systems):

        #atoms = structures[i]
        nlistoptions = wrapper.requested_neighbor_lists()[0]
        print(nlistoptions)
        
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

    systems = systems_new
    selected_atoms = Labels(
        names=["system", "atom"],
        values=torch.tensor([[0, j] for j in np.arange(0, len(structures[0]), 3)]))
    
    outputs={"features": ModelOutput(per_atom=False)}
    if per_atom:
        outputs["features/per_atom"] = ModelOutput(per_atom=True)

    evaloptions = ModelEvaluationOptions(
        length_unit='',
        outputs=outputs,
        selected_atoms=selected_atoms,
    )
    print("evaluating wrapper ...")
    cv = wrapper(
        systems=systems,
        options=evaloptions,
        check_consistency=True,
    )
