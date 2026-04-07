from curses import wrapper
import os
from typing import Dict, List, Optional
import argparse
from xml.parsers.expat import model

import torch
import numpy as np
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelEvaluationOptions,
    ModelOutput,
    System,
    systems_to_torch,
    load_atomistic_model
)
from pathlib import Path
from torch.profiler import record_function
from smoothsoap.descriptors.model_soap import SOAP_CV


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

    def save_checkpoint(self, path='.', name='wrapper'):
        os.makedirs(path, exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),          # all buffers, incl. model.mu, model.proj_dims, etc.
            'soap_init_params': self.model._init_params,
            'soap_hypers': self.model.hypers,
            'wrapper_params': {
                'zmin': self.zmin,
                'zmax': self.zmax,
                'nlistoptions': self.nlistoptions,
            },
        }, f"{path}/{name}.ckpt")

    @classmethod
    def load_checkpoint(cls, path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)

        # reconstruct the inner model
        soap = SOAP_CV(**ckpt['soap_init_params'])
        soap.hypers = ckpt['soap_hypers']

        # reconstruct the wrapper
        wp = ckpt['wrapper_params']
        wrapper = cls(soap, wp['zmin'], wp['zmax'], wp['nlistoptions'])

        # load all buffers in one go — state_dict handles model.mu, model.proj_dims, etc.
        wrapper.load_state_dict(ckpt['state_dict'])

        return wrapper


class Wrapper(torch.nn.Module):
    def __init__(self, model, zmin, zmax, nlistoptions):
        super().__init__()
        self.model = model
        self.zmin = zmin
        self.zmax = zmax
        self.nlistoptions = nlistoptions
        self._soap_init_params = {}
        self._soap_hypers = {}
        self._capabilities = None
        self._metadata = None

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
                return Labels(
                    names=["system", "atom"],
                    values=torch.tensor([[0, int(i)] for i in [0]]),
                )
            return Labels(
                names=["system", "atom"],
                values=torch.tensor([[0, int(i)] for i in indices[mask]]),
            )
        else:
            pos = systems[0].positions[selected_atoms.values[:, 1]]
            mask = (pos[:, 2] > self.zmin) & (pos[:, 2] < self.zmax)
            return Labels(
                names=selected_atoms.names,
                values=selected_atoms.values[mask],
            )

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        newselected_atoms = self.pre_selected(systems, selected_atoms)
        return self.model(systems, outputs, newselected_atoms)

    def requested_neighbor_lists(self):
        return self.nlistoptions

    def save_checkpoint(self, path='.', name='wrapper'):
        os.makedirs(path, exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'soap_init_params': self._soap_init_params,
            'soap_hypers': self._soap_hypers,
            'capabilities': self._capabilities,
            'metadata': self._metadata,
            'wrapper_params': {
                'zmin': self.zmin,
                'zmax': self.zmax,
                'nlistoptions': self.nlistoptions,
            },
        }, f"{path}/{name}.ckpt")
        print(f'Checkpoint saved at {path}/{name}.ckpt')

    @classmethod
    def load_checkpoint(cls, path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        soap = SOAP_CV(**ckpt['soap_init_params'])
        soap.hypers = ckpt['soap_hypers']
        wp = ckpt['wrapper_params']
        wrapper = cls(soap, wp['zmin'], wp['zmax'], wp['nlistoptions'])
        wrapper.load_state_dict(ckpt['state_dict'])
        wrapper._capabilities = ckpt['capabilities']
        wrapper._metadata = ckpt['metadata']
        return wrapper

    @classmethod
    def load_checkpoint(cls, path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        soap = SOAP_CV(**ckpt['soap_init_params'])
        soap.hypers = ckpt['soap_hypers']
        wp = ckpt['wrapper_params']
        wrapper = cls(soap, wp['zmin'], wp['zmax'], wp['nlistoptions'])

        # register buffers directly — None buffers don't survive state_dict round-trip
        state = ckpt['state_dict']
        for key in ['projection_matrix', 'mu', 'proj_dims', 'atomic_types']:
            tensor = state.get(f'model.{key}')
            if tensor is not None:
                wrapper.model.register_buffer(key, tensor)

        wrapper._capabilities = ckpt.get('capabilities')
        wrapper._metadata = ckpt.get('metadata')
        return wrapper
    #def save_model(self, metadata, capabilities, path='.', name='wrapper'):
    #    os.makedirs(path, exist_ok=True)
    ##    wrapper = AtomisticModel(self, metadata, capabilities)
    #    print(f"Saving to {path}/{name}.pt")
    #    wrapper.save(f"{path}/{name}.pt", collect_extensions=f"{path}/extensions")

    def save_model(self, path='.', name='wrapper'):
        os.makedirs(path, exist_ok=True)
        atomistic = AtomisticModel(self, self._metadata, self._capabilities)
        atomistic.save(f"{path}/{name}.pt", collect_extensions=f"{path}/extensions")
        print(f"Saved to {path}/{name}.pt")
        
    @classmethod
    def from_torchscript(cls, pt_path, zmin, zmax, soap_hypers=None):
        pt_path = Path(pt_path)
        model = load_atomistic_model(
            str(pt_path),
            extensions_directory=str(pt_path.parent / 'extensions')
        )

        inner = model.module
        soap_init_params = {
            'cutoff': inner.cutoff,
            'max_angular': inner.max_angular,
            'max_radial': inner.max_radial,
            'centers': list(inner.centers),
            'neighbors': list(inner.neighbors),
        }

        wrapper = cls(inner, zmin, zmax, nlistoptions=model.requested_neighbor_lists())
        wrapper._soap_init_params = soap_init_params
        wrapper._soap_hypers = soap_hypers or {}
        wrapper._capabilities = model.capabilities()
        wrapper._metadata = model.metadata()
        return wrapper

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

    wrapper = Wrapper.from_torchscript(str(model_path), zmin=zmin, zmax=zmax)
    wrapper.save_checkpoint(path='.', name=f'soap_wrapper_zmin{zmin}_zmax{zmax}')

    # later — load and re-export, capabilities fully preserved
    wrapper = Wrapper.load_checkpoint(f'soap_wrapper_zmin{zmin}_zmax{zmax}.ckpt')
    wrapper.eval()
    wrapper.save_model(path=".", name="soap_wrapper")
    exit()


    model = load_atomistic_model(str(model_path), extensions_directory=f"{model_path.parent / 'extensions'}")
    print(model.module.cutoff)
    print(model.capabilities().outputs)
    wrapper = Wrapper(model.module, zmin, zmax, nlistoptions=model.requested_neighbor_lists())
    wrapper.save_checkpoint(path='.', name=f'soap_wrapper_zmin{zmin}_zmax{zmax}')
    wrapper.eval()
    print("saving wrapper ...")
    wrapper.save_model(model.metadata(), model.capabilities(), path='.', name=f'soap_wrapper_zmin{zmin}_zmax{zmax}')
    #wrapper = load_atomistic_model(f'./soap_wrapper_zmin{zmin}_zmax{zmax}.pt', extensions_directory='./extensions')
   
    exit()
    wrapper = Wrapper.from_torchscript(
    pt_path="path/to/soap_model.pt",
    zmin=0.5,
    zmax=15.0,
    nlistoptions=your_nlist_options,
    )
    wrapper.save_checkpoint(path="output", name="wrapper")