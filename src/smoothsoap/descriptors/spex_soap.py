import math
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
    systems_to_torch,
)

from spex import SphericalExpansion


class SPEX_CV(torch.nn.Module):
    """SOAP-PS -> PCA projection -> collective variable, fully torch-scriptable.

    Drop-in replacement for the featomic-backed ``SOAP_CV``: identical
    constructor signature, identical method names and signatures, identical
    ``forward`` output (``{"features": TensorMap, "features/per_atom": TensorMap}``),
    and identical ``self.soap_block`` layout::

        samples    (n_centers): ['system', 'atom', 'center_type']
        components (        ): []
        properties (n_feats  ): ['neighbor_1_type', 'neighbor_2_type',
                                 'l', 'n_1', 'n_2']

    Internally the SOAP power spectrum is computed with ``torch-spex``
    (``SphericalExpansion`` + manual PS contraction). Neighbor lists use
    metatomic's script-safe ``NeighborListOptions`` mechanism: PLUMED/ASE/
    whatever runtime hosts the saved model attaches the NL to each ``System``
    and the model retrieves it via ``system.get_neighbor_list(options)``. No
    C++-bound neighbor-list class is stored on the module, so
    ``torch.jit.script`` works.
    """

    def __init__(self, cutoff, max_angular, max_radial, centers, neighbors,
                 projection_matrix=None):
        super().__init__()

        # ------------------------------------------------------------------
        # Hyperparameters
        # ------------------------------------------------------------------
        self.cutoff = float(cutoff)
        self.max_angular = int(max_angular)
        self.max_radial = int(max_radial)
        self.centers = sorted([int(c) for c in centers])
        self.neighbors = sorted([int(n) for n in neighbors])

        # ------------------------------------------------------------------
        # Featomic-equivalent ``selected_keys`` (kept for parity / inspection)
        # ------------------------------------------------------------------
        self.selected_keys = Labels(
            names=["center_type", "neighbor_1_type", "neighbor_2_type"],
            values=torch.tensor(
                [[i, j, k] for i in self.centers
                           for j in self.neighbors
                           for k in self.neighbors if j <= k],
                dtype=torch.int32,
            ),
        )

        self.id = f"SOAP_{cutoff}{max_angular}{max_radial}_{centers}"

        # ------------------------------------------------------------------
        # Script-safe neighbor list: declare options, request from System.
        # ``NeighborListOptions`` is a TorchScript class, so this is fine
        # to keep as an attribute.
        # ------------------------------------------------------------------
        self.nl_options = NeighborListOptions(
            cutoff=self.cutoff,
            full_list=True,
            strict=True,
        )

        # ------------------------------------------------------------------
        # spex calculator: ``self.calculator`` plays the role that
        # ``SoapPowerSpectrum`` played in the featomic version.
        # ------------------------------------------------------------------
        self.calculator = SphericalExpansion(
            cutoff=self.cutoff,
            max_angular=self.max_angular,
            radial={"LaplacianEigenstates":
                    {"max_radial": self.max_radial, "trim": True}},
            angular="SphericalHarmonics",
            species={"Orthogonal": {"species": self.neighbors}},
            cutoff_function={"ShiftedCosine": {"width": 0.5}},
        ).to(dtype=torch.float64)

        # ------------------------------------------------------------------
        # Projection matrix (mirrors SOAP_CV; the original had a typo
        # referring to ``trans_matrix`` -- here we use the actual argument).
        # ------------------------------------------------------------------
        if projection_matrix is not None:
            self.register_buffer(
                "projection_matrix",
                torch.tensor(projection_matrix.copy()).T,
            )
        else:
            self.register_buffer("projection_matrix", None)

        self.register_buffer("mu", torch.zeros(1, dtype=torch.float64))
        self.hypers: Dict[str, str] = {}

        # ------------------------------------------------------------------
        # Static tables for the PS contraction (script-friendly inner loops)
        # ------------------------------------------------------------------
        ps_pairs: List[Tuple[int, int, bool]] = []
        for nt1_idx in range(len(self.neighbors)):
            for nt2_idx in range(len(self.neighbors)):
                if self.neighbors[nt1_idx] <= self.neighbors[nt2_idx]:
                    ps_pairs.append((
                        nt1_idx, nt2_idx,
                        self.neighbors[nt1_idx] != self.neighbors[nt2_idx],
                    ))
        self._ps_pairs = ps_pairs

        norms: List[float] = []
        for l in range(self.max_angular + 1):
            norms.append((-1.0) ** l / math.sqrt(2.0 * l + 1.0))
        self._norms = norms
        self._cross_factor = math.sqrt(2.0)

        # n_radial per l (LaplacianEigenstates with trim=True varies per l).
        n_per_l: List[int] = list(self.calculator.radial.n_per_l)
        self._n_per_l = n_per_l

        # Property index table: [neighbor_1_type, neighbor_2_type, l, n_1, n_2]
        # in the exact order ``_compute_ps`` emits features.
        prop_rows: List[List[int]] = []
        for nt1_idx, nt2_idx, _is_cross in ps_pairs:
            nt1 = self.neighbors[nt1_idx]
            nt2 = self.neighbors[nt2_idx]
            for l in range(self.max_angular + 1):
                n_rad_l = n_per_l[l]
                for n1 in range(n_rad_l):
                    for n2 in range(n_rad_l):
                        prop_rows.append([nt1, nt2, l, n1, n2])
        self.register_buffer(
            "_ps_property_values",
            torch.tensor(prop_rows, dtype=torch.int32),
        )
        self.proj_dims = [0]

    # ----------------------------------------------------------------------
    # metatomic NL contract: declare what NL the engine should provide.
    # ----------------------------------------------------------------------
    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [self.nl_options]

    # ----------------------------------------------------------------------
    # Internal: SOAP-PS computation (single system).
    # Pulls the NL from the System (script-safe), then computes spex
    # expansion + manual power-spectrum contraction.
    # ----------------------------------------------------------------------
    def _compute_soap(
        self,
        systems: List[System],
        selected_atoms: Optional[Labels],
    ) -> Tuple[torch.Tensor, Labels]:
        """Returns (ps_values, samples_with_center_type)."""
        system = systems[0]
        positions = system.positions
        cell = system.cell
        species = system.types

        # --- Pull NL from System (attached by the runtime / by calculate) ---
        nl_block = system.get_neighbor_list(self.nl_options)
        nl_samples = nl_block.samples.values
        i_idx = nl_samples[:, 0].to(torch.long)
        j_idx = nl_samples[:, 1].to(torch.long)
        S = nl_samples[:, 2:5]  # [n_pairs, 3], integer cell shifts

        # --- Center mask: atoms whose type is in self.centers ---
        center_mask = torch.zeros(
            species.shape[0], dtype=torch.bool, device=species.device,
        )
        for ct in self.centers:
            center_mask = center_mask | (species == ct)

        # --- Optional restriction by metatomic ``selected_atoms`` ---
        if selected_atoms is not None:
            sel_mask = torch.zeros(
                species.shape[0], dtype=torch.bool, device=species.device,
            )
            sel_atom_idx = selected_atoms.values[:, -1].to(torch.long)
            sel_mask[sel_atom_idx] = True
            center_mask = center_mask & sel_mask

        # --- Pairs where i is a center ---
        cmask = center_mask[i_idx]
        i_idx = i_idx[cmask]
        j_idx = j_idx[cmask]
        S = S[cmask]

        # --- Self-pairs for centers (n=n', a=a' diagonal) ---
        center_indices = center_mask.nonzero().squeeze(-1)
        n_self = center_indices.shape[0]
        i_idx = torch.cat([i_idx, center_indices])
        j_idx = torch.cat([j_idx, center_indices])
        S = torch.cat([S, torch.zeros(
            n_self, 3, dtype=S.dtype, device=S.device,
        )])

        # --- Differentiable R_ij from positions (autograd flows through) ---
        S_f = S.to(positions.dtype)
        R_ij = positions[j_idx] - positions[i_idx] + S_f @ cell

        # tiny epsilon on self-pairs to dodge sphericart NaN (torch-spex #27)
        eps = torch.zeros_like(R_ij)
        if n_self > 0:
            eps[-n_self:, 0] = 1e-30
        R_ij = R_ij + eps

        # --- Spherical expansion + power spectrum ---
        expansion = self.calculator(R_ij, i_idx, j_idx, species)
        ps = self._compute_ps(expansion, center_mask)

        # --- featomic-shaped per-atom Labels ---
        n_centers = center_indices.shape[0]
        sys_col = torch.zeros((n_centers, 1), dtype=torch.int32, device=species.device)
        atom_col = center_indices.to(torch.int32).unsqueeze(-1)
        ctype_col = species[center_indices].to(torch.int32).unsqueeze(-1)
        samples_per_atom = Labels(
            ["system", "atom", "center_type"],
            torch.cat([sys_col, atom_col, ctype_col], dim=-1),
        )

        return ps, samples_per_atom

    def _compute_ps(
        self,
        expansion: List[torch.Tensor],
        center_mask: torch.Tensor,
    ) -> torch.Tensor:
        """SOAP power spectrum, featomic-compatible feature ordering.

        For each (nt1, nt2) with nt1 <= nt2, for each l, for each n1, n2:
            PS = (-1)^l / sqrt(2l+1) * sum_m c_{n1,l,m,nt1} c_{n2,l,m,nt2}
        with an extra sqrt(2) factor for cross-species (nt1 != nt2).
        """
        ps_blocks: List[torch.Tensor] = []
        for nt1_idx, nt2_idx, is_cross in self._ps_pairs:
            for l in range(self.max_angular + 1):
                c = expansion[l][center_mask]    # [n_centers, 2l+1, n_rad_l, n_species]
                a = c[:, :, :, nt1_idx]
                b = c[:, :, :, nt2_idx]
                ps_l = torch.einsum("smn,smN->snN", a, b)
                norm = self._norms[l]
                if is_cross:
                    norm = norm * self._cross_factor
                ps_blocks.append((ps_l * norm).reshape(c.shape[0], -1))
        return torch.cat(ps_blocks, dim=-1)

    # ----------------------------------------------------------------------
    # Public API (matches SOAP_CV).
    # ``calculate`` and the setters use Python-only / vesin / ase code; we
    # mark them ``@torch.jit.ignore`` so they don't have to be scriptable.
    # ----------------------------------------------------------------------
    @torch.jit.ignore
    def calculate(self, systems, selected_samples=None):
        if selected_samples is None:
            selected_samples = self.selected_samples

        # Make sure each system has the NL we need attached. This is the
        # offline path (training/setup); at PLUMED runtime the engine
        # attaches NLs based on requested_neighbor_lists().
        for system in systems:
            self._ensure_nl(system)

        ps, samples_per_atom = self._compute_soap(systems, selected_samples)

        properties = Labels(
            names=["neighbor_1_type", "neighbor_2_type", "l", "n_1", "n_2"],
            values=self._ps_property_values,
        )

        self.soap_block = TensorBlock(
            values=ps,
            samples=samples_per_atom,
            components=[],
            properties=properties,
        )
        return self.soap_block.values

    @torch.jit.ignore
    def _ensure_nl(self, system):
        """Attach a NL matching ``self.nl_options`` to ``system`` if missing.

        Uses ``vesin.torch.NeighborList`` *locally* (function-scoped import)
        so vesin never becomes part of the module state -- which would
        otherwise break ``torch.jit.script`` because the underlying C++
        class lacks pickle/serialization hooks.
        """
        # If already attached, nothing to do.
        for opts in system.known_neighbor_lists():
            if opts == self.nl_options:
                return

        from vesin.torch import NeighborList as _VesinNL

        nl = _VesinNL(cutoff=self.cutoff, full_list=True)
        positions = system.positions
        cell = system.cell

        i, j, S, D = nl.compute(
            points=positions.detach(),
            box=cell.detach(),
            periodic=True,
            quantities="ijSD",
        )

        samples = Labels(
            names=[
                "first_atom", "second_atom",
                "cell_shift_a", "cell_shift_b", "cell_shift_c",
            ],
            values=torch.stack([
                i.to(torch.int32), j.to(torch.int32),
                S[:, 0].to(torch.int32),
                S[:, 1].to(torch.int32),
                S[:, 2].to(torch.int32),
            ], dim=-1),
        )
        components = [Labels(
            names=["xyz"],
            values=torch.tensor([[0], [1], [2]], dtype=torch.int32),
        )]
        properties = Labels(
            names=["distance"],
            values=torch.tensor([[0]], dtype=torch.int32),
        )
        nl_block = TensorBlock(
            values=D.reshape(-1, 3, 1).to(positions.dtype),
            samples=samples,
            components=components,
            properties=properties,
        )
        system.add_neighbor_list(self.nl_options, nl_block)

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        
        device = systems[0].positions.device
        if "features" not in outputs:
            return {}

        if outputs["features"].per_atom:
            raise ValueError(
                "per_atom=True is not supported directly, "
                "output will be in features/per_atom"
            )

        if len(systems[0]) == 0:
            # PLUMED is trying to determine the size of the output
            projected = torch.zeros(
                (0, len(self.proj_dims)), dtype=torch.float64, device=device
            )
            projected_mean = torch.zeros(
                (0, len(self.proj_dims)), dtype=torch.float64, device=device
            )
            samples = Labels(
                ["system"], torch.zeros((0, 1), dtype=torch.int32, device=device),
            )
            samples_per_atom = Labels(
                ["system", "atom"], torch.zeros((0, 2), dtype=torch.int32, device=device),
            )
        else:
            ps, soap_samples = self._compute_soap(systems, selected_atoms)

            projected = torch.einsum(
                'ij,jk->ik',
                (ps - self.mu),
                self.projection_matrix[:, self.proj_dims],
            )

            # Drop "center_type" -> ("system","atom"), like SOAP_CV.forward.
            samples_per_atom = soap_samples.remove("center_type")
            samples = Labels(
                ["system"], torch.zeros((1, 1), dtype=torch.int32, device=device),
            )

            projected_mean = torch.mean(projected, dim=0)
            projected_mean = projected_mean.unsqueeze(0)

            # existing line stays the same
            ps, soap_samples = self._compute_soap(systems, selected_atoms)

            projected = torch.einsum(
                'ij,jk->ik',
                (ps - self.mu),
                self.projection_matrix[:, self.proj_dims],
            )

            samples_per_atom = soap_samples.remove("center_type")
            samples = Labels(
                ["system"], torch.zeros((1, 1), dtype=torch.int32, device=device),  # add device
            )
            projected_mean = torch.mean(projected, dim=0).unsqueeze(0)

        block_per_atom = TensorBlock(
            values=projected,
            samples=samples_per_atom,
            components=[],
            properties=Labels(
                "soap_pca",
                torch.tensor(self.proj_dims, dtype=torch.int, device=device).unsqueeze(-1)
            ),
        )
        cv_per_atom = TensorMap(
            keys=Labels("_", torch.tensor([[0]])),
            blocks=[block_per_atom],
        )

        block = TensorBlock(
            values=projected_mean,
            samples=samples,
            components=[],
            properties=Labels(
                "soap_pca",
                torch.tensor(self.proj_dims, dtype=torch.int, device=device).unsqueeze(-1)
            ),
        )
        cv = TensorMap(
            keys=Labels("_", torch.tensor([[0]])),
            blocks=[block],
        )
        return {"features": cv, "features/per_atom": cv_per_atom}

    # ----------------------------------------------------------------------
    # Setters / utilities (identical to the featomic SOAP_CV).
    # All ignored by the scripter -- they're setup-time helpers.
    # ----------------------------------------------------------------------
    @torch.jit.ignore
    def set_samples(self, selected_atoms):
        self.selected_samples = Labels(
            names=["atom"],
            values=torch.tensor(selected_atoms, dtype=torch.int32).unsqueeze(-1),
        )

    @torch.jit.ignore
    def set_atom_types(self, trj):
        types = [i.number for j in trj for w in j for i in w]
        self.atomic_types = sorted(set(types), key=types.index)

    @torch.jit.ignore
    def set_projection_dims(self, dims):
        self.proj_dims = dims

    def set_projection_mu(self, mu):
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float64))

    @torch.jit.ignore
    def update_hypers(self, hypers):  # hypers has to be dict
        self.hypers.update({key: str(val) for key, val in hypers.items()})
 
    def set_projection_matrix(self, matrix):
        if isinstance(matrix, torch.Tensor):
            mat_t = matrix.detach().clone()
        else:
            mat_t = torch.tensor(matrix.copy())
        self.register_buffer("projection_matrix", mat_t)

    @torch.jit.ignore
    def save_model(self, path='.', name='soap_model'):
        capabilities = ModelCapabilities(
            outputs={
                "features": ModelOutput(per_atom=False),
                "features/per_atom": ModelOutput(per_atom=True),
            },
            interaction_range=10.0,
            supported_devices=["cuda", "cpu"],
            length_unit="A",
            atomic_types=self.atomic_types,
            dtype="float64",
        )

        metadata = ModelMetadata(
            name="SOAP based CV",
            authors=['SmoothSOAP'],
            description='Hyperparameters in extra',
            extra=self.hypers,
        )
        model = AtomisticModel(self, metadata, capabilities)
        model.save(
            "{}/{}.pt".format(path, name),
            collect_extensions=f"{path}/extensions",
        )
        print(f'model saved at {path}/{name}.pt')