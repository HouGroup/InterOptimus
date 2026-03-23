"""
Level-1 interface inpainting pipeline built on top of MatterGen + InterOptimus.

Design goals:
- Keep implementation lightweight and optional-dependency friendly.
- Use MatterGen pre-trained checkpoints without fine-tuning.
- Only inpaint local interface atoms (20-40 atoms typical), not full supercells.
- Preserve known bulk environments and rank candidates via fast MLIP relaxation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import os
import sys

import numpy as np
import torch
from pymatgen.core import Structure


@dataclass
class InterfacePartition:
    substrate_bulk_mask: np.ndarray
    film_bulk_mask: np.ndarray
    interface_mask: np.ndarray
    split_frac: float
    axis: int


@dataclass
class PatchData:
    patch_structure: Structure
    global_indices: List[int]
    # Local index -> global index in full supercell.
    # None means this local site is a virtual candidate (no original atom).
    local_to_global: List[Optional[int]]
    fixed_local_indices: List[int]
    unknown_local_indices: List[int]
    virtual_unknown_local_indices: List[int]


@dataclass
class InpaintingConfig:
    axis: int = 2
    split_frac: float = 0.5
    # Thickness for each interface diffusion region (Angstrom).
    # In a double-interface cell this gives "two x 5 A" when set to 5.
    interface_region_thickness_angstrom: float = 5.0
    # Backward-compatible fractional half-width fallback.
    interface_half_width: float = 0.12
    # Keep generated unknown atoms inside interface bands.
    enforce_interface_region: bool = True
    # Scheme-2 (overcomplete sites + occupancy selection)
    use_variable_occupancy: bool = True
    num_virtual_unknown_sites: int = 8
    max_added_atoms: Optional[int] = None
    occupancy_neighbor_cutoff: float = 3.0
    occupancy_min_neighbors: int = 1
    occupancy_min_distance: float = 1.6
    random_seed: int = 42
    max_patch_atoms: int = 40
    min_atoms_per_bulk_side: int = 8
    num_samples: int = 32
    batch_size: int = 8
    mattergen_pretrained_name: str = "mattergen_base"
    mattergen_model_path: Optional[str] = None
    mattergen_repo_path: Optional[str] = None
    sampling_config_name: str = "interface_fast"
    sampling_config_overrides: Optional[List[str]] = None
    record_trajectories: bool = False
    enforce_allowed_elements: bool = True
    # Keep this False by default so diffusion can discover termination
    # (i.e., how many atoms from either slab side should populate interface regions).
    enforce_unknown_stoichiometry: bool = False
    mlip_calc: str = "orb-models"
    mlip_user_settings: Optional[Dict[str, Any]] = None
    mlip_relax_steps: int = 30
    mlip_relax_fmax: float = 0.1
    top_k: int = 5


def _periodic_1d_distance(a: np.ndarray, b: float) -> np.ndarray:
    """Shortest distance on [0,1) circle."""
    d = np.abs(a - b)
    return np.minimum(d, 1.0 - d)


def partition_double_interface(
    structure: Structure,
    axis: int = 2,
    split_frac: float = 0.5,
    interface_region_thickness_angstrom: Optional[float] = None,
    interface_half_width: float = 0.12,
) -> InterfacePartition:
    """
    Partition a double-interface supercell into substrate-bulk / film-bulk / interface.

    Assumption: along the selected axis, substrate occupies [0, split_frac),
    film occupies [split_frac, 1), and periodic boundaries create two interfaces
    near split_frac and 0.
    """
    frac = np.mod(np.asarray(structure.frac_coords), 1.0)
    z = frac[:, axis]

    dist_to_split = _periodic_1d_distance(z, split_frac)
    dist_to_zero = _periodic_1d_distance(z, 0.0)
    dist_to_interface = np.minimum(dist_to_split, dist_to_zero)
    if interface_region_thickness_angstrom is not None:
        axis_len = float(np.linalg.norm(np.asarray(structure.lattice.matrix)[axis]))
        half_width_frac = 0.5 * float(interface_region_thickness_angstrom) / max(axis_len, 1e-8)
        interface_mask = dist_to_interface <= half_width_frac
    else:
        interface_mask = dist_to_interface <= interface_half_width

    substrate_bulk_mask = (z < split_frac) & (~interface_mask)
    film_bulk_mask = (z >= split_frac) & (~interface_mask)

    return InterfacePartition(
        substrate_bulk_mask=substrate_bulk_mask,
        film_bulk_mask=film_bulk_mask,
        interface_mask=interface_mask,
        split_frac=split_frac,
        axis=axis,
    )


def _select_patch_indices(
    structure: Structure,
    partition: InterfacePartition,
    max_patch_atoms: int = 40,
    min_atoms_per_bulk_side: int = 8,
) -> List[int]:
    """Select all interface atoms + nearest bulk context atoms until max_patch_atoms."""
    interface_idx = np.where(partition.interface_mask)[0].tolist()
    sub_idx = np.where(partition.substrate_bulk_mask)[0].tolist()
    film_idx = np.where(partition.film_bulk_mask)[0].tolist()

    if not interface_idx:
        raise ValueError("No interface atoms selected. Increase interface_half_width.")

    if len(interface_idx) >= max_patch_atoms:
        frac = np.mod(np.asarray(structure.frac_coords), 1.0)
        z = frac[:, partition.axis]
        d = np.minimum(_periodic_1d_distance(z, partition.split_frac), _periodic_1d_distance(z, 0.0))
        sorted_if = sorted(interface_idx, key=lambda i: float(d[i]))
        return sorted(sorted_if[:max_patch_atoms])

    dmat = np.asarray(structure.distance_matrix)

    def nearest_to_interface(candidates: Sequence[int], n_pick: int) -> List[int]:
        if n_pick <= 0 or not candidates:
            return []
        scored = []
        for i in candidates:
            dmin = float(np.min(dmat[i, interface_idx]))
            scored.append((dmin, i))
        scored.sort(key=lambda x: x[0])
        return [i for _, i in scored[: min(n_pick, len(scored))]]

    selected = set(interface_idx)
    budget = max_patch_atoms - len(selected)
    half = budget // 2

    pick_sub = nearest_to_interface(sub_idx, max(min_atoms_per_bulk_side, half))
    pick_film = nearest_to_interface(film_idx, max(min_atoms_per_bulk_side, half))
    selected.update(pick_sub)
    selected.update(pick_film)

    if len(selected) < max_patch_atoms:
        rest_pool = [i for i in (sub_idx + film_idx) if i not in selected]
        rest = nearest_to_interface(rest_pool, max_patch_atoms - len(selected))
        selected.update(rest)

    selected_list = sorted(selected)
    return selected_list[:max_patch_atoms]


def build_patch(
    structure: Structure,
    partition: InterfacePartition,
    max_patch_atoms: int = 40,
    min_atoms_per_bulk_side: int = 8,
) -> PatchData:
    """Build local patch structure and local/global index maps."""
    selected_global = _select_patch_indices(
        structure=structure,
        partition=partition,
        max_patch_atoms=max_patch_atoms,
        min_atoms_per_bulk_side=min_atoms_per_bulk_side,
    )
    species = [structure[i].specie for i in selected_global]
    fcoords = [structure.frac_coords[i] for i in selected_global]
    patch = Structure(lattice=structure.lattice, species=species, coords=fcoords, coords_are_cartesian=False)

    sub_global = set(np.where(partition.substrate_bulk_mask)[0].tolist())
    film_global = set(np.where(partition.film_bulk_mask)[0].tolist())
    interface_global = set(np.where(partition.interface_mask)[0].tolist())

    fixed_local = []
    unknown_local = []
    for local_i, global_i in enumerate(selected_global):
        if global_i in interface_global:
            unknown_local.append(local_i)
        elif global_i in sub_global or global_i in film_global:
            fixed_local.append(local_i)

    return PatchData(
        patch_structure=patch,
        global_indices=selected_global,
        local_to_global=selected_global.copy(),
        fixed_local_indices=fixed_local,
        unknown_local_indices=unknown_local,
        virtual_unknown_local_indices=[],
    )


def _add_virtual_unknown_sites(
    supercell: Structure,
    patch: PatchData,
    partition: InterfacePartition,
    num_virtual_unknown_sites: int,
    seed: int = 42,
) -> PatchData:
    if num_virtual_unknown_sites <= 0:
        return patch
    rng = np.random.default_rng(seed)
    axis = int(partition.axis)
    split = float(partition.split_frac) % 1.0
    frac = np.mod(np.asarray(supercell.frac_coords), 1.0)
    z = frac[:, axis]
    # Estimate interface half-width from current unknown atoms.
    if patch.unknown_local_indices:
        unknown_globals = [patch.global_indices[i] for i in patch.unknown_local_indices if i < len(patch.global_indices)]
        if unknown_globals:
            uz = z[np.asarray(unknown_globals, dtype=int)]
            d_split = _periodic_1d_distance(uz, split)
            d_zero = _periodic_1d_distance(uz, 0.0)
            half = float(np.max(np.minimum(d_split, d_zero)))
        else:
            half = 0.12
    else:
        half = 0.12

    species = [s.specie for s in patch.patch_structure]
    fcoords = [np.asarray(c, dtype=float) for c in patch.patch_structure.frac_coords]
    local_to_global = list(patch.local_to_global)
    fixed_local = list(patch.fixed_local_indices)
    unknown_local = list(patch.unknown_local_indices)
    virtual_local: List[int] = []

    # Use allowed species pool from existing patch.
    allowed_species = [str(s) for s in species]
    if not allowed_species:
        return patch

    def _sample_interface_z() -> float:
        center = 0.0 if rng.random() < 0.5 else split
        dz = rng.uniform(-half, half)
        return (center + dz) % 1.0

    for _ in range(num_virtual_unknown_sites):
        base = fcoords[rng.integers(0, len(fcoords))].copy()
        base[axis] = _sample_interface_z()
        sp = allowed_species[rng.integers(0, len(allowed_species))]
        species.append(sp)
        fcoords.append(base)
        local_to_global.append(None)
        li = len(fcoords) - 1
        unknown_local.append(li)
        virtual_local.append(li)

    st = Structure(lattice=patch.patch_structure.lattice, species=species, coords=fcoords, coords_are_cartesian=False)
    return PatchData(
        patch_structure=st,
        global_indices=patch.global_indices,
        local_to_global=local_to_global,
        fixed_local_indices=fixed_local,
        unknown_local_indices=unknown_local,
        virtual_unknown_local_indices=virtual_local,
    )


def _select_virtual_occupancy(
    patch_structure: Structure,
    patch: PatchData,
    cfg: InpaintingConfig,
) -> List[int]:
    if not patch.virtual_unknown_local_indices:
        return []
    dist = np.asarray(patch_structure.distance_matrix)
    non_virtual = [i for i in range(len(patch_structure)) if i not in patch.virtual_unknown_local_indices]
    keep_cap = cfg.max_added_atoms
    if keep_cap is None:
        keep_cap = max(1, len(patch.unknown_local_indices) // 3)

    scored: List[Tuple[float, int]] = []
    for i in patch.virtual_unknown_local_indices:
        d_all = dist[i]
        d_nonself = np.delete(d_all, i)
        dmin = float(np.min(d_nonself)) if len(d_nonself) else 999.0
        if dmin < cfg.occupancy_min_distance:
            continue
        neigh = 0
        for j in non_virtual:
            if d_all[j] <= cfg.occupancy_neighbor_cutoff:
                neigh += 1
        if neigh < cfg.occupancy_min_neighbors:
            continue
        # Larger neighbor support and not-too-close distances are preferred.
        score = float(neigh) + min(dmin, cfg.occupancy_neighbor_cutoff) * 0.2
        scored.append((score, i))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [i for _, i in scored[: int(keep_cap)]]


def _pack_patch_candidate(
    patch_structure: Structure,
    patch: PatchData,
    keep_virtual_locals: List[int],
) -> Tuple[Structure, List[Optional[int]]]:
    keep_set = set(range(len(patch_structure))) - set(patch.virtual_unknown_local_indices)
    keep_set.update(keep_virtual_locals)
    keep_idx = sorted(keep_set)
    species = [patch_structure[i].specie for i in keep_idx]
    fcoords = [patch_structure.frac_coords[i] for i in keep_idx]
    out = Structure(
        lattice=patch_structure.lattice,
        species=species,
        coords=fcoords,
        coords_are_cartesian=False,
    )
    local_to_global = [patch.local_to_global[i] for i in keep_idx]
    return out, local_to_global


def _import_mattergen_modules(mattergen_repo_path: Optional[str] = None):
    """Import MatterGen lazily to keep InterOptimus import light."""
    repo_path = (
        mattergen_repo_path
        or os.getenv("MATTERGEN_REPO")
        or "/Users/jason/Documents/GitHub/mattergen"
    )
    if repo_path and repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    from hydra.utils import instantiate
    from torch.utils.data import DataLoader

    from mattergen.common.data.chemgraph import ChemGraph
    from mattergen.common.data.collate import collate
    from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
    from mattergen.diffusion.sampling.pc_sampler import PredictorCorrector
    from mattergen.generator import CrystalGenerator, draw_samples_from_sampler

    return {
        "instantiate": instantiate,
        "DataLoader": DataLoader,
        "ChemGraph": ChemGraph,
        "collate": collate,
        "MatterGenCheckpointInfo": MatterGenCheckpointInfo,
        "PredictorCorrector": PredictorCorrector,
        "CrystalGenerator": CrystalGenerator,
        "draw_samples_from_sampler": draw_samples_from_sampler,
    }


def _structure_to_chemgraph(structure: Structure, ChemGraph) -> Any:
    atomic_numbers = torch.tensor([int(s.specie.Z) for s in structure], dtype=torch.long)
    frac = torch.tensor(np.asarray(structure.frac_coords), dtype=torch.float32)
    cell = torch.tensor(np.asarray(structure.lattice.matrix), dtype=torch.float32).reshape(1, 3, 3)
    num_atoms = torch.tensor([len(structure)], dtype=torch.long)
    return ChemGraph(atomic_numbers=atomic_numbers, pos=frac, cell=cell, num_atoms=num_atoms)


class _FixedMaskLoader:
    """Yields (conditioning_data, mask) pairs for MatterGen inpainting sampling."""

    def __init__(self, graph, mask_pos_single: torch.Tensor, batch_size: int, num_samples: int, collate_fn):
        self.graph = graph
        self.mask_pos_single = mask_pos_single
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.collate_fn = collate_fn

    def __iter__(self):
        n = self.num_samples
        b = self.batch_size
        for start in range(0, n, b):
            cur = min(b, n - start)
            batch_graphs = [self.graph] * cur
            conditioning_data = self.collate_fn(batch_graphs)
            mask = {
                "pos": self.mask_pos_single.repeat(cur, 1).to(conditioning_data["pos"].device),
                "cell": torch.ones((cur, 3, 3), dtype=torch.float32, device=conditioning_data["cell"].device),
            }
            yield conditioning_data, mask

    def __len__(self):
        return int(math.ceil(self.num_samples / self.batch_size))


def sample_patch_with_mattergen(
    patch: PatchData,
    num_samples: int = 32,
    batch_size: int = 8,
    pretrained_name: str = "mattergen_base",
    model_path: Optional[str] = None,
    mattergen_repo_path: Optional[str] = None,
    sampling_config_name: str = "default",
    sampling_config_overrides: Optional[List[str]] = None,
    record_trajectories: bool = False,
    enforce_allowed_elements: bool = True,
    enforce_unknown_stoichiometry: bool = True,
) -> List[Structure]:
    """
    Inpaint unknown interface atoms in a local patch using MatterGen pre-trained model.

    Notes:
    - We fix only positions via `mask['pos']`; atomic types of fixed atoms are restored post hoc.
    - Cell is fixed for Level-1.
    """
    mg = _import_mattergen_modules(mattergen_repo_path=mattergen_repo_path)
    ChemGraph = mg["ChemGraph"]
    MatterGenCheckpointInfo = mg["MatterGenCheckpointInfo"]
    CrystalGenerator = mg["CrystalGenerator"]
    instantiate = mg["instantiate"]
    collate = mg["collate"]
    draw_samples_from_sampler = mg["draw_samples_from_sampler"]

    if model_path:
        ckpt_info = MatterGenCheckpointInfo(
            model_path=Path(model_path).resolve(),
            load_epoch="last",
            config_overrides=[],
            strict_checkpoint_loading=True,
        )
    else:
        ckpt_info = MatterGenCheckpointInfo.from_hf_hub(pretrained_name, config_overrides=[])

    generator = CrystalGenerator(
        checkpoint_info=ckpt_info,
        batch_size=batch_size,
        num_batches=int(math.ceil(num_samples / batch_size)),
        sampling_config_name=sampling_config_name,
        sampling_config_overrides=sampling_config_overrides,
        record_trajectories=record_trajectories,
    )

    graph = _structure_to_chemgraph(patch.patch_structure, ChemGraph=ChemGraph)
    mask_pos_single = torch.zeros((len(patch.patch_structure), 3), dtype=torch.float32)
    mask_pos_single[patch.fixed_local_indices] = 1.0
    loader = _FixedMaskLoader(
        graph=graph,
        mask_pos_single=mask_pos_single,
        batch_size=batch_size,
        num_samples=num_samples,
        collate_fn=collate,
    )

    sampling_config = generator.load_sampling_config(
        batch_size=batch_size,
        num_batches=int(math.ceil(num_samples / batch_size)),
        target_compositions_dict=None,
    )
    sampler_partial = instantiate(sampling_config.sampler_partial)
    sampler = sampler_partial(pl_module=generator.model)

    generated = draw_samples_from_sampler(
        sampler=sampler,
        condition_loader=loader,
        cfg=generator.cfg,
        output_path=None,
        properties_to_condition_on={},
        record_trajectories=False,
    )

    # Build hard constraints from reference patch
    reference = patch.patch_structure
    allowed_elements = {str(reference[i].specie) for i in range(len(reference))}
    target_unknown_counts: Dict[str, int] = {}
    for i in patch.unknown_local_indices:
        key = str(reference[i].specie)
        target_unknown_counts[key] = target_unknown_counts.get(key, 0) + 1

    def _project_unknown_species(st: Structure) -> Structure:
        """Project unknown-region species to satisfy allowed elements and exact stoichiometry."""
        if not (enforce_allowed_elements or enforce_unknown_stoichiometry):
            return st
        out = st.copy()
        unknown_idx = patch.unknown_local_indices

        # Current unknown species list
        cur = [str(out[i].specie) for i in unknown_idx]
        cur_counts: Dict[str, int] = {}
        for s in cur:
            cur_counts[s] = cur_counts.get(s, 0) + 1

        # Replace disallowed species first.
        if enforce_allowed_elements:
            for li in unknown_idx:
                s = str(out[li].specie)
                if s not in allowed_elements:
                    # choose a valid replacement with highest target deficit first
                    deficits = {
                        e: target_unknown_counts.get(e, 0) - cur_counts.get(e, 0)
                        for e in target_unknown_counts
                    }
                    repl = max(deficits, key=lambda e: deficits[e]) if deficits else next(iter(allowed_elements))
                    cur_counts[s] = cur_counts.get(s, 0) - 1
                    cur_counts[repl] = cur_counts.get(repl, 0) + 1
                    out.replace(li, species=repl, coords=out.frac_coords[li], coords_are_cartesian=False)

        # Enforce exact unknown stoichiometry by greedy balancing.
        if enforce_unknown_stoichiometry:
            # Recompute after disallowed replacement.
            cur = [str(out[i].specie) for i in unknown_idx]
            cur_counts = {}
            for s in cur:
                cur_counts[s] = cur_counts.get(s, 0) + 1

            surplus_sites: Dict[str, List[int]] = {}
            for li in unknown_idx:
                s = str(out[li].specie)
                surplus_sites.setdefault(s, []).append(li)

            deficits = {
                e: target_unknown_counts.get(e, 0) - cur_counts.get(e, 0)
                for e in target_unknown_counts
            }
            surpluses = {e: -d for e, d in deficits.items() if d < 0}
            deficits = {e: d for e, d in deficits.items() if d > 0}

            if deficits and surpluses:
                for e_need, n_need in deficits.items():
                    remain = n_need
                    for e_have, n_extra in list(surpluses.items()):
                        if remain <= 0:
                            break
                        take = min(remain, n_extra)
                        if take <= 0:
                            continue
                        pool = surplus_sites.get(e_have, [])
                        for _ in range(min(take, len(pool))):
                            li = pool.pop()
                            out.replace(li, species=e_need, coords=out.frac_coords[li], coords_are_cartesian=False)
                        surpluses[e_have] -= take
                        remain -= take

        return out

    # Restore fixed sites exactly (species + coordinates).
    restored = []
    for st in generated:
        st_new = st.copy()
        for i in patch.fixed_local_indices:
            st_new.replace(
                i,
                species=reference[i].specie,
                coords=reference.frac_coords[i],
                coords_are_cartesian=False,
            )
        st_new = _project_unknown_species(st_new)
        restored.append(st_new)
    return restored


def rank_structures_with_mlip(
    structures: Sequence[Structure],
    calc: str = "orb-models",
    user_settings: Optional[Dict[str, Any]] = None,
    relax_steps: int = 30,
    relax_fmax: float = 0.1,
) -> List[Dict[str, Any]]:
    """Short-relax and rank candidates by MLIP energy."""
    from InterOptimus.mlip import MlipCalc

    settings = dict(user_settings or {})
    if "fmax" not in settings:
        settings["fmax"] = relax_fmax
    if "steps" not in settings:
        settings["steps"] = relax_steps
    if "fix_cell_booleans" not in settings:
        settings["fix_cell_booleans"] = [False, False, True, False, False, False]
    if "device" not in settings:
        settings["device"] = "cpu"

    mc = MlipCalc(calc=calc, user_settings=settings)
    ranked: List[Dict[str, Any]] = []
    for i, st in enumerate(structures):
        try:
            relaxed, energy = mc.optimize(st, optimizer="FIRE", **settings)
            ranked.append({"idx": i, "energy": float(energy), "structure": relaxed, "status": "ok"})
        except Exception:
            try:
                energy = mc.calculate(st)
                ranked.append({"idx": i, "energy": float(energy), "structure": st, "status": "single_point"})
            except Exception as exc:
                ranked.append({"idx": i, "energy": float("inf"), "structure": st, "status": f"failed: {exc}"})
    ranked.sort(key=lambda x: x["energy"])
    return ranked


def reinsert_patch(
    full_structure: Structure,
    patch_structure: Structure,
    global_indices: Sequence[int],
    update_global_indices: Optional[Sequence[int]] = None,
) -> Structure:
    """
    Reinsert local patch sites into full supercell.

    If update_global_indices is None, all patch-mapped sites are updated.
    """
    out = full_structure.copy()
    update_set = set(update_global_indices) if update_global_indices is not None else None

    for local_i, global_i in enumerate(global_indices):
        if update_set is not None and global_i not in update_set:
            continue
        out.replace(
            global_i,
            species=patch_structure[local_i].specie,
            coords=patch_structure.frac_coords[local_i],
            coords_are_cartesian=False,
        )
    return out


def reinsert_patch_with_mapping(
    full_structure: Structure,
    patch_structure: Structure,
    local_to_global: Sequence[Optional[int]],
) -> Structure:
    out = full_structure.copy()
    for local_i, global_i in enumerate(local_to_global):
        if global_i is None:
            out.append(
                species=patch_structure[local_i].specie,
                coords=patch_structure.frac_coords[local_i],
                coords_are_cartesian=False,
            )
        else:
            out.replace(
                global_i,
                species=patch_structure[local_i].specie,
                coords=patch_structure.frac_coords[local_i],
                coords_are_cartesian=False,
            )
    return out


def run_level1_interface_inpainting(
    supercell: Structure,
    cfg: Optional[InpaintingConfig] = None,
) -> Dict[str, Any]:
    """
    End-to-end Level-1 pipeline:
    partition -> patch -> mattergen inpaint -> mlip rank -> reinsert top-k.
    """
    cfg = cfg or InpaintingConfig()

    partition = partition_double_interface(
        structure=supercell,
        axis=cfg.axis,
        split_frac=cfg.split_frac,
        interface_region_thickness_angstrom=cfg.interface_region_thickness_angstrom,
        interface_half_width=cfg.interface_half_width,
    )
    patch = build_patch(
        structure=supercell,
        partition=partition,
        max_patch_atoms=cfg.max_patch_atoms,
        min_atoms_per_bulk_side=cfg.min_atoms_per_bulk_side,
    )
    if cfg.use_variable_occupancy:
        patch = _add_virtual_unknown_sites(
            supercell=supercell,
            patch=patch,
            partition=partition,
            num_virtual_unknown_sites=cfg.num_virtual_unknown_sites,
            seed=cfg.random_seed,
        )

    generated_patches = sample_patch_with_mattergen(
        patch=patch,
        num_samples=cfg.num_samples,
        batch_size=cfg.batch_size,
        pretrained_name=cfg.mattergen_pretrained_name,
        model_path=cfg.mattergen_model_path,
        mattergen_repo_path=cfg.mattergen_repo_path,
        sampling_config_name=cfg.sampling_config_name,
        sampling_config_overrides=cfg.sampling_config_overrides,
        record_trajectories=cfg.record_trajectories,
        enforce_allowed_elements=cfg.enforce_allowed_elements,
        enforce_unknown_stoichiometry=cfg.enforce_unknown_stoichiometry,
    )

    def _interface_half_width_frac() -> float:
        if cfg.interface_region_thickness_angstrom is not None:
            axis_len = float(np.linalg.norm(np.asarray(supercell.lattice.matrix)[cfg.axis]))
            return 0.5 * float(cfg.interface_region_thickness_angstrom) / max(axis_len, 1e-8)
        return float(cfg.interface_half_width)

    def _project_to_interface_bands(st: Structure) -> Structure:
        if not cfg.enforce_interface_region:
            return st
        out = st.copy()
        axis = int(cfg.axis)
        split = float(cfg.split_frac) % 1.0
        half = _interface_half_width_frac()
        centers = (0.0, split)

        for local_i in patch.unknown_local_indices:
            f = np.mod(np.asarray(out.frac_coords[local_i], dtype=float), 1.0)
            z = float(f[axis])
            # Pick nearest interface center under periodic metric.
            best_center = 0.0
            best_delta = 0.0
            best_abs = 1e9
            for c in centers:
                delta = ((z - c + 0.5) % 1.0) - 0.5
                ad = abs(delta)
                if ad < best_abs:
                    best_abs = ad
                    best_delta = delta
                    best_center = c
            # Hard clamp to interface half-width.
            clamped = min(max(best_delta, -half), half)
            f[axis] = (best_center + clamped) % 1.0
            out.replace(local_i, species=out[local_i].specie, coords=f, coords_are_cartesian=False)
        return out

    if cfg.enforce_interface_region:
        generated_patches = [_project_to_interface_bands(st) for st in generated_patches]

    candidate_patch_structures: List[Structure] = []
    candidate_local_to_global: List[List[Optional[int]]] = []
    if cfg.use_variable_occupancy and patch.virtual_unknown_local_indices:
        for st in generated_patches:
            keep_virtual = _select_virtual_occupancy(st, patch, cfg)
            st_eff, l2g = _pack_patch_candidate(st, patch, keep_virtual)
            candidate_patch_structures.append(st_eff)
            candidate_local_to_global.append(l2g)
    else:
        candidate_patch_structures = generated_patches
        candidate_local_to_global = [patch.local_to_global for _ in generated_patches]

    ranked = rank_structures_with_mlip(
        structures=candidate_patch_structures,
        calc=cfg.mlip_calc,
        user_settings=cfg.mlip_user_settings,
        relax_steps=cfg.mlip_relax_steps,
        relax_fmax=cfg.mlip_relax_fmax,
    )

    top = ranked[: max(1, cfg.top_k)]
    full_candidates = []
    for item in top:
        idx = int(item["idx"])
        full_candidates.append(
            reinsert_patch_with_mapping(
                full_structure=supercell,
                patch_structure=item["structure"],
                local_to_global=candidate_local_to_global[idx],
            )
        )

    return {
        "partition": partition,
        "patch": patch,
        "generated_patch_structures": generated_patches,
        "ranked_patch_candidates": ranked,
        "top_full_candidates": full_candidates,
    }

