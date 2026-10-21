"""Vectorised helpers for ``Faults.build_faulted_property_geomodels``.

Each helper replaces the per-layer ``for k in range(thickness_map_max + 1)``
loop with one broadcast over k and keeps the loop's output unchanged,
including its edge cases (see each docstring).
"""
from __future__ import annotations

import numpy as np


def fill_layer_partial_voxels(
    work_cube_lith: np.ndarray,
    work_cube_sealed: np.ndarray,
    work_cube_depth: np.ndarray,
    work_cube_net_to_gross: np.ndarray,
    current_depth_map: np.ndarray,
    previous_depth_map: np.ndarray,
    top_map_index: np.ndarray,
    thickness_map_max: int,
    tvdml_map: np.ndarray,
    ng_map: np.ndarray,
    facies_value: float,
    facies_above: float,
) -> None:
    """Fill one layer with fractional-voxel weights (``cfg.partial_voxels``).

    Negative depth indices clip to 0, so several k can hit the same voxel.
    The loop accumulated those writes in ascending k order, and so does
    ``np.add.at``. Legacy quirk kept for parity: the loop's first write into a
    water voxel (-1) stored the value and then added it again (2x), because the
    ``!= -1`` add ran after the ``== -1`` assignment.
    """
    n_k = thickness_map_max + 1
    if n_k <= 0:
        return
    nz = work_cube_lith.shape[2]
    k_i = np.arange(n_k)
    k_f = k_i.astype("float32")
    tmi = top_map_index[:, :, None]
    tmi_f = top_map_index.astype("float32")[:, :, None]
    cdm = current_depth_map[:, :, None]
    pdm = previous_depth_map[:, :, None]

    top = np.minimum(np.maximum(cdm, tmi_f + k_f), pdm)
    base = np.minimum(pdm, (tmi_f + k_f) + np.float32(1))
    fraction = np.clip(base - top, 0.0, 1.0)
    depth_index = (tmi + k_i).astype("int")
    valid = (fraction > 0.0) & (depth_index < nz)

    il, jl, kl = np.nonzero(valid)
    if il.size == 0:
        return
    zi = np.clip(depth_index[il, jl, kl], 0, None)
    frac = fraction[il, jl, kl]
    idx = (il, jl, zi)

    lith_values = facies_value * frac
    _, first = np.unique(
        np.ravel_multi_index(idx, work_cube_lith.shape), return_index=True
    )
    first_idx = (il[first], jl[first], zi[first])
    water = work_cube_lith[first_idx] == -1.0
    work_cube_lith[tuple(a[water] for a in first_idx)] = lith_values[first][water]
    np.add.at(work_cube_lith, idx, lith_values)
    np.add.at(work_cube_sealed, idx, (1 - facies_above) * frac)
    np.add.at(work_cube_depth, idx, tvdml_map[il, jl] * frac)
    np.add.at(work_cube_net_to_gross, idx, ng_map[il, jl] * frac)


def fill_layer_whole_voxels(
    work_cube_lith: np.ndarray,
    work_cube_sealed: np.ndarray,
    work_cube_depth: np.ndarray,
    work_cube_net_to_gross: np.ndarray,
    previous_depth_map: np.ndarray,
    thickness_map: np.ndarray,
    thickness_map_max: int,
    max_depth_index: int,
    tvdml_map: np.ndarray,
    ng_map: np.ndarray,
    facies_value: float,
    facies_above: float,
) -> None:
    """Fill one layer upward from its base, one whole voxel per k.

    Each trace writes depth indices ``base - k``, so no voxel repeats within a
    layer. Indices below 0 are not filtered: NumPy wraps them as the loop did.
    """
    n_k = thickness_map_max + 1
    if n_k <= 0:
        return
    k_i = np.arange(n_k)
    base_index = (0.5 + np.clip(previous_depth_map, 0, max_depth_index)).astype("int")
    il, jl, kl = np.nonzero(thickness_map[:, :, None] > k_i)
    if il.size == 0:
        return
    idx = (il, jl, base_index[il, jl] - kl)

    work_cube_lith[idx] = facies_value
    work_cube_sealed[idx] = 1 - facies_above
    work_cube_depth[idx] = tvdml_map[il, jl]
    np.add.at(work_cube_net_to_gross, idx, ng_map[il, jl])
