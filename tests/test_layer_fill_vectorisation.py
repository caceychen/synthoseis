"""Parity tests for the vectorised Faults / Closures k-loops.

Each reference below is the per-k Python loop copied verbatim from
``Faults.build_faulted_property_geomodels`` and
``Closures.create_closure_labels_from_*depth_maps`` before vectorisation.
The vectorised helpers must reproduce them bit for bit, including the edge
cases: negative depth indices, clipped duplicate voxels, NaN maps and a
``voxel_change_count`` smaller than ``closure_segments``.
"""
from __future__ import annotations

import numpy as np
import pytest

from datagenerator._closures_vectorised import stamp_closure_slab
from datagenerator._faults_vectorised import (
    fill_layer_partial_voxels,
    fill_layer_whole_voxels,
)


# --------------------------------------------------------------------------- #
# Faults references
# --------------------------------------------------------------------------- #
def _ref_partial(cubes, cdm, pdm, tmi, tmax, tvdml, ng, fac_i, fac_above):
    lith, sealed, depth, n2g = cubes
    cube_shape = lith.shape
    ii, jj = np.meshgrid(
        np.arange(cdm.shape[0]), np.arange(cdm.shape[1]), indexing="ij"
    )
    for k in range(tmax + 1):
        top_map = np.max(np.dstack((cdm, tmi.astype("float32") + k)), axis=-1)
        top_map = np.min(np.dstack((top_map, pdm)), axis=-1)
        base_map = np.min(
            np.dstack((pdm, tmi.astype("float32") + k + 1)), axis=-1
        )
        fraction_of_voxel = np.clip(base_map - top_map, 0.0, 1.0)
        valid_k = np.where(
            (fraction_of_voxel > 0.0) & ((tmi + k).astype("int") < cube_shape[2]),
            1,
            0,
        )
        sublayer_ii = ii[valid_k == 1]
        sublayer_jj = jj[valid_k == 1]
        if sublayer_ii.shape[0] > 0:
            sublayer_depth_map = (tmi + k).astype("int")[valid_k == 1]
            z = np.clip(sublayer_depth_map, 0, None)
            sublayer_ng_map = ng[valid_k == 1]
            sublayer_tvdml_map = tvdml[valid_k == 1]
            frac = fraction_of_voxel[valid_k == 1]

            input_cube = lith[sublayer_ii, sublayer_jj, z]
            values = fac_i * frac
            input_cube[input_cube == -1.0] = values[input_cube == -1.0] * 1.0
            input_cube[input_cube != -1.0] += values[input_cube != -1.0]
            lith[sublayer_ii, sublayer_jj, z] = input_cube * 1.0

            input_cube = sealed[sublayer_ii, sublayer_jj, z]
            values = (1 - fac_above) * frac
            input_cube[input_cube == -1.0] = values[input_cube == -1.0] * 1.0
            input_cube[input_cube != -1.0] += values[input_cube != -1.0]
            sealed[sublayer_ii, sublayer_jj, z] = input_cube * 1.0

            depth[sublayer_ii, sublayer_jj, z] += sublayer_tvdml_map * frac
            n2g[sublayer_ii, sublayer_jj, z] += sublayer_ng_map * frac


def _ref_whole(cubes, pdm, thick, tmax, max_idx, tvdml, ng, fac_i, fac_above):
    lith, sealed, depth, n2g = cubes
    ii, jj = np.meshgrid(
        np.arange(pdm.shape[0]), np.arange(pdm.shape[1]), indexing="ij"
    )
    for k in range(tmax + 1):
        sublayer_ii = ii[thick > k]
        sublayer_jj = jj[thick > k]
        if sublayer_ii.shape[0] > 0:
            z = (0.5 + np.clip(pdm[thick > k], 0, max_idx)).astype("int") - k
            lith[sublayer_ii, sublayer_jj, z] = fac_i
            sealed[sublayer_ii, sublayer_jj, z] = 1 - fac_above
            depth[sublayer_ii, sublayer_jj, z] = tvdml[thick > k]
            n2g[sublayer_ii, sublayer_jj, z] += ng[thick > k]


def _layer_stack(seed, shape, n_layers, dtype, partial, negative_top):
    """Depth maps for a stack of layers, optionally poking above sample 0."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = shape
    thick = rng.uniform(0.3, 6.0, size=(nx, ny, n_layers))
    maps = np.cumsum(thick, axis=-1) + rng.uniform(0, 3, size=(nx, ny, 1))
    if negative_top:
        maps -= rng.uniform(4, 9)
    maps = np.concatenate([np.zeros((nx, ny, 1)), maps], axis=-1).astype(dtype)
    facies = np.hstack(([-1.0], rng.integers(0, 2, n_layers).astype(float)))
    return maps, facies, rng


def _run_layers(fill, maps, facies, rng, nz, dtype, partial):
    """Drive the layer loop of build_faulted_property_geomodels."""
    nx, ny, _ = maps.shape
    cubes = [
        np.ones((nx, ny, nz), dtype) * -1,
        np.zeros((nx, ny, nz), dtype),
        np.zeros((nx, ny, nz), dtype),
        np.zeros((nx, ny, nz), dtype),
    ]
    maps = maps.copy()
    previous = maps[:, :, -1]
    if partial:
        previous += 0.5
    for i in range(maps.shape[2] - 2, 0, -1):
        current = maps[:, :, i]
        if partial:
            current += 0.5
        tmi = current.copy().astype("int")
        thick = maps[:, :, i + 1].copy().astype("int") + 1 - tmi
        tvdml = previous - maps[:, :, 0]
        ng = rng.uniform(0, 1, size=(nx, ny))
        fill(cubes, current, previous, tmi, thick, int(thick.max()), tvdml, ng,
             facies[i], facies[i - 1])
        previous = current.copy()
    return cubes


def _ref_partial_fill(cubes, cur, prev, tmi, thick, tmax, tvdml, ng, fi, fa):
    _ref_partial(cubes, cur, prev, tmi, tmax, tvdml, ng, fi, fa)


def _vec_partial_fill(cubes, cur, prev, tmi, thick, tmax, tvdml, ng, fi, fa):
    fill_layer_partial_voxels(*cubes, cur, prev, tmi, tmax, tvdml, ng, fi, fa)


def _make_whole(max_idx, vectorised):
    def fill(cubes, cur, prev, tmi, thick, tmax, tvdml, ng, fi, fa):
        if vectorised:
            fill_layer_whole_voxels(
                *cubes, prev, thick, tmax, max_idx, tvdml, ng, fi, fa
            )
        else:
            _ref_whole(cubes, prev, thick, tmax, max_idx, tvdml, ng, fi, fa)
    return fill


def _assert_cubes_equal(got, want):
    for name, g, w in zip(("lith", "sealed", "depth", "n2g"), got, want):
        np.testing.assert_array_equal(g, w, err_msg=name)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("negative_top", [False, True])
@pytest.mark.parametrize("seed", [0, 7, 42])
def test_partial_voxels_parity(seed, negative_top, dtype):
    shape = (9, 7, 64)
    maps, facies, _ = _layer_stack(seed, shape, 12, dtype, True, negative_top)
    want = _run_layers(_ref_partial_fill, maps, facies,
                       np.random.default_rng(seed + 100), shape[2], dtype, True)
    got = _run_layers(_vec_partial_fill, maps, facies,
                      np.random.default_rng(seed + 100), shape[2], dtype, True)
    _assert_cubes_equal(got, want)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("negative_top", [False, True])
@pytest.mark.parametrize("seed", [0, 7, 42])
def test_whole_voxels_parity(seed, negative_top, dtype):
    shape = (9, 7, 64)
    maps, facies, _ = _layer_stack(seed, shape, 12, dtype, False, negative_top)
    max_idx = shape[2] - 1
    want = _run_layers(_make_whole(max_idx, False), maps, facies,
                       np.random.default_rng(seed + 100), shape[2], dtype, False)
    got = _run_layers(_make_whole(max_idx, True), maps, facies,
                      np.random.default_rng(seed + 100), shape[2], dtype, False)
    _assert_cubes_equal(got, want)


def test_partial_voxels_negative_top_hits_sample_zero_repeatedly():
    """Guard that the edge case the parity test covers is actually exercised."""
    cdm = np.full((2, 2), -3.4)
    tmi = cdm.astype("int")
    k = np.arange(5)
    assert np.count_nonzero(np.clip(tmi[0, 0] + k, 0, None) == 0) > 1


# --------------------------------------------------------------------------- #
# Closures reference
# --------------------------------------------------------------------------- #
def _ref_stamp(closure_segments, voxel_change_count, top, closure, max_closure):
    ii, jj = np.meshgrid(
        np.arange(top.shape[0]), np.arange(top.shape[1]), indexing="ij"
    )
    slices_with_substitution = 0
    for k in range(max_closure + 1):
        horizon_slice = (k + top).clip(0, closure_segments.shape[2] - 2)
        sublayer_kk = horizon_slice[horizon_slice < closure.astype("int")]
        sublayer_ii = ii[horizon_slice < closure.astype("int")]
        sublayer_jj = jj[horizon_slice < closure.astype("int")]
        if sublayer_ii.size > 0:
            slices_with_substitution += 1
            k_indices = sublayer_kk + 1
            try:
                closure_segments[sublayer_ii, sublayer_jj, k_indices.astype("int")] += 1.0
                voxel_change_count[sublayer_ii, sublayer_jj, k_indices.astype("int")] += 1
            except IndexError:
                pass
    return slices_with_substitution > 0


@pytest.mark.parametrize("vcc_nz", [48, 30])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("seed", [1, 5, 9, 13])
def test_stamp_closure_slab_parity(seed, dtype, vcc_nz):
    rng = np.random.default_rng(seed)
    nx, ny, nz = 11, 8, 48
    top = rng.uniform(-6, nz + 4, size=(nx, ny)).astype(dtype)
    closure = (top + rng.uniform(-2, 12, size=(nx, ny))).astype(dtype)
    top[rng.random((nx, ny)) < 0.1] = np.nan
    max_closure = int(np.around(np.nanmax(closure - top), 0))

    seg_ref = rng.integers(0, 3, (nx, ny, nz)).astype("float32")
    vcc_ref = rng.integers(0, 3, (nx, ny, vcc_nz)).astype(np.uint8)
    seg_vec, vcc_vec = seg_ref.copy(), vcc_ref.copy()

    with np.errstate(invalid="ignore"):
        want = _ref_stamp(seg_ref, vcc_ref, top, closure, max_closure)
        got = stamp_closure_slab(seg_vec, vcc_vec, top, closure, max_closure)

    assert got == want
    np.testing.assert_array_equal(seg_vec, seg_ref)
    np.testing.assert_array_equal(vcc_vec, vcc_ref)


def test_stamp_closure_slab_no_closure():
    seg = np.zeros((3, 3, 10), "float32")
    vcc = np.zeros((3, 3, 10), np.uint8)
    top = np.full((3, 3), 5.0)
    assert stamp_closure_slab(seg, vcc, top, top.copy(), 0) is False
    assert not seg.any() and not vcc.any()
