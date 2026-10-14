"""Small-cube and NumPy 2 compatibility fixes."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from datagenerator import _noise_shim
from datagenerator.Closures import Closures
from datagenerator.Horizons import Onlaps, RandomHorizonStack


def _cfg(nz, **extra):
    base = dict(
        cube_shape=(16, 16, nz),
        num_lyr_lut=300,
        dip_factor_max=1.0,
        include_channels=False,
        verbose=False,
    )
    base.update(extra)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("seed", range(10))
def test_random_horizon_stack_onlaps_fit_small_cube(seed):
    """Onlap layers are drawn up to 200 but the table has 0.4 * nz entries."""
    stub = SimpleNamespace(cfg=_cfg(64), rng=np.random.default_rng(seed))
    RandomHorizonStack._generate_lookup_tables(stub)
    assert stub.onlaps.shape == (int(500 / 1250 * 64),)


@pytest.mark.parametrize("seed", range(10))
def test_onlaps_lookup_fits_small_cube(seed):
    stub = SimpleNamespace(cfg=_cfg(64), rng=np.random.default_rng(seed), max_layers=120)
    Onlaps._generate_onlap_lookup_table(stub)
    assert stub.onlaps.shape == (25,)
    assert stub.onlaps.sum() >= 1


def test_onlaps_lookup_unchanged_on_large_cube():
    """Draws are unchanged where the table was already big enough."""
    stub = SimpleNamespace(cfg=_cfg(1250), rng=np.random.default_rng(3), max_layers=120)
    Onlaps._generate_onlap_lookup_table(stub)
    rng = np.random.default_rng(3)
    expected = np.sort(
        rng.uniform(low=5, high=120, size=int(rng.triangular(1, 4, 7) + 0.5)).astype("int")
    )
    assert set(np.flatnonzero(stub.onlaps)) == set(expected)


def test_closure_statistics_report_plain_ints():
    from skimage import measure

    arr = np.zeros((6, 6, 6), np.uint8)
    arr[1:4, 2:5, 1:3] = 1
    stub = SimpleNamespace(_measure=measure, cfg=SimpleNamespace(digi=4))
    msg = Closures.calculate_closure_statistics(stub, arr, "Oil")
    assert "np.int64" not in msg
    assert "Num X,Y,Z Samples: [2, 2, 1]" in msg


def test_noise_shim_reseeds_only_when_base_changes(monkeypatch):
    calls = []
    real_seed = _noise_shim._osx.seed
    monkeypatch.setattr(_noise_shim, "_current_seed", None)
    monkeypatch.setattr(_noise_shim._osx, "seed", lambda b: (calls.append(b), real_seed(b)))

    first = [_noise_shim.pnoise2(x * 0.1, 0.3, octaves=3, base=7) for x in range(20)]
    other = _noise_shim.pnoise2(0.5, 0.3, base=8)
    again = [_noise_shim.pnoise2(x * 0.1, 0.3, octaves=3, base=7) for x in range(20)]

    assert calls == [7, 8, 7]
    assert first == again
    assert other != first[5]


def test_simplex_noise_imports_without_noise_package():
    import importlib
    import sys

    import datagenerator.simplexNoise as sn

    saved = sys.modules.get("noise")
    sys.modules["noise"] = None  # makes `import noise` raise ModuleNotFoundError
    try:
        mod = importlib.reload(sn)
        assert mod.noise is _noise_shim
    finally:
        if saved is None:
            sys.modules.pop("noise", None)
        else:
            sys.modules["noise"] = saved
        importlib.reload(sn)
