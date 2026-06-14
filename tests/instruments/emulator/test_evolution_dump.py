"""Testing structured emulator evolution dumps."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import BSpline

from qibolab._core.instruments.emulator.engine.evolution_dump import (
    DENSITY_MATRICES_FILENAME,
    DUMP_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    TIME_COEFFICIENTS_FILENAME,
    TIME_GRID_FILENAME,
    dump_evolution,
    load_evolution_dump,
)


class FakeQutip:
    """Small qsave/qload stand-in used to test dump path handling."""

    def qsave(self, obj, path: str) -> None:
        Path(path).with_suffix(".qu").write_text(json.dumps(obj))

    def qload(self, path: str):
        return json.loads(Path(path).with_suffix(".qu").read_text())


def test_dump_evolution_writes_manifest_and_arrays(tmp_path: Path):
    engine = FakeQutip()
    time_grid = np.linspace(0, 1, 4)
    coefficients = np.ones((2, time_grid.size))
    density_matrices = np.zeros((1, 2, 2, 2))

    dump_dir = dump_evolution(
        tmp_path,
        engine=engine,
        operators=[{"static": True}, {"drive": 1}],
        time_grid=time_grid,
        time_coefficients=coefficients,
        density_matrices=density_matrices,
    )

    manifest = json.loads((dump_dir / MANIFEST_FILENAME).read_text())

    assert manifest["version"] == DUMP_SCHEMA_VERSION
    assert manifest["engine"] == "qutip"
    assert manifest["files"]["time_grid"] == TIME_GRID_FILENAME
    assert manifest["files"]["density_matrices"] == DENSITY_MATRICES_FILENAME
    assert manifest["files"]["time_coefficients"] == TIME_COEFFICIENTS_FILENAME
    assert np.allclose(np.load(dump_dir / TIME_GRID_FILENAME), time_grid)
    assert np.allclose(np.load(dump_dir / TIME_COEFFICIENTS_FILENAME), coefficients)
    assert np.allclose(np.load(dump_dir / DENSITY_MATRICES_FILENAME), density_matrices)


def test_load_evolution_dump_round_trip(tmp_path: Path):
    engine = FakeQutip()
    time_grid = np.linspace(0, 1, 5)
    coefficients = np.arange(10).reshape(2, 5)
    density_matrices = np.arange(8).reshape(2, 2, 2)

    dump_evolution(
        tmp_path,
        engine=engine,
        operators=["static", "drive"],
        time_grid=time_grid,
        time_coefficients=coefficients,
        density_matrices=density_matrices,
    )

    dump = load_evolution_dump(tmp_path, engine=engine)

    assert dump.path == tmp_path
    assert dump.operators == ["static", "drive"]
    assert np.allclose(dump.time_grid, time_grid)
    assert np.allclose(dump.time_coefficients, coefficients)
    assert np.allclose(dump.density_matrices, density_matrices)


def test_evolution_dump_reconstructs_splines(tmp_path: Path):
    engine = FakeQutip()
    time_grid = np.linspace(0, 1, 6)
    coefficients = np.stack(
        [np.sin(2 * np.pi * time_grid), np.cos(2 * np.pi * time_grid)]
    )
    density_matrices = np.zeros((2, 2, 2))

    dump_evolution(
        tmp_path,
        engine=engine,
        operators=["static", "drive_a", "drive_b"],
        time_grid=time_grid,
        time_coefficients=coefficients,
        density_matrices=density_matrices,
    )
    dump = load_evolution_dump(tmp_path, engine=engine)

    splines = dump.coefficient_splines()

    assert len(splines) == 2
    assert all(isinstance(spline, BSpline) for spline in splines)
    assert pytest.approx(splines[0](time_grid[2]), abs=1e-6) == coefficients[0, 2]


def test_evolution_dump_supports_sweep_indexing(tmp_path: Path):
    engine = FakeQutip()
    time_grid = np.linspace(0, 1, 3)
    coefficients = np.arange(12).reshape(2, 2, 3)
    density_matrices = np.arange(16).reshape(2, 2, 2, 2)

    dump_evolution(
        tmp_path,
        engine=engine,
        operators=["static"],
        time_grid=time_grid,
        time_coefficients=coefficients,
        density_matrices=density_matrices,
    )
    dump = load_evolution_dump(tmp_path, engine=engine)

    assert np.allclose(dump.coefficient_splines((1, 0))[0](0.5), 7.0)
    assert np.allclose(dump.density_matrices_at((0, 1)), np.arange(4, 8).reshape(2, 2))
