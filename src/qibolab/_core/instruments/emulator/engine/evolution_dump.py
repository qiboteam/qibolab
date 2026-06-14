"""Utilities for storing and loading emulator evolution data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline, make_interp_spline

SPLINE_ORDER = 3

__all__ = [
    "DENSITY_MATRICES_FILENAME",
    "DUMP_SCHEMA_VERSION",
    "EvolutionDump",
    "MANIFEST_FILENAME",
    "OPERATORS_FILENAME",
    "TIME_COEFFICIENTS_FILENAME",
    "TIME_GRID_FILENAME",
    "dump_evolution",
    "load_evolution_dump",
]

DUMP_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
OPERATORS_FILENAME = "operators"
TIME_GRID_FILENAME = "time_grid.npy"
TIME_COEFFICIENTS_FILENAME = "time_coefficients.npy"
DENSITY_MATRICES_FILENAME = "density_matrices.npy"
SPLINE_ORDER = 3


@dataclass(frozen=True)
class EvolutionDump:
    """Saved Hamiltonian operators, drive coefficients, and state traces."""

    path: Path
    """Directory containing the dump files."""
    manifest: dict[str, Any]
    """Dump metadata."""
    operators: Any
    """Static and time-dependent operators in the engine-native format."""
    time_grid: NDArray
    """Time samples used for the drive coefficients."""
    time_coefficients: NDArray | None
    """Drive coefficients with shape ``(*sweep_shape, n_channels, n_times)``."""
    density_matrices: NDArray
    """Density matrices with shape ``(*sweep_shape, n_measurements, dim, dim)``."""

    def coefficient_splines(
        self, sweep_index: tuple[int, ...] | int = ()
    ) -> list[BSpline]:
        """Build B-splines for a selected sweep point."""
        if self.time_coefficients is None:
            return []

        if sweep_index == ():
            coeffs = self.time_coefficients
        elif isinstance(sweep_index, int):
            coeffs = self.time_coefficients[sweep_index]
        else:
            coeffs = self.time_coefficients[sweep_index]
        if coeffs.ndim == 1:
            coeffs = coeffs[None]

        return [
            make_interp_spline(
                self.time_grid,
                channel_coeff,
                k=min(SPLINE_ORDER, len(self.time_grid) - 1),
            )
            for channel_coeff in coeffs
        ]

    def density_matrices_at(self, sweep_index: tuple[int, ...] | int = ()) -> NDArray:
        """Return density matrices for a selected sweep point."""
        if sweep_index == ():
            return self.density_matrices
        if isinstance(sweep_index, int):
            return self.density_matrices[sweep_index]
        return self.density_matrices[sweep_index]


def dump_evolution(
    dump_dir: Path,
    *,
    engine: Any,
    operators: Any,
    time_grid: NDArray,
    time_coefficients: NDArray | None,
    density_matrices: NDArray,
) -> Path:
    """Persist a structured evolution dump for one experiment execution."""
    dump_dir.mkdir(parents=True, exist_ok=True)

    engine.qsave(operators, str(dump_dir / OPERATORS_FILENAME))
    np.save(dump_dir / TIME_GRID_FILENAME, time_grid)
    if time_coefficients is not None:
        np.save(dump_dir / TIME_COEFFICIENTS_FILENAME, time_coefficients)
    np.save(dump_dir / DENSITY_MATRICES_FILENAME, density_matrices)

    manifest = {
        "version": DUMP_SCHEMA_VERSION,
        "engine": "qutip",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            "operators": f"{OPERATORS_FILENAME}.qu",
            "time_grid": TIME_GRID_FILENAME,
            "density_matrices": DENSITY_MATRICES_FILENAME,
        },
        "shapes": {
            "time_grid": list(time_grid.shape),
            "density_matrices": list(density_matrices.shape),
        },
    }
    if time_coefficients is not None:
        manifest["files"]["time_coefficients"] = TIME_COEFFICIENTS_FILENAME
        manifest["shapes"]["time_coefficients"] = list(time_coefficients.shape)

    (dump_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return dump_dir


def load_evolution_dump(dump_dir: Path, *, engine: Any) -> EvolutionDump:
    """Load a structured evolution dump written by :func:`dump_evolution`."""
    if not (dump_dir / MANIFEST_FILENAME).is_file():
        raise FileNotFoundError(f"No evolution dump manifest found in {dump_dir}.")

    manifest = json.loads((dump_dir / MANIFEST_FILENAME).read_text())
    files = manifest["files"]

    operators = engine.qload(_qload_path(dump_dir / files["operators"]))
    time_grid = np.load(dump_dir / files["time_grid"])
    density_matrices = np.load(dump_dir / files["density_matrices"])
    time_coefficients = None
    if "time_coefficients" in files:
        time_coefficients = np.load(dump_dir / files["time_coefficients"])

    return EvolutionDump(
        path=dump_dir,
        manifest=manifest,
        operators=operators,
        time_grid=time_grid,
        time_coefficients=time_coefficients,
        density_matrices=density_matrices,
    )


def _qload_path(path: Path) -> str:
    """Return the filename format expected by ``qutip.qload``."""
    if path.suffix == ".qu":
        return str(path.with_suffix(""))
    return str(path)
