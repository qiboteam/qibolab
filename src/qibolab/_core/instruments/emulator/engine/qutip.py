from collections.abc import Iterable
from functools import cached_property
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .abstract import (
    HAMILTONIAN_FILENAME,
    INTEGRATION_MIN_TIME_STEP,
    INTEGRATION_MULTIPLIER,
    STATE_FILENAME,
    Operator,
    OperatorEvolution,
    SimulationEngine,
    _spline_function,
)

__all__ = ["QutipEngine"]

INTEGRATION_MAX_TIME_STEP = 0.02
"""ns, min resolution of the integrator"""

GPU_DTYPE_PLUGINS = {"jax": "qutip_jax", "jaxdia": "qutip_jax", "cupyd": "qutip_cupy"}
"""QuTiP data-layer dtypes usable on GPU and the plugin packages providing them."""
JAX_GPU_DTYPES = {"jax", "jaxdia"}
"""QuTiP GPU dtypes backed by JAX."""


class QutipEngine(SimulationEngine):
    """Qutip simulation engine."""

    device: Literal["cpu", "gpu"] = "cpu"
    """Device used by QuTiP operators."""
    gpu_dtype: str = "jax"
    """QuTiP data-layer dtype used for GPU arrays.

    ``jax``/``jaxdia`` use the qutip-jax plugin (recommended, available on
    PyPI) together with the diffrax integrator, so the whole evolution stays
    on the accelerator. ``cupyd`` uses the experimental qutip-cupy plugin
    (not released on PyPI), where the ODE integration remains on the CPU.
    """

    @cached_property
    def engine(self):
        """Return the qutip engine."""
        # TODO: maybe it can be improved
        import qutip as qt

        if self.device == "gpu":
            plugin = GPU_DTYPE_PLUGINS.get(self.gpu_dtype)
            if plugin is None:
                raise ValueError(
                    f"Unknown gpu_dtype {self.gpu_dtype!r}; "
                    f"supported values: {sorted(GPU_DTYPE_PLUGINS)}."
                )
            try:
                import_module(plugin)
            except ImportError as exc:
                raise ImportError(
                    f"QutipEngine(device='gpu', gpu_dtype={self.gpu_dtype!r}) "
                    f"requires the optional {plugin.replace('_', '-')} package "
                    "and a working CUDA-enabled backend."
                ) from exc
            if self.gpu_dtype in JAX_GPU_DTYPES:
                self._require_jax_gpu_backend()

        return qt

    def _require_jax_gpu_backend(self) -> None:
        """Validate that qutip-jax is connected to an accelerator backend."""
        import jax

        devices = jax.devices()
        if not any(device.platform == "gpu" for device in devices):
            available = (
                ", ".join(f"{device.platform}:{device}" for device in devices) or "none"
            )
            raise ImportError(
                f"QutipEngine(device='gpu', gpu_dtype={self.gpu_dtype!r}) requires "
                "a CUDA-enabled JAX backend; available JAX devices: "
                f"{available}. Install a CUDA-enabled JAX runtime, for example "
                "`jax[cuda12]`, or use device='cpu'."
            )

    def _to_device(self, op: Operator) -> Operator:
        """Move a QuTiP operator to the selected data layer."""
        return op.to(self.gpu_dtype) if self.device == "gpu" else op

    def dump_results(
        self, hamiltonian: Operator, sim_results: Any, dump_dir: Path
    ) -> None:
        """Save the Hamiltonian and simulation results to files with incremented naming."""

        dump_dir.mkdir(parents=True, exist_ok=True)

        count_1 = sum(
            1
            for file in dump_dir.iterdir()
            if file.is_file() and HAMILTONIAN_FILENAME in file.name
        )
        count_2 = sum(
            1
            for file in dump_dir.iterdir()
            if file.is_file() and STATE_FILENAME in file.name
        )
        count = max(count_1, count_2)

        self.engine.qsave(
            hamiltonian, str(dump_dir) + f"/{HAMILTONIAN_FILENAME}_{count}"
        )
        self.engine.qsave(sim_results, str(dump_dir) + f"/{STATE_FILENAME}_{count}")

    def evolve(
        self,
        hamiltonian: Operator,
        initial_state: Operator,
        time: Iterable[float],
        time_hamiltonian: OperatorEvolution = None,
        collapse_operators: list[Operator] = None,
        save_evolution: Path | None = None,
        **kwargs,
    ):
        """Evolve the system."""

        # force the qutip engine (and, on GPU, its data-layer plugin) to be
        # imported and registered before any operator is built: ``_to_device``
        # below calls ``op.to(self.gpu_dtype)``, which only works once the
        # qutip-jax/qutip-cupy plugin has registered that dtype; this also
        # validates ``gpu_dtype`` early instead of part-way through the evolution
        _ = self.engine
        time_diff = np.diff(time)
        nsteps = max(time_diff) / INTEGRATION_MIN_TIME_STEP * INTEGRATION_MULTIPLIER
        # not every SciPy solvers accepts as parameters min_step, that's why we
        # define nsteps instead
        options = {"max_step": INTEGRATION_MAX_TIME_STEP, "nsteps": nsteps}
        if self.device == "gpu" and self.gpu_dtype in ("jax", "jaxdia"):
            # the default scipy integrators would transfer the state between
            # device and host on every step; the diffrax integrator provided
            # by qutip-jax keeps the whole evolution on the accelerator, with
            # the same maximum-step bound enforced through its controller
            from diffrax import PIDController

            options = {
                "method": "diffrax",
                "stepsize_controller": PIDController(
                    rtol=1e-6, atol=1e-8, dtmax=INTEGRATION_MAX_TIME_STEP
                ),
                "max_steps": int(nsteps),
            }

        if time_hamiltonian is not None:
            coefficients = [
                coefficient for _, coefficient in time_hamiltonian.operators
            ]
            if "method" in options:
                # diffrax jits over the coefficients, and the spline objects
                # are not traceable by JAX; convert them with the shared spline
                # helper so the coefficients become jittable functions
                from jax import jit

                coefficients = [
                    jit(_spline_function(coefficient)) for coefficient in coefficients
                ]
            hamiltonian = [self._to_device(hamiltonian)] + [
                [self._to_device(operator), coefficient]
                for (operator, _), coefficient in zip(
                    time_hamiltonian.operators, coefficients, strict=True
                )
            ]
        else:
            hamiltonian = self._to_device(hamiltonian)

        sim_results = self.engine.mesolve(
            hamiltonian,
            self._to_device(initial_state),
            time,
            [self._to_device(op) for op in collapse_operators or []],
            options=options,
            **kwargs,
        )

        if save_evolution is not None:
            self.dump_results(
                hamiltonian=hamiltonian,
                sim_results=sim_results,
                dump_dir=save_evolution,
            )

        return sim_results

    def create(self, n: int) -> Operator:
        """Create operator for n levels system."""
        return self._to_device(self.engine.create(n))

    def destroy(self, n: int) -> Operator:
        """Destroy operator for n levels system."""
        return self._to_device(self.engine.destroy(n))

    def identity(self, n: int) -> Operator:
        """Identity operator for n levels system."""
        return self._to_device(self.engine.qeye(n))

    def tensor(self, operators: list[Operator]) -> Operator:
        """Tensor product of a list of operators."""
        return self._to_device(self.engine.tensor(*operators))

    def expand(self, op: Operator, targets: int | list[int], dims: list[int]):
        """Expand operator in larger Hilbert space."""
        return self._to_device(self.engine.expand_operator(op, targets, dims))

    def basis(self, dim: int, state: int) -> Operator:
        """Basis operator for n levels system."""
        return self._to_device(self.engine.basis(dimensions=dim, n=state))
