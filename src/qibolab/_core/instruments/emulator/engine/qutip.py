import json
import os
from collections.abc import Callable, Iterable
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import make_interp_spline

from .abstract import (
    HAMILTONIAN_FILENAME,
    INTEGRATION_MAX_TIME_STEP,
    INTEGRATION_MIN_TIME_STEP,
    INTEGRATION_MULTIPLIER,
    SIMULATOR_CONFIG,
    SPLINE_INTERP_ORDER,
    SWEEP_SIMULATION_FILENAME,
    Operator,
    OperatorEvolution,
    SimulationEngine,
    TimeDependentOperator,
)

__all__ = ["QutipEngine", "load_simulation"]


class QutipEngine(SimulationEngine):
    """Qutip simulation engine."""

    device: Literal["cpu", "gpu"] = "cpu"

    @cached_property
    def engine(self):
        """Return the qutip engine."""
        # TODO: maybe it can be improved
        import qutip as qt

        if self.device == "gpu":
            import jax

            devices = jax.devices()
            if self.device == "gpu" and any(
                device.platform == "gpu" for device in devices
            ):
                import qutip_jax  # noqa: F401
            else:
                object.__setattr__(self, "device", "cpu")

        return qt

    def _to_device(self, op: Operator) -> Operator:
        """Move a QuTiP operator to the selected data layer."""
        return op.to("jax") if self.device == "gpu" else op

    def interpolate_coeffs(
        self, x: NDArray, y: NDArray
    ) -> Callable[[NDArray], Iterable[float]]:

        spline = make_interp_spline(x, y, k=SPLINE_INTERP_ORDER)
        if self.device == "gpu":
            import jax

            from .gpu_interpolation import jax_interpolation

            # TODO: check if there is a more efficient implementation for jax_interpolation
            return jax.jit(jax_interpolation(spline))
        else:
            return spline

    def evolve(
        self,
        hamiltonian: Operator,
        initial_state: Operator,
        time: Iterable[float],
        time_hamiltonian: OperatorEvolution,
        collapse_operators: list[Operator] | None = None,
        **kwargs,
    ):
        """Evolve the system."""

        # force the qutip engine (and, on GPU, its data-layer plugin) to be
        # imported and registered before any operator is built: ``_to_device``
        # below calls ``op.to(self.gpu_dtype)``, which only works once the
        # qutip-jax/qutip-cupy plugin has registered that dtype; this also
        # validates ``gpu_dtype`` early instead of part-way through the evolution
        time_diff = np.diff(time)
        nsteps = max(time_diff) / INTEGRATION_MIN_TIME_STEP * INTEGRATION_MULTIPLIER

        if self.device == "gpu":
            import diffrax
            # the default scipy integrators would transfer the state between
            # device and host on every step; the diffrax integrator provided
            # by qutip-jax keeps the whole evolution on the accelerator, with
            # the same maximum-step bound enforced through its controller

            options = {
                "method": "diffrax",
                "stepsize_controller": diffrax.PIDController(
                    rtol=1e-6, atol=1e-8, dtmax=INTEGRATION_MAX_TIME_STEP
                ),
                "max_steps": int(nsteps),
            }

        else:  # only cpu available, using SciPy solver
            # not every SciPy solvers accepts as parameters min_step, that's why we
            # define nsteps instead
            options = {"max_step": INTEGRATION_MAX_TIME_STEP, "nsteps": nsteps}

        hamiltonian = [self._to_device(hamiltonian)] + [
            [
                self._to_device(operator),
                self.interpolate_coeffs(time_hamiltonian.times, coefficient),
            ]
            for operator, coefficient in time_hamiltonian.operators
        ]

        sim_results = self.engine.mesolve(
            hamiltonian,
            self._to_device(initial_state),
            time,
            [self._to_device(op) for op in collapse_operators or []],
            options=options,
            **kwargs,
        )

        return sim_results, options

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

    def expand(self, op: Operator, dims: list[int], targets: int | list[int]):
        """Expand operator in larger Hilbert space."""
        return self._to_device(self.engine.expand_operator(op, dims, targets))

    def basis(self, dim: int, state: int) -> Operator:
        """Basis operator for n levels system."""
        return self._to_device(self.engine.basis(dimensions=dim, n=state))


def load_simulation(
    simulation_path: os.PathLike, sequence_index: int, sweep_index: int
) -> tuple[OperatorEvolution, np.typing.NDArray, dict]:
    """Load simulation results from disk.

    Args:
        simulation_path: Path to the simulation directory.
        sequence_index: Index of the sequence to load.
        sweep_index: Index of the sweep to load.

    Returns:
        A tuple containing:
            - system: List with Hamiltonian and time-dependent operators.
            - result_states: Array of simulated result states.
            - sim_configs: Dictionary of simulation configuration parameters.

    Raises:
        NotADirectoryError: If the sequence directory does not exist.
    """
    simulation_path = Path(simulation_path)

    simulated_sequence_path = simulation_path / f"sequence_{sequence_index}"
    if not simulated_sequence_path.is_dir():
        raise NotADirectoryError

    hamiltonians = np.load(simulated_sequence_path / (HAMILTONIAN_FILENAME + ".npy"))
    sweep_sim = np.load(
        simulated_sequence_path / (SWEEP_SIMULATION_FILENAME + f"_{sweep_index}.npz")
    )
    result_states = sweep_sim["results"]
    time_coeffs = sweep_sim["time_coeffs"]

    with open(simulated_sequence_path / (SIMULATOR_CONFIG + ".json")) as f:
        sim_configs = json.load(f)

    system = [Operator(hamiltonians[0])] + [
        TimeDependentOperator([ham, coeffs])
        for ham, coeffs in zip(hamiltonians[1:], time_coeffs[1:])
    ]
    timesteps = time_coeffs[0]
    system_evo = OperatorEvolution(operators=system, times=timesteps)

    return system_evo, result_states, sim_configs
