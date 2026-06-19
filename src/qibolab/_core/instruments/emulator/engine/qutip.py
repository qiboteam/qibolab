from collections.abc import Callable, Iterable
from functools import cached_property
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import make_interp_spline

from .abstract import (
    INTEGRATION_MAX_TIME_STEP,
    INTEGRATION_MIN_TIME_STEP,
    INTEGRATION_MULTIPLIER,
    SPLINE_INTERP_ORDER,
    Operator,
    OperatorEvolution,
    SimulationEngine,
    jax_interpolation,
)

__all__ = ["QutipEngine"]


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

    def expand(self, op: Operator, targets: int | list[int], dims: list[int]):
        """Expand operator in larger Hilbert space."""
        return self._to_device(self.engine.expand_operator(op, targets, dims))

    def basis(self, dim: int, state: int) -> Operator:
        """Basis operator for n levels system."""
        return self._to_device(self.engine.basis(dimensions=dim, n=state))
