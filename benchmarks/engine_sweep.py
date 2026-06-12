"""Solver-level benchmarks for the emulator engines.

The system is a chain of coupled transmons (Duffing oscillators with `levels`
levels each) with a spline-interpolated resonant drive on the first transmon
and amplitude damping on every element — the same Hamiltonian structure the
emulator builds for its platforms, exercised directly through the engine API
so the spline conversion and solver configuration of each engine are included
in the measurement.

Subcommands:

``single``
    One evolution per case over increasing Hilbert-space size, reporting cold
    (first call, including JIT compilation for Dynamiqs) and warm timings.

``batched``
    A drive-amplitude scan, the canonical calibration workload: QuTiP runs a
    serial loop of evolutions, while Dynamiqs broadcasts the amplitude axis
    through a single batched ``mesolve`` call.

``validate``
    Cross-engine and cross-precision accuracy comparison on the same resonant
    drive, reporting the maximum density-matrix deviation from the QuTiP CPU
    double-precision reference.

Examples:

    python benchmarks/engine_sweep.py single --engine qutip --cases 2x2,3x2,3x3
    python benchmarks/engine_sweep.py single --engine dynamiqs --device gpu \\
        --precision single --cases 3x3,3x4,3x5
    python benchmarks/engine_sweep.py batched --engine dynamiqs --device gpu \\
        --case 3x3 --amplitudes 200
    python benchmarks/engine_sweep.py validate --case 3x2
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import time

import numpy as np
from scipy.interpolate import make_interp_spline

from qibolab._core.instruments.emulator.engine.abstract import OperatorEvolution
from qibolab.instruments.emulator import DynamiqsEngine, QutipEngine

GIGA = 1e9
FREQUENCIES = (4.5, 5.5)
"""GHz, range of the transmon frequencies along the chain."""
ANHARMONICITY = -0.3
"""GHz, transmon anharmonicity."""
COUPLING = 0.05
"""GHz, nearest-neighbor coupling."""
T1 = 50_000
"""ns, relaxation time of every element."""
RABI = 0.02
"""GHz, peak Rabi rate of the drive."""
MAX_DIM = 4096
"""Refuse larger Hilbert spaces to avoid accidental memory exhaustion."""


def parse_case(case: str) -> tuple[int, int]:
    """Parse a ``<levels>x<elements>`` case label."""
    levels, n = (int(x) for x in case.strip().split("x"))
    return levels, n


def chain_operators(engine, levels: int, n: int):
    """Chain Hamiltonian terms built through the engine operator API."""
    dims = [levels] * n
    lowerings = [
        engine.expand(engine.destroy(levels), dims, i)
        if n > 1
        else engine.destroy(levels)
        for i in range(n)
    ]
    frequencies = np.linspace(*FREQUENCIES, n)
    hamiltonian = 0
    for i, lowering in enumerate(lowerings):
        number = lowering.dag() * lowering
        hamiltonian += 2 * np.pi * frequencies[i] * number
        hamiltonian += (
            np.pi
            * ANHARMONICITY
            * (lowering.dag() * lowering.dag() * lowering * lowering)
        )
    for left, right in zip(lowerings[:-1], lowerings[1:]):
        hamiltonian += 2 * np.pi * COUPLING * (left.dag() * right + left * right.dag())
    collapse = [(1 / T1) ** 0.5 * lowering for lowering in lowerings]
    drive = lowerings[0] + lowerings[0].dag()
    initial = engine.basis(dims, [0] * n) if n > 1 else engine.basis(levels, 0)
    return hamiltonian, drive, collapse, initial


def drive_spline(duration: float, amplitude: float = 1.0):
    """Cubic spline of a sin^2 envelope with a carrier resonant with qubit 0."""
    times = np.linspace(0.0, duration, int(duration * 50) + 1)
    waveform = (
        2
        * np.pi
        * RABI
        * amplitude
        * np.sin(np.pi * times / duration) ** 2
        * np.cos(2 * np.pi * FREQUENCIES[0] * times)
    )
    return make_interp_spline(times, waveform, k=3)


def evolve_case(engine, levels: int, n: int, duration: float, points: int, **kwargs):
    """One evolution through the public engine API."""
    hamiltonian, drive, collapse, initial = chain_operators(engine, levels, n)
    evolution = OperatorEvolution(operators=[[drive, drive_spline(duration)]])
    return engine.evolve(
        hamiltonian=hamiltonian,
        initial_state=initial,
        time=np.linspace(0.0, duration, points),
        time_hamiltonian=evolution,
        collapse_operators=collapse,
        **kwargs,
    )


def memory_metrics(device: str) -> dict:
    metrics = {
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    }
    if device == "gpu":
        try:
            import jax

            stats = jax.local_devices()[0].memory_stats() or {}
            metrics["gpu_peak_bytes"] = stats.get("peak_bytes_in_use")
        except Exception:
            metrics["gpu_peak_bytes"] = None
        try:
            metrics["nvidia_smi"] = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.used",
                    "--format=csv,noheader",
                ],
                text=True,
                timeout=5,
            ).strip()
        except (FileNotFoundError, subprocess.SubprocessError):
            metrics["nvidia_smi"] = None
    return metrics


def build_engine(args: argparse.Namespace):
    if args.engine == "qutip":
        return QutipEngine(device=args.device, gpu_dtype=args.gpu_dtype)
    return DynamiqsEngine(
        device=args.device,
        precision=args.precision,
        matmul_precision=args.matmul_precision,
        method=args.method,
    )


def check_dim(levels: int, n: int) -> int:
    dim = levels**n
    if dim > MAX_DIM:
        raise ValueError(f"Hilbert space dimension {dim} exceeds MAX_DIM={MAX_DIM}.")
    return dim


def run_single(args: argparse.Namespace) -> None:
    engine = build_engine(args)
    for case in args.cases.split(","):
        levels, n = parse_case(case)
        dim = check_dim(levels, n)
        start = time.perf_counter()
        evolve_case(engine, levels, n, args.duration, args.points)
        cold = time.perf_counter() - start
        start = time.perf_counter()
        result = evolve_case(engine, levels, n, args.duration, args.points)
        warm = time.perf_counter() - start
        populations = np.real(np.diagonal(result.states[-1].full()))
        record = {
            "mode": "single",
            "engine": args.engine,
            "device": args.device,
            "precision": args.precision if args.engine == "dynamiqs" else "double",
            "method": args.method if args.engine == "dynamiqs" else "qutip-default",
            "case": case,
            "dim": dim,
            "cold_seconds": cold,
            "warm_seconds": warm,
            "ground_population": float(populations[0]),
            **memory_metrics(args.device),
        }
        print(json.dumps(record), flush=True)


def run_batched(args: argparse.Namespace) -> None:
    levels, n = parse_case(args.case)
    dim = check_dim(levels, n)
    amplitudes = np.linspace(0.25, 1.0, args.amplitudes)

    if args.engine == "qutip":
        engine = QutipEngine(device=args.device, gpu_dtype=args.gpu_dtype)
        hamiltonian, drive, collapse, initial = chain_operators(engine, levels, n)
        times = np.linspace(0.0, args.duration, args.points)
        start = time.perf_counter()
        for amplitude in amplitudes:
            evolution = OperatorEvolution(
                operators=[[drive, drive_spline(args.duration, amplitude)]]
            )
            engine.evolve(
                hamiltonian=hamiltonian,
                initial_state=initial,
                time=times,
                time_hamiltonian=evolution,
                collapse_operators=collapse,
            )
        cold = warm = time.perf_counter() - start
    else:
        # Dynamiqs broadcasts a batch axis through a single mesolve call: the
        # engine API is built around one evolution at a time, so the batched
        # coefficient is assembled here from the same engine pieces (operators
        # and the JAX-traceable spline conversion) to demonstrate the headroom
        # available to a future batched-sweep execution path.
        import jax.numpy as jnp

        from qibolab._core.instruments.emulator.engine.dynamiqs import (
            _spline_function,
            _unwrap,
        )

        engine = build_engine(args)
        dq = engine.engine
        hamiltonian, drive, collapse, initial = chain_operators(engine, levels, n)
        envelope = _spline_function(drive_spline(args.duration))
        batch = jnp.asarray(amplitudes)

        def coefficient(t):
            return batch * envelope(t)

        batched_hamiltonian = dq.constant(_unwrap(hamiltonian)) + dq.modulated(
            coefficient, _unwrap(drive)
        )
        times = np.linspace(0.0, args.duration, args.points)
        method = (
            dq.method.Rouchon3(dt=engine.fixed_step_dt)
            if args.method == "fixed"
            else dq.method.Tsit5(rtol=engine.rtol, atol=engine.atol)
        )

        def execute():
            result = dq.mesolve(
                batched_hamiltonian,
                [_unwrap(op) for op in collapse],
                _unwrap(initial),
                times,
                method=method,
                options=dq.Options(progress_meter=False),
            )
            np.asarray(result.states.to_jax())  # synchronize

        start = time.perf_counter()
        execute()
        cold = time.perf_counter() - start
        start = time.perf_counter()
        execute()
        warm = time.perf_counter() - start

    record = {
        "mode": "batched",
        "engine": args.engine,
        "device": args.device,
        "precision": args.precision if args.engine == "dynamiqs" else "double",
        "case": args.case,
        "dim": dim,
        "amplitudes": args.amplitudes,
        "cold_seconds": cold,
        "warm_seconds": warm,
        "seconds_per_amplitude": warm / args.amplitudes,
        **memory_metrics(args.device),
    }
    print(json.dumps(record), flush=True)


def run_validate(args: argparse.Namespace) -> None:
    levels, n = parse_case(args.case)
    dim = check_dim(levels, n)

    def states(engine):
        result = evolve_case(engine, levels, n, args.duration, args.points)
        return np.stack([state.full() for state in result.states])

    reference = states(QutipEngine())
    configurations = [
        ("dynamiqs-cpu-double-adaptive", DynamiqsEngine()),
        ("dynamiqs-cpu-double-fixed", DynamiqsEngine(method="fixed")),
    ]
    if args.device == "gpu":
        configurations += [
            ("dynamiqs-gpu-double", DynamiqsEngine(device="gpu")),
            ("dynamiqs-gpu-single", DynamiqsEngine(device="gpu", precision="single")),
            ("qutip-gpu-jax", QutipEngine(device="gpu", gpu_dtype="jax")),
        ]

    for label, engine in configurations:
        try:
            deviation = float(np.abs(states(engine) - reference).max())
            record = {
                "mode": "validate",
                "configuration": label,
                "case": args.case,
                "dim": dim,
                "max_deviation": deviation,
            }
        except Exception as error:  # pragma: no cover - diagnostic path
            record = {
                "mode": "validate",
                "configuration": label,
                "case": args.case,
                "dim": dim,
                "error": repr(error),
            }
        print(json.dumps(record), flush=True)


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--engine", choices=["qutip", "dynamiqs"], default="dynamiqs")
        sub.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
        sub.add_argument(
            "--gpu-dtype", choices=["jax", "jaxdia", "cupyd"], default="jax"
        )
        sub.add_argument("--precision", choices=["single", "double"], default="double")
        sub.add_argument(
            "--matmul-precision", choices=["low", "high", "highest"], default="highest"
        )
        sub.add_argument("--method", choices=["adaptive", "fixed"], default="adaptive")
        sub.add_argument("--duration", type=float, default=40.0, help="ns")
        sub.add_argument("--points", type=int, default=11)

    single = subparsers.add_parser("single", help="size sweep of one evolution")
    common(single)
    single.add_argument("--cases", default="2x2,3x2,3x3,3x4,3x5")

    batched = subparsers.add_parser("batched", help="drive-amplitude scan")
    common(batched)
    batched.add_argument("--case", default="3x3")
    batched.add_argument("--amplitudes", type=int, default=200)

    validate = subparsers.add_parser("validate", help="accuracy comparison")
    common(validate)
    validate.add_argument("--case", default="3x2")

    return parser


def main() -> None:
    args = parser().parse_args()
    if args.mode == "single":
        run_single(args)
    elif args.mode == "batched":
        run_batched(args)
    else:
        run_validate(args)


if __name__ == "__main__":
    main()
