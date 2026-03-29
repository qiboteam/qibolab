#!/usr/bin/env python3
"""Benchmark CUDA-Q and QuTiP emulator backends on the same workloads."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import numpy as np
    import qibolab
    from qibolab import PulseSequence


DEFAULT_PLATFORM_ROOT = (
    Path(__file__).resolve().parents[1] / "tests" / "instruments" / "emulator" / "platforms"
)


class BenchmarkError(RuntimeError):
    """Raised when benchmark setup or execution fails."""


@dataclass
class Workload:
    """Pulse-level workload to benchmark."""

    name: str
    description: str
    build: Callable[[Any], list[Any]]


@dataclass
class BenchmarkResult:
    """Timing summary for one backend/workload pair."""

    backend: str
    workload: str
    repeats: int
    warmups: int
    shots: int
    mean_seconds: float
    median_seconds: float
    min_seconds: float
    max_seconds: float
    stdev_seconds: float
    sample_seconds: list[float]
    measurement_summary: list[float]


def _build_single_qubit(platform: Any) -> list[Any]:
    q0 = platform.natives.single_qubit[0]
    return [q0.RX() | q0.MZ()]


def _build_qutrit_ladder(platform: Any) -> list[Any]:
    q0 = platform.natives.single_qubit[0]
    if q0.RX12 is None:
        raise BenchmarkError(f"Platform {platform} does not expose RX12.")
    return [q0.RX() | q0.RX12() | q0.MZ()]


def _build_two_qubit_cz(platform: Any) -> list[Any]:
    from qibolab import PulseSequence

    if platform.nqubits < 2:
        raise BenchmarkError(f"Platform {platform} does not have two qubits.")

    pair = platform.natives.two_qubit[0, 1]
    if pair.CZ is None:
        raise BenchmarkError(f"Platform {platform} does not expose CZ.")

    q0 = platform.natives.single_qubit[0]
    q1 = platform.natives.single_qubit[1]

    seq = PulseSequence()
    seq += q0.RX90()
    seq += q1.RX90()
    seq |= pair.CZ()
    seq |= q0.MZ() + q1.MZ()
    return [seq]


WORKLOADS: dict[str, Workload] = {
    "single-qubit": Workload(
        name="single-qubit",
        description="1q RX followed by readout",
        build=_build_single_qubit,
    ),
    "qutrit-ladder": Workload(
        name="qutrit-ladder",
        description="1q RX + RX12 followed by readout",
        build=_build_qutrit_ladder,
    ),
    "two-qubit-cz": Workload(
        name="two-qubit-cz",
        description="2q RX90 + CZ followed by dual readout",
        build=_build_two_qubit_cz,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark emulator execution time for the QuTiP and CUDA-Q backends "
            "using the repository's emulator test platforms."
        )
    )
    parser.add_argument(
        "--platform-root",
        type=Path,
        default=DEFAULT_PLATFORM_ROOT,
        help="Directory containing the emulator platform folders.",
    )
    parser.add_argument(
        "--qutip-platform",
        default="split-transmons",
        help="Platform folder name for the QuTiP-backed emulator.",
    )
    parser.add_argument(
        "--cudaq-platform",
        default="split-transmons-cudaq",
        help="Platform folder name for the CUDA-Q-backed emulator.",
    )
    parser.add_argument(
        "--workload",
        dest="workloads",
        action="append",
        choices=sorted(WORKLOADS),
        help="Workload to run. Can be passed multiple times. Defaults to all workloads.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="Number of shots for each execution.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of untimed warmup executions per backend/workload pair.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of timed executions per backend/workload pair.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path to write raw benchmark results as JSON.",
    )
    parser.add_argument(
        "--check-tolerance",
        type=float,
        default=0.05,
        help=(
            "Maximum allowed absolute difference between backend measurement summaries "
            "for the same workload."
        ),
    )
    return parser.parse_args()


def ensure_dependencies() -> None:
    missing = [module for module in ("qutip", "cudaq") if __importable(module) is False]
    if missing:
        missing_csv = ", ".join(missing)
        raise BenchmarkError(
            f"Missing optional dependency/dependencies: {missing_csv}. "
            "Install QuTiP and CUDA-Q before running this benchmark."
        )


def __importable(module: str) -> bool:
    try:
        __import__(module)
    except ImportError:
        return False
    return True


def load_platforms(
    platform_root: Path,
    qutip_platform_name: str,
    cudaq_platform_name: str,
) -> dict[str, Any]:
    import qibolab

    if not platform_root.exists():
        raise BenchmarkError(f"Platform root does not exist: {platform_root}")

    os.environ["QIBOLAB_PLATFORMS"] = str(platform_root)

    platforms = {
        "qutip": qibolab.create_platform(qutip_platform_name),
        "cudaq": qibolab.create_platform(cudaq_platform_name),
    }
    return platforms


def summarize_measurements(
    result: dict[int, Any],
    sequences: list[Any],
) -> list[float]:
    import numpy as np

    summaries = []
    for sequence in sequences:
        for _, acquisition in sequence.acquisitions:
            data = np.asarray(result[acquisition.id])
            summaries.append(float(np.real(data).mean()))
    return summaries


def benchmark_backend(
    backend: str,
    workload: Workload,
    platform: Any,
    *,
    shots: int,
    warmups: int,
    repeats: int,
) -> BenchmarkResult:
    sequences = workload.build(platform)

    for _ in range(warmups):
        platform.execute(sequences, nshots=shots)

    sample_seconds = []
    measurement_summary = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = platform.execute(sequences, nshots=shots)
        sample_seconds.append(time.perf_counter() - started)
        measurement_summary = summarize_measurements(result, sequences)

    stdev = statistics.stdev(sample_seconds) if len(sample_seconds) > 1 else 0.0

    return BenchmarkResult(
        backend=backend,
        workload=workload.name,
        repeats=repeats,
        warmups=warmups,
        shots=shots,
        mean_seconds=statistics.mean(sample_seconds),
        median_seconds=statistics.median(sample_seconds),
        min_seconds=min(sample_seconds),
        max_seconds=max(sample_seconds),
        stdev_seconds=stdev,
        sample_seconds=sample_seconds,
        measurement_summary=measurement_summary,
    )


def check_measurements(
    qutip_result: BenchmarkResult,
    cudaq_result: BenchmarkResult,
    tolerance: float,
) -> None:
    if len(qutip_result.measurement_summary) != len(cudaq_result.measurement_summary):
        raise BenchmarkError(
            "Backends produced different numbers of acquisition summaries for "
            f"workload {qutip_result.workload}."
        )

    for index, (lhs, rhs) in enumerate(
        zip(qutip_result.measurement_summary, cudaq_result.measurement_summary)
    ):
        if math.isfinite(lhs) and math.isfinite(rhs) and abs(lhs - rhs) <= tolerance:
            continue
        raise BenchmarkError(
            "Backends produced mismatched measurement summaries for "
            f"workload {qutip_result.workload} at acquisition {index}: "
            f"qutip={lhs:.6f}, cudaq={rhs:.6f}, tolerance={tolerance:.6f}"
        )


def print_report(results: list[BenchmarkResult]) -> None:
    print()
    print("Emulator backend benchmark")
    print("=" * 78)
    print(
        f"{'Workload':20} {'Backend':8} {'Mean [s]':>10} {'Median [s]':>10} "
        f"{'Min [s]':>10} {'Std [s]':>10}"
    )
    print("-" * 78)
    for result in results:
        print(
            f"{result.workload:20} {result.backend:8} "
            f"{result.mean_seconds:10.6f} {result.median_seconds:10.6f} "
            f"{result.min_seconds:10.6f} {result.stdev_seconds:10.6f}"
        )

    print("-" * 78)
    qutip_by_workload = {result.workload: result for result in results if result.backend == "qutip"}
    cudaq_by_workload = {result.workload: result for result in results if result.backend == "cudaq"}
    for workload in sorted(qutip_by_workload):
        qutip_result = qutip_by_workload[workload]
        cudaq_result = cudaq_by_workload[workload]
        speedup = qutip_result.mean_seconds / cudaq_result.mean_seconds
        print(
            f"{workload:20} speedup (qutip/cudaq): {speedup:0.2f}x "
            f"| measurement summary qutip={qutip_result.measurement_summary} "
            f"cudaq={cudaq_result.measurement_summary}"
        )


def write_json(path: Path, results: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(result) for result in results], indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    ensure_dependencies()

    workloads = (
        [WORKLOADS[name] for name in args.workloads]
        if args.workloads
        else list(WORKLOADS.values())
    )
    platforms = load_platforms(
        args.platform_root,
        args.qutip_platform,
        args.cudaq_platform,
    )

    results = []
    for workload in workloads:
        qutip_result = benchmark_backend(
            "qutip",
            workload,
            platforms["qutip"],
            shots=args.shots,
            warmups=args.warmup,
            repeats=args.repeats,
        )
        cudaq_result = benchmark_backend(
            "cudaq",
            workload,
            platforms["cudaq"],
            shots=args.shots,
            warmups=args.warmup,
            repeats=args.repeats,
        )
        check_measurements(qutip_result, cudaq_result, args.check_tolerance)
        results.extend([qutip_result, cudaq_result])

    print_report(results)

    if args.json_output is not None:
        write_json(args.json_output, results)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as exc:
        print(f"Benchmark setup failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
