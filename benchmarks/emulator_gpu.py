"""Benchmark emulator engines on CPU or accelerator devices.

Example:
    python benchmarks/emulator_gpu.py --engine dynamiqs --device cpu --runs 5
    python benchmarks/emulator_gpu.py --engine dynamiqs --device gpu --precision single
    python benchmarks/emulator_gpu.py --engine qutip --device gpu
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from qibolab import AcquisitionType, AveragingMode, create_platform
from qibolab._core.platform.load import PLATFORMS
from qibolab.instruments.emulator import DynamiqsEngine, EmulatorController, QutipEngine


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_PLATFORMS = REPO / "tests" / "instruments" / "emulator" / "platforms"


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="qubit")
    parser.add_argument("--platforms-path", type=Path, default=DEFAULT_PLATFORMS)
    parser.add_argument("--engine", choices=["qutip", "dynamiqs"], default="dynamiqs")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--gpu-dtype", choices=["jax", "jaxdia", "cupyd"], default="jax"
    )
    parser.add_argument("--precision", choices=["single", "double"], default="double")
    parser.add_argument(
        "--matmul-precision", choices=["low", "high", "highest"], default="highest"
    )
    parser.add_argument("--method", choices=["adaptive", "fixed"], default="adaptive")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--nshots", type=int, default=100)
    return parser


def nvidia_smi() -> str | None:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def jax_devices() -> list[str]:
    try:
        import jax

        return [str(device) for device in jax.devices()]
    except Exception as exc:  # pragma: no cover - diagnostic path
        return [f"unavailable: {exc}"]


def cupy_device() -> str | None:
    try:
        import cupy as cp

        device = cp.cuda.Device()
        return f"id={device.id}, memory={cp.cuda.runtime.memGetInfo()}"
    except Exception:
        return None


def build_engine(args: argparse.Namespace):
    if args.engine == "qutip":
        return QutipEngine(device=args.device, gpu_dtype=args.gpu_dtype)
    return DynamiqsEngine(
        precision=args.precision,
        device=args.device,
        device_index=args.device_index,
        matmul_precision=args.matmul_precision,
        method=args.method,
    )


def emulator_controller(platform) -> EmulatorController:
    return next(
        instrument
        for instrument in platform.instruments.values()
        if isinstance(instrument, EmulatorController)
    )


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    os.environ[PLATFORMS] = str(args.platforms_path)
    platform = create_platform(args.platform)
    emulator_controller(platform).engine = build_engine(args)

    gates = platform.natives.single_qubit[0]
    sequence = gates.RX() | gates.MZ()
    acquisition = list(sequence.channel(platform.qubits[0].acquisition))[-1].id
    execute_kwargs = dict(
        nshots=args.nshots,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    for _ in range(args.warmups):
        platform.execute([sequence], **execute_kwargs)

    timings = []
    for _ in range(args.runs):
        start = time.perf_counter()
        result = platform.execute([sequence], **execute_kwargs)
        timings.append(time.perf_counter() - start)

    return {
        "engine": args.engine,
        "device": args.device,
        "platform": args.platform,
        "runs": args.runs,
        "warmups": args.warmups,
        "precision": args.precision,
        "matmul_precision": args.matmul_precision,
        "method": args.method,
        "gpu_dtype": args.gpu_dtype if args.engine == "qutip" else None,
        "mean_seconds": float(np.mean(timings)),
        "std_seconds": float(np.std(timings)),
        "min_seconds": float(np.min(timings)),
        "max_seconds": float(np.max(timings)),
        # peak resident set size of the process, covering the C/JAX buffers
        # that tracemalloc cannot see (Linux reports KiB)
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "result": np.asarray(result[acquisition]).tolist(),
        "nvidia_smi": nvidia_smi(),
        "jax_devices": jax_devices(),
        "cupy_device": cupy_device(),
    }


def main() -> None:
    args = parser().parse_args()
    print(json.dumps(benchmark(args), indent=2))


if __name__ == "__main__":
    main()
