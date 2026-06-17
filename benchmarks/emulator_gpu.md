# Emulator GPU benchmarks

GPU support for the emulator engines (issue #1414) comes with two questions
attached: *when* does the GPU pay off, and *what does it cost in accuracy*?
This document answers both with measurements, produced by the two helpers in
this directory:

- `emulator_gpu.py` — end-to-end benchmark of `platform.execute()` on the
  bundled emulator platforms;
- `engine_sweep.py` — solver-level size sweep (`single`), drive-amplitude scan
  (`batched`), and accuracy comparison (`validate`) on a chain of coupled
  transmons with a spline-interpolated resonant drive, built through the same
  engine API the emulator uses.

## TL;DR — when to use which engine

1. **Single evolutions of small systems (Hilbert dimension ≲ 100):** the
   QuTiP engine on CPU is the fastest option by 1–2 orders of magnitude. GPU
   time is dominated by kernel-launch latency times the many integration steps
   needed to resolve lab-frame qubit oscillations.
2. **Batched parameter scans** (the typical calibration workload — the same
   system solved at many drive amplitudes): the Dynamiqs engine on GPU
   broadcasts the scan through one `mesolve` call and overtakes the serial
   QuTiP CPU loop already at dimension 27 (two 3-level transmons) by an order
   of magnitude (×13 at 200 scan points), with the advantage growing with the
   number of points.
3. **Single evolutions of larger systems:** on a resonantly driven chain the
   Dynamiqs GPU engine overtakes QuTiP CPU around dimension ~60–100 in
   **single precision** (×11 at dimension 243, at an accuracy cost that is
   negligible compared to the solver differences — see the validation table)
   and around dimension ~150–250 even in double precision, despite the
   measurements below coming from a consumer GPU whose double-precision
   throughput is rate-limited in hardware. On data-center GPUs with full-rate
   FP64 the double-precision crossover is expected substantially earlier.

## Environment

All numbers below were measured on:

- NVIDIA GeForce RTX 5090 (32 GB, Blackwell, driver 595.71.05), CUDA 12.9,
  Ubuntu 24.04, Python 3.12, 24-core CPU;
- `qutip 5.3.0`, `dynamiqs 0.3.4`, `jax 0.6.2` + `jax-cuda12-plugin`
  (consistent with `poetry.lock`: dynamiqs requires `jax<0.7`, and the CUDA 12
  plugin of the locked JAX version supports current GPUs):

  ```bash
  pip install qibolab[emulator] "jax[cuda12]"   # Dynamiqs engine on GPU
  pip install qutip-jax                          # QuTiP engine on GPU
  ```

- `XLA_PYTHON_CLIENT_PREALLOCATE=false` so the reported GPU memory reflects
  actual usage rather than the JAX pre-allocation pool.

The `cupyd` data layer additionally needs `qutip-cupy`, which is not released
on PyPI (`pip install git+https://github.com/qutip/qutip-cupy`) — it is
therefore treated as experimental and not benchmarked in detail.

## Workload

A chain of `n` transmons with `levels` levels each (Duffing oscillators,
4.5–5.5 GHz lab-frame frequencies, −0.3 GHz anharmonicity, 50 MHz
nearest-neighbor coupling, T1 = 50 µs on every element), driven on the first
transmon by a 40 ns sin²-envelope pulse with a carrier resonant with that
transmon, interpolated by the same cubic-spline machinery the emulator uses,
and integrated with each engine's defaults: QuTiP `zvode` with
`max_step = 0.02 ns`, Dynamiqs adaptive Tsit5 with `rtol = atol = 1e-8` (the
difference between 1e-6 and 1e-8 tolerances measured below 3% of runtime, so
the strict setting is used throughout). "Cold" timings include JIT
compilation; "warm" timings are a second identical call in the same process.

## Accuracy validation (`engine_sweep.py validate --case 3x2 --device gpu`)

Maximum absolute deviation of the density matrices (all saved times) from the
QuTiP-CPU double-precision reference, two coupled 3-level transmons,
resonantly driven:

| configuration | max deviation |
| --- | ---: |
| Dynamiqs CPU double, adaptive | 6.1e-4 |
| Dynamiqs CPU double, fixed (Rouchon3, dt=5e-3) | 9.1e-4 |
| Dynamiqs GPU double | 6.1e-4 |
| Dynamiqs GPU **single** | 6.2e-4 |
| QuTiP GPU (qutip-jax, diffrax) | 5.8e-4 |

Two observations: the GPU reproduces the CPU result of the same engine to
machine precision, and **single precision adds ~1e-5 on top of a ~6e-4
cross-solver deviation** — the engine/solver choice dominates, not the float
width. All deviations are far below the probability tolerances used by the
emulator tests (1e-2).

## Single evolutions: size sweep (`engine_sweep.py single`)

Warm seconds per evolution (double precision unless stated):

| case | dim | QuTiP CPU | Dynamiqs CPU | Dynamiqs GPU | Dynamiqs GPU (single) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2×2 | 4 | 0.023 | 0.81 | 9.8 | — |
| 3×2 | 9 | 0.076 | 0.76 | 9.7 | — |
| 3×3 | 27 | 1.50 | 2.10 | 16.1 | 8.9 |
| 3×4 | 81 | 21.5 | 30.4 | 51.4 | **14.2** |
| 3×5 | 243 | 229.0 | — | **187.9** | **20.6** |

QuTiP's runtime grows steeply once the density matrix stops fitting in cache
(21.5 s → 229 s between dimensions 81 and 243, ≈ dim^2.2), while the GPU
columns stay launch-latency-bound far longer (≈ dim^1.1 up to 243 in single
precision). The resulting crossovers on this workload: **single precision
overtakes QuTiP CPU between dimensions 81 and 243** (1.5× at 81, 11× at 243),
and **double precision crosses over just below dimension 243** (1.2× at 243)
even on FP64-limited consumer hardware. Peak GPU memory at dimension 243 is
58 MB (double) / 28 MB (single) — far from being the constraint; very large
Hilbert spaces remain reachable.

GPU columns measure the same machine-verified physics: the ground-state
population reported per record agrees with the QuTiP value at every size
(e.g. 0.1973 vs 0.1972 at dimension 81; 0.2704 vs 0.2729 at dimension 243,
consistent with the cross-solver deviations in the validation table).

## Batched amplitude scans (`engine_sweep.py batched`)

The same system solved at many drive amplitudes — QuTiP loops, Dynamiqs
broadcasts the amplitude axis through a single batched `mesolve`:

| case | dim | points | QuTiP CPU serial | Dynamiqs GPU batched (warm) | speed-up |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3×3 | 27 | 200 | 299.7 | **22.7** | **×13.2** |
| 3×4 | 81 | 50 | 991.5 | **127.7** | **×7.8** |

Batching amortizes both the launch latency and the integration steps across
the whole scan: at dimension 27 the batched GPU evolution costs 0.11 s per
amplitude against 1.50 s for each serial QuTiP evolution — and these batched
numbers are in *double* precision. Peak GPU memory for the 200-point batch is
105 MB.

The emulator currently executes sweeps as a loop of independent evolutions,
so these numbers demonstrate the headroom available to a batched-sweep
execution path rather than an immediate end-to-end speed-up; wiring the
qibolab sweepers to batched evolution is left as follow-up work, since it
touches the execution pipeline outside the engine boundary.

## End-to-end emulator benchmark (`emulator_gpu.py`, `qubit` fixture)

| engine | device | mean seconds |
| --- | --- | ---: |
| QuTiP | CPU | 0.018 |
| Dynamiqs | CPU | 0.75 |
| Dynamiqs | GPU | 11.9 |

At dimension 2 the GPU is pure overhead, consistent with the size sweep. The
gap between the Dynamiqs CPU column here and the solver-level sweep is JIT
compilation and per-call retracing in `platform.execute()`.

## Caveats and outlook

- The double-precision GPU column reflects GeForce hardware, whose FP64
  throughput is ~1/64 of FP32; on A100/H100-class devices the
  double-precision crossover moves to much smaller systems. The benchmark
  scripts print the detected hardware so results remain attributable.
- Runtime on both engines is dominated by the number of integration steps
  needed to track lab-frame oscillations (tolerance settings change runtimes
  by <3%); a rotating-frame/interaction-picture formulation would be the
  single largest speed lever for both engines, independent of GPU support.
- `qutip-jax` evolution emits a diffrax warning that complex-dtype support is
  work in progress; the validation table shows the result agrees with the
  CPU reference for this workload, but the warning is worth keeping in mind.
- Dynamiqs batching applies to coefficient batches (drive parameter scans).
  Scans that change the Hamiltonian structure or the readout cannot be
  batched this way.
