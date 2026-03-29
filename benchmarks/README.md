# Emulator Benchmarks

Use [`emulator_compare.py`](./emulator_compare.py) to compare the emulator runtime of the QuTiP and CUDA-Q backends on the same pulse workloads.

The benchmark loads the existing emulator platforms from [`tests/instruments/emulator/platforms`](../tests/instruments/emulator/platforms), runs warmups plus repeated timed executions, and checks that both backends produce comparable measurement summaries before reporting timings.

Example:

```bash
python benchmarks/emulator_compare.py --repeats 10 --warmup 2
```

Or through Poe:

```bash
poe benchmark-emulator --repeats 10 --warmup 2
```

Requirements:

- `qutip` must be installed.
- `cudaq` must be installed.
- The script assumes the repository checkout layout is unchanged so it can find the bundled emulator platforms.
