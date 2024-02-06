# Rust API

## Install

You just need to install Qibolab as usual (see [main README.md](../README.md)) and
specify the `qibolab` crate inside `Cargo.toml`.

See the example below.

## Example

To run the example in the [`examples/`](./examples) folder just use Cargo:

```sh
cargo run --example example
```

## Environment

In order to properly locate `qibolab` in a virtual environment, on some platform (e.g.
MacOS) it is required to manually modify the `$PYTHONPATH`.

```sh
PYTHONPATH=$(realpath ${VIRTUAL_ENV}/lib/python3.11/site-packages):$PYTHONPATH
```

In editable mode installation, you might also need to manually add the `src/` folder:

```sh
PYTHONPATH=$(realpath qibolab/src/):$PYTHONPATH
```
