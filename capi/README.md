Qibolab C-API
=============

This repository contains a C library to access Qibolab from programming languages different from Python.

## Installation

Make sure you have installed the `qibolab` module in Python, then in order to install proceed with the usual cmake steps:
```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=<your install prefix>
cmake --build build
cmake --install build
```

## Usage

The compiler flags to include this library in your package can be
retrieved with:
```bash
pkg-config qibolab --cflags
pkg-config qibolab --libs
```

If you installed to a non-standard location, you need to set up the `PKG_CONFIG_PATH` and `LD_LIBRARY_PATH`, e.g.:
```bash
export PKG_CONFIG_PATH=${VIRTUAL_ENV}/lib/pkgconfig/:${PKG_CONFIG_PATH}:
export LD_LIBRARY_PATH=${VIRTUAL_ENV}/lib/:${LD_LIBRARY_PATH}:
```

Sample programs using this library are provided in the `capi/examples/` directory.
