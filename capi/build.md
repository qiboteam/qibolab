# Further compilation instructions

## Shared libraries on MacOS

On MacOS the environment variable `LD_LIBRARY_PATH` is replaced by `DYLD_LIBRARY_PATH`.

## Nix vs Clang

Same of what is specified in the [README](./README.md), but to use the GCC compiler,
which is not necessarily the default (e.g. on MacOS), add a suitable CMake flag

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=<your install prefix> -D CMAKE_CXX_COMPILER=g++
cmake --build build
cmake --install build
```
