# This file is part of
from cqibolab import ffi
from qibolab import execute_qasm as py_execute_qasm
import numpy as np


@ffi.def_extern()
def execute_qasm(circuit, platform, nshots):
    """Generate samples for a given circuit qasm, platform and number of shots."""
    py_circuit = ffi.string(circuit).decode('utf-8')
    py_platform = ffi.string(platform).decode('utf-8')
    qasm_res = py_execute_qasm(py_circuit, py_platform, nshots)
    return ffi.cast("int*", ffi.from_buffer(qasm_res.samples()))
