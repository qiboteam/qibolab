# This file is part of
from cqibolab import ffi

from qibolab import execute_qasm as py_execute_qasm


@ffi.def_extern()
def execute_qasm(circuit, platform, nshots):
    """Generate samples for a given circuit qasm, platform and number of
    shots."""
    py_circuit = ffi.string(circuit).decode("utf-8")
    py_platform = ffi.string(platform).decode("utf-8")
    qasm_res = py_execute_qasm(circuit=py_circuit, platform=py_platform, nshots=nshots)
    return ffi.cast("int*", ffi.from_buffer(qasm_res.samples().flatten()))
