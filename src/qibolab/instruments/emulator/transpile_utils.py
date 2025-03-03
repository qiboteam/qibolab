import cudaq
from cudaq.kernel.kernel_builder import PyKernel
from cudaq.kernel.kernel_decorator import PyKernelDecorator

from qibo import gates
from qibo.models import Circuit
from qibo.transpiler.pipeline import Passes
from qibo.transpiler.optimizer import Preprocessing
from qibo.transpiler.router import ShortestPaths
from qibo.transpiler.unroller import Unroller, NativeGates
from qibo.transpiler.placer import Random
from qibo.transpiler.asserts import assert_transpiling

DEFAULT_NATIVE_GATES=NativeGates.default()

def get_emulator_connectivity(platform):
    connectivity=platform.topology 
    #exception_qubits = None
    try:
        exception_qubits = platform.instruments["pulse_simulator"].simulation_engine.model_config["exception_qubits"]
        for q in exception_qubits:
            q = int(q)
            if q in list(connectivity.nodes):
                connectivity.remove_node(q)
    except:
        pass

    return connectivity
    

def make_pipeline(platform, native_gates=DEFAULT_NATIVE_GATES):
    # Define the general transpilation pipeline
    custom_passes = [Preprocessing(), Random(), ShortestPaths(), Unroller(native_gates)]
    connectivity = get_emulator_connectivity(platform)
    custom_pipeline = Passes(custom_passes, connectivity)   

    return custom_pipeline

def transpile(circuit, platform, native_gates=DEFAULT_NATIVE_GATES):
    custom_pipeline = make_pipeline(platform, native_gates)
    transpiled_circ, final_layout = custom_pipeline(circuit)

    return transpiled_circ, final_layout

def get_pulse_sequence(circuit, backend, add_measure_all=False, native_gates=DEFAULT_NATIVE_GATES):
    platform = backend.platform
    
    # Set up circuit
    if type(circuit) in [PyKernel, PyKernelDecorator]: # note: no measurement gates 
        openqasm = cudaq.translate(circuit, format='openqasm2')
        circuit = Circuit.from_qasm(openqasm)
        nqubit = circuit.nqubits
        circuit.wire_names = list(range(nqubit))
        
    if add_measure_all:
        circuit.add(gates.M(*range(nqubit)))

    connectivity = get_emulator_connectivity(platform)
    
    # Transpile circuit
    transpiled_circ, final_layout = transpile(circuit, platform, native_gates)
    
    assert_transpiling(
    original_circuit=circuit,
    transpiled_circuit=transpiled_circ,
    connectivity=connectivity,
    final_layout=final_layout,
    native_gates=native_gates
    )

    # Compile pulse sequence
    pulse_sequence, measurement_map = backend.compiler.compile(transpiled_circ, platform=platform)
    
    return pulse_sequence