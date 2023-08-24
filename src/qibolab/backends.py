import itertools

import numpy as np
from qibo import __version__ as qibo_version
from qibo.backends import NumpyBackend
from qibo.config import raise_error
from qibo.states import CircuitResult

from qibolab import ExecutionParameters
from qibolab import __version__ as qibolab_version
from qibolab import create_platform
from qibolab.compilers import Compiler
from qibolab.platform import Platform
from qibolab.transpilers.pipeline import Passes


class QibolabBackend(NumpyBackend):
    def __init__(self, platform, runcard=None):
        super().__init__()
        self.name = "qibolab"
        if isinstance(platform, Platform):
            self.platform = platform
        else:
            self.platform = create_platform(platform, runcard)
        self.versions = {
            "qibo": qibo_version,
            "numpy": self.np.__version__,
            "qibolab": qibolab_version,
        }
        self.compiler = Compiler.default()
        self.transpiler = Passes(connectivity=self.platform.topology)

    def apply_gate(self, gate, state, nqubits):  # pragma: no cover
        raise_error(NotImplementedError, "Qibolab cannot apply gates directly.")

    def apply_gate_density_matrix(self, gate, state, nqubits):  # pragma: no cover
        raise_error(NotImplementedError, "Qibolab cannot apply gates directly.")

    def assign_measurements(self, measurement_map, circuit_result):
        """Assigning measurement outcomes to :class:`qibo.states.MeasurementResult` for each gate.

        This allows properly obtaining the measured shots from the :class:`qibo.states.CircuitResult`
        object returned by the circuit execution.

        Args:
            measurement_map (dict): Map from each measurement gate to the sequence of
                readout pulses implementing it.
            circuit_result (:class:`qibo.states.CircuitResult`): Circuit result object
                containing the readout measurement shots. This is created in ``execute_circuit``.
        """
        readout = circuit_result.execution_result
        for gate, sequence in measurement_map.items():
            _samples = (readout[pulse.serial].samples for pulse in sequence.pulses)
            samples = list(filter(lambda x: x is not None, _samples))
            gate.result.backend = self
            gate.result.register_samples(np.array(samples).T)

    def execute_circuit(
        self, circuit, initial_state=None, nshots=None, fuse_one_qubit=False, check_transpiled=False
    ):  # pragma: no cover
        """Executes a quantum circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to execute.
            initial_state (:class:`qibo.models.circuit.Circuit`): Circuit to prepare the initial state.
                If ``None`` the default ``|00...0>`` state is used.
            nshots (int): Number of shots to sample from the experiment.
                If ``None`` the default value provided as hardware_avg in the
                calibration yml will be used.
            fuse_one_qubit (bool): If ``True`` it fuses one qubit gates during
                transpilation to reduce circuit depth.
            check_transpiled (bool): If ``True`` it checks that the transpiled
                circuit is equivalent to the original using simulation.

        Returns:
            CircuitResult object containing the results acquired from the execution.
        """
        if isinstance(initial_state, type(circuit)):
            return self.execute_circuit(
                circuit=initial_state + circuit,
                nshots=nshots,
                fuse_one_qubit=fuse_one_qubit,
                check_transpiled=check_transpiled,
            )
        if initial_state is not None:
            raise_error(
                ValueError,
                "Hardware backend only supports circuits as initial states.",
            )

        if self.transpiler is None or self.transpiler.is_satisfied(circuit):
            native_circuit = circuit
        else:
            # Transform a circuit into proper connectivity and native gates
            native_circuit, qubit_map = self.transpiler(circuit)
            # TODO: Use the qubit map to properly map measurements
            if check_transpiled:
                self.transpiler.check_execution(circuit, native_circuit)

        # Transpile the native circuit into a sequence of pulses ``PulseSequence``
        sequence, measurement_map = self.compiler.compile(native_circuit, self.platform)

        if not self.platform.is_connected:
            self.platform.connect()
            self.platform.setup()

        # Execute the pulse sequence on the platform
        self.platform.start()
        readout = self.platform.execute_pulse_sequence(
            sequence,
            ExecutionParameters(nshots=nshots),
        )
        self.platform.stop()
        result = CircuitResult(self, circuit, readout, nshots)
        self.assign_measurements(measurement_map, result)
        return result

    def circuit_result_tensor(self, result):
        raise_error(
            NotImplementedError,
            "Qibolab cannot return state vector in tensor representation.",
        )

    def circuit_result_representation(self, result: CircuitResult):
        # TODO: Consider changing this to a more readable format.
        # this must return a ``str`` because it is used in ``CircuitResult.__repr__``.
        return str(result.execution_result)

    def circuit_result_probabilities(self, result: CircuitResult, qubits=None):
        """Returns the probability of the qubit being in state ``|0>``."""
        if qubits is None:  # pragma: no cover
            qubits = [self.platform.get_qubit(q) for q in result.measurement_gate.qubits]

        # basic classification
        probabilities = []
        for qubit in qubits:
            # execution_result[qubit] provides the latest acquisition data for the corresponding qubit
            qubit_result = result.execution_result[qubit]
            if qubit_result.samples is None:
                mean_state0 = complex(self.platform.qubits[qubit].mean_gnd_states)
                mean_state1 = complex(self.platform.qubits[qubit].mean_exc_states)
                measurement = complex(qubit_result.I, qubit_result.Q)
                d0 = abs(measurement - mean_state0)
                d1 = abs(measurement - mean_state1)
                d01 = abs(mean_state0 - mean_state1)
                p = (d1**2 + d01**2 - d0**2) / 2 / d01**2
                probabilities.append([p, 1 - p])
            else:
                outcomes, counts = np.unique(qubit_result.samples, return_counts=True)
                probabilities.append([0, 0])
                for i, c in zip(outcomes.astype(int), counts):
                    probabilities[-1][i] = c / result.nshots

        # bring probabilities to the format returned by simulation
        return np.array(
            [
                np.prod([p[b] for p, b in zip(probabilities, bitstring)])
                for bitstring in itertools.product([0, 1], repeat=len(qubits))
            ]
        )
