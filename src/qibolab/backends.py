import itertools

import numpy as np
from qibo import __version__ as qibo_version
from qibo import gates
from qibo.backends import NumpyBackend
from qibo.config import log, raise_error
from qibo.states import CircuitResult

from qibolab import __version__ as qibolab_version
from qibolab.platform import Platform
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.transpilers import can_execute, transpile


class QibolabBackend(NumpyBackend):
    def __init__(self, platform, runcard=None):
        super().__init__()
        self.name = "qibolab"
        if isinstance(platform, AbstractPlatform):
            self.platform = platform
        else:
            self.platform = Platform(platform, runcard)
        self.versions = {
            "qibo": qibo_version,
            "numpy": self.np.__version__,
            "qibolab": qibolab_version,
        }

    def apply_gate(self, gate, state, nqubits):  # pragma: no cover
        raise_error(NotImplementedError, "Qibolab cannot apply gates directly.")

    def apply_gate_density_matrix(self, gate, state, nqubits):  # pragma: no cover
        raise_error(NotImplementedError, "Qibolab cannot apply gates directly.")

    def execute_circuit(
        self, circuit, initial_state=None, nshots=None, fuse_one_qubit=False, check_transpiled=False
    ):  # pragma: no cover
        """Executes a quantum circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to execute.
            initial_state (:class:`qibo.models.circuit.Circuit`): Circuit to prepare the initial state.
                If ``None`` the default |00...0> state is used.
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
            self.execute_circuit(
                circuit=initial_state + circuit,
                nshots=nshots,
                fuse_one_qubit=fuse_one_qubit,
                check_transpiled=check_transpiled,
            )
        elif initial_state is not None:
            raise_error(
                ValueError,
                "Hardware backend only supports circuits as initial states.",
            )

        two_qubit_natives = self.platform.two_qubit_natives
        if can_execute(circuit, two_qubit_natives, verbose=False):
            native_circuit = circuit
        else:
            # Transform a circuit into proper connectivity and native gates
            log.info("Transpiling circuit.")
            native_circuit, _ = transpile(circuit, two_qubit_natives)
            if check_transpiled:
                backend = NumpyBackend()
                target_state = backend.execute_circuit(circuit).state()
                final_state = backend.execute_circuit(native_circuit).state()
                fidelity = np.abs(np.dot(np.conj(target_state), final_state))
                np.testing.assert_allclose(fidelity, 1.0)
                log.info("Transpiler test passed.")

        # Transpile the native circuit into a sequence of pulses ``PulseSequence``
        sequence = self.platform.transpile(native_circuit)

        if not self.platform.is_connected:
            self.platform.connect()
            self.platform.setup()

        # Execute the pulse sequence on the platform
        self.platform.start()
        readout = self.platform.execute_pulse_sequence(sequence, nshots)
        self.platform.stop()
        result = CircuitResult(self, native_circuit, readout, nshots)

        # Register measurement outcomes
        if isinstance(readout, dict):
            for gate in native_circuit.queue:
                if isinstance(gate, gates.M):
                    samples = []
                    for serial in gate.pulses:
                        shots = readout[serial].shots
                        if shots is not None:
                            samples.append(shots)
                    gate.result.backend = self
                    gate.result.register_samples(np.array(samples).T)
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
        """Returns the probability of the qubit being in state |0>"""
        if qubits is None:  # pragma: no cover
            qubits = result.measurement_gate.qubits

        # basic classification
        probabilities = []
        for qubit in qubits:
            # execution_result[qubit] provides the latest acquisition data for the corresponding qubit
            qubit_result = result.execution_result[qubit]
            if qubit_result.shots is None:
                mean_state0 = complex(self.platform.qubits[qubit].mean_gnd_states)
                mean_state1 = complex(self.platform.qubits[qubit].mean_exc_states)
                measurement = complex(qubit_result.I, qubit_result.Q)
                d0 = abs(measurement - mean_state0)
                d1 = abs(measurement - mean_state1)
                d01 = abs(mean_state0 - mean_state1)
                p = (d1**2 + d01**2 - d0**2) / 2 / d01**2
                probabilities.append([p, 1 - p])
            else:
                outcomes, counts = np.unique(qubit_result.shots, return_counts=True)
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
