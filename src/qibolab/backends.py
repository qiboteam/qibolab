import itertools

import numpy as np
from qibo.backends import NumpyBackend
from qibo.config import log, raise_error
from qibo.states import CircuitResult

from qibolab.transpilers import can_execute, transpile


class QibolabBackend(NumpyBackend):
    def __init__(self, platform, runcard=None):
        from qibo import __version__ as qibo_version

        from qibolab import __version__
        from qibolab.platform import Platform

        super().__init__()
        self.name = "qibolab"
        self.platform = Platform(platform, runcard)
        self.platform.connect()
        self.platform.setup()
        self.versions = {
            "qibo": qibo_version,
            "numpy": self.np.__version__,
            "qibolab": __version__,
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
            circuit (:class:`qibo.core.circuit.Circuit`): Circuit to execute.
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
        if initial_state is not None:
            raise_error(
                ValueError,
                "Hardware backend does not support initial state in circuits.",
            )

        if can_execute(circuit):
            native_circuit = circuit
        else:
            # Transform a circuit into proper connectivity and native gates
            native_circuit, hardware_qubits = transpile(circuit)
            if check_transpiled:
                backend = NumpyBackend()
                target_state = backend.execute_circuit(circuit).state()
                final_state = backend.execute_circuit(native_circuit).state()
                fidelity = np.abs(np.dot(np.conj(target_state), final_state))
                np.testing.assert_allclose(fidelity, 1.0)
                log.info("Transpiler test passed.")

        # Transpile the native circuit into a sequence of pulses ``PulseSequence``
        sequence = self.platform.transpile(native_circuit)

        # Execute the pulse sequence on the platform
        self.platform.start()
        readout = self.platform(sequence, nshots)
        self.platform.stop()
        result = CircuitResult(self, native_circuit, readout, nshots)

        shots = readout.get("binned_classified")
        # Register measurement outcomes
        if shots is not None:
            for gate in native_circuit.measurements:
                samples = np.array([shots.get(pulse) for pulse in gate.pulses])
                gate.result.register_samples(samples.T)
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
            qubits = result.circuit.measurement_gate.qubits

        # basic classification
        probabilities = []
        for qubit in qubits:
            mean_state0: complex = complex(self.platform.characterization["single_qubit"][qubit]["mean_gnd_states"])
            mean_state1: complex = complex(self.platform.characterization["single_qubit"][qubit]["mean_exc_states"])
            i = result.execution_result[qubit][2]  # execution_result[qubit] provides the latest
            q = result.execution_result[qubit][3]  # acquisition data for the corresponding qubit
            measurement: complex = complex(i, q)
            d0 = abs(measurement - mean_state0)
            d1 = abs(measurement - mean_state1)
            d01 = abs(mean_state0 - mean_state1)
            p = (d1**2 + d01**2 - d0**2) / 2 / d01**2
            probabilities.append([p, 1 - p])

        # bring probabilities to the format returned by simulation
        return np.array(
            [
                np.prod([p[b] for p, b in zip(probabilities, bitstring)])
                for bitstring in itertools.product([0, 1], repeat=len(qubits))
            ]
        )
