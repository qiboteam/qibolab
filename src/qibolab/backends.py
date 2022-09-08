# -*- coding: utf-8 -*-
from qibo.backends import NumpyBackend
from qibo.config import raise_error
from qibo.states import CircuitResult


class QibolabBackend(NumpyBackend):
    def __init__(self, platform, runcard=None):
        from qibolab.platform import Platform

        super().__init__()
        self.name = "qibolab"
        self.platform = Platform(platform, runcard)
        self.platform.connect()
        self.platform.setup()

    def apply_gate(self, gate, state, nqubits):  # pragma: no cover
        raise_error(NotImplementedError, "Qibolab cannot apply gates directly.")

    def apply_gate_density_matrix(self, gate, state, nqubits):  # pragma: no cover
        raise_error(NotImplementedError, "Qibolab cannot apply gates directly.")

    def execute_circuit(self, circuit, initial_state=None, nshots=None):  # pragma: no cover
        """Executes a quantum circuit.

        Args:
            circuit (:class:`qibo.core.circuit.Circuit`): Circuit to execute.
            nshots (int): Number of shots to sample from the experiment.
                If ``None`` the default value provided as hardware_avg in the
                calibration yml will be used.

        Returns:
            CircuitResult object containing the results acquired from the execution.
        """
        from qibolab.pulses import PulseSequence

        if initial_state is not None:
            raise_error(
                ValueError,
                "Hardware backend does not support " "initial state in circuits.",
            )

        # Translate gates to pulses and create a ``PulseSequence``
        if circuit.measurement_gate is None:
            raise_error(RuntimeError, "No measurement register assigned.")

        sequence = self.platform.transpile(circuit)

        # Execute the pulse sequence on the platform
        self.platform.start()
        readout = self.platform(sequence, nshots)
        self.platform.stop()
        return CircuitResult(self, circuit, readout, nshots)

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
        # Returns the probability of the qubit being in state 0
        if qubits is None:  # pragma: no cover
            qubits = result.circuit.measurement_gate.qubits
        # naive normalization
        qubit = qubits[0]
        readout = list(list(result.execution_result.values())[0].values())[0]
        state1_voltage = self.platform.settings["characterization"]["single_qubit"][qubit]["state1_voltage"]
        state0_voltage = self.platform.settings["characterization"]["single_qubit"][qubit]["state0_voltage"]
        import numpy as np

        p = np.abs(readout[0] * 1e6 - state1_voltage) / np.abs(state1_voltage - state0_voltage)
        return [p, 1 - p]
        # TODO: calculate probabilities based on the euclidean distance to state 0 and state 1 average points
