# -*- coding: utf-8 -*-
import numpy as np
from qibo.backends import NumpyBackend
from qibo.config import raise_error
from qibo.states import CircuitResult


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

        if circuit.measurement_gate is None:
            raise_error(RuntimeError, "No measurement register assigned.")

        # Transpile a circuit into a sequence of pulses ``PulseSequence``
        sequence: PulseSequence = self.platform.transpile(circuit)

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
        """Returns the probability of the qubit being in state |0>"""
        if qubits is None:  # pragma: no cover
            qubits = result.circuit.measurement_gate.qubits

        def distance(a, b):
            return abs(a - b)

        # basic classification
        probabilities = []
        for qubit in qubits:
            mean_state0: complex = complex(self.platform.characterization["single_qubit"][qubit]["mean_gnd_states"])
            mean_state1: complex = complex(self.platform.characterization["single_qubit"][qubit]["mean_exc_states"])
            i = result.execution_result[qubit][2]  # execution_result[qubit] provides the latest
            q = result.execution_result[qubit][3]  # acquisition data for the corresponding qubit
            measurement: complex = complex(i, q)
            d0 = distance(measurement, mean_state0)
            d1 = distance(measurement, mean_state1)
            d01 = distance(mean_state0, mean_state1)
            p = (d1**2 + d01**2 - d0**2) / 2 / d01**2
            probabilities.append([p, 1 - p])
        return probabilities
