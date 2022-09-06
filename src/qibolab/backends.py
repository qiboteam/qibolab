# -*- coding: utf-8 -*-
from qibo.backends import NumpyBackend
from qibo.config import raise_error
from qibo.states import CircuitResult


class QibolabBackend(NumpyBackend):
    def __init__(self, platform, runcard=None):
        from qibolab.native import NativeGates
        from qibolab.platform import Platform

        super().__init__()
        self.name = "qibolab"
        self.platform = Platform(platform, runcard)
        self.native_gates = NativeGates(self)

    def apply_gate(self, gate, state, nqubits):  # pragma: no cover
        raise_error(NotImplementedError, "Qibolab cannot apply gates directly.")

    def apply_gate_density_matrix(self, gate, state, nqubits):  # pragma: no cover
        raise_error(NotImplementedError, "Qibolab cannot apply gates directly.")

    def asnative(self, gate):
        """Transforms an arbitrary gate to a hardware native gate."""
        name = gate.__class__.__name__
        return getattr(self.native_gates, name)(gate)

    def transpile(self, circuit, fuse_one_qubit=True):
        from qibo import gates

        from qibolab.transpilers.connectivity import fix_connecivity

        circuit1, hardware_qubits = fix_connecivity(circuit)
        circuit2 = circuit.__class__(circuit.nqubits)
        # two-qubit gates to native
        for gate in circuit1.queue:
            if len(gate.qubits) > 1:
                circuit2.add(self.asnative(gate))
            else:
                circuit2.add(gate)

        # fuse one-qubit gates
        if fuse_one_qubit:
            circuit2 = circuit2.fuse(max_qubits=1)

        # one-qubit gates to native
        circuit3 = circuit.__class__(circuit.nqubits)
        for gate in circuit2.queue:
            if isinstance(gate, gates.FusedGate):
                matrix = gate.asmatrix(self)
                circuit3.add(self.asnative(gates.Unitary(matrix, *gate.qubits)))
            else:
                circuit3.add(self.asnative(gate))

        return circuit3

    def execute_circuit(
        self, circuit, initial_state=None, nshots=None
    ):  # pragma: no cover
        """Executes a quantum circuit.

        Args:
            circuit (:class:`qibo.core.circuit.Circuit`): Circuit to execute.
            nshots (int): Number of shots to sample from the experiment.
                If ``None`` the default value provided as hardware_avg in the
                calibration yml will be used.

        Returns:
            Readout results acquired by after execution.
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

        sequence = PulseSequence()
        for gate in circuit.queue:
            native_gate = self.asnative(gate)
            self.platform.to_sequence(sequence, native_gate)
        self.platform.to_sequence(sequence, circuit.measurement_gate)

        # Execute the pulse sequence on the platform
        self.platform.connect()
        self.platform.setup()
        self.platform.start()
        readout = self.platform(sequence, nshots)
        self.platform.stop()
        return CircuitResult(self, circuit, readout, nshots)

    def circuit_result_tensor(self, result):
        raise_error(
            NotImplementedError,
            "Qibolab cannot return state vector in tensor representation.",
        )

    def circuit_result_representation(self, result):
        # TODO: Consider changing this to a more readable format.
        # this must return a ``str`` because it is used in ``CircuitResult.__repr__``.
        return str(result.execution_result)

    def circuit_result_probabilities(self, result, qubits=None):
        if qubits is None:  # pragma: no cover
            qubits = result.circuit.measurement_gate.qubits
        # naive normalization
        qubit = qubits[0]
        readout = list(list(result.execution_result.values())[0].values())[0]
        state1_voltage = self.platform.settings["characterization"]["single_qubit"][
            qubit
        ]["state1_voltage"]
        state0_voltage = self.platform.settings["characterization"]["single_qubit"][
            qubit
        ]["state0_voltage"]
        import numpy as np

        p = np.abs(readout[0] * 1e6 - state1_voltage) / np.abs(
            state1_voltage - state0_voltage
        )
        return [p, 1 - p]
