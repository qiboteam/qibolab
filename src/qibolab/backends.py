from collections import deque

import numpy as np
from qibo import __version__ as qibo_version
from qibo.backends import NumpyBackend
from qibo.config import raise_error
from qibo.models import Circuit
from qibo.result import MeasurementOutcomes

from qibolab.compilers import Compiler
from qibolab.execution_parameters import ExecutionParameters
from qibolab.platform import Platform, create_platform
from qibolab.platform.load import available_platforms
from qibolab.qubits import QubitId, QubitPairId
from qibolab.version import __version__ as qibolab_version


def execute_qasm(circuit: str, platform, initial_state=None, nshots=1000):
    """Executes a QASM circuit.

    Args:
        circuit (str): the QASM circuit.
        platform (str): the platform where to execute the circuit.
        initial_state (:class:`qibo.models.circuit.Circuit`): Circuit to prepare the initial state.
                If ``None`` the default ``|00...0>`` state is used.
        nshots (int): Number of shots to sample from the experiment.

    Returns:
        ``MeasurementOutcomes`` object containing the results acquired from the execution.
    """
    from qibolab.backends import QibolabBackend

    circuit = Circuit.from_qasm(circuit)
    return QibolabBackend(platform).execute_circuit(
        circuit, initial_state=initial_state, nshots=nshots
    )


class QibolabBackend(NumpyBackend):
    def __init__(self, platform):
        super().__init__()
        self.name = "qibolab"
        if isinstance(platform, Platform):
            self.platform = platform
        else:
            self.platform = create_platform(platform)
        self.versions = {
            "qibo": qibo_version,
            "numpy": self.np.__version__,
            "qibolab": qibolab_version,
        }
        self.compiler = Compiler.default()

    @property
    def qubits(self) -> list[QubitId]:
        """Returns the qubits in the platform."""
        return list(self.platform.qubits)

    @property
    def connectivity(self) -> list[QubitPairId]:
        """Returns the list of connected qubits."""
        return list(self.platform.pairs)

    @property
    def natives(self) -> list[str]:
        """Returns the list of native gates supported by the platform."""
        compiler = Compiler.default()
        natives = [g.__name__ for g in list(compiler.rules)]
        calibrated = self.platform.pairs

        check_2q = ["CZ", "CNOT"]
        for gate in check_2q:
            if gate in natives and all(
                getattr(calibrated[p].native_gates, gate) is None
                for p in self.connectivity
            ):
                natives.remove(gate)

        return natives

    def apply_gate(self, gate, state, nqubits):  # pragma: no cover
        raise_error(NotImplementedError, "Qibolab cannot apply gates directly.")

    def apply_gate_density_matrix(self, gate, state, nqubits):  # pragma: no cover
        raise_error(NotImplementedError, "Qibolab cannot apply gates directly.")

    def assign_measurements(self, measurement_map, readout):
        """Assigning measurement outcomes to
        :class:`qibo.states.MeasurementResult` for each gate.

        This allows properly obtaining the measured shots from the :class:`qibolab.pulses.ReadoutPulse` object obtaned after pulse sequence execution.

        Args:
            measurement_map (dict): Map from each measurement gate to the sequence of
                readout pulses implementing it.
            readout (:class:`qibolab.pulses.ReadoutPulse`): Readout result object
                containing the readout measurement shots. This is created in ``execute_circuit``.
        """
        for gate, sequence in measurement_map.items():
            _samples = (readout[pulse.serial].samples for pulse in sequence.pulses)
            samples = list(filter(lambda x: x is not None, _samples))
            gate.result.backend = self
            gate.result.register_samples(np.array(samples).T)

    def execute_circuit(self, circuit, initial_state=None, nshots=1000):
        """Executes a quantum circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to execute.
            initial_state (:class:`qibo.models.circuit.Circuit`): Circuit to prepare the initial state.
                If ``None`` the default ``|00...0>`` state is used.
            nshots (int): Number of shots to sample from the experiment.

        Returns:
            ``MeasurementOutcomes`` object containing the results acquired from the execution.
        """
        if isinstance(initial_state, Circuit):
            return self.execute_circuit(
                circuit=initial_state + circuit,
                nshots=nshots,
            )
        if initial_state is not None:
            raise_error(
                ValueError,
                "Hardware backend only supports circuits as initial states.",
            )

        # This should be done in qibo side
        # Temporary fix: overwrite the wire names
        if not all(q in self.qubits for q in circuit.wire_names):
            circuit._wire_names = self.qubits[: circuit.nqubits]

        sequence, measurement_map = self.compiler.compile(circuit, self.platform)

        if not self.platform.is_connected:
            self.platform.connect()

        readout = self.platform.execute_pulse_sequence(
            sequence,
            ExecutionParameters(nshots=nshots),
        )

        self.platform.disconnect()

        result = MeasurementOutcomes(circuit.measurements, self, nshots=nshots)
        self.assign_measurements(measurement_map, readout)
        return result

    def execute_circuits(self, circuits, initial_states=None, nshots=1000):
        """Executes multiple quantum circuits with a single communication with
        the control electronics.

        Circuits are unrolled to a single pulse sequence.

        Args:
            circuits (list): List of circuits to execute.
            initial_states (:class:`qibo.models.circuit.Circuit`): Circuit to prepare the initial state.
                If ``None`` the default ``|00...0>`` state is used.
            nshots (int): Number of shots to sample from the experiment.

        Returns:
            List of ``MeasurementOutcomes`` objects containing the results acquired from the execution of each circuit.
        """
        if isinstance(initial_states, Circuit):
            return self.execute_circuits(
                circuits=[initial_states + circuit for circuit in circuits],
                nshots=nshots,
            )
        if initial_states is not None:
            raise_error(
                ValueError,
                "Hardware backend only supports circuits as initial states.",
            )

        # This should be done in qibo side
        # Temporary fix: overwrite the wire names
        for circuit in circuits:
            if not all(q in self.qubits for q in circuit.wire_names):
                circuit._wire_names = self.qubits[: circuit.nqubits]

        # TODO: Maybe these loops can be parallelized
        sequences, measurement_maps = zip(
            *(self.compiler.compile(circuit, self.platform) for circuit in circuits)
        )

        if not self.platform.is_connected:
            self.platform.connect()

        readout = self.platform.execute_pulse_sequences(
            sequences,
            ExecutionParameters(nshots=nshots),
        )

        self.platform.disconnect()

        results = []
        readout = {k: deque(v) for k, v in readout.items()}
        for circuit, measurement_map in zip(circuits, measurement_maps):
            results.append(
                MeasurementOutcomes(circuit.measurements, self, nshots=nshots)
            )
            for gate, sequence in measurement_map.items():
                samples = [
                    readout[pulse.serial].popleft().samples for pulse in sequence.pulses
                ]
                gate.result.backend = self
                gate.result.register_samples(np.array(samples).T)
        return results


class MetaBackend:
    """Meta-backend class which takes care of loading the qibolab backend."""

    @staticmethod
    def load(platform: str):
        """Loads the backend.

        Args:
            platform (str): Name of the platform to load.
        Returns:
            qibo.backends.abstract.Backend: The loaded backend.
        """
        from qibolab.backends import QibolabBackend

        return QibolabBackend(platform=platform)

    def list_available(self) -> dict:
        """Lists all the available qibolab platforms."""
        return {platform: True for platform in available_platforms()}
