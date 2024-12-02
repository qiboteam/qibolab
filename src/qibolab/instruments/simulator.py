from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from qibo import Circuit, gates
from qibo.config import log

from qibolab.couplers import Coupler
from qibolab.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab.native import SingleQubitNatives, TwoQubitNatives
from qibolab.pulses import Pulse, PulseSequence, PulseType
from qibolab.qubits import Qubit, QubitId, QubitPairId
from qibolab.result import SampleResults
from qibolab.sweeper import Sweeper
from qibolab.unrolling import Bounds

from .abstract import Controller
from .port import Port

SAMPLING_RATE = 1


@dataclass
class SimulatorPort(Port):
    name: str
    offset: float = 0.0
    lo_frequency: int = 0
    lo_power: int = 0
    gain: int = 0
    attenuation: int = 0
    power_range: int = 0
    filters: Optional[dict] = None


def is_equal(pulse1: Pulse, pulse2: Pulse) -> bool:
    """Check if two pulses are equal."""
    attrs = ("duration", "amplitude", "frequency", "relative_phase", "shape", "type")
    return all(getattr(pulse1, a) == getattr(pulse2, a) for a in attrs)


def pulse_to_gate(
    pulse: Pulse, natives: SingleQubitNatives, qubit_map: dict[QubitId, int]
):
    """Convert drive and readout pulses to Qibo gates."""
    qubit = qubit_map.get(pulse.qubit, pulse.qubit)

    if pulse.type is PulseType.READOUT:
        assert is_equal(pulse, natives.MZ.pulse(start=0))
        return gates.M(qubit)

    assert pulse.type is PulseType.DRIVE
    if is_equal(pulse, natives.RX.pulse(start=0)):
        return gates.RX(qubit, theta=np.pi)
    if is_equal(pulse, natives.RX90.pulse(start=0)):
        return gates.RX(qubit, theta=np.pi / 2)
    if is_equal(pulse, natives.RX.pulse(start=0, relative_phase=np.pi / 2)):
        return gates.RY(qubit, theta=np.pi)
    if is_equal(pulse, natives.RX90.pulse(start=0, relative_phase=np.pi / 2)):
        return gates.RY(qubit, theta=np.pi / 2)
    raise ValueError(f"Unsupported pulse: {pulse}")


def flux_pulse_to_gate(
    pulse: Pulse,
    two_qubit_natives: dict[QubitPairId, TwoQubitNatives],
    qubit_map: dict[QubitId, int],
):
    """Convert flux pulse to Qibo CZ gate."""
    assert pulse.type is PulseType.FLUX
    for pair, natives in two_qubit_natives.items():
        if natives.CZ is not None:
            native_flux_pulse = natives.CZ.sequence(start=0)[0].pulses[0]
            if (
                is_equal(pulse, native_flux_pulse)
                and pulse.qubit == native_flux_pulse.qubit
            ):
                qubit0 = qubit_map.get(pair[0], pair[0])
                qubit1 = qubit_map.get(pair[1], pair[1])
                return gates.CZ(qubit0, qubit1)
    raise ValueError(f"Unsupported pulse: {pulse}")


def sequence_to_circuit(sequence: PulseSequence, qubits, two_qubit_natives, qubit_map):
    """Convert sequence to Qibo circuit."""
    clock = defaultdict(int)
    all_gates = []
    for pulse in sorted(sequence.pulses, key=lambda p: p.start):
        if pulse.type is PulseType.FLUX:
            gate = flux_pulse_to_gate(pulse, two_qubit_natives, qubit_map)
        else:
            natives = qubits[pulse.qubit].native_gates
            gate = pulse_to_gate(pulse, natives, qubit_map)

        for qubit in gate.qubits:
            if pulse.start < clock[qubit]:
                raise ValueError("Overlapping pulses.")
            clock[qubit] = pulse.start + pulse.duration

        all_gates.append(gate)

    circuit = Circuit(max(clock.keys()) + 1)
    circuit.add(all_gates)
    return circuit


class SimulatorInstrument(Controller):

    BOUNDS = Bounds(1, 1, 1)

    PortType = SimulatorPort

    def __init__(self, name, address, pairs, qubit_map):
        super().__init__(name, address)
        self.qubit_map = qubit_map
        self.two_qubit_natives = {p: pair.native_gates for p, pair in pairs.items()}

    @property
    def sampling_rate(self):
        return SAMPLING_RATE

    def connect(self):
        log.info(f"Connecting to {self.name} instrument.")

    def disconnect(self):
        log.info(f"Disconnecting {self.name} instrument.")

    def setup(self, *args, **kwargs):
        log.info(f"Setting up {self.name} instrument.")

    def get_values(self, options, ro_pulse, shape):
        if options.acquisition_type is AcquisitionType.DISCRIMINATION:
            if options.averaging_mode is AveragingMode.SINGLESHOT:
                values = np.random.randint(2, size=shape)
            elif options.averaging_mode is AveragingMode.CYCLIC:
                values = np.random.rand(*shape)
        elif options.acquisition_type is AcquisitionType.RAW:
            samples = int(ro_pulse.duration * SAMPLING_RATE)
            waveform_shape = tuple(samples * dim for dim in shape)
            values = (
                np.random.rand(*waveform_shape) * 100
                + 1j * np.random.rand(*waveform_shape) * 100
            )
        elif options.acquisition_type is AcquisitionType.INTEGRATION:
            values = np.random.rand(*shape) * 100 + 1j * np.random.rand(*shape) * 100
        return values

    def play(
        self,
        qubits: Dict[QubitId, Qubit],
        couplers: Dict[QubitId, Coupler],
        sequence: PulseSequence,
        options: ExecutionParameters,
    ):
        if options.acquisition_type is not AcquisitionType.DISCRIMINATION:
            raise NotImplementedError("Only DISCRIMINATION acquisition is supported.")
        if options.averaging_mode is not AveragingMode.SINGLESHOT:
            raise NotImplementedError("Only SINGLESHOT averaging mode is supported.")

        circuit = sequence_to_circuit(
            sequence, qubits, self.two_qubit_natives, self.qubit_map
        )

        result = circuit(nshots=options.nshots)
        samples = result.samples()

        results = {}
        for ro_pulse in sequence.ro_pulses:
            q = self.qubit_map.get(ro_pulse.qubit, ro_pulse.qubit)
            results[ro_pulse.qubit] = results[ro_pulse.serial] = SampleResults(
                samples[:, q]
            )

        return results

    def sweep(
        self,
        qubits: Dict[QubitId, Qubit],
        couplers: Dict[QubitId, Coupler],
        sequence: PulseSequence,
        options: ExecutionParameters,
        *sweepers: List[Sweeper],
    ):
        raise NotImplementedError
