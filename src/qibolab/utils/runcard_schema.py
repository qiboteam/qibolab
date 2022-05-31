"""RuncardSchema class. Represents the structure of the multiqubit runcard."""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from qibolab.utils.nested_dataclass import nested_dataclass

@dataclass
class PulseSequence:
    """Pulse sequence dictionary."""
    start: int
    duration: int
    amplitude: float
    frequency: float
    shape: str
    phase: float
    type: str
    channel: int | None = None

@nested_dataclass
class RuncardSchema:
    """Representation of the multiqubit runcard structure."""

    @nested_dataclass
    class SharedSettings:
        """Shared settings dictionary."""
        hardware_avg: int
        sampling_rate: int
        repetition_duration: int
        minimum_delay_between_instructions: int
        qc_spectroscopy_pulse: PulseSequence
        readout_pulse: PulseSequence

    @nested_dataclass
    class Instruments:
        """Instruments dictionary."""
        @nested_dataclass
        class QRM:
            """QRM dictionary."""
            @dataclass
            class QRMSetup:
                """QRM setup dictionary."""
                ref_clock: str
                sync_en: bool
                scope_acq_avg_mode_en: bool
                scope_acq_trigger_mode: str
                gain: float
                acquisition_start: int
                acquisition_duration: int
                mode: str
                channel_port_map: Dict[int, str]
                lo: str

            lib: str
            classname: str
            ip: str
            setup: QRMSetup

        @nested_dataclass
        class QCM:
            """QCM dictionary."""
            @dataclass
            class QCMSetup:
                """QCM setup dictionary."""
                ref_clock: str
                sync_en: bool
                gain: float
                channel_port_map: Dict[int, str]
                lo: str
            lib: str
            classname: str
            ip: str
            setup: QCMSetup

        @nested_dataclass
        class LO:
            """LO dictionary."""
            @dataclass
            class LOSetup:
                """LO setup dictionary."""
                frequency: float
                power: float

            lib: str
            classname: str
            ip: str
            setup: LOSetup

        qrm: QRM
        lo_qrm: LO
        qcm1: QCM
        lo_qcm1: LO

    @nested_dataclass
    class NativeGates:
        """Native gates dictionary."""

        @nested_dataclass
        class Gate:
            name: str
            pulse_sequence: PulseSequence | List[PulseSequence]

            def __post_init__(self):
                if isinstance(self.pulse_sequence, list):
                    self.pulse_sequence = [PulseSequence(**sequence) for sequence in self.pulse_sequence]

        single_qubit: Dict[int, Gate]
        two_qubit: Dict[str, Gate]

        def __post_init__(self):
            """Cast the gate dictionaries into the Gate class."""
            for _, gates in self.single_qubit.items():
                gates = [self.Gate(**gate) for gate in gates]
            for _, gates in self.two_qubit.items():
                gates = [self.Gate(**gate) for gate in gates]

    @nested_dataclass
    class Characterisation:
        """Characterisation dictionary."""
        @dataclass
        class QubitCharacteristics:
            """Qubit characteristics dictionary."""
            resonator_freq: float
            qubit_freq: float
            T1: float
            T2: float
            mean_gnd_states: Tuple
            mean_exc_states: Tuple

        single_qubit: Dict[int, QubitCharacteristics]

        def __post_init__(self):
            """Cast the qubit characteristic dictionaries into the QubitCharacteristics class."""
            for _, characteristic in self.single_qubit.items():
                characteristic = self.QubitCharacteristics(**characteristic)

    nqubits: int
    description: str
    shared_settings: SharedSettings
    topology: np.ndarray
    channels: np.ndarray
    qubit_channel_map: Dict[int, List[int]]  # dictionary with integer keys and list values
    instruments: Instruments
    native_gates: NativeGates
    characterization: Characterisation
