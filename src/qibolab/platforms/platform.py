from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from qibo.config import raise_error

from qibolab.platforms.abstract import AbstractPlatform


class DesignPlatform(AbstractPlatform):
    """Platform that using an instrument design.

    This will maybe replace the ``AbstractPlatform`` object
    and work as a generic platform that works with an arbitrary
    ``InstrumentDesign``.
    """

    def __init__(self, name, design, runcard):
        super().__init__(name, runcard)
        self.design = design

    def connect(self):
        self.design.connect()
        self.is_connected = True

    def setup(self):
        self.design.setup()

    def start(self):
        self.design.start()

    def stop(self):
        self.design.stop()

    def disconnect(self):
        self.design.disconnect()
        self.is_connected = False

    def execute_pulse_sequence(self, sequence, **kwargs):
        options = ExecutionParameters(**kwargs)

        if options.relaxation_time is None:
            options.relaxation_time = self.relaxation_time

        return self.design.play(
            self.qubits,
            sequence,
            options=options,
        )

    def sweep(self, sequence, *sweepers, **kwargs):
        options = ExecutionParameters(**kwargs)

        if options.relaxation_time is None:
            options.relaxation_time = self.relaxation_time

        return self.design.sweep(
            self.qubits,
            sequence,
            *sweepers,
            options=options,
        )

    def set_lo_drive_frequency(self, qubit, freq):
        self.qubits[qubit].drive.local_oscillator.frequency = freq

    def get_lo_drive_frequency(self, qubit):
        return self.qubits[qubit].drive.local_oscillator.frequency

    def set_lo_readout_frequency(self, qubit, freq):
        self.qubits[qubit].readout.local_oscillator.frequency = freq

    def get_lo_readout_frequency(self, qubit):
        return self.qubits[qubit].readout.local_oscillator.frequency

    def set_lo_twpa_frequency(self, qubit, freq):
        self.qubits[qubit].twpa.local_oscillator.frequency = freq

    def get_lo_twpa_frequency(self, qubit):
        return self.qubits[qubit].twpa.local_oscillator.frequency

    def set_lo_twpa_power(self, qubit, power):
        self.qubits[qubit].twpa.local_oscillator.power = power

    def get_lo_twpa_power(self, qubit):
        return self.qubits[qubit].twpa.local_oscillator.power

    def set_attenuation(self, qubit, att):
        raise_error(NotImplementedError, f"{self.name} does not support attenuation.")

    def get_attenuation(self, qubit):
        raise_error(NotImplementedError, f"{self.name} does not support attenuation.")

    def set_gain(self, qubit, gain):
        raise_error(NotImplementedError, f"{self.name} does not support gain.")

    def get_gain(self, qubit):
        raise_error(NotImplementedError, f"{self.name} does not support gain.")

    def set_bias(self, qubit, bias):
        self.qubits[qubit].flux.bias = bias

    def get_bias(self, qubit):
        return self.qubits[qubit].flux.bias


class AcquisitionType(Enum):
    """
    Types of data acquisition from hardware.

    SPECTROSCOPY: Zurich Integration mode for RO frequency sweeps,
    INTEGRATION: Demodulate and integrate the waveform,
    RAW: Acquire the waveform as it is,
    DISCRIMINATION: Demodulate, integrate the waveform and discriminate among states based on the voltages

    """

    RAW = auto()
    INTEGRATION = auto()
    DISCRIMINATION = auto()


class AveragingMode(Enum):
    """
    Types of data averaging from hardware.

    CYLIC: Better averaging for noise,
    SINGLESHOT: False averaging,
    [SEQUENTIAL: Worse averaging for noise]

    """

    CYCLIC = auto()
    SINGLESHOT = auto()


@dataclass
class ExecutionParameters:
    """Data structure to deal with execution parameters

    :nshots: Number of shots per point on the experiment
    :relaxation_time: Relaxation time for the qubit [s]
    :fast_reset: Enable or disable fast reset
    :acquisition_type: Data acquisition mode
    :averaging_mode: Data averaging mode
    """

    nshots: Optional[int] = 1024
    relaxation_time: Optional[float] = None
    fast_reset: bool = False
    acquisition_type: AcquisitionType = AcquisitionType.DISCRIMINATION
    averaging_mode: AveragingMode = AveragingMode.SINGLESHOT
