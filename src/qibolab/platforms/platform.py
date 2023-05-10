from dataclasses import InitVar, dataclass
from enum import Enum, auto
from typing import Optional

from qibo.config import raise_error

from qibolab.platforms.abstract import AbstractPlatform
from qibolab.result import (
    AveragedIntegratedResults,
    AveragedRawWaveformResults,
    AveragedStateResults,
    IntegratedResults,
    RawWaveformResults,
    StateResults,
)


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


RESULTS_TYPE = {
    AveragingMode.CYCLIC: {
        AcquisitionType.INTEGRATION: AveragedIntegratedResults,
        AcquisitionType.RAW: AveragedRawWaveformResults,
        AcquisitionType.DISCRIMINATION: AveragedStateResults,
    },
    AveragingMode.SINGLESHOT: {
        AcquisitionType.INTEGRATION: IntegratedResults,
        AcquisitionType.RAW: RawWaveformResults,
        AcquisitionType.DISCRIMINATION: StateResults,
    },
}


@dataclass(frozen=True)
class ExecutionParameters:
    """Data structure to deal with execution parameters

    :nshots: nshots (int): Number of shots to sample from the experiment. Default is 1024.
    relaxation_time (int): Time to wait for the qubit to relax to its ground state between shots in s.
                If ``None`` the default value provided as ``relaxation_time`` in the runcard will be used.
    :fast_reset (bool): Enable or disable fast reset
    :acquisition_type (AcquisitionType): Data acquisition mode
    :averaging_mode (AveragingMode): Data averaging mode
    """

    nshots: Optional[int] = 1024
    relaxation_time: Optional[float] = None
    fast_reset: bool = False
    acquisition_type: AcquisitionType = AcquisitionType.DISCRIMINATION
    averaging_mode: AveragingMode = AveragingMode.SINGLESHOT

    def __post_init__(self):
        if not isinstance(self.acquisition_type, AcquisitionType):
            raise TypeError("acquisition_type is not valid")
        if not isinstance(self.averaging_mode, AveragingMode):
            raise TypeError("averaging mode is not valid")

    @property
    def results_type(self):
        """Returns corresponding results class"""
        return RESULTS_TYPE[self.averaging_mode][self.acquisition_type]


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

    def execute_pulse_sequence(self, sequence, options, **kwargs):
        """Executes a pulse sequence.

        Args:
            sequence (:class:`qibolab.pulses.PulseSequence`): Pulse sequence to execute.
            options (:class:`qibolab.platforms.platform.ExecutionParameters`) Class holding the execution options.
            **kwargs: May need them for something
        Returns:
            Readout results acquired by after execution.
        """
        if options.relaxation_time is None:
            options.relaxation_time = self.relaxation_time

        return self.design.play(self.qubits, options, sequence)

    def sweep(self, sequence, options, *sweepers, **kwargs):
        """Executes a pulse sequence for different values of sweeped parameters.
        Useful for performing chip characterization.

        Args:
            sequence (:class:`qibolab.pulses.PulseSequence`): Pulse sequence to execute.
            options (:class:`qibolab.platforms.platform.ExecutionParameters`) Class holding the execution options.
            *sweepers (:class:`qibolab.sweeper.Sweeper`): Sweeper objects that specify which
                parameters are being sweeped.
            **kwargs: May need them for something

        Returns:
            Readout results acquired by after execution.
        """
        if options.relaxation_time is None:
            options.relaxation_time = self.relaxation_time

        return self.design.sweep(
            self.qubits,
            options,
            sequence,
            *sweepers,
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
        if self.qubits[qubit].flux is None:
            raise_error(NotImplementedError, f"{self.name} does not have flux.")
        self.qubits[qubit].flux.bias = bias

    def get_bias(self, qubit):
        return self.qubits[qubit].flux.bias
