from dataclasses import dataclass, replace
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
    """Data acquisition from hardware"""

    DISCRIMINATION = auto()
    """Demodulate, integrate the waveform and discriminate among states based on the voltages"""
    INTEGRATION = auto()
    """Demodulate and integrate the waveform"""
    RAW = auto()
    """Acquire the waveform as it is"""
    SPECTROSCOPY = auto()
    """Zurich Integration mode for RO frequency sweeps"""


class AveragingMode(Enum):
    """Data averaging modes from hardware"""

    CYCLIC = auto()
    """Better averaging for short timescale noise"""
    SINGLESHOT = auto()
    """SINGLESHOT: No averaging"""
    SEQUENTIAL = auto()
    """SEQUENTIAL: Worse averaging for noise[Avoid]"""


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
    """Data structure to deal with execution parameters"""

    nshots: Optional[int] = None
    """Number of shots to sample from the experiment. Default is the runcard value."""
    relaxation_time: Optional[int] = None
    """Time to wait for the qubit to relax to its ground state between shots in ns. Default is the runcard value."""
    fast_reset: bool = False
    """Enable or disable fast reset"""
    acquisition_type: AcquisitionType = AcquisitionType.DISCRIMINATION
    """Data acquisition type"""
    averaging_mode: AveragingMode = AveragingMode.SINGLESHOT
    """Data averaging mode"""

    def __post_init__(self):
        if not isinstance(self.acquisition_type, AcquisitionType):
            raise TypeError(f"acquisition_type: {self.acquisition_type} is not valid as acquisition")
        if not isinstance(self.averaging_mode, AveragingMode):
            raise TypeError(f"averaging mode: {self.averaging_mode} is not valid as averaging")

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
            options (:class:`qibolab.platforms.platform.ExecutionParameters`): Object holding the execution options.
            **kwargs: May need them for something
        Returns:
            Readout results acquired by after execution.
        """
        if options.nshots is None:
            options = replace(options, nshots=self.nshots)

        if options.relaxation_time is None:
            options = replace(options, relaxation_time=self.relaxation_time)

        return self.design.play(self.qubits, sequence, options)

    def sweep(self, sequence, options, *sweepers, **kwargs):
        """Executes a pulse sequence for different values of sweeped parameters.
        Useful for performing chip characterization.

        Args:
            sequence (:class:`qibolab.pulses.PulseSequence`): Pulse sequence to execute.
            options (:class:`qibolab.platforms.platform.ExecutionParameters`): Object holding the execution options.
            *sweepers (:class:`qibolab.sweeper.Sweeper`): Sweeper objects that specify which
                parameters are being sweeped.
            **kwargs: May need them for something

        Returns:
            Readout results acquired by after execution.
        """

        if options.nshots is None:
            options = replace(options, nshots=self.nshots)

        if options.relaxation_time is None:
            options = replace(options, relaxation_time=self.relaxation_time)

        return self.design.sweep(
            self.qubits,
            sequence,
            options,
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
