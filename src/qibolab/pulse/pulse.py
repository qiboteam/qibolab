"""Pulse abstractions."""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from qibolab.constants import PULSE, RUNCARD
from qibolab.pulse.pulse_shape.pulse_shape import PulseShape
from qibolab.typings import PulseName
from qibolab.utils import Factory, Waveforms


@dataclass
class Pulse:
    """Describes a single pulse to be added to waveform array.

    Args:
        start (float): Start time of pulse in ns.
        duration (float): Pulse duration in ns.
        amplitude (float): Pulse digital amplitude (unitless) [0 to 1].
        frequency (float): Pulse Intermediate Frequency in Hz [10e6 to 300e6].
        phase (float): To be added.
        shape: (str): {'Rectangular', 'Gaussian(rel_sigma)', 'DRAG(rel_sigma, beta)'} Pulse shape.
            See :py:mod:`qibolab.pulses_shapes` for list of available shapes.
        channel (int/str): Specifies the device that will execute this pulse.
        type (str): {'ro', 'qd', 'qf'} type of pulse {ReadOut, Qubit Drive, Qubit Flux}   
        offset_i (float): Optional pulse I offset (unitless).
            (amplitude + offset) should be between [0 and 1].
        offset_q (float): Optional pulse Q offset (unitless).
            (amplitude + offset) should be between [0 and 1].
        qubit (int): qubit associated with the pulse

    Example:
        .. code-block:: python

            from qibolab.pulses import Pulse
            from qibolab.pulse_shapes import Gaussian

            # define pulse with Gaussian shape
            pulse = Pulse(start=0,
                          duration=60,
                          amplitude=0.3,
                          frequency=200000000.0,
                          phase=0,
                          shape=Gaussian(5),
                          channel=1,
                          type='qd')
    """

    name: ClassVar[PulseName] = PulseName.PULSE
    amplitude: float
    phase: float
    duration: int
    pulse_shape: PulseShape
    frequency: float | None = None

    def __post_init__(self):
        """Cast qubit_ids to list."""
        if isinstance(self.pulse_shape, dict):
            self.pulse_shape = Factory.get(name=self.pulse_shape.pop(RUNCARD.NAME))(
                **self.pulse_shape  # pylint: disable=not-a-mapping
            )

    def modulated_waveforms(self, frequency: float, resolution: float = 1.0) -> Waveforms:
        """Applies digital quadrature amplitude modulation (QAM) to the pulse envelope.

        Args:
            resolution (float, optional): The resolution of the pulses in ns. Defaults to 1.0.

        Returns:
            NDArray: I and Q modulated waveforms.
        """
        envelope = self.envelope(resolution=resolution)
        envelopes = [np.real(envelope), np.imag(envelope)]
        time = np.arange(self.duration / resolution) * 1e-9 * resolution
        cosalpha = np.cos(2 * np.pi * frequency * time + self.phase)
        sinalpha = np.sin(2 * np.pi * frequency * time + self.phase)
        mod_matrix = np.array([[cosalpha, sinalpha], [-sinalpha, cosalpha]])
        imod, qmod = np.transpose(np.einsum("abt,bt->ta", mod_matrix, envelopes))
        return Waveforms(i=imod.tolist(), q=qmod.tolist())
    
    @property
    def start(self):
        """Pulse 'start' property.

        Raises:
            ValueError: Is start time is not defined.

        Returns:
            int: Start time of the pulse.
        """
        return self.start_time

    @property
    def serial(self):
        return str(self.to_dict())

    def envelope(self, amplitude: float | None = None, resolution: float = 1.0):
        """Pulse 'envelope' property.

        Returns:
            List[float]: Amplitudes of the envelope of the pulse. Max amplitude is fixed to 1.
        """
        if amplitude is None:
            amplitude = self.amplitude
        return self.pulse_shape.envelope(duration=self.duration, amplitude=amplitude, resolution=resolution)

    def to_dict(self):
        """Return dictionary of pulse.

        Returns:
            dict: Dictionary describing the pulse.
        """
        return {
            PULSE.NAME: self.name.value,
            PULSE.AMPLITUDE: self.amplitude,
            PULSE.FREQUENCY: self.frequency,
            PULSE.PHASE: self.phase,
            PULSE.DURATION: self.duration,
            PULSE.PULSE_SHAPE: self.pulse_shape.to_dict(),
        }

    def __repr__(self):
        return self.serial
