"""PulseSequence class."""
from dataclasses import dataclass, field
from typing import List

import numpy as np

from qibolab.constants import PULSESEQUENCE, RUNCARD
from qibolab.pulse.pulse import Pulse
from qibolab.pulse.readout_pulse import ReadoutPulse
from qibolab.typings import PulseName
from qibolab.utils import Waveforms


@dataclass
class PulseSequence:
    """Container of Pulse objects addressed to the same bus. All pulses should be addressed to the same port and
    have the same frequency."""

    pulses: List[Pulse]
    port: int
    frequency: float = field(init=False)
    _name: PulseName = field(init=False)

    def __post_init__(self):
        """Get port and frequency values from pulse."""
        self.frequency = self.pulses[0].frequency
        self._name = self.pulses[0].name

    def add(self, pulse: Pulse):
        """Add pulse to sequence.

        Args:
            pulse (Pulse): Pulse object.
        """
        if pulse.name != self.name:
            raise ValueError(
                "All Pulse objects inside a PulseSequence should have the same type (Pulse or ReadoutPulse)."
            )
        if pulse.frequency != self.frequency:
            raise ValueError("All Pulse objects inside a PulseSequence should have the same frequency.")
        self.pulses.append(pulse)
    
    def add_u3(self, theta: float, phi: float, lam:float, qubit:int = 0):
        """Add pulses tha implement a U3 gate.

        Args:
            theta (float): theta parameter of the U3 gate.
            phi (float): phi parameter of the U3 gate.
            lam (float): lambda parameter of the U3 gate.
            qubit (int, optional): qubit number. Defaults to 0.
        """
        #TODO: Implement method

    def __iter__(self):
        """Redirect __iter__ magic method."""
        return self.pulses.__iter__()

    def waveforms(self, frequency: float, resolution: float = 1.0) -> Waveforms:
        """PulseSequence 'waveforms' property.

        Args:
            resolution (float): The resolution of the pulses in ns.

        Returns:
            Waveforms: Class containing the I, Q waveforms for a specific qubit.
        """
        waveforms = Waveforms()
        time = 0
        for pulse in self.pulses:
            wait_time = round((pulse.start - time) / resolution)
            if wait_time > 0:
                waveforms.add(imod=np.zeros(shape=wait_time), qmod=np.zeros(shape=wait_time))
            time += pulse.start
            pulse_waveforms = pulse.modulated_waveforms(frequency=frequency, resolution=resolution)
            waveforms += pulse_waveforms
            time += pulse.duration

        return waveforms

    @property
    def name(self):
        """Name of the pulses of the pulse sequence.

        Returns:
            str: Name of the pulses. Options are "Pulse" or "ReadoutPulse"""
        return self._name

    def to_dict(self):
        """Return dictionary representation of the class.

        Returns:
            dict: Dictionary representation of the class.
        """
        return {
            PULSESEQUENCE.PULSES: [pulse.to_dict() for pulse in self.pulses],
            PULSESEQUENCE.PORT: self.port,
        }

    @classmethod
    def from_dict(cls, dictionary: dict):
        """Load PulseSequence object from dictionary.

        Args:
            dictionary (dict): Dictionary representation of the PulseSequence object.

        Returns:
            PulseSequence: Loaded class.
        """
        pulses = [
            Pulse(**settings) if Pulse.name == PulseName(settings.pop(RUNCARD.NAME)) else ReadoutPulse(**settings)
            for settings in dictionary[PULSESEQUENCE.PULSES]
        ]
        port = dictionary[PULSESEQUENCE.PORT]
        return PulseSequence(pulses=pulses, port=port)
