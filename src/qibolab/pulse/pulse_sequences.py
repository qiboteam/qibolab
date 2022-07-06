"""PulseSequence class."""
from dataclasses import dataclass, field
from typing import List

from qibolab.constants import PULSESEQUENCES
from qibolab.pulse.pulse import Pulse
from qibolab.pulse.pulse_sequence import PulseSequence


@dataclass
class PulseSequences:
    """Class containing a list of PulseSequence objects. It is the pulsed representation of a Qibo circuit.

    Args:
        elements (List[PulseSequences]): List of pulse sequences.
    """

    elements: List[PulseSequence] = field(default_factory=list)

    def add(self, pulse: Pulse, port: int):
        """Add pulse sequence.

        Args:
            pulse_sequence (PulseSequence): Pulse object.
        """
        for pulse_sequence in self.elements:
            if port == pulse_sequence.port and pulse.name == pulse_sequence.name:
                pulse_sequence.add(pulse=pulse)
                return
        self.elements.append(PulseSequence(pulses=[pulse], port=port))

    def to_dict(self):
        """Return dictionary representation of the class.

        Returns:
            dict: Dictionary representation of the class.
        """
        return {PULSESEQUENCES.ELEMENTS: [pulse_sequence.to_dict() for pulse_sequence in self.elements]}

    @classmethod
    def from_dict(cls, dictionary: dict):
        """Build PulseSequence instance from dictionary.

        Args:
            dictionary (dict): Dictionary description of the class.

        Returns:
            PulseSequence: Class instance.
        """
        elements = [PulseSequence.from_dict(dictionary=settings) for settings in dictionary[PULSESEQUENCES.ELEMENTS]]

        return PulseSequences(elements=elements)

    def __iter__(self):
        """Redirect __iter__ magic method to elements."""
        return self.elements.__iter__()

    def __len__(self):
        """Redirect __len__ magic method to elements."""
        return len(self.elements)
