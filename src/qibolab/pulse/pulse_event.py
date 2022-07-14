"""PulseEvent class."""
from dataclasses import dataclass, field

from qibolab.pulse import Pulse


@dataclass
class PulseEvent:
    """Describes a single pulse with a start time."""

    sort_index = field(init=False)
    pulse: Pulse
    start_time: int

    def __post_init__(self):
        self.sort_index = self.start_time
