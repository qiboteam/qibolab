from typing import Literal

from qibolab._core.components.configs import Config


class QbloxIqMixerConfig(Config):
    kind: Literal["qblox-iq-mixer"] = "qblox-iq-mixer"

    offset_i: float = 0.0
    """DC offset for the I component."""
    offset_q: float = 0.0
    """DC offset for the Q component."""

    """qblox has 6 sequencers per module.
    scale_q_*: The relative amplitude scale/factor of the q channel, to account for I-Q
    amplitude imbalance.
    phase_q_*: The phase offset of the q channel, to account for I-Q phase
    imbalance.
    """
    scale_q_sequencer0: float = 1.0
    phase_q_sequencer0: float = 0.0
    scale_q_sequencer1: float = 1.0
    phase_q_sequencer1: float = 0.0
    scale_q_sequencer2: float = 1.0
    phase_q_sequencer2: float = 0.0
    scale_q_sequencer3: float = 1.0
    phase_q_sequencer3: float = 0.0
    scale_q_sequencer4: float = 1.0
    phase_q_sequencer4: float = 0.0
    scale_q_sequencer5: float = 1.0
    phase_q_sequencer5: float = 0.0
