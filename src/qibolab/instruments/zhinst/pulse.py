"""Wrapper for qibolab and laboneq pulses and sweeps."""

from typing import Optional

import laboneq.simple as lo
import numpy as np
from laboneq.dsl.experiment.pulse_library import (
    sampled_pulse_complex,
    sampled_pulse_real,
)

from qibolab.pulses import Drag, Gaussian, GaussianSquare, Pulse, PulseType, Rectangular
from qibolab.sweeper import Parameter

from .util import NANO_TO_SECONDS, SAMPLING_RATE


def select_pulse(pulse: Pulse):
    """Return laboneq pulse object corresponding to the given qibolab pulse."""
    if isinstance(pulse.shape, Rectangular):
        can_compress = pulse.type is not PulseType.READOUT
        return lo.pulse_library.const(
            length=round(pulse.duration * NANO_TO_SECONDS, 9),
            amplitude=pulse.amplitude,
            can_compress=can_compress,
        )
    if isinstance(pulse.shape, Gaussian):
        sigma = pulse.shape.rel_sigma
        return lo.pulse_library.gaussian(
            length=round(pulse.duration * NANO_TO_SECONDS, 9),
            amplitude=pulse.amplitude,
            sigma=2 / sigma,
            zero_boundaries=False,
        )

    if isinstance(pulse.shape, GaussianSquare):
        sigma = pulse.shape.rel_sigma
        width = pulse.shape.width
        can_compress = pulse.type is not PulseType.READOUT
        return lo.pulse_library.gaussian_square(
            length=round(pulse.duration * NANO_TO_SECONDS, 9),
            width=round(pulse.duration * NANO_TO_SECONDS, 9) * width,
            amplitude=pulse.amplitude,
            can_compress=can_compress,
            sigma=2 / sigma,
            zero_boundaries=False,
        )

    if isinstance(pulse.shape, Drag):
        sigma = pulse.shape.rel_sigma
        beta = pulse.shape.beta
        return lo.pulse_library.drag(
            length=round(pulse.duration * NANO_TO_SECONDS, 9),
            amplitude=pulse.amplitude,
            sigma=2 / sigma,
            beta=beta,
            zero_boundaries=False,
        )

    if np.all(pulse.envelope_waveform_q(SAMPLING_RATE) == 0):
        return sampled_pulse_real(
            samples=pulse.envelope_waveform_i(SAMPLING_RATE),
            can_compress=True,
        )
    else:
        return sampled_pulse_complex(
            samples=pulse.envelope_waveform_i(SAMPLING_RATE)
            + (1j * pulse.envelope_waveform_q(SAMPLING_RATE)),
            can_compress=True,
        )


class ZhPulse:
    """Wrapper data type that holds a qibolab pulse, the corresponding laboneq
    pulse object, and any sweeps associated with this pulse."""

    def __init__(self, pulse):
        self.pulse: Pulse = pulse
        """Qibolab pulse."""
        self.zhpulse = select_pulse(pulse)
        """Laboneq pulse."""
        self.zhsweepers: list[tuple[Parameter, lo.SweepParameter]] = []
        """Parameters to be swept, along with their laboneq sweep parameter
        definitions."""
        self.delay_sweeper: Optional[lo.SweepParameter] = None
        """Laboneq sweep parameter if the delay of the pulse should be
        swept."""

    # pylint: disable=R0903,E1101
    def add_sweeper(self, param: Parameter, sweeper: lo.SweepParameter):
        """Add sweeper to list of sweepers associated with this pulse."""
        if param in {
            Parameter.amplitude,
            Parameter.frequency,
            Parameter.duration,
            Parameter.relative_phase,
        }:
            self.zhsweepers.append((param, sweeper))
        elif param is Parameter.start:
            # TODO: Change this case to ``Delay.duration``
            if self.delay_sweeper:
                raise ValueError(
                    "Cannot have multiple delay sweepers for a single pulse"
                )
            self.delay_sweeper = sweeper
        else:
            raise ValueError(f"Sweeping {param} is not supported")
