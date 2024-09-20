"""Wrapper for qibolab and laboneq pulses and sweeps."""

import laboneq.simple as laboneq
import numpy as np
from laboneq.dsl.experiment.pulse_library import (
    sampled_pulse_complex,
    sampled_pulse_real,
)

from qibolab._core.pulses import Drag, Gaussian, GaussianSquare, Pulse, Rectangular

from .constants import NANO_TO_SECONDS, SAMPLING_RATE


def select_pulse(pulse: Pulse):
    """Return laboneq pulse object corresponding to the given qibolab pulse."""
    if isinstance(pulse.envelope, Rectangular):
        # FIXME:
        # can_compress = pulse.type is not PulseType.READOUT
        can_compress = False
        return laboneq.pulse_library.const(
            length=round(pulse.duration * NANO_TO_SECONDS, 9),
            amplitude=pulse.amplitude,
            can_compress=can_compress,
        )
    if isinstance(pulse.envelope, Gaussian):
        sigma = pulse.envelope.rel_sigma
        return laboneq.pulse_library.gaussian(
            length=round(pulse.duration * NANO_TO_SECONDS, 9),
            amplitude=pulse.amplitude,
            sigma=2 / sigma,
            zero_boundaries=False,
        )

    if isinstance(pulse.envelope, GaussianSquare):
        sigma = pulse.envelope.rel_sigma
        width = pulse.envelope.width
        # FIXME:
        # can_compress = pulse.type is not PulseType.READOUT
        can_compress = False
        return laboneq.pulse_library.gaussian_square(
            length=round(pulse.duration * NANO_TO_SECONDS, 9),
            width=round(pulse.duration * NANO_TO_SECONDS, 9) * width,
            amplitude=pulse.amplitude,
            can_compress=can_compress,
            sigma=2 / sigma,
            zero_boundaries=False,
        )

    if isinstance(pulse.envelope, Drag):
        sigma = pulse.envelope.rel_sigma
        beta = pulse.envelope.beta
        return laboneq.pulse_library.drag(
            length=round(pulse.duration * NANO_TO_SECONDS, 9),
            amplitude=pulse.amplitude,
            sigma=2 / sigma,
            beta=beta,
            zero_boundaries=False,
        )

    if np.all(pulse.q(SAMPLING_RATE) == 0):
        return sampled_pulse_real(
            samples=pulse.i(SAMPLING_RATE),
            can_compress=True,
        )
    else:
        return sampled_pulse_complex(
            samples=pulse.i(SAMPLING_RATE) + (1j * pulse.q(SAMPLING_RATE)),
            can_compress=True,
        )
