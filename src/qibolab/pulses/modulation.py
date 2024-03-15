import numpy as np

from .envelope import IqWaveform

__all__ = ["modulate", "demodulate"]

SAMPLING_RATE = 1
"""Default sampling rate in gigasamples per second (GSps).

Used for generating waveform envelopes if the instruments do not provide
a different value.
"""


def modulate(
    envelope: IqWaveform,
    freq: float,
    rate: float = SAMPLING_RATE,
    phase: float = 0.0,
) -> IqWaveform:
    """Modulate the envelope waveform with a carrier.

    `envelope` is a `(2, n)`-shaped array of I and Q (first dimension) envelope signals,
    as a function of time (second dimension), and `freq` the frequency of the carrier to
    modulate with (usually the IF) in GHz.
    `rate` is an optional sampling rate, in Gs/s, to sample the carrier.

    .. note::

        Only the combination `freq / rate` is actually relevant, but it is frequently
        convenient to specify one in GHz and the other in Gs/s. Thus the two arguments
        are provided for the simplicity of their interpretation.

    `phase` is an optional initial phase for the carrier.
    """
    samples = np.arange(envelope.shape[1])
    phases = (2 * np.pi * freq / rate) * samples + phase
    cos = np.cos(phases)
    sin = np.sin(phases)
    mod = np.array([[cos, -sin], [sin, cos]])

    # the normalization is related to `mod`, but only applied at the end for the sake of
    # performances
    return np.einsum("ijt,jt->it", mod, envelope) / np.sqrt(2)


def demodulate(
    modulated: IqWaveform,
    freq: float,
    rate: float = SAMPLING_RATE,
) -> IqWaveform:
    """Demodulate the acquired pulse.

    The role of the arguments is the same of the corresponding ones in :func:`modulate`,
    which is essentially the inverse of this function.
    """
    # in case the offsets have not been removed in hardware
    modulated = modulated - np.mean(modulated)

    samples = np.arange(modulated.shape[1])
    phases = (2 * np.pi * freq / rate) * samples
    cos = np.cos(phases)
    sin = np.sin(phases)
    demod = np.array([[cos, sin], [-sin, cos]])

    # the normalization is related to `demod`, but only applied at the end for the sake
    # of performances
    return np.sqrt(2) * np.einsum("ijt,jt->it", demod, modulated)
