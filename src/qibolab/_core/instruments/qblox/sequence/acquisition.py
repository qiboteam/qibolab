from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
from pydantic import UUID4

from qibolab._core import pulses
from qibolab._core.pulses.pulse import PulseLike
from qibolab._core.serialize import Model

from .waveforms import Waveform

__all__ = []

MeasureId = UUID4
Weight = Waveform
Weights = dict[str, Weight]

KERNEL_WEIGHTS = ("kernel_i", "kernel_q")
"""Names of the I and Q path weights, stored at indices 0 and 1."""


class Acquisition(Model):
    num_bins: int
    index: int


Acquisitions = dict[MeasureId, Acquisition]


class AcquisitionSpec(Model):
    acquisition: Acquisition
    duration: int
    weighted: bool = False


def kernel_weights(kernel: npt.NDArray) -> Weights:
    """Split a complex kernel into the weights of the I and Q paths.

    The matched filter is ``Re(k s) = Re(k) I - Im(k) Q``. Each path is normalized to
    unit L2 norm, such that the two integrated components carry the same noise, and the
    IQ-plane rotation used for discrimination realizes the full matched filter. The
    weights are then scaled to the hardware range ``[-1, 1]``.
    """
    w = np.stack([kernel.real, -kernel.imag]).astype(float)
    norms = np.linalg.norm(w, axis=1, keepdims=True)
    if not np.any(norms):
        raise ValueError("Integration kernel is identically zero.")
    w = np.divide(w, norms, out=np.zeros_like(w), where=norms > 0)
    w /= np.max(np.abs(w))
    return {
        name: Weight(data=data, index=i)
        for i, (name, data) in enumerate(zip(KERNEL_WEIGHTS, w))
    }


def acquisitions(
    sequence: Iterable[PulseLike], num_bins: int, kernel_length: int | None = None
) -> dict[MeasureId, AcquisitionSpec]:
    """Collect acquisitions, weighted with the kernel when its length is given."""
    acqs = [p for p in sequence if isinstance(p, (pulses.Acquisition, pulses.Readout))]
    if kernel_length is not None:
        for acq in acqs:
            window = acq.acquisition if isinstance(acq, pulses.Readout) else acq
            # the weight length sets the integration length
            if int(window.duration) != kernel_length:
                raise ValueError(
                    f"Integration kernel length ({kernel_length} samples) differs from "
                    f"the acquisition duration ({int(window.duration)} samples)."
                )
    return {
        acq.id: AcquisitionSpec(
            acquisition=Acquisition(num_bins=num_bins, index=i),
            duration=int(acq.duration),
            weighted=kernel_length is not None,
        )
        for i, acq in enumerate(acqs)
    }
