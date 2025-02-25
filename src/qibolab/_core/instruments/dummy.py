import numpy as np
from pydantic import Field
from qibo.config import log

from qibolab._core.components import Channel, Config
from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.identifier import ChannelId
from qibolab._core.pulses import Acquisition
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers
from qibolab._core.unrolling import Bounds

from .abstract import Controller
from .oscillator import LocalOscillator

SAMPLING_RATE = 1
BOUNDS = Bounds(waveforms=1, readout=1, instructions=1)


__all__ = ["DummyLocalOscillator", "DummyInstrument"]


class DummyDevice:
    """Dummy device that does nothing but follows the QCoDeS interface.

    Used by :class:`qibolab.instruments.dummy.DummyLocalOscillator`.
    """

    def set(self, name, value):
        """Set device property."""

    def get(self, name):
        """Get device property."""
        return 0

    def on(self):
        """Turn device on."""

    def off(self):
        """Turn device on."""

    def close(self):
        """Close connection with device."""


class DummyLocalOscillator(LocalOscillator):
    """Dummy local oscillator instrument.

    Useful for testing the interface defined in :class:`qibolab.instruments.oscillator.LocalOscillator`.
    """

    def create(self):
        return DummyDevice()


class DummyInstrument(Controller):
    """Dummy instrument that returns random voltage values.

    Useful for testing code without requiring access to hardware.

    Args:
        name (str): name of the instrument.
        address (int): address to connect to the instrument.
            Not used since the instrument is dummy, it only
            exists to keep the same interface with other
            instruments.
    """

    address: str
    bounds: str = "dummy/bounds"
    channels: dict[ChannelId, Channel] = Field(default_factory=dict)

    @property
    def sampling_rate(self) -> int:
        return SAMPLING_RATE

    def connect(self):
        log.info("Connecting to dummy instrument.")

    def disconnect(self):
        log.info("Disconnecting dummy instrument.")

    def values(self, options: ExecutionParameters, shape: tuple[int, ...]):
        if options.acquisition_type is AcquisitionType.DISCRIMINATION:
            if options.averaging_mode is AveragingMode.SINGLESHOT:
                return np.random.randint(2, size=shape)
            return np.random.rand(*shape)
        return np.random.rand(*shape) * 100

    def play(
        self,
        configs: dict[str, Config],
        sequences: list[PulseSequence],
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers],
    ):
        def values(acq: Acquisition):
            samples = int(acq.duration * self.sampling_rate)
            return np.array(
                self.values(options, options.results_shape(sweepers, samples))
            )

        return {
            acq.id: values(acq) for seq in sequences for (_, acq) in seq.acquisitions
        }
