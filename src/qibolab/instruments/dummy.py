import numpy as np
from qibo.config import log

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.pulses.pulse import Acquisition
from qibolab.sequence import PulseSequence
from qibolab.sweeper import ParallelSweepers
from qibolab.unrolling import Bounds

from ..components import Config
from .abstract import Controller
from .oscillator import LocalOscillator

SAMPLING_RATE = 1
BOUNDS = Bounds(waveforms=1, readout=1, instructions=1)


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

    name: str
    address: str
    bounds: str = "dummy/bounds"

    @property
    def sampling_rate(self) -> int:
        return SAMPLING_RATE

    def connect(self):
        log.info(f"Connecting to {self.name} instrument.")

    def disconnect(self):
        log.info(f"Disconnecting {self.name} instrument.")

    def setup(self, *args, **kwargs):
        log.info(f"Setting up {self.name} instrument.")

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
