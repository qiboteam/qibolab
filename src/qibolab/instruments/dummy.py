from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from qibo.config import log

from qibolab.couplers import Coupler
from qibolab.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab.pulses import PulseSequence
from qibolab.qubits import Qubit, QubitId
from qibolab.sweeper import Sweeper
from qibolab.unrolling import Bounds

from .abstract import Controller
from .oscillator import LocalOscillator
from .port import Port

SAMPLING_RATE = 1


@dataclass
class DummyPort(Port):
    name: str
    offset: float = 0.0
    lo_frequency: int = 0
    lo_power: int = 0
    gain: int = 0
    attenuation: int = 0
    power_range: int = 0
    filters: Optional[dict] = None


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

    BOUNDS = Bounds(1, 1, 1)

    PortType = DummyPort

    @property
    def sampling_rate(self):
        return SAMPLING_RATE

    def connect(self):
        log.info(f"Connecting to {self.name} instrument.")

    def disconnect(self):
        log.info(f"Disconnecting {self.name} instrument.")

    def setup(self, *args, **kwargs):
        log.info(f"Setting up {self.name} instrument.")

    def get_values(self, options, ro_pulse, shape):
        if options.acquisition_type is AcquisitionType.DISCRIMINATION:
            if options.averaging_mode is AveragingMode.SINGLESHOT:
                values = np.random.randint(2, size=shape)
            elif options.averaging_mode is AveragingMode.CYCLIC:
                values = np.random.rand(*shape)
        elif options.acquisition_type is AcquisitionType.RAW:
            samples = int(ro_pulse.duration * SAMPLING_RATE)
            waveform_shape = tuple(samples * dim for dim in shape)
            values = (
                np.random.rand(*waveform_shape) * 100
                + 1j * np.random.rand(*waveform_shape) * 100
            )
        elif options.acquisition_type is AcquisitionType.INTEGRATION:
            values = np.random.rand(*shape) * 100 + 1j * np.random.rand(*shape) * 100
        return values

    def play(
        self,
        qubits: Dict[QubitId, Qubit],
        couplers: Dict[QubitId, Coupler],
        sequence: PulseSequence,
        options: ExecutionParameters,
        *sweepers: List[Sweeper],
    ):
        results = {}

        if options.averaging_mode is not AveragingMode.CYCLIC:
            shape = (options.nshots,) + tuple(
                len(sweeper.values) for sweeper in sweepers
            )
        else:
            shape = tuple(len(sweeper.values) for sweeper in sweepers)

        for ro_pulse in sequence.ro_pulses:
            values = self.get_values(options, ro_pulse, shape)
            results[ro_pulse.qubit] = results[ro_pulse.id] = options.results_type(
                values
            )

        return results
