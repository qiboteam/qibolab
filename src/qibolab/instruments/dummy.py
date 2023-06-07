import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Union

import numpy as np
from qibo.config import log, raise_error

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.instruments.abstract import Controller
from qibolab.instruments.port import Port
from qibolab.platform import Qubit
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Sweeper


@dataclass
class DummyPort(Port):
    name: str
    _offset: float = 0.0
    _lo_frequency: int = 0
    _lo_power: int = 0
    _gain: int = 0
    _attenuation: int = 0
    _power_range: int = 0
    _filter: dict = field(default_factory=dict)

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value):
        self._offset = value

    @property
    def lo_frequency(self):
        return self._lo_frequency

    @lo_frequency.setter
    def lo_frequency(self, value):
        self._lo_frequency = value

    @property
    def lo_power(self):
        return self._lo_power

    @lo_power.setter
    def lo_power(self, value):
        self._lo_power = value

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, value):
        self._gain = value

    @property
    def attenuation(self):
        return self._attenuation

    @attenuation.setter
    def attenuation(self, value):
        self._attenuation = value

    @property
    def power_range(self):
        return self._power_range

    @power_range.setter
    def power_range(self, value):
        self._power_range = value

    @property
    def filter(self):
        return self._filter

    @filter.setter
    def filter(self, value):
        self._filter = value


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

    def __init__(self, name, address):
        super().__init__(name, address)
        self.ports = {}

    def __getitem__(self, value):
        if value not in self.ports:
            self.ports[value] = DummyPort(value)
        return self.ports[value]

    def connect(self):
        log.info("Connecting to dummy instrument.")

    def setup(self, *args, **kwargs):
        log.info("Setting up dummy instrument.")

    def start(self):
        log.info("Starting dummy instrument.")

    def stop(self):
        log.info("Stopping dummy instrument.")

    def disconnect(self):
        log.info("Disconnecting dummy instrument.")

    def play(self, qubits: Dict[Union[str, int], Qubit], sequence: PulseSequence, options: ExecutionParameters):
        ro_pulses = {pulse.qubit: pulse.serial for pulse in sequence.ro_pulses}

        expts = 1 if options.averaging_mode is AveragingMode.CYCLIC else options.nshots
        results = {}
        for ro_pulse in sequence.ro_pulses:
            if options.acquisition_type is AcquisitionType.DISCRIMINATION:
                values = np.random.rand(expts)
            else:
                values = np.random.rand(expts) * 100 + 1j * np.random.rand(expts) * 100
            results[ro_pulse.qubit] = results[ro_pulse.serial] = options.results_type(values)

        return results

    def sweep(
        self,
        qubits: Dict[Union[str, int], Qubit],
        sequence: PulseSequence,
        options: ExecutionParameters,
        *sweepers: List[Sweeper],
    ):
        expts = 1
        for sweeper in sweepers:
            expts *= len(sweeper.values)
        if options.averaging_mode is not AveragingMode.CYCLIC:
            expts *= options.nshots

        results = {}

        for ro_pulse in sequence.ro_pulses:
            if options.acquisition_type is AcquisitionType.DISCRIMINATION:
                values = np.random.rand(expts)
            else:
                values = np.random.rand(expts) * 100 + 1j * np.random.rand(expts) * 100
            results[ro_pulse.qubit] = results[ro_pulse.serial] = options.results_type(values)

        return results
