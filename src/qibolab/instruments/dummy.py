from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
from qibo.config import log

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.instruments.abstract import Controller
from qibolab.instruments.port import Port
from qibolab.platform import Qubit
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Sweeper


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

    PortType = DummyPort
    sampling_rate = 1

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

    def get_values(self, options, sequence, exp_points):
        results = {}
        for ro_pulse in sequence.ro_pulses:
            if options.acquisition_type is AcquisitionType.DISCRIMINATION:
                if options.averaging_mode is AveragingMode.SINGLESHOT:
                    values = np.random.randint(2, size=exp_points)
                elif options.averaging_mode is AveragingMode.CYCLIC:
                    values = np.random.rand(exp_points)
            elif options.acquisition_type is AcquisitionType.RAW:
                samples = int(ro_pulse.duration * self.sampling_rate)
                values = np.random.rand(samples * exp_points) * 100 + 1j * np.random.rand(samples * exp_points) * 100
            elif options.acquisition_type is AcquisitionType.INTEGRATION:
                values = np.random.rand(exp_points) * 100 + 1j * np.random.rand(exp_points) * 100
            results[ro_pulse.qubit] = results[ro_pulse.serial] = options.results_type(values)
        return results

    def play(self, qubits: Dict[Union[str, int], Qubit], sequence: PulseSequence, options: ExecutionParameters):
        exp_points = 1 if options.averaging_mode is AveragingMode.CYCLIC else options.nshots

        return self.get_values(options, sequence, exp_points)

    def sweep(
        self,
        qubits: Dict[Union[str, int], Qubit],
        sequence: PulseSequence,
        options: ExecutionParameters,
        *sweepers: List[Sweeper],
    ):
        exp_points = 1
        for sweeper in sweepers:
            exp_points *= len(sweeper.values)
        if options.averaging_mode is not AveragingMode.CYCLIC:
            exp_points *= options.nshots

        return self.get_values(options, sequence, exp_points)
