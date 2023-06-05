import copy
import time
from typing import Dict, List, Union

import numpy as np
from qibo.config import log, raise_error

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.instruments.abstract import AbstractInstrument
from qibolab.platform import Qubit
from qibolab.pulses import PulseSequence, PulseType
from qibolab.result import IntegratedResults, SampleResults
from qibolab.sweeper import Parameter, Sweeper


class DummyInstrument(AbstractInstrument):
    """Dummy instrument that returns random voltage values.

    Useful for testing code without requiring access to hardware.

    Args:
        name (str): name of the instrument.
        address (int): address to connect to the instrument.
            Not used since the instrument is dummy, it only
            exists to keep the same interface with other
            instruments.
    """

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
