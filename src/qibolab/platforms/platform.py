from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

from qibolab.platforms.abstract import AbstractPlatform


class DesignPlatform(AbstractPlatform):
    """Platform that using an instrument design.

    This will maybe replace the ``AbstractPlatform`` object
    and work as a generic platform that works with an arbitrary
    ``InstrumentDesign``.
    """

    def __init__(self, name, design, runcard):
        super().__init__(name, runcard)
        self.design = design

    def connect(self):
        self.design.connect()
        self.is_connected = True

    def setup(self):
        self.design.setup(self.qubits, **self.settings["settings"])

    def start(self):
        self.design.start()

    def stop(self):
        self.design.stop()

    def disconnect(self):
        self.design.disconnect()
        self.is_connected = False

    def execute_pulse_sequence(self, sequence, **kwargs):
        options = ExecutionParameters(**kwargs)

        if options.relaxation_time is None:
            options.relaxation_time = self.relaxation_time
        if options.fast_reset:
            options.fast_reset = {
                qubit.name: self.create_RX_pulse(qubit=qubit.name, start=sequence.finish)
                for qubit in self.qubits.values()
                if not qubit.flux_coupler
            }

        return self.design.play(
            self.qubits,
            sequence,
            options=options,
        )

    def sweep(self, sequence, *sweepers, **kwargs):
        options = ExecutionParameters(**kwargs)

        if options.relaxation_time is None:
            options.relaxation_time = self.relaxation_time
        if options.fast_reset:
            options.fast_reset = {
                qubit.name: self.create_RX_pulse(qubit=qubit.name, start=sequence.finish)
                for qubit in self.qubits.values()
                if not qubit.flux_coupler
            }

        return self.design.sweep(
            self.qubits,
            sequence,
            *sweepers,
            options=options,
        )


class AcquisitionType(Enum):
    """
    Types of data acquisition from hardware.
    (SPECTROSCOPY[Weird Zurich mode], INTEGRATION, RAW, DISCRIMINATION)
    """

    RAW = auto()
    INTEGRATION = auto()
    DISCRIMINATION = auto()


class AveragingMode(Enum):
    """
    Types of data averaging from hardware.
    (CYLIC[True averaging], SINGLESHOT[False averaging], [SEQUENTIAL, bad averaging])
    """

    CYCLIC = auto()
    SINGLESHOT = auto()


@dataclass
class ExecutionParameters:
    """Data structure to deal with execution parameters

    :nshots: Number of shots per point on the experiment
    :relaxation_time: Relaxation time for the qubit
    :fast_reset: Enable or disable fast reset
    :sim_time: Time for the simulation execution
    :acquisition_type: Data acquisition mode
    :averaging_mode: Data averaging mode
    """

    nshots: Optional[np.uint32] = None
    relaxation_time: Optional[np.uint32] = None
    fast_reset: Optional[bool] = False
    sim_time: Optional[np.float64] = 10e-6
    acquisition_type: AcquisitionType = AcquisitionType.INTEGRATION
    averaging_mode: AveragingMode = AveragingMode.CYCLIC
