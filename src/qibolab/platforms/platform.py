from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
import numpy.typing as npt

from qibolab.platforms.abstract import AbstractPlatform, Qubit


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

    def check_coupler(self, qubit: Qubit) -> bool:
        """Checks if the qubit is a coupler qubit."""
        return not qubit.flux_coupler

    def execute_pulse_sequence(self, sequence, **kwargs):
        options = ExecutionParameters(kwargs)

        if options.relaxation_time is None:
            options.relaxation_time = self.relaxation_time
        if options.fast_reset:
            options.fast_reset = {
                qubit.name: self.create_RX_pulse(qubit=qubit.name, start=sequence.finish)
                for qubit in self.qubits.values()
                if self.check_coupler(qubit)
            }

        return self.design.play(
            self.qubits,
            sequence,
            nshots=options.nshots,
            relaxation_time=options.relaxation_time,
            fast_reset=options.fast_reset,
            sim_time=options.sim_time,
            acquisition_type=options.acquisition_type,
            averaging_type=options.averaging_type,
        )

    def sweep(self, sequence, *sweepers, **kwargs):
        options = ExecutionParameters(kwargs)

        if options.relaxation_time is None:
            options.relaxation_time = self.relaxation_time
        if options.fast_reset:
            options.fast_reset = {
                qubit.name: self.create_RX_pulse(qubit=qubit.name, start=sequence.finish)
                for qubit in self.qubits.values()
                if self.check_coupler(qubit)
            }
        return self.design.sweep(
            self.qubits,
            sequence,
            *sweepers,
            nshots=options.nshots,
            relaxation_time=options.relaxation_time,
            fast_reset=options.fast_reset,
            sim_time=options.sim_time,
            acquisition_type=options.acquisition_type,
            averaging_type=options.averaging_type,
        )


@dataclass
class ExecutionParameters:
    """Data structure to deal with execution parameters

    :nshots: Number of shots per point on the experiment
    :relaxation_time: Relaxation time for the qubit
    :fast_reset: Enable or disable fast reset
    :sim_time: Time for the simulation execution
    :acquisition_type: Data acquisition mode (INTEGRATION, RAW, DISCRIMINATION)
    :averaging_type: Data averaging mode (CYLIC, SINGLESHOT)
    """

    nshots: Optional[np.uint32] = None
    relaxation_time: Optional[np.uint32] = None
    fast_reset: Optional[bool] = False
    sim_time: Optional[np.float64] = 10e-6
    acquisition_type: Optional[str] = None
    averaging_type: Optional[str] = None

    def __init__(self, dict):
        for k, v in dict.items():
            setattr(self, k, v)
