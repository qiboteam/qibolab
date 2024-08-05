from typing import Union

from proprietary_instruments import ControllerDriver

from qibolab.execution_parameters import ExecutionParameters
from qibolab.instruments.abstract import Controller
from qibolab.pulses import PulseSequence
from qibolab.qubits import Qubit
from qibolab.result import IntegratedResults, SampleResults
from qibolab.sweeper import Sweeper


class MyController(Controller):
    def __init__(self, name, address):
        self.device = ControllerDriver(address)
        super().__init__(name, address)

    def connect(self):
        """Empty method to comply with Instrument interface."""

    def start(self):
        """Empty method to comply with Instrument interface."""

    def stop(self):
        """Empty method to comply with Instrument interface."""

    def disconnect(self):
        """Empty method to comply with Instrument interface."""

    # FIXME:: *args, **kwargs are not passed on
    def setup(self):
        """Empty method to comply with Instrument interface."""

    # FIXME:: this seems to be incompatbile with the ABC, too
    def play(
        self,
        qubits: dict[int, Qubit],
        sequence: PulseSequence,
        execution_parameters: ExecutionParameters,
    ) -> dict[str, Union[IntegratedResults, SampleResults]]:
        """Executes a PulseSequence."""

        # usually, some modification on the qubit objects, sequences or
        # parameters is needed so that the qibolab interface comply with the one
        # of the device here these are equal
        results = self.device.run_experiment(qubits, sequence, execution_parameters)

        # also the results are, in qibolab, specific objects that need some kind
        # of conversion. Refer to the results section in the documentation.
        return results

    # FIXME:: this seems to be incompatbile with the ABC, too
    def sweep(
        self,
        qubits: dict[int, Qubit],
        sequence: PulseSequence,
        execution_parameters: ExecutionParameters,
        *sweepers: Sweeper,
    ) -> dict[str, Union[IntegratedResults, SampleResults]]:
        # usually, some modification on the qubit objects, sequences or
        # parameters is needed so that the qibolab interface comply with the one
        # of the device here these are equal
        results = self.device.run_scan(qubits, sequence, sweepers, execution_parameters)

        # also the results are, in qibolab, specific objects that need some kind
        # of conversion. Refer to the results section in the documentation.
        return results

    def play_sequences(
        self,
        qubits: dict[int, Qubit],
        sequences: list[PulseSequence],
        execution_parameters: ExecutionParameters,
    ) -> dict[str, Union[IntegratedResults, SampleResults]]:
        """This method is used for sequence unrolling sweeps.

        Here not implemented.
        """
        raise NotImplementedError
