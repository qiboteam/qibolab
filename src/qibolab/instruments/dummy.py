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
        nshots = options.nshots
        time.sleep(options.relaxation_time * 1e-9)

        ro_pulses = {pulse.qubit: pulse.serial for pulse in sequence.ro_pulses}

        results = {}
        for qubit, serial in ro_pulses.items():
            if options.acquisition_type is AcquisitionType.DISCRIMINATION:
                samples = (
                    np.random.rand(1) if options.averaging_mode is AveragingMode.CYCLIC else np.random.rand(nshots)
                )
                results[qubit] = results[serial] = options.results_type(samples)

            else:
                i = np.random.rand(1) if options.averaging_mode is AveragingMode.CYCLIC else np.random.rand(nshots)
                q = np.random.rand(1) if options.averaging_mode is AveragingMode.CYCLIC else np.random.rand(nshots)
                exp_res = i + 1j * q
                results[qubit] = results[serial] = options.results_type(data=exp_res)

            # results[qubit] = results[serial] = ExecutionResults.from_components(i, q, shots)

        return results

    def sweep(
        self,
        qubits: Dict[Union[str, int], Qubit],
        sequence: PulseSequence,
        options: ExecutionParameters,
        *sweepers: List[Sweeper],
    ):
        results = {}
        sweeper_pulses = {}

        # create copy of the sequence
        copy_sequence = copy.deepcopy(sequence)
        # perform sweeping recursively
        results = self._sweep_recursion(
            qubits,
            copy_sequence,
            sequence,
            options,
            *sweepers,
        )
        sequence = copy_sequence
        return results

    def play_sequence_in_sweep_recursion(
        self,
        qubits: List[Qubit],
        sequence: PulseSequence,
        or_sequence: PulseSequence,
        options: ExecutionParameters,
    ) -> Dict[str, Union[IntegratedResults, SampleResults]]:
        """Last recursion layer, if no sweeps are present

        After playing the sequence, the resulting dictionary keys need
        to be converted to the correct values.
        Even indexes correspond to qubit number and are not changed.
        Odd indexes correspond to readout pulses serials and are convert
        to match the original sequence (of the sweep) and not the one just executed.
        """
        res = self.play(qubits, sequence, options)
        newres = {}
        serials = [pulse.serial for pulse in or_sequence.ro_pulses]
        for idx, key in enumerate(res):
            if idx % 2 == 1:
                newres[serials[idx // 2]] = res[key]
            else:
                newres[key] = res[key]
        return newres

    def get_sequences_for_sweep(self, sequence, sweeper):
        sequences = []
        for kdx, val in enumerate(sweeper.values):
            new_sequence = copy.deepcopy(sequence)
            for pulse in sweeper.pulses:
                idx = [p.serial for p in new_sequence].index(pulse.serial)
                if sweeper.parameter in {Parameter.delay, Parameter.duration}:
                    if sweeper.parameter is Parameter.duration:
                        base_val = getattr(pulse, "duration")
                        new_val = sweeper.get_values(base_val)[kdx]
                        setattr(new_sequence[idx], "start", new_val)
                        delta = new_val - base_val
                    else:
                        delta = val
                    for seq_pulse in new_sequence[idx:]:
                        base_val = getattr(seq_pulse, "start")
                        setattr(seq_pulse, "start", base_val + delta)
                else:
                    base_val = getattr(pulse, sweeper.parameter.name)
                    setattr(new_sequence[idx], sweeper.parameter.name, sweeper.get_values(base_val)[kdx])
                    delta = getattr(pulse, sweeper.parameter.name)
            sequences.append(new_sequence)
        return sequences

    def get_qubits_for_sweep(self, qubits, sweeper):
        sweep_qubits = []
        for kdx, val in enumerate(sweeper.values):
            new_qubits = copy.deepcopy(qubits)
            for idx, qubit in enumerate(sweeper.qubits):
                base_val = getattr(qubit, sweeper.parameter.name)
                setattr(new_qubits[idx], sweeper.parameter.name, sweeper.get_values(base_val)[kdx])
            sweep_qubits.append(new_qubits)
        return sweep_qubits

    def _sweep_recursion(
        self,
        qubits,
        sequence,
        original_sequence,
        options,
        *sweepers,
    ):
        if len(sweepers) == 0:
            return self.play_sequence_in_sweep_recursion(qubits, sequence, original_sequence, options)

        sweeper = sweepers[0]

        results = {}
        if sweeper.pulses is not None:
            sequences = self.get_sequences_for_sweep(original_sequence, sweeper)
            for sequence in sequences:
                res = self._sweep_recursion(qubits, sequence, original_sequence, options, *sweepers[1:])
                for key in res:
                    if key in results:
                        results[key] = results[key] + res[key]
                    else:
                        results[key] = res[key]
            return results
        else:
            qubits_sweep = self.get_qubits_for_sweep(qubits, sweeper)
            for new_qubits in qubits_sweep:
                res = self._sweep_recursion(new_qubits, sequence, original_sequence, options, *sweepers[1:])
                for key in res:
                    if key in results:
                        results[key] = results[key] + res[key]
                    else:
                        results[key] = res[key]
            return results
