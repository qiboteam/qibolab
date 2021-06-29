import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future
from qibo.config import raise_error
from qiboicarusq import pulses, experiment


class TaskScheduler:
    """Scheduler class for organizing FPGA calibration and pulse sequence execution."""

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pi_trig = None # NIY
        #self._qubit_config = experiment.static.initial_config #NIU

    def fetch_config(self):
        """Fetches the qubit configuration data

        Returns:
            List of dicts representing qubit metadata or false if data is not ready yet
        """
        if self._qubit_config is None:
            raise_error(RuntimeError, "Cannot fetch qubit configuration "
                                      "because calibration is not complete.")
        return self._qubit_config

    def poll_config(self):
        """Blocking command to wait until qubit calibration is complete."""
        raise_error(NotImplementedError)

    def config_ready(self):
        """Checks if qubit calibration is complete.

        Returns:
            Boolean flag representing status of qubit calibration complete
        """
        return self._qubit_config is not None

    def execute_pulse_sequence(self, pulse_sequence, nshots):
        """Submits a pulse sequence to the queue for execution.

        Args:
            pulse_sequence: Pulse sequence object.
            shots: Number of trials.

        Returns:
            concurrent.futures.Future object representing task status
        """
        from qiboicarusq.circuit import PulseSequence
        if not isinstance(pulse_sequence, PulseSequence):
            raise_error(TypeError, "Pulse sequence {} has invalid type."
                                   "".format(pulse_sequence))
        if not isinstance(nshots, int) or nshots < 1:
            raise_error(ValueError, "Invalid number of shots {}.".format(nshots))
        future = self._executor.submit(self._execute_pulse_sequence,
                                       pulse_sequence=pulse_sequence,
                                       nshots=nshots)
        return future

    @staticmethod
    def _execute_pulse_sequence(pulse_sequence, nshots):
        wfm = pulse_sequence.compile()
        experiment.upload(wfm)
        experiment.start(nshots)
        experiment.stop()
        res = experiment.download()
        return res

    def execute_batch_sequence(self, pulse_batch, nshots):
        if not isinstance(nshots, int) or nshots < 1:
            raise_error(ValueError, "Invalid number of shots {}.".format(nshots))
        future = self._executor.submit(self._execute_batch_sequence,
                                       pulse_batch=pulse_batch,
                                       nshots=nshots)
        return future

    @staticmethod
    def _execute_batch_sequence(pulse_batch, nshots):
        wfm = pulse_batch[0].compile()
        steps = len(pulse_batch)
        sample_size = len(wfm[0])
        wfm_batch = np.zeros((steps, experiment.static.nchannels, sample_size))
        for i in range(steps):
            wfm = pulse_batch[i].compile()
            for j in range(experiment.static.nchannels):
                wfm_batch[i, j] = wfm[j]

        experiment.upload_batch(wfm_batch, nshots)
        experiment.start_batch(steps, nshots)
        experiment.stop()
        res = experiment.download()
        return res

    @staticmethod
    def check_tomography_required(target_qubits):
        return experiment.check_tomography_required(target_qubits)

    def execute_circuit(self, pulse_sequence, nshots, target_qubits):
        job = self.execute_pulse_sequence(pulse_sequence, nshots)
        raw_data = job.result()
        iq_signals = experiment.parse_raw(raw_data, target_qubits)
        qubit_states = experiment.parse_iq(iq_signals, target_qubits)
        
        if experiment.static.measurement_level == 2:
            return (qubit_states,)
        elif experiment.static.measurement_level == 1:
            return (qubit_states, iq_signals)
        else:
            return (qubit_states, iq_signals, raw_data)

    def execute_tomography(self, pulse_batch, nshots, target_qubits):
        job = self.execute_batch_sequence(pulse_batch, nshots)
        raw_data = job.result()
        iq_signals = [experiment.parse_raw(raw_signals, target_qubits) for raw_signals in raw_data]
        density_matrix = experiment.parse_tomography(raw_data, target_qubits)
        
        if experiment.static.measurement_level == 2:
            return (density_matrix,)
        elif experiment.static.measurement_level == 1:
            return (density_matrix, iq_signals)
        else:
            return (density_matrix, iq_signals, raw_data)

