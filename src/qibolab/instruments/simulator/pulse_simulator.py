"""Pulse simulator module for running quantum dynamics simulation of model of
device."""

import operator
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
from qibo.config import log

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.instruments.abstract import Controller
from qibolab.instruments.port import Port
from qibolab.instruments.simulator.backends.generic import make_comp_basis
from qibolab.instruments.simulator.backends.qutip_backend import Qutip_Simulator
from qibolab.platform import Coupler, Qubit
from qibolab.pulses import PulseSequence, ReadoutPulse
from qibolab.qubits import QubitId
from qibolab.result import IntegratedResults, SampleResults
from qibolab.sweeper import Parameter, Sweeper, SweeperType


@dataclass
class DummyPort(Port):
    """Placeholder.

    Copied over from dummy, may not be required.
    """

    name: str
    offset: float = 0.0
    lo_frequency: int = 0
    lo_power: int = 0
    gain: int = 0
    attenuation: int = 0
    power_range: int = 0
    filters: Optional[dict] = None


def get_default_simulation_config(sim_sampling_boost=20, default_nshots=100):
    """Returns the default simulation configuration for the pulse simulator.

    Args:
        sim_sampling_boost (int): The boost factor to be multiplied to the device sampling rate for the sampling rate to be used in simulation.
        default_nshots (int): The default number of shots for the simulation.

    Returns:
        dict: The default simulation configuration.
    """
    default_CR_drive_simulation_config = {
        "simulation_backend_name": "Qutip",
        "default_nshots": default_nshots,
        "sim_sampling_boost": sim_sampling_boost,
        "simulate_dissipation": True,
        "instant_measurement": True,
    }
    return default_CR_drive_simulation_config


class PulseSimulator(Controller):
    """Runs quantum dynamics simulation of model of device.

    Interfaces with Qibolab. Useful for testing code without requiring
    access to hardware.
    """

    PortType = DummyPort  # Placeholder. Copied over from dummy, may not be required.

    def __init__(
        self,
        simulation_config: dict,
        model_config: dict,
        sim_opts: Optional[None] = None,
    ):
        """Set emulator configuration.

        Args:
            simulation_config (dict): Simulation configuration dictionary.
            model_config (dict): Model configuration dictionary.
            sim_opts (optional): Simulation backend specific object specifying simulation options.
        """
        self.simulation_config = simulation_config
        self.model_config = model_config
        self.sim_opts = sim_opts

        self.all_simulation_backends = {"Qutip": Qutip_Simulator}

        self.available_sweep_parameters = {
            Parameter.amplitude,
            Parameter.duration,
            Parameter.frequency,
            Parameter.relative_phase,
            Parameter.start,
        }

        self.update()

    @property
    def sampling_rate(self):
        return self.model_config["sampling_rate"]

    def update(self):
        """Updates the pulse simulator by loading all parameters from
        `self.model_config` and `self.simulation_config`."""
        self.simulation_backend_name = self.simulation_config["simulation_backend_name"]
        self.device_name = self.model_config["device_name"]
        self.model_name = self.model_config["model_name"]
        self.emulator_name = f"{self.device_name} emulator running {self.model_name} on {self.simulation_backend_name} backend"
        self.simulation_backend = self.all_simulation_backends[
            self.simulation_backend_name
        ](self.model_config, self.sim_opts)

        self.platform2simulator_channels = self.model_config[
            "platform2simulator_channels"
        ]

        self.sim_sampling_boost = self.simulation_config["sim_sampling_boost"]
        self.instant_measurement = self.simulation_config["instant_measurement"]
        self.runcard_duration_in_dt_units = self.model_config[
            "runcard_duration_in_dt_units"
        ]
        self.readout_error = self.model_config["readout_error"]
        self.exec_params = ExecutionParameters
        self.default_nshots = self.simulation_config["default_nshots"]
        self.exec_params.nshots = self.default_nshots
        self.exec_params.acquisition_type = AcquisitionType.DISCRIMINATION
        self.simulate_dissipation = self.simulation_config["simulate_dissipation"]

        self.pulse_sequence_history = []
        self.channel_waveforms_history = []

    def update_sim_opts(self, updated_sim_opts):
        self.sim_opts = updated_sim_opts
        self.simulation_backend.update_sim_opts(updated_sim_opts)

    def connect(self):
        log.info(f"Connecting to {self.emulator_name}.")

    def disconnect(self):
        log.info(f"Disconnecting {self.emulator_name}.")

    def setup(self, *args, **kwargs):
        log.info(f"Setting up {self.emulator_name}.")

    def start(self):
        log.info(f"Starting {self.emulator_name}.")

    def stop(self):
        log.info(f"Stopping {self.emulator_name}.")

    def get_samples(
        self,
        nshots: int,
        ro_reduced_dm: np.ndarray,
        ro_qubit_list: list,
        readout_error: Optional[dict] = None,
    ) -> dict[Union[str, int], list]:
        """Gets samples from a the density matrix corresponding to the system
        or subsystem specified by the ordered qubit indices.

        Args:
            nshots (int): Number of shots corresponding to the number of samples in the output.
            ro_reduced_dm (np.ndarray): Input density matrix.
            rdm_qubit_list (list): Qubit indices corresponding to the Hilbert space structure of the reduced density matrix (ro_reduced_dm).
            readout_error (dict, optional): Dictionary that specifies the prepare 0 measure 1 and prepare 1 measure 0 probability for each qubit.

        Returns:
            dict: The sampled qubit values for each qubit labelled by its index.
        """
        # load readout error from model_config if not specified
        if readout_error is None:
            readout_error = self.readout_error
        # check if readout error is consistent with ro qubits
        if readout_error is not None:
            for ro_qubit in ro_qubit_list:
                if ro_qubit not in readout_error.keys():
                    raise ValueError(
                        f"ro_qubit {ro_qubit} not present in ro_error_dict"
                    )

        # use the real diagonal part of the reduced density matrix as the probability distribution
        ro_probability_distribution = np.diag(ro_reduced_dm).real

        # preprocess distribution
        ro_probability_distribution = [
            (elem if np.isclose(1e-10, elem) == False else 0)
            for elem in ro_probability_distribution
        ]  # to remove small negative values
        ro_probability_sum = np.sum(ro_probability_distribution)
        print(f"ro_probability_sum: {ro_probability_sum}")
        ro_probability_distribution = ro_probability_distribution / ro_probability_sum
        ro_qubits_dim = len(ro_probability_distribution)

        # create array of computational basis states of the reduced (measured) Hilbert space
        reduced_computation_basis = make_comp_basis(
            ro_qubit_list, self.simulation_backend.qid_nlevels_map
        )

        # sample computation basis index nshots times from distribution
        sample_all_ro_list = np.random.choice(
            range(ro_qubits_dim), nshots, True, ro_probability_distribution
        )

        samples = {}
        for ind, ro_qubit in enumerate(ro_qubit_list):
            # extracts sampled values for each readout qubit from sample_all_ro_list
            outcomes = [
                reduced_computation_basis[outcome][ind]
                for outcome in sample_all_ro_list
            ]
            samples[ro_qubit] = outcomes

        if readout_error is not None:
            samples = apply_readout_noise(samples, readout_error)

        return samples

    def get_results_from_samples(
        self,
        ro_pulse_list: list,
        samples: dict[Union[str, int], list],
        execution_parameters: ExecutionParameters,
        append_to_shape: list = [],
    ) -> dict[str, Union[IntegratedResults, SampleResults]]:
        """Converts samples into Qibolab results format.

        Args:
            ro_pulse_list (list): List of readout pulse sequences.
            samples (dict): Samples generated by self.get_samples.
            append_to_shape (list): Specifies additional dimensions for the shape of the results. Defaults to empty list.
            execution_parameters (`qibolab.ExecutionParameters`): Parameters (nshots,
                                                        relaxation_time,
                                                        fast_reset,
                                                        acquisition_type,
                                                        averaging_mode)

        Returns:
            dict: Qibolab results for AcquisitionType.DISCRIMINATION.

        Raises:
            TypeError: If execution_parameters.acquisition_type is AcquisitionType.RAW or AcquisitionType.INTEGRATION.
        """
        shape = [execution_parameters.nshots] + append_to_shape
        results = {}
        for ro_pulse in ro_pulse_list:
            if execution_parameters.acquisition_type is AcquisitionType.DISCRIMINATION:
                values = np.array(samples[ro_pulse.qubit]).reshape(shape)
                processed_values = SampleResults(values)

                if execution_parameters.averaging_mode is AveragingMode.SINGLESHOT:
                    pass
                elif execution_parameters.averaging_mode is AveragingMode.CYCLIC:
                    processed_values = (
                        processed_values.average
                    )  # generaetes AveragedSampleResults

            elif execution_parameters.acquisition_type is AcquisitionType.RAW:
                raise TypeError("Emulator does not support raw measurement")
            elif execution_parameters.acquisition_type is AcquisitionType.INTEGRATION:
                raise TypeError("Emulator does not support integrated measurement")

            results[ro_pulse.qubit] = results[ro_pulse.serial] = processed_values
        return results

    def run_pulse_simulation(
        self,
        sequence: PulseSequence,
        instant_measurement: bool = True,
    ) -> tuple[np.ndarray, list]:
        """Simulates the input pulse sequence.

        Args:
            sequence (`qibolab.pulses.PulseSequence`): Pulse sequece to simulate.
            instant_measurement (bool): Collapses readout pulses to duration of 1 if True. Defaults to True.

        Returns:
            tuple: Reduced density matrix of final state specified by readout qubits as well as the list of readout qubits in, both in little endian order.
        """
        # reduces measurement time to 1 dt to save simulation time
        if instant_measurement:
            for i in range(len(sequence)):
                if type(sequence[i]) is ReadoutPulse:
                    sequence[i].duration = 1

        # extract waveforms from pulse sequence
        channel_waveforms = ps_to_waveform_dict(
            sequence,
            self.platform2simulator_channels,
            self.sampling_rate,
            self.sim_sampling_boost,
            self.runcard_duration_in_dt_units,
        )

        self.pulse_sequence_history.append(sequence.copy())
        self.channel_waveforms_history.append(channel_waveforms)
        # execute pulse simulation in emulator
        ro_reduced_dm, rdm_qubit_list = self.simulation_backend.qevolve(
            channel_waveforms, self.simulate_dissipation
        )

        return ro_reduced_dm, rdm_qubit_list

    def play(
        self,
        qubits: Dict[QubitId, Qubit],
        couplers: Dict[QubitId, Coupler],
        sequence: PulseSequence,
        execution_parameters: Optional[ExecutionParameters] = None,
    ) -> dict[str, Union[IntegratedResults, SampleResults]]:
        """Executes the sequence of instructions and generates readout results.

        Args:
            qubits (dict): Qubits involved in the device. Does not affect emulator.
            couplers (dict): Couplers involved in the device. Does not affect emulator.
            sequence (`qibolab.pulses.PulseSequence`): Pulse sequence to simulate.
            execution_parameters (`qibolab.ExecutionParameters`): Parameters (nshots,
                                                        relaxation_time,
                                                        fast_reset,
                                                        acquisition_type,
                                                        averaging_mode)
                                                        Defaults to None, for which case it uses the values stored in self.exec_params.

        Returns:
            dict: A dictionary mapping the readout pulses serial and respective qubits to
            Qibolab results object.
        """
        if execution_parameters == None:
            execution_parameters = self.exec_params

        nshots = execution_parameters.nshots
        ro_pulse_list = sequence.ro_pulses
        ro_reduced_dm, rdm_qubit_list = self.run_pulse_simulation(
            sequence, self.instant_measurement
        )
        samples = self.get_samples(nshots, ro_reduced_dm, rdm_qubit_list)
        results = self.get_results_from_samples(
            ro_pulse_list, samples, execution_parameters
        )

        return results

    def split_batches(self, sequences):
        """Placeholder.

        Copied over from dummy, may not be required.
        """
        return [sequences]

    def print_sim_details(self, sim_index=-1):
        """Print simulation details of any of the previously run simulations.

        Defaults to -1, i.e. the last simulation.
        """
        full_time_list = self.simulation_backend.pulse_sim_time_list[sim_index]
        print("Hamiltonian:", self.simulation_backend.H[sim_index])
        print("Initial state:", self.simulation_backend.psi0)
        # print("Full time list:", full_time_list)
        print("Initial simualtion time:", full_time_list[0])
        print("Final simualtion time:", full_time_list[-1])
        print("Simualtion time step (dt):", full_time_list[1])
        print("Total number of time steps:", len(full_time_list))
        print("Static dissipators:", self.simulation_backend.static_dissipators)
        print("Simulation options:", self.simulation_backend.sim_opts)

    ### sweeper adapted from icarusqfpga ###
    def sweep(
        self,
        qubits: Dict[QubitId, Qubit],
        couplers: Dict[QubitId, Coupler],
        sequence: PulseSequence,
        execution_parameters: ExecutionParameters,
        *sweeper: List[Sweeper],
    ) -> dict[str, Union[IntegratedResults, SampleResults]]:
        """Executes the sweep and generates readout results.

        Args:
            qubits (dict): Qubits involved in the device. Does not affect emulator.
            couplers (dict): Couplers involved in the device. Does not affect emulator.
            sequence (`qibolab.pulses.PulseSequence`): Pulse sequence to simulate.
            execution_parameters (`qibolab.ExecutionParameters`): Parameters (nshots,
                                                        relaxation_time,
                                                        fast_reset,
                                                        acquisition_type,
                                                        averaging_mode)
            *sweepers (`qibolab.Sweeper`): Sweeper objects.

        Returns:
            dict: A dictionary mapping the readout pulses serial and respective qubits to
            results objects.
        """
        append_to_shape = [len(sweep.values) for sweep in sweeper]

        # Record pulse values before sweeper modification
        bsv = []
        for sweep in sweeper:
            param_name = sweep.parameter.name.lower()
            if sweep.parameter not in self.available_sweep_parameters:
                raise NotImplementedError(
                    "Sweep parameter requested not available", param_name
                )
            base_sweeper_values = [getattr(pulse, param_name) for pulse in sweep.pulses]
            bsv.append(base_sweeper_values)

        sweep_samples = self._sweep_recursion(
            qubits, couplers, sequence, execution_parameters, *sweeper
        )

        # reshape and reformat samples to results format
        results = self.get_results_from_samples(
            sequence.ro_pulses, sweep_samples, execution_parameters, append_to_shape
        )

        # Reset pulse values back to original values
        for sweep, base_sweeper_values in zip(sweeper, bsv):
            param_name = sweep.parameter.name.lower()
            for pulse, value in zip(sweep.pulses, base_sweeper_values):
                setattr(pulse, param_name, value)

        return results

    def _sweep_recursion(
        self,
        qubits: Dict[QubitId, Qubit],
        couplers: Dict[QubitId, Coupler],
        sequence: PulseSequence,
        execution_parameters: ExecutionParameters,
        *sweeper: Sweeper,
    ) -> dict[Union[str, int], list]:
        """Performs sweep by recursion. Appends sampled lists obtained from
        each call of `self._sweep_play`.

        Args:
            qubits (dict): Qubits involved in the device. Does not affect emulator.
            couplers (dict): Couplers involved in the device. Does not affect emulator.
            sequence (`qibolab.pulses.PulseSequence`): Pulse sequence to simulate.
            execution_parameters (`qibolab.ExecutionParameters`): Parameters (nshots,
                                                        relaxation_time,
                                                        fast_reset,
                                                        acquisition_type,
                                                        averaging_mode)
            *sweepers (`qibolab.Sweeper`): Sweeper objects.

        Returns:
            dict: A dictionary mapping the qubit indices to list of sampled values.
        """

        if len(sweeper) == 0:
            samples = self._sweep_play(qubits, couplers, sequence, execution_parameters)
            return samples

        sweep = sweeper[0]
        param = sweep.parameter
        param_name = param.name.lower()

        if param not in self.available_sweep_parameters:
            raise NotImplementedError(
                "Sweep parameter requested not available", param_name
            )

        base_sweeper_values = [getattr(pulse, param_name) for pulse in sweep.pulses]
        sweeper_op = _sweeper_operation.get(sweep.type)
        ret = {}

        print("sweep param:", param_name)
        print("values", sweep.values)

        for value in sweep.values:
            for idx, pulse in enumerate(sweep.pulses):
                base = base_sweeper_values[idx]
                setattr(pulse, param_name, sweeper_op(value, base))

            self.merge_sweep_results(
                ret,
                self._sweep_recursion(
                    qubits, couplers, sequence, execution_parameters, *sweeper[1:]
                ),
            )

        return ret

    def _sweep_play(
        self,
        qubits: Dict[QubitId, Qubit],
        couplers: Dict[QubitId, Coupler],
        sequence: PulseSequence,
        execution_parameters: Optional[ExecutionParameters] = None,
    ) -> dict[Union[str, int], list]:
        """Generates samples list labelled by qubit index.

        Args:
            qubits (dict): Qubits involved in the device. Does not affect emulator.
            couplers (dict): Couplers involved in the device. Does not affect emulator.
            sequence (`qibolab.pulses.PulseSequence`): Pulse sequence to simulate.
            execution_parameters (`qibolab.ExecutionParameters`): Parameters (nshots,
                                                        relaxation_time,
                                                        fast_reset,
                                                        acquisition_type,
                                                        averaging_mode)
                                                        Defaults to None, for which case it uses the values stored in self.exec_params.

        Returns:
            dict: A dictionary mapping the qubit indices to list of sampled values.
        """
        if execution_parameters == None:
            execution_parameters = self.exec_params

        nshots = execution_parameters.nshots
        ro_pulse_list = sequence.ro_pulses

        # run pulse simulation
        ro_reduced_dm, rdm_qubit_list = self.run_pulse_simulation(
            sequence, self.instant_measurement
        )
        # generate samples
        samples = self.get_samples(nshots, ro_reduced_dm, rdm_qubit_list)

        return samples

    @staticmethod
    def merge_sweep_results(
        dict_a: """dict[str, Union[IntegratedResults, SampleResults]]""",
        dict_b: """dict[str, Union[IntegratedResults, SampleResults]]""",
    ) -> """dict[str, Union[IntegratedResults, SampleResults]]""":
        """Merges two dictionary mapping pulse serial to Qibolab results
        object.

        If dict_b has a key (serial) that dict_a does not have, simply add it,
        otherwise sum the two results

        Args:
            dict_a (dict): dict mapping ro pulses serial to qibolab res objects
            dict_b (dict): dict mapping ro pulses serial to qibolab res objects
        Returns:
            A dict mapping the readout pulses serial to Qibolab results objects
        """
        for serial in dict_b:
            if serial in dict_a:
                dict_a[serial] = dict_a[serial] + dict_b[serial]
            else:
                dict_a[serial] = dict_b[serial]
        return dict_a


_sweeper_operation = {
    SweeperType.ABSOLUTE: lambda value, base: value,
    SweeperType.OFFSET: operator.add,
    SweeperType.FACTOR: operator.mul,
}


def ps_to_waveform_dict(
    sequence: PulseSequence,
    platform2simulator_channels: dict,
    sampling_rate=1,
    sim_sampling_boost=1,
    runcard_duration_in_dt_units=False,
) -> dict[str, Union[np.ndarray, dict[str, np.ndarray]]]:
    """Converts pulse sequence to dictionary of time and channel separated
    waveforms.

    Args:
        sequence (`qibolab.pulses.PulseSequence`): Pulse sequence to simulate.
        platform2simulator_channels (dict): A dictionary that maps platform channel names to simulator channel names.
        sampling_rate (float): Sampling rate in units of samples/ns. Defaults to 1.
        sim_sampling_boost (int): Additional factor multiplied to sampling_rate for improving numerical accuracy in simulation. Defaults to 1.
        runcard_duration_in_dt_units (bool): If True, assumes that all time-related quantities in the runcard are expressed in units of inverse sampling rate and implements the necessary routines to account for that. If False, assumes that runcard time units are in ns. Defaults to False.

    Returns:
        dict: A dictionary containing the full list of simulation time steps, as well as the corresponding discretized channel waveforms labelled by their respective simulation channel names.
    """
    times_list = []
    signals_list = []
    emulator_channel_name_list = []

    def channel_translator(platform_channel_name, frequency):
        """Option to add frequency specific channel operators."""
        try:
            # frequency dependent channel operation
            return platform2simulator_channels[platform_channel_name + frequency]
        except:
            # frequency independent channel operation (default)
            return platform2simulator_channels[platform_channel_name]

    if runcard_duration_in_dt_units:
        """Assumes pulse duration in runcard is in units of dt=1/sampling_rate,
        i.e. time interval between samples.

        Pulse duration in this case is simply the total number of time
        samples; pulse.start and pulse.duration are therefore integers,
        and sampling_rate is only used to construct the value of dt in
        ns.
        """
        for qubit in sequence.qubits:
            qubit_pulses = sequence.get_qubit_pulses(qubit)
            for channel in qubit_pulses.channels:
                channel_pulses = qubit_pulses.get_channel_pulses(channel)
                for i, pulse in enumerate(channel_pulses):

                    start = pulse.start
                    actual_pulse_frequency = (
                        pulse.frequency
                    )  # store actual pulse frequency for channel_translator
                    # rescale frequency to be compatible with sampling_rate = 1
                    pulse.frequency = pulse.frequency / sampling_rate
                    # need to first set pulse._if in GHz to use modulated_waveform_i method
                    pulse._if = pulse.frequency / 1e9

                    pulse_signal = pulse.modulated_waveform_i(
                        sim_sampling_boost
                    ).data  # *np.sqrt(2)
                    end = start + len(pulse_signal) / sim_sampling_boost
                    t = (
                        np.arange(start * sim_sampling_boost, end * sim_sampling_boost)
                        / sampling_rate
                        / sim_sampling_boost
                    )

                    times_list.append(t)
                    signals_list.append(pulse_signal)

                    if pulse.type.value == "qd":
                        platform_channel_name = f"drive-{qubit}"
                    elif pulse.type.value == "qf":
                        platform_channel_name = f"flux-{qubit}"
                    elif pulse.type.value == "ro":
                        platform_channel_name = f"readout-{qubit}"

                    # restore pulse frequency values
                    pulse.frequency = actual_pulse_frequency
                    pulse._if = pulse.frequency / 1e9

                    emulator_channel_name_list.append(
                        channel_translator(platform_channel_name, pulse._if)
                    )

        tmin, tmax = [], []
        for times in times_list:
            tmin.append(np.amin(times))
            tmax.append(np.amax(times))

        tmin = np.amin(tmin)
        tmax = np.amax(tmax)
        Nt = int(np.round((tmax - tmin) * sampling_rate * sim_sampling_boost) + 1)
        full_time_list = np.linspace(tmin, tmax, Nt)

    else:
        """Assumes pulse duration in runcard is in ns."""
        for qubit in sequence.qubits:
            qubit_pulses = sequence.get_qubit_pulses(qubit)
            for channel in qubit_pulses.channels:
                channel_pulses = qubit_pulses.get_channel_pulses(channel)
                for i, pulse in enumerate(channel_pulses):
                    sim_sampling_rate = sampling_rate * sim_sampling_boost
                    start = int(pulse.start * sim_sampling_rate)
                    # need to first set pulse._if in GHz to use modulated_waveform_i method
                    pulse._if = pulse.frequency / 1e9

                    pulse_signal = pulse.modulated_waveform_i(
                        sim_sampling_rate
                    ).data  # *np.sqrt(2)
                    end = start + len(pulse_signal)
                    t = np.arange(start, end) / sim_sampling_rate

                    times_list.append(t)
                    signals_list.append(pulse_signal)

                    if pulse.type.value == "qd":
                        platform_channel_name = f"drive-{qubit}"
                    elif pulse.type.value == "qf":
                        platform_channel_name = f"flux-{qubit}"
                    elif pulse.type.value == "ro":
                        platform_channel_name = f"readout-{qubit}"

                    emulator_channel_name_list.append(
                        channel_translator(platform_channel_name, pulse._if)
                    )

        tmin, tmax = [], []
        for times in times_list:
            tmin.append(np.amin(times))
            tmax.append(np.amax(times))

        tmin = np.amin(tmin)
        tmax = np.amax(tmax)
        Nt = int(np.round((tmax - tmin) * sampling_rate * sim_sampling_boost) + 1)
        full_time_list = np.linspace(tmin, tmax, Nt)

    channel_waveforms = {"time": full_time_list, "channels": {}}

    unique_channel_names = np.unique(emulator_channel_name_list)
    for channel_name in unique_channel_names:
        waveform = np.zeros(len(full_time_list))

        for i, pulse_signal in enumerate(signals_list):
            if emulator_channel_name_list[i] == channel_name:
                for t_ind, t in enumerate(times_list[i]):
                    full_t_ind = int(
                        np.round((t - tmin) * sampling_rate * sim_sampling_boost)
                    )
                    waveform[full_t_ind] += pulse_signal[t_ind]

        channel_waveforms["channels"].update({channel_name: waveform})

    return channel_waveforms


def apply_readout_noise(
    samples: dict[Union[str, int], list],
    readout_error: dict[Union[int, str], list],
) -> dict[Union[str, int], list]:
    """Applies readout noise to samples.

    Args:
        samples (dict): Samples generated from self.get_samples.
        readout_error (dict): Dictionary specifying the readout noise for each qubit. Readout noise is specified by a list containing probabilities of prepare 0 measure 1, and prepare 1 measure 0.

    Returns:
        dict: The noisy sampled qubit values for each qubit labelled by its index.
    """
    noisy_samples = {}
    for ro_qubit in samples.keys():
        noisy_samples.update({ro_qubit: []})
        p0m1, p1m0 = readout_error[ro_qubit]
        qubit_values = samples[ro_qubit]

        for i, v in enumerate(qubit_values):
            if v == 0:
                noisy_samples[ro_qubit].append(
                    np.random.choice([0, 1], p=[1 - p0m1, p0m1])
                )
            else:
                noisy_samples[ro_qubit].append(
                    np.random.choice([0, 1], p=[p1m0, 1 - p1m0])
                )

    return noisy_samples
