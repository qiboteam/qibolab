"""Pulse simulator module for running quantum dynamics simulation of model of
device."""

import copy
import operator
from typing import Union

import numpy as np
import numpy.typing as npt

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.instruments.abstract import Controller
from qibolab.instruments.emulator.engines.qutip_engine import QutipSimulator
from qibolab.instruments.emulator.models import general_no_coupler_model
from qibolab.pulses import PulseSequence
from qibolab.qubits import Qubit, QubitId
from qibolab.result import average, collect
from qibolab.sweeper import Parameter, Sweeper, SweeperType

AVAILABLE_SWEEP_PARAMETERS = {
    Parameter.amplitude,
    Parameter.duration,
    Parameter.frequency,
    Parameter.relative_phase,
}

SIMULATION_ENGINES = {
    "Qutip": QutipSimulator,
}

MODELS = {
    "general_no_coupler_model": general_no_coupler_model,
}

GHZ = 1e9


class PulseSimulator(Controller):
    """Runs quantum dynamics simulation of model of device.

    Interfaces with Qibolab. Useful for testing code without requiring
    access to hardware.
    """

    PortType = None

    def __init__(self):
        super().__init__(name=None, address=None)

    @property
    def sampling_rate(self):
        return self._sampling_rate

    def setup(self, **kwargs):
        """Updates the pulse simulator by loading all parameters from
        `model_config` and `simulation_config`."""
        super().setup(kwargs["bounds"])
        self.settings = kwargs

        simulation_config = kwargs["simulation_config"]
        model_params = kwargs["model_params"]
        sim_opts = kwargs["sim_opts"]

        model_name = model_params["model_name"]
        model_config = MODELS[model_name].generate_model_config(model_params)

        simulation_engine_name = simulation_config["simulation_engine_name"]
        self.simulation_engine = SIMULATION_ENGINES[simulation_engine_name](
            model_config, sim_opts
        )

        # Settings for pulse processing
        self._sampling_rate = simulation_config["sampling_rate"]
        self.sim_sampling_boost = simulation_config["sim_sampling_boost"]
        self.runcard_duration_in_dt_units = simulation_config[
            "runcard_duration_in_dt_units"
        ]
        self.instant_measurement = simulation_config["instant_measurement"]
        self.platform_to_simulator_channels = model_config[
            "platform_to_simulator_channels"
        ]

        self.readout_error = {
            int(k): v for k, v in model_config["readout_error"].items()
        }
        self.simulate_dissipation = simulation_config["simulate_dissipation"]
        self.output_state_history = simulation_config["output_state_history"]

    def connect(self):
        pass

    def disconnect(self):
        pass

    def dump(self):
        return self.settings

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
            tuple: A tuple containing a dictionary of time-related information (sequence duration, simulation time step, and simulation time), the reduced density matrix of the quantum state at the end of simulation in the Hilbert space specified by the qubits present in the readout channels (little endian), as well as the corresponding list of qubit indices.
        """
        # reduces measurement time to 1 dt to save simulation time
        if instant_measurement:
            sequence = truncate_ro_pulses(sequence)

        # extract waveforms from pulse sequence
        channel_waveforms = ps_to_waveform_dict(
            sequence,
            self.platform_to_simulator_channels,
            self.sampling_rate,
            self.sim_sampling_boost,
            self.runcard_duration_in_dt_units,
        )

        # execute pulse simulation in emulator
        simulation_results = self.simulation_engine.qevolve(
            channel_waveforms, self.simulate_dissipation
        )

        return simulation_results

    def play(
        self,
        qubits: dict[QubitId, Qubit],
        couplers: dict[QubitId, Qubit],
        sequence: PulseSequence,
        execution_parameters: ExecutionParameters,
    ) -> dict[str, npt.NDArray]:
        """Executes the sequence of instructions and generates readout results,
        as well as simulation-related time and states data.

        Args:
            qubits (dict): Qubits involved in the device. Does not affect emulator.
            couplers (dict): Couplers involved in the device. Does not affect emulator.
            sequence (`qibolab.pulses.PulseSequence`): Pulse sequence to simulate.
            execution_parameters (`qibolab.ExecutionParameters`): Parameters (nshots,
                                                        relaxation_time,
                                                        fast_reset,
                                                        acquisition_type,
                                                        averaging_mode)

        Returns:
            dict: A dictionary mapping the readout pulses serial and respective qubits to
            Qibolab results object, as well as simulation-related time and states data.
        """
        nshots = execution_parameters.nshots
        ro_pulse_list = sequence.probe_pulses
        times_dict, output_states, ro_reduced_dm, rdm_qubit_list = (
            self.run_pulse_simulation(sequence, self.instant_measurement)
        )
        if not self.output_state_history:
            output_states = output_states[-1]

        samples = get_samples(
            nshots,
            ro_reduced_dm,
            rdm_qubit_list,
            self.simulation_engine.qid_nlevels_map,
        )
        # apply default readout noise
        if self.readout_error is not None:
            samples = apply_readout_noise(samples, self.readout_error)
        # generate result object
        results = get_results_from_samples(ro_pulse_list, samples, execution_parameters)
        results["simulation"] = {
            "sequence_duration": times_dict["sequence_duration"],
            "simulation_dt": times_dict["simulation_dt"],
            "simulation_time": times_dict["simulation_time"],
            "output_states": output_states,
        }

        return results

    ### sweeper adapted from icarusqfpga ###
    def sweep(
        self,
        qubits: dict[QubitId, Qubit],
        couplers: dict[QubitId, Qubit],
        sequence: PulseSequence,
        execution_parameters: ExecutionParameters,
        *sweeper: list[Sweeper],
    ) -> dict[str, Union[npt.NDArray, dict]]:
        """Executes the sweep and generates readout results, as well as
        simulation-related time and states data.

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
            Qibolab results objects, as well as simulation-related time and states data.

        Raises:
            NotImplementedError: If sweep.parameter is not in AVAILABLE_SWEEP_PARAMETERS.
        """
        sweeper_shape = [len(sweep.values) for sweep in sweeper]

        # Record pulse values before sweeper modification
        bsv = []
        for sweep in sweeper:
            param_name = sweep.parameter.name.lower()
            if sweep.parameter not in AVAILABLE_SWEEP_PARAMETERS:
                raise NotImplementedError(
                    "Sweep parameter requested not available", param_name
                )
            base_sweeper_values = [getattr(pulse, param_name) for pulse in sweep.pulses]
            bsv.append(base_sweeper_values)

        sweep_samples = self._sweep_recursion(
            qubits, couplers, sequence, execution_parameters, *sweeper
        )
        output_states_list = sweep_samples.pop("output_states")
        sequence_duration_array = sweep_samples.pop("sequence_duration")
        simulation_dt_array = sweep_samples.pop("simulation_dt")
        simulation_time_array = sweep_samples.pop("simulation_time")

        # reshape output_states to sweeper dimensions
        output_states_array = np.ndarray(sweeper_shape, dtype=list)
        listlen = len(output_states_list) // np.prod(sweeper_shape)
        array_indices = make_array_index_list(sweeper_shape)
        for index in array_indices:
            output_states_array[tuple(index)] = output_states_list[:listlen]
            output_states_list = output_states_list[listlen:]

        # reshape time data to sweeper dimensions
        sequence_duration_array = np.array(sequence_duration_array).reshape(
            sweeper_shape
        )
        simulation_dt_array = np.array(simulation_dt_array).reshape(sweeper_shape)
        simulation_time_array = np.array(simulation_time_array).reshape(sweeper_shape)

        # reshape and reformat samples to results format
        results = get_results_from_samples(
            sequence.probe_pulses, sweep_samples, execution_parameters, sweeper_shape
        )

        # Reset pulse values back to original values (following icarusqfpga)
        for sweep, base_sweeper_values in zip(sweeper, bsv):
            param_name = sweep.parameter.name.lower()
            for pulse, value in zip(sweep.pulses, base_sweeper_values):
                setattr(pulse, param_name, value)
                # FIXME: this is copy-pasted from IcarusQ, check the comment in there
                # Since the sweeper will modify the readout pulse serial, we collate the results with the qubit number.
                # This is only for qibocal compatiability and will be removed with IcarusQ v2.
                # if pulse.type is PulseType.READOUT:
                #     results[pulse.serial] = results[pulse.qubit]

        results.update(
            {
                "simulation": {
                    "sequence_duration": sequence_duration_array,
                    "simulation_dt": simulation_dt_array,
                    "simulation_time": simulation_time_array,
                    "output_states": output_states_array,
                }
            }
        )

        return results

    def _sweep_recursion(
        self,
        qubits: dict[QubitId, Qubit],
        couplers: dict[QubitId, Qubit],
        sequence: PulseSequence,
        execution_parameters: ExecutionParameters,
        *sweeper: Sweeper,
    ) -> dict[Union[str, int], list]:
        """Performs sweep by recursion. Appends sampled lists and other
        simulation data obtained from each call of `self._sweep_play`.

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
            dict: A dictionary mapping the qubit indices to list of sampled values, simulation-related time data, and simulated states data.
        """

        if len(sweeper) == 0:
            times_dict, output_states, samples = self._sweep_play(
                qubits, couplers, sequence, execution_parameters
            )
            if not self.output_state_history:
                output_states = [output_states[-1]]
            samples.update({"output_states": output_states})
            for k, v in times_dict.items():
                samples.update({k: [v]})
            return samples

        sweep = sweeper[0]
        param = sweep.parameter
        param_name = param.name.lower()

        base_sweeper_values = [getattr(pulse, param_name) for pulse in sweep.pulses]
        sweeper_op = _sweeper_operation.get(sweep.type)
        ret = {}

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
        qubits: dict[QubitId, Qubit],
        couplers: dict[QubitId, Qubit],
        sequence: PulseSequence,
        execution_parameters: ExecutionParameters,
    ) -> dict[Union[str, int], list]:
        """Generates simulation-related time data, simulated states data, and
        samples list labelled by qubit index.

        Args:
            qubits (dict): Qubits involved in the device. Does not affect emulator.
            couplers (dict): Couplers involved in the device. Does not affect emulator.
            sequence (`qibolab.pulses.PulseSequence`): Pulse sequence to simulate.
            execution_parameters (`qibolab.ExecutionParameters`): Parameters (nshots,
                                                        relaxation_time,
                                                        fast_reset,
                                                        acquisition_type,
                                                        averaging_mode)

        Returns:
            dict: A tuple with dictionary containing simulation-related time data, a list of states at each time step in the simulation, and a dictionary mapping the qubit indices to list of sampled values.
        """
        nshots = execution_parameters.nshots
        ro_pulse_list = sequence.probe_pulses

        # run pulse simulation
        times_dict, state_history, ro_reduced_dm, rdm_qubit_list = (
            self.run_pulse_simulation(sequence, self.instant_measurement)
        )
        # generate samples
        samples = get_samples(
            nshots,
            ro_reduced_dm,
            rdm_qubit_list,
            self.simulation_engine.qid_nlevels_map,
        )
        # apply default readout noise
        if self.readout_error is not None:
            samples = apply_readout_noise(samples, self.readout_error)

        return times_dict, state_history, samples

    @staticmethod
    def merge_sweep_results(
        dict_a: """dict[str, Union[npt.NDArray, list]]""",
        dict_b: """dict[str, Union[npt.NDArray, list]]""",
    ) -> """dict[str, Union[npt.NDArray, list]]""":
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
    platform_to_simulator_channels: dict,
    sampling_rate: float = 1.0,
    sim_sampling_boost: int = 1,
    runcard_duration_in_dt_units: bool = False,
) -> dict[str, Union[np.ndarray, dict[str, np.ndarray]]]:
    """Converts pulse sequence to dictionary of time and channel separated
    waveforms.

    Args:
        sequence (`qibolab.pulses.PulseSequence`): Pulse sequence to simulate.
        platform_to_simulator_channels (dict): A dictionary that maps platform channel names to simulator channel names.
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
            return platform_to_simulator_channels[platform_channel_name + frequency]
        except:
            # frequency independent channel operation (default)
            return platform_to_simulator_channels[platform_channel_name]

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
                    start = int(pulse.start * sim_sampling_boost)
                    actual_pulse_frequency = (
                        pulse.frequency
                    )  # store actual pulse frequency for channel_translator
                    # rescale frequency to be compatible with sampling_rate = 1
                    pulse.frequency = pulse.frequency / sampling_rate
                    # need to first set pulse._if in GHz to use modulated_waveform_i method
                    pulse._if = pulse.frequency / GHZ

                    i_env = pulse.envelope_waveform_i(sim_sampling_boost).data
                    q_env = pulse.envelope_waveform_q(sim_sampling_boost).data

                    # Qubit drive microwave signals
                    end = start + len(i_env)
                    t = np.arange(start, end) / sampling_rate / sim_sampling_boost
                    cosalpha = np.cos(
                        2 * np.pi * pulse._if * sampling_rate * t + pulse.relative_phase
                    )
                    sinalpha = np.sin(
                        2 * np.pi * pulse._if * sampling_rate * t + pulse.relative_phase
                    )
                    pulse_signal = i_env * sinalpha + q_env * cosalpha
                    # pulse_signal = pulse_signal/np.sqrt(2) # uncomment for ibm runcard

                    times_list.append(t)
                    signals_list.append(pulse_signal)

                    if pulse.type.value == "qd":
                        platform_channel_name = f"drive-{qubit}"
                    # TODO: to add during flux pulse update
                    # elif pulse.type.value == "qf":
                    #   platform_channel_name = f"flux-{qubit}"
                    elif pulse.type.value == "ro":
                        platform_channel_name = f"readout-{qubit}"

                    # restore pulse frequency values
                    pulse.frequency = actual_pulse_frequency
                    pulse._if = pulse.frequency / GHZ

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
                    pulse._if = pulse.frequency / GHZ

                    i_env = pulse.envelope_waveform_i(sim_sampling_rate).data
                    q_env = pulse.envelope_waveform_q(sim_sampling_rate).data

                    # Qubit drive microwave signals
                    end = start + len(i_env)
                    t = np.arange(start, end) / sim_sampling_rate
                    cosalpha = np.cos(2 * np.pi * pulse._if * t + pulse.relative_phase)
                    sinalpha = np.sin(2 * np.pi * pulse._if * t + pulse.relative_phase)
                    pulse_signal = i_env * sinalpha + q_env * cosalpha
                    # pulse_signal = pulse_signal/np.sqrt(2) # uncomment for ibm runcard

                    times_list.append(t)
                    signals_list.append(pulse_signal)

                    if pulse.type.value == "qd":
                        platform_channel_name = f"drive-{qubit}"
                    # TODO: add during flux pulse update
                    # elif pulse.type.value == "qf":
                    # platform_channel_name = f"flux-{qubit}"
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
        samples (dict): Samples generated from get_samples.
        readout_error (dict): Dictionary specifying the readout noise for each qubit. Readout noise is specified by a list containing probabilities of prepare 0 measure 1, and prepare 1 measure 0.

    Returns:
        dict: The noisy sampled qubit values for each qubit labelled by its index.

    Raises:
        ValueError: If the readout qubits given by samples.keys() is not a subset of the qubits with readout errors specified in readout_error.
    """
    # check if readout error is specified for all ro qubits
    ro_qubit_list = list(samples.keys())
    if not set(ro_qubit_list).issubset(readout_error.keys()):
        raise ValueError(f"Not all readout qubits are present in readout_error!")
    noisy_samples = {}
    for ro_qubit in ro_qubit_list:  # samples.keys():
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


def make_comp_basis(
    qubit_list: list[Union[int, str]], qid_nlevels_map: dict[Union[int, str], int]
) -> np.ndarray:
    """Generates the computational basis states of the Hilbert space.

    Args:
        qubit_list (list): List of target qubit indices to generate the local Hilbert space of the qubits that respects the order given by qubit_list.
        qid_nlevels_map (dict): Dictionary mapping the qubit IDs given in qubit_list to their respective Hilbert space dimensions.

    Returns:
        `np.ndarray`: The list of computation basis states of the local Hilbert space in a numpy array.
    """
    nqubits = len(qubit_list)
    qid_list = [str(qubit) for qubit in qubit_list]
    nlevels = [qid_nlevels_map[qid] for qid in qid_list]

    return make_array_index_list(nlevels)


def make_array_index_list(array_shape: list):
    """Generates all indices of an array of arbitrary shape in ascending
    order."""
    return np.indices(array_shape).reshape(len(array_shape), -1).T


def get_results_from_samples(
    ro_pulse_list: list,
    samples: dict[Union[str, int], list],
    execution_parameters: ExecutionParameters,
    prepend_to_shape: list = [],
) -> dict[str, npt.NDArray]:
    """Converts samples into Qibolab results format.

    Args:
        ro_pulse_list (list): List of readout pulse sequences.
        samples (dict): Samples generated by get_samples.
        append_to_shape (list): Specifies additional dimensions for the shape of the results. Defaults to empty list.
        execution_parameters (`qibolab.ExecutionParameters`): Parameters (nshots,
                                                    relaxation_time,
                                                    fast_reset,
                                                    acquisition_type,
                                                    averaging_mode)

    Returns:
        dict: Qibolab result data.

    Raises:
        ValueError: If execution_parameters.acquisition_type is not supported.
    """
    shape = prepend_to_shape + [execution_parameters.nshots]
    tshape = [-1] + list(range(len(prepend_to_shape)))

    results = {}
    for ro_pulse in ro_pulse_list:
        values = np.array(samples[ro_pulse.qubit]).reshape(shape).transpose(tshape)

        if execution_parameters.acquisition_type is AcquisitionType.DISCRIMINATION:
            processed_values = values

        elif execution_parameters.acquisition_type is AcquisitionType.INTEGRATION:
            vals = values.astype(np.complex128)
            processed_values = collect(vals.real, vals.imag)

        else:
            raise ValueError(
                f"Current emulator does not support requested AcquisitionType {execution_parameters.acquisition_type}"
            )

        if execution_parameters.averaging_mode is AveragingMode.CYCLIC:
            processed_values = average(processed_values)

        results[ro_pulse.qubit] = results[ro_pulse.serial] = processed_values
    return results


def get_samples(
    nshots: int,
    ro_reduced_dm: np.ndarray,
    ro_qubit_list: list,
    qid_nlevels_map: dict[Union[int, str], int],
) -> dict[Union[str, int], list]:
    """Gets samples from the density matrix corresponding to the system or
    subsystem specified by the ordered qubit indices.

    Args:
        nshots (int): Number of shots corresponding to the number of samples in the output.
        ro_reduced_dm (np.ndarray): Input density matrix.
        ro_qubit_list (list): Qubit indices corresponding to the Hilbert space structure of the reduced density matrix (ro_reduced_dm).
        qid_nlevels_map (dict): Dictionary mapping the qubit IDs given in qubit_list to their respective Hilbert space dimensions.

    Returns:
        dict: The sampled qubit values for each qubit labelled by its index.
    """
    # use the real diagonal part of the reduced density matrix as the probability distribution
    ro_probability_distribution = np.diag(ro_reduced_dm).real

    # preprocess distribution
    ro_probability_distribution = np.maximum(
        ro_probability_distribution, 0
    )  # to remove small negative values
    ro_probability_sum = np.sum(ro_probability_distribution)
    ro_probability_distribution = ro_probability_distribution / ro_probability_sum
    ro_qubits_dim = len(ro_probability_distribution)

    # create array of computational basis states of the reduced (measured) Hilbert space
    reduced_computation_basis = make_comp_basis(ro_qubit_list, qid_nlevels_map)

    # sample computation basis index nshots times from distribution
    sample_all_ro_list = np.random.choice(
        ro_qubits_dim, nshots, True, ro_probability_distribution
    )

    samples = {}
    for ind, ro_qubit in enumerate(ro_qubit_list):
        # extracts sampled values for each readout qubit from sample_all_ro_list
        outcomes = [
            reduced_computation_basis[outcome][ind] for outcome in sample_all_ro_list
        ]
        samples[ro_qubit] = outcomes

    return samples


def truncate_ro_pulses(
    sequence: PulseSequence,
) -> PulseSequence:
    """Creates a deepcopy of the original sequence with truncated readout
    pulses to one time step.

    Args:
        sequence (`qibolab.pulses.PulseSequence`): Pulse sequence.

    Returns:
        `qibolab.pulses.PulseSequence`: Modified pulse sequence with one time step readout pulses.
    """
    sequence = copy.deepcopy(sequence)
    for pulse in sequence.probe_pulses:
        pulse.duration = 1

    return sequence
