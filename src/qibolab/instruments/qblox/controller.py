import signal
from dataclasses import replace

import numpy as np
from qblox_instruments.qcodes_drivers.cluster import Cluster as QbloxCluster
from qibo.config import log, raise_error

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.instruments.abstract import Controller
from qibolab.instruments.qblox.cluster_qcm_bb import QcmBb
from qibolab.instruments.qblox.cluster_qcm_rf import QcmRf
from qibolab.instruments.qblox.cluster_qrm_rf import QrmRf
from qibolab.instruments.qblox.sequencer import SAMPLING_RATE
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from qibolab.unrolling import Bounds

MAX_NUM_BINS = 98304
"""Maximum number of bins that should be used per sequencer in a readout
module.

One sequencer can have up to 2 ** 17 bins, however the 6 sequencers in
module allocate bins from a shared memory, and this memory is smaller
than 6 * 2 ** 17. Hence, if all sequencers are used at once, each cannot
support 2 ** 17 bins. In fact, the total bin memory for the module is 3
* (2 ** 17 + 2 ** 16). This means up to three sequencers used at once
can support 2 ** 17 bins each, but four or more sequencers cannot. So
the limitation on the number of bins technically depends on the number
of sequencers used, but to keep things simple we limit ourselves to max
number of bins that works regardless of situation.
"""


class QbloxController(Controller):
    """A controller to manage qblox devices.

    Attributes:
        is_connected (bool): .
        modules (dict): A dictionay with the qblox modules connected to the experiment.
    """

    def __init__(
        self, name, address: str, modules, internal_reference_clock: bool = True
    ):
        """Initialises the controller."""
        super().__init__(name=name, address=address)
        self.is_connected = False
        self.cluster: QbloxCluster = None
        self.modules: dict = modules
        self._reference_clock = "internal" if internal_reference_clock else "external"
        self.bounds = Bounds(
            waveforms=int(
                4e4
            ),  # Translate SEQUENCER_MEMORY = 2**17 into pulse duration
            readout=int(1e6),
            instructions=int(1e6),
        )
        signal.signal(signal.SIGTERM, self._termination_handler)

    @property
    def sampling_rate(self):
        return SAMPLING_RATE

    def connect(self):
        """Connects to the modules."""

        if self.is_connected:
            return
        try:
            # Connect cluster
            QbloxCluster.close_all()
            self.cluster = QbloxCluster(self.name, self.address)
            self.cluster.reset()
            self.cluster.set("reference_source", self._reference_clock)
            # Connect modules
            for module in self.modules.values():
                module.connect(self.cluster)
            self.is_connected = True
            log.info("QbloxController: all modules connected.")

        except Exception as exception:
            raise ConnectionError(f"Unable to connect:\n{str(exception)}\n")
            # TODO: check for exception 'The module qrm_rf0 does not have parameters in0_att' and reboot the cluster

    def disconnect(self):
        """Disconnects all modules."""
        if self.is_connected:
            for module in self.modules.values():
                module.disconnect()
            self.cluster.close()
            self.is_connected = False

    def _termination_handler(self, signum, frame):
        """Calls all modules to stop if the program receives a termination
        signal."""

        log.warning("Termination signal received, disconnecting modules.")
        if self.is_connected:
            for name in self.modules:
                self.modules[name].disconnect()
        log.warning("QbloxController: all modules are disconnected.")
        exit(0)

    def _set_module_channel_map(self, module: QrmRf, qubits: dict):
        """Retrieve all the channels connected to a specific Qblox module.

        This method updates the `channel_port_map` attribute of the
        specified Qblox module based on the information contained in the
        provided qubits dictionary (dict of `qubit` objects).

        Return the list of channels connected to module_name
        """
        for qubit in qubits.values():
            for channel in qubit.channels:
                if channel.port and channel.port.module.name == module.name:
                    module.channel_map[channel.name] = channel
        return list(module.channel_map)

    def _execute_pulse_sequence(
        self,
        qubits: dict,
        sequence: PulseSequence,
        options: ExecutionParameters,
        sweepers: list() = [],  # list(Sweeper) = []
        **kwargs,
        # nshots=None,
        # navgs=None,
        # relaxation_time=None,
    ):
        """Executes a sequence of pulses or a sweep.

        Args:
            sequence (:class:`qibolab.pulses.PulseSequence`): The sequence of pulses to execute.
            options (:class:`qibolab.platforms.platform.ExecutionParameters`): Object holding the execution options.
            sweepers (list(Sweeper)): A list of Sweeper objects defining parameter sweeps.
        """
        if not self.is_connected:
            raise_error(
                RuntimeError, "Execution failed because modules are not connected."
            )

        if options.averaging_mode is AveragingMode.SINGLESHOT:
            nshots = options.nshots
            navgs = 1
        else:
            navgs = options.nshots
            nshots = 1

        relaxation_time = options.relaxation_time
        repetition_duration = sequence.finish + relaxation_time

        # shots results are stored in separate bins
        # calculate number of shots
        num_bins = nshots
        for sweeper in sweepers:
            num_bins *= len(sweeper.values)

        # DEBUG: Plot Pulse Sequence
        # sequence.plot('plot.png')
        # DEBUG: sync_en
        # from qblox_instruments.qcodes_drivers.cluster import Cluster
        # cluster:Cluster = self.modules['cluster'].device
        # for module in cluster.modules:
        #     if module.get("present"):
        #         for sequencer in module.sequencers:
        #             if sequencer.get('sync_en'):
        #                 print(f"type: {module.module_type}, sequencer: {sequencer.name}, sync_en: True")

        # Process Pulse Sequence. Assign pulses to modules and generate waveforms & program
        module_pulses = {}
        data = {}
        for name, module in self.modules.items():
            # from the pulse sequence, select those pulses to be synthesised by the module
            module_channels = self._set_module_channel_map(module, qubits)
            module_pulses[name] = sequence.get_channel_pulses(*module_channels)

            #  ask each module to generate waveforms & program and upload them to the device
            module.process_pulse_sequence(
                qubits,
                module_pulses[name],
                navgs,
                nshots,
                repetition_duration,
                sweepers,
            )

            # log.info(f"{self.modules[name]}: Uploading pulse sequence")
            module.upload()

        # play the sequence or sweep
        for module in self.modules.values():
            if isinstance(module, (QrmRf, QcmRf, QcmBb)):
                module.play_sequence()

        # retrieve the results
        acquisition_results = {}
        for name, module in self.modules.items():
            if (
                isinstance(module, QrmRf)
                and not module_pulses[name].probe_pulses.is_empty
            ):
                results = module.acquire()
                for key, value in results.items():
                    acquisition_results[key] = value
        # TODO: move to QRM_RF.acquire()
        shape = tuple(len(sweeper.values) for sweeper in reversed(sweepers))
        shots_shape = (nshots,) + shape
        for ro_pulse in sequence.probe_pulses:
            if options.acquisition_type is AcquisitionType.DISCRIMINATION:
                _res = acquisition_results[ro_pulse.id].classified
                _res = np.reshape(_res, shots_shape)
                if options.averaging_mode is not AveragingMode.SINGLESHOT:
                    _res = np.mean(_res, axis=0)
            elif options.acquisition_type is AcquisitionType.RAW:
                i_raw = acquisition_results[ro_pulse.id].raw_i
                q_raw = acquisition_results[ro_pulse.id].raw_q
                _res = i_raw + 1j * q_raw
            elif options.acquisition_type is AcquisitionType.INTEGRATION:
                ires = acquisition_results[ro_pulse.id].shots_i
                qres = acquisition_results[ro_pulse.id].shots_q
                _res = ires + 1j * qres
                if options.averaging_mode is AveragingMode.SINGLESHOT:
                    _res = np.reshape(_res, shots_shape)
                else:
                    _res = np.reshape(_res, shape)

            acquisition = options.results_type(np.squeeze(_res))
            data[ro_pulse.id] = data[ro_pulse.qubit] = acquisition

        return data

    def play(self, qubits, couplers, sequence, options):
        return self._execute_pulse_sequence(qubits, sequence, options)

    def sweep(
        self,
        qubits: dict,
        couplers: dict,
        sequence: PulseSequence,
        options: ExecutionParameters,
        *sweepers,
    ):
        """Executes a sequence of pulses while sweeping one or more parameters.

        The parameters to be swept are defined in :class:`qibolab.sweeper.Sweeper` object.
        Args:
            sequence (:class:`qibolab.pulses.PulseSequence`): The sequence of pulses to execute.
            options (:class:`qibolab.platforms.platform.ExecutionParameters`): Object holding the execution options.
            sweepers (list(Sweeper)): A list of Sweeper objects defining parameter sweeps.
        """
        id_results = {}
        map_id_serial = {}

        # during the sweep, pulse parameters need to be changed
        # to avoid affecting the user, make a copy of the pulse sequence
        # and the sweepers, as they contain references to pulses
        sequence_copy = sequence.copy()
        sweepers_copy = []
        for sweeper in sweepers:
            if sweeper.pulses:
                ps = [
                    sequence_copy[sequence_copy.index(pulse)]
                    for pulse in sweeper.pulses
                    if pulse in sequence_copy
                ]
            else:
                ps = None
            sweepers_copy.append(
                Sweeper(
                    parameter=sweeper.parameter,
                    values=sweeper.values,
                    pulses=ps,
                    qubits=sweeper.qubits,
                    type=sweeper.type,
                )
            )

        # reverse sweepers exept for res punchout att
        contains_attenuation_frequency = any(
            sweepers_copy[i].parameter == Parameter.attenuation
            and sweepers_copy[i + 1].parameter == Parameter.frequency
            for i in range(len(sweepers_copy) - 1)
        )

        if not contains_attenuation_frequency:
            sweepers_copy.reverse()

        # create a map between the pulse id, which never changes, and the original serial
        for pulse in sequence_copy.probe_pulses:
            map_id_serial[pulse.id] = pulse.id
            id_results[pulse.id] = None
            id_results[pulse.qubit] = None

        # execute the each sweeper recursively
        self._sweep_recursion(
            qubits,
            sequence_copy,
            options,
            *tuple(sweepers_copy),
            results=id_results,
        )

        # return the results using the original serials
        serial_results = {}
        for pulse in sequence_copy.probe_pulses:
            serial_results[map_id_serial[pulse.id]] = id_results[pulse.id]
            serial_results[pulse.qubit] = id_results[pulse.id]
        return serial_results

    def _sweep_recursion(
        self,
        qubits,
        sequence,
        options: ExecutionParameters,
        *sweepers,
        results,
    ):
        """Executes a sweep recursively.

        Args:
            sequence (:class:`qibolab.pulses.PulseSequence`): The sequence of pulses to execute.
            sweepers (list(Sweeper)): A list of Sweeper objects defining parameter sweeps.
            results (:class:`qibolab.results.ExecutionResults`): A results object to update with the reults of the execution.
            nshots (int): The number of times the sequence of pulses should be executed.
            average (bool): A flag to indicate if the results of the shots should be averaged.
            relaxation_time (int): The the time to wait between repetitions to allow the qubit relax to ground state.
        """

        for_loop_sweepers = [Parameter.attenuation, Parameter.lo_frequency]
        sweeper: Sweeper = sweepers[0]

        # until sweeper contains the information to determine whether the sweep should be relative or
        # absolute:

        # elif sweeper.parameter is Parameter.relative_phase:
        #     initial = {}
        #     for pulse in sweeper.pulses:
        #         initial[pulse.id] = pulse.relative_phase

        # elif sweeper.parameter is Parameter.frequency:
        #     initial = {}
        #     for pulse in sweeper.pulses:
        #         initial[pulse.id] = pulse.frequency

        if sweeper.parameter in for_loop_sweepers:
            # perform sweep recursively
            for value in sweeper.values:
                if sweeper.parameter is Parameter.attenuation:
                    initial = {}
                    for qubit in sweeper.qubits:
                        initial[qubit.name] = qubits[qubit.name].readout.attenuation
                        if sweeper.type == SweeperType.ABSOLUTE:
                            qubit.readout.attenuation = value
                        elif sweeper.type == SweeperType.OFFSET:
                            qubit.readout.attenuation = initial[qubit.name] + value
                        elif sweeper.type == SweeperType.FACTOR:
                            qubit.readout.attenuation = initial[qubit.name] * value

                elif sweeper.parameter is Parameter.lo_frequency:
                    initial = {}
                    for pulse in sweeper.pulses:
                        # FIXME:
                        # if pulse.type == PulseType.READOUT:
                        if True:
                            initial[pulse.id] = qubits[pulse.qubit].readout.lo_frequency
                            if sweeper.type == SweeperType.ABSOLUTE:
                                qubits[pulse.qubit].readout.lo_frequency = value
                            elif sweeper.type == SweeperType.OFFSET:
                                qubits[pulse.qubit].readout.lo_frequency = (
                                    initial[pulse.id] + value
                                )
                            elif sweeper.type == SweeperType.FACTOR:
                                qubits[pulse.qubit].readout.lo_frequency = (
                                    initial[pulse.id] * value
                                )
                        # FIXME:
                        # elif pulse.type == PulseType.DRIVE:
                        elif True:
                            initial[pulse.id] = qubits[pulse.qubit].drive.lo_frequency
                            if sweeper.type == SweeperType.ABSOLUTE:
                                qubits[pulse.qubit].drive.lo_frequency = value
                            elif sweeper.type == SweeperType.OFFSET:
                                qubits[pulse.qubit].drive.lo_frequency = (
                                    initial[pulse.id] + value
                                )
                            elif sweeper.type == SweeperType.FACTOR:
                                qubits[pulse.qubit].drive.lo_frequency = (
                                    initial[pulse.id] * value
                                )

                if len(sweepers) > 1:
                    self._sweep_recursion(
                        qubits,
                        sequence,
                        options,
                        *sweepers[1:],
                        results=results,
                    )
                else:
                    result = self._execute_pulse_sequence(
                        qubits=qubits, sequence=sequence, options=options
                    )
                    for pulse in sequence.probe_pulses:
                        if results[pulse.id]:
                            results[pulse.id] += result[pulse.id]
                        else:
                            results[pulse.id] = result[pulse.id]
                        results[pulse.qubit] = results[pulse.id]
        else:
            # rt sweeps
            # relative phase sweeps that cross 0 need to be split in two separate sweeps
            split_relative_phase = False
            rt_sweepers = [
                Parameter.frequency,
                Parameter.gain,
                Parameter.bias,
                Parameter.amplitude,
                Parameter.duration,
                Parameter.relative_phase,
            ]

            if sweeper.parameter == Parameter.relative_phase:
                if sweeper.type != SweeperType.ABSOLUTE:
                    raise_error(
                        ValueError,
                        "relative_phase sweeps other than ABSOLUTE are not supported by qblox yet",
                    )
                from qibolab.instruments.qblox.q1asm import convert_phase

                c_values = np.array([convert_phase(v) for v in sweeper.values])
                if any(np.diff(c_values) < 0):
                    split_relative_phase = True
                    _from = 0
                    for idx in np.append(
                        np.where(np.diff(c_values) < 0), len(c_values) - 1
                    ):
                        _to = idx + 1
                        _values = sweeper.values[_from:_to]
                        split_sweeper = Sweeper(
                            parameter=sweeper.parameter,
                            values=_values,
                            pulses=sweeper.pulses,
                            qubits=sweeper.qubits,
                        )
                        self._sweep_recursion(
                            qubits,
                            sequence,
                            options,
                            *((split_sweeper,) + sweepers[1:]),
                            results=results,
                        )
                        _from = _to

            if not split_relative_phase:
                if any(s.parameter not in rt_sweepers for s in sweepers):
                    # TODO: reorder the sequence of the sweepers and the results
                    raise Exception(
                        "cannot execute a for-loop sweeper nested inside of a rt sweeper"
                    )
                nshots = (
                    options.nshots
                    if options.averaging_mode == AveragingMode.SINGLESHOT
                    else 1
                )
                navgs = (
                    options.nshots
                    if options.averaging_mode != AveragingMode.SINGLESHOT
                    else 1
                )
                num_bins = nshots
                for sweeper in sweepers:
                    num_bins *= len(sweeper.values)

                    # split the sweep if the number of bins is larget than the memory of the sequencer (2**17)
                if num_bins < MAX_NUM_BINS:
                    # for sweeper in sweepers:
                    #     if sweeper.parameter is Parameter.amplitude:
                    #         # qblox cannot sweep amplitude in real time, but sweeping gain is quivalent
                    #         for pulse in sweeper.pulses:
                    #             pulse.amplitude = 1

                    #     elif sweeper.parameter is Parameter.gain:
                    #         for pulse in sweeper.pulses:
                    #             # qblox has an external and an internal gains
                    #             # when sweeping the internal, set the external to 1
                    #             # TODO check if it needs to be restored after execution
                    #             if pulse.type == PulseType.READOUT:
                    #                 qubits[pulse.qubit].readout.gain = 1
                    #             elif pulse.type == PulseType.DRIVE:
                    #                 qubits[pulse.qubit].drive.gain = 1

                    result = self._execute_pulse_sequence(
                        qubits, sequence, options, sweepers
                    )
                    self._add_to_results(sequence, results, result)
                else:
                    sweepers_repetitions = 1
                    for sweeper in sweepers:
                        sweepers_repetitions *= len(sweeper.values)
                    if sweepers_repetitions > MAX_NUM_BINS:
                        raise ValueError(
                            f"Requested sweep has {sweepers_repetitions} total number of sweep points. "
                            f"Maximum supported is {MAX_NUM_BINS}"
                        )

                    max_rt_nshots = MAX_NUM_BINS // sweepers_repetitions
                    num_full_sft_iterations = nshots // max_rt_nshots
                    result_chunks = []
                    for sft_iteration in range(num_full_sft_iterations + 1):
                        _nshots = min(
                            max_rt_nshots, nshots - sft_iteration * max_rt_nshots
                        )

                        res = self._execute_pulse_sequence(
                            qubits,
                            sequence,
                            replace(options, nshots=_nshots),
                            sweepers,
                        )
                        result_chunks.append(res)
                    result = self._combine_result_chunks(result_chunks)
                    self._add_to_results(sequence, results, result)

    @staticmethod
    def _combine_result_chunks(chunks):
        some_chunk = next(iter(chunks))
        some_result = next(iter(some_chunk.values()))
        return {
            key: some_result.__class__(
                np.concatenate([chunk[key] for chunk in chunks], axis=0)
            )
            for key in some_chunk.keys()
        }

    @staticmethod
    def _add_to_results(sequence, results, results_to_add):
        for pulse in sequence.probe_pulses:
            if results[pulse.id]:
                results[pulse.id] += results_to_add[pulse.id]
            else:
                results[pulse.id] = results_to_add[pulse.id]
            results[pulse.qubit] = results[pulse.id]
