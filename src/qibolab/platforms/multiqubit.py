import copy

import numpy as np
import yaml
from qibo.config import log, raise_error

from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence, PulseType
from qibolab.result import ExecutionResults
from qibolab.sweeper import Sweeper, Parameter


class MultiqubitPlatform(AbstractPlatform):
    def __init__(self, name, runcard):
        super().__init__(name, runcard)
        self.instruments = {}
        # Instantiate instruments
        for name in self.settings["instruments"]:
            lib = self.settings["instruments"][name]["lib"]
            i_class = self.settings["instruments"][name]["class"]
            address = self.settings["instruments"][name]["address"]
            from importlib import import_module

            InstrumentClass = getattr(import_module(f"qibolab.instruments.{lib}"), i_class)
            instance = InstrumentClass(name, address)
            self.instruments[name] = instance

        # Generate qubit_instrument_map from qubit_channel_map and the instruments' channel_port_maps
        self.qubit_instrument_map = {}
        for qubit in self.qubit_channel_map:
            self.qubit_instrument_map[qubit] = [None, None, None, None]
            for name in self.instruments:
                if "channel_port_map" in self.settings["instruments"][name]["settings"]:
                    for channel in self.settings["instruments"][name]["settings"]["channel_port_map"]:
                        if channel in self.qubit_channel_map[qubit]:
                            self.qubit_instrument_map[qubit][self.qubit_channel_map[qubit].index(channel)] = name
                if "s4g_modules" in self.settings["instruments"][name]["settings"]:
                    for channel in self.settings["instruments"][name]["settings"]["s4g_modules"]:
                        if channel in self.qubit_channel_map[qubit]:
                            self.qubit_instrument_map[qubit][self.qubit_channel_map[qubit].index(channel)] = name

    def reload_settings(self):
        super().reload_settings()
        self.characterization = self.settings["characterization"]
        self.qubit_channel_map = self.settings["qubit_channel_map"]
        self.hardware_avg = self.settings["settings"]["hardware_avg"]
        self.relaxation_time = self.settings["settings"]["relaxation_time"]

    def set_lo_drive_frequency(self, qubit, freq):
        self.qd_port[qubit].lo_frequency = freq

    def get_lo_drive_frequency(self, qubit):
        return self.qd_port[qubit].lo_frequency

    def set_lo_readout_frequency(self, qubit, freq):
        self.ro_port[qubit].lo_frequency = freq

    def get_lo_readout_frequency(self, qubit):
        return self.ro_port[qubit].lo_frequency

    def set_attenuation(self, qubit, att):
        self.ro_port[qubit].attenuation = att

    def set_gain(self, qubit, gain):
        self.qd_port[qubit].gain = gain

    def set_bias(self, qubit, bias):
        if qubit in self.qbm:
            self.qb_port[qubit].current = bias
        elif qubit in self.qfm:
            self.qf_port[qubit].offset = bias

    def get_attenuation(self, qubit):
        return self.ro_port[qubit].attenuation

    def get_bias(self, qubit):
        if qubit in self.qbm:
            return self.qb_port[qubit].current
        elif qubit in self.qfm:
            return self.qf_port[qubit].offset

    def get_gain(self, qubit):
        return self.qd_port[qubit].gain

    def connect(self):
        """Connects to lab instruments using the details specified in the calibration settings."""
        if not self.is_connected:
            try:
                for name in self.instruments:
                    log.info(f"Connecting to {self.name} instrument {name}.")
                    self.instruments[name].connect()
                self.is_connected = True
            except Exception as exception:
                raise_error(
                    RuntimeError,
                    "Cannot establish connection to " f"{self.name} instruments. " f"Error captured: '{exception}'",
                )

    def setup(self):
        if not self.is_connected:
            raise_error(
                RuntimeError,
                "There is no connection to the instruments, the setup cannot be completed",
            )

        for name in self.instruments:
            # Set up every with the platform settings and the instrument settings
            self.instruments[name].setup(
                **self.settings["settings"],
                **self.settings["instruments"][name]["settings"],
            )

        # Generate ro_channel[qubit], qd_channel[qubit], qf_channel[qubit], qrm[qubit], qcm[qubit], lo_qrm[qubit], lo_qcm[qubit]
        self.ro_channel = {}  # readout
        self.qd_channel = {}  # qubit drive
        self.qf_channel = {}  # qubit flux
        self.qb_channel = {}  # qubit flux biassing
        self.qrm = {}  # qubit readout module
        self.qdm = {}  # qubit drive module
        self.qfm = {}  # qubit flux module
        self.qbm = {}  # qubit flux biassing module
        self.ro_port = {}
        self.qd_port = {}
        self.qf_port = {}
        self.qb_port = {}
        for qubit in self.qubit_channel_map:
            self.ro_channel[qubit] = self.qubit_channel_map[qubit][0]
            self.qd_channel[qubit] = self.qubit_channel_map[qubit][1]
            self.qb_channel[qubit] = self.qubit_channel_map[qubit][2]
            self.qf_channel[qubit] = self.qubit_channel_map[qubit][3]

            if not self.qubit_instrument_map[qubit][0] is None:
                self.qrm[qubit] = self.instruments[self.qubit_instrument_map[qubit][0]]
                self.ro_port[qubit] = self.qrm[qubit].ports[
                    self.qrm[qubit].channel_port_map[self.qubit_channel_map[qubit][0]]
                ]
            if not self.qubit_instrument_map[qubit][1] is None:
                self.qdm[qubit] = self.instruments[self.qubit_instrument_map[qubit][1]]
                self.qd_port[qubit] = self.qdm[qubit].ports[
                    self.qdm[qubit].channel_port_map[self.qubit_channel_map[qubit][1]]
                ]
            if not self.qubit_instrument_map[qubit][2] is None:
                self.qfm[qubit] = self.instruments[self.qubit_instrument_map[qubit][2]]
                self.qf_port[qubit] = self.qfm[qubit].ports[
                    self.qfm[qubit].channel_port_map[self.qubit_channel_map[qubit][2]]
                ]
            if not self.qubit_instrument_map[qubit][3] is None:
                self.qbm[qubit] = self.instruments[self.qubit_instrument_map[qubit][3]]
                self.qb_port[qubit] = self.qbm[qubit].dacs[self.qubit_channel_map[qubit][3]]

    def start(self):
        if self.is_connected:
            for name in self.instruments:
                self.instruments[name].start()

    def stop(self):
        if self.is_connected:
            for name in self.instruments:
                self.instruments[name].stop()

    def disconnect(self):
        if self.is_connected:
            for name in self.instruments:
                self.instruments[name].disconnect()
            self.is_connected = False

    def execute_pulse_sequence(self, sequence: PulseSequence, nshots=None, navgs=None, relaxation_time=None, sweepers:list = []):
        if not self.is_connected:
            raise_error(RuntimeError, "Execution failed because instruments are not connected.")
        if nshots is None:
            nshots = 1
        if navgs is None:
            navgs = self.hardware_avg
        if relaxation_time is None:
            relaxation_time = self.relaxation_time

        # DEBUG: Plot Pulse Sequence
        # sequence.plot('plot.png')
        # DEBUG: sync_en
        # from qblox_instruments.qcodes_drivers.cluster import Cluster
        # cluster:Cluster = self.instruments['cluster'].device
        # for module in cluster.modules:
        #     if module.get("present"):
        #         for sequencer in module.sequencers:
        #             if sequencer.get('sync_en'):
        #                 print(f"type: {module.module_type}, sequencer: {sequencer.name}, sync_en: True")

        # Process Pulse Sequence. Assign pulses to instruments and generate waveforms & program
        instrument_pulses = {}
        roles = {}
        data = {}
        for name in self.instruments:
            roles[name] = self.settings["instruments"][name]["roles"]
            if "control" in roles[name] or "readout" in roles[name]:
                instrument_pulses[name] = sequence.get_channel_pulses(*self.instruments[name].channels)

                # until we have frequency planning use the ifs stored in the runcard to change the los
                if self.instruments[name].__class__.__name__.split('.')[-1] in ["ClusterQRM_RF", "ClusterQCM_RF", "ClusterQCM"]:
                    for port in self.instruments[name].ports:
                        _los = []
                        _ifs = []
                        port_pulses = instrument_pulses[name].get_channel_pulses(self.instruments[name]._port_channel_map[port])
                        for pulse in port_pulses:
                            if pulse.type == PulseType.READOUT:
                                _if = int(self.native_gates["single_qubit"][pulse.qubit]["MZ"]["if_frequency"])
                            elif pulse.type == PulseType.DRIVE:
                                _if = int(self.native_gates["single_qubit"][pulse.qubit]["RX"]["if_frequency"])
                            pulse._if = _if
                            _los.append(int(pulse.frequency - _if))
                            _ifs.append(int(_if))
                        if len(_los) > 1:
                            for _ in range(1, len(_los)):
                                if _los[0] != _los[_]:
                                    raise ValueError(f"Pulses:\n{instrument_pulses[name]}\nsharing the lo at device: {name} - port: {port}\ncannot be synthesised with intermediate frequencies:\n{_ifs}")
                        if len(_los) > 0:
                            self.instruments[name].ports[port].lo_frequency = _los[0]

                self.instruments[name].process_pulse_sequence(instrument_pulses[name], nshots, navgs, relaxation_time, sweepers)
                self.instruments[name].upload()
        for name in self.instruments:
            if "control" in roles[name] or "readout" in roles[name]:
                if not instrument_pulses[name].is_empty:
                    self.instruments[name].play_sequence()

        acquisition_results = {}
        for name in self.instruments:
            if "readout" in roles[name]:
                if not instrument_pulses[name].is_empty:
                    if not instrument_pulses[name].ro_pulses.is_empty:
                        results = self.instruments[name].acquire(sweepers)
                        existing_keys = set(acquisition_results.keys()) & set(results.keys())
                        for key, value in results.items():
                            if key in existing_keys:
                                acquisition_results[key].update(value)
                            else:
                                acquisition_results[key] = value

        for ro_pulse in sequence.ro_pulses:
            data[ro_pulse.serial] = ExecutionResults.from_components(*acquisition_results[ro_pulse.serial])
            data[ro_pulse.qubit] = copy.copy(data[ro_pulse.serial])
        return data

    def sweep(self, sequence, *sweepers, nshots=1024, average=True, relaxation_time=None):

        id_results = {}
        map_id_serial = {}
        sequence_copy = sequence.copy()

        if average:
            navgs = nshots
            nshots = 1
        else:
            navgs = 1

        sweepers_copy = []
        for sweeper in sweepers:
            sweepers_copy.append(Sweeper(
                parameter=sweeper.parameter,
                values=sweeper.values,
                pulses=[sequence_copy[sequence_copy.index(pulse)] for pulse in sweeper.pulses],
                qubits=sweeper.qubits
                 ))

        for pulse in sequence_copy.ro_pulses:
            map_id_serial[pulse.id] = pulse.serial
            id_results[pulse.id] = ExecutionResults.from_components(np.array([]), np.array([]))
            id_results[pulse.qubit] = id_results[pulse.id]

        # for-loop-based sweepers
        # self._sweep_recursion_for_loops(
        #     sequence_copy,
        #     *tuple(sweepers_copy),
        #     results=id_results,
        #     nshots=nshots,
        #     navgs=navgs,
        #     average=average,
        #     relaxation_time=relaxation_time
        # )
        
        # rt-based sweepers
        result = self.execute_pulse_sequence(sequence_copy, nshots, navgs, relaxation_time, sweepers_copy)
        for pulse in sequence_copy.ro_pulses:
            id_results[pulse.id] += result[pulse.serial]
            id_results[pulse.qubit] = id_results[pulse.id]

        
        serial_results = {}
        for pulse in sequence_copy.ro_pulses:
            serial_results[map_id_serial[pulse.id]] = id_results[pulse.id]
            serial_results[pulse.qubit] = id_results[pulse.id]
        return serial_results

    def _sweep_recursion_for_loops(
        self,
        sequence,
        *sweepers,
        results,
        nshots,
        navgs,
        average=False,
        relaxation_time=None,
    ):
        sweeper = sweepers[0]

        initial = {}
        if sweeper.parameter is Parameter.attenuation:
            for qubit in sweeper.qubits:
                initial[qubit] = self.get_attenuation(qubit)
        elif sweeper.parameter is Parameter.gain:
            initial = {}
            for qubit in sweeper.qubits:
                initial[qubit] = self.get_gain(qubit)
        elif sweeper.parameter is Parameter.bias:
            initial = {}
            for qubit in sweeper.qubits:
                initial[qubit] = self.get_bias(qubit)
        elif sweeper.parameter is Parameter.frequency:
            initial = {}
            for pulse in sweeper.pulses:
                initial[pulse.id] = pulse.frequency
        elif sweeper.parameter is Parameter.lo_frequency:
            initial = {}
            for pulse in sweeper.pulses:
                if pulse.type == PulseType.READOUT:
                    initial[pulse.id] = self.get_lo_readout_frequency(pulse.qubit)
                elif pulse.type == PulseType.DRIVE:
                    initial[pulse.id] = self.get_lo_readout_frequency(pulse.qubit)
        elif sweeper.parameter is Parameter.amplitude:
            initial = {}
            for pulse in sweeper.pulses:
                initial[pulse.id] = pulse.amplitude

        # perform sweep recursively
        for value in sweeper.values:
            if sweeper.parameter is Parameter.attenuation:
                for qubit in sweeper.qubits:
                    self.set_attenuation(qubit, initial[qubit] + value)
            elif sweeper.parameter is Parameter.gain:
                for qubit in sweeper.qubits:
                    self.set_gain(qubit, initial[qubit] + value)
            elif sweeper.parameter is Parameter.bias:
                for qubit in sweeper.qubits:
                    self.set_bias(qubit, initial[qubit] + value)
            elif sweeper.parameter is Parameter.frequency:
                for pulse in sweeper.pulses:
                    pulse.frequency = initial[pulse.id] + value
            elif sweeper.parameter is Parameter.lo_frequency:
                for pulse in sweeper.pulses:
                    if pulse.type == PulseType.READOUT:
                        self.set_lo_readout_frequency(initial[pulse.id] + value)
                    elif pulse.type == PulseType.DRIVE:
                        self.set_lo_readout_frequency(initial[pulse.id] + value)
            elif sweeper.parameter is Parameter.amplitude:
                for pulse in sweeper.pulses:
                    pulse.amplitude = initial[pulse.id] + value

            if len(sweepers)>1:
                self._sweep_recursion(
                    sequence,
                    *sweepers[1:],
                    results=results,
                    nshots=nshots,
                    average=average,
                    relaxation_time=relaxation_time,
                )
            else:
                result = self.execute_pulse_sequence(sequence, nshots, navgs, relaxation_time)
                for pulse in sequence.ro_pulses:
                    results[pulse.id] += result[pulse.serial].compute_average() if average else result[pulse.serial]
                    results[pulse.qubit] = results[pulse.id]


    def measure_fidelity(self, qubits=None, nshots=None):
        self.reload_settings()
        if not qubits:
            qubits = self.qubits
        results = {}
        for qubit in qubits:
            self.qrm[qubit].ports["i1"].hardware_demod_en = True  # required for binning
            # create exc sequence
            sequence_exc = PulseSequence()
            RX_pulse = self.create_RX_pulse(qubit, start=0)
            ro_pulse = self.create_qubit_readout_pulse(qubit, start=RX_pulse.duration)
            sequence_exc.add(RX_pulse)
            sequence_exc.add(ro_pulse)
            amplitude, phase, i, q = self.execute_pulse_sequence(sequence_exc, nshots=nshots)[
                "demodulated_integrated_binned"
            ][ro_pulse.serial]

            iq_exc = i + 1j * q

            sequence_gnd = PulseSequence()
            ro_pulse = self.create_qubit_readout_pulse(qubit, start=0)
            sequence_gnd.add(ro_pulse)

            amplitude, phase, i, q = self.execute_pulse_sequence(sequence_gnd, nshots=nshots)[
                "demodulated_integrated_binned"
            ][ro_pulse.serial]
            iq_gnd = i + 1j * q

            iq_mean_exc = np.mean(iq_exc)
            iq_mean_gnd = np.mean(iq_gnd)
            origin = iq_mean_gnd

            iq_gnd_translated = iq_gnd - origin
            iq_exc_translated = iq_exc - origin
            rotation_angle = np.angle(np.mean(iq_exc_translated))
            # rotation_angle = np.angle(iq_mean_exc - origin)
            iq_exc_rotated = iq_exc_translated * np.exp(-1j * rotation_angle) + origin
            iq_gnd_rotated = iq_gnd_translated * np.exp(-1j * rotation_angle) + origin

            # sort both lists of complex numbers by their real components
            # combine all real number values into one list
            # for each item in that list calculate the cumulative distribution
            # (how many items above that value)
            # the real value that renders the biggest difference between the two distributions is the threshold
            # that is the one that maximises fidelity

            real_values_exc = iq_exc_rotated.real
            real_values_gnd = iq_gnd_rotated.real
            real_values_combined = np.concatenate((real_values_exc, real_values_gnd))
            real_values_combined.sort()

            cum_distribution_exc = [
                sum(map(lambda x: x.real >= real_value, real_values_exc)) for real_value in real_values_combined
            ]
            cum_distribution_gnd = [
                sum(map(lambda x: x.real >= real_value, real_values_gnd)) for real_value in real_values_combined
            ]
            cum_distribution_diff = np.abs(np.array(cum_distribution_exc) - np.array(cum_distribution_gnd))
            argmax = np.argmax(cum_distribution_diff)
            threshold = real_values_combined[argmax]
            errors_exc = nshots - cum_distribution_exc[argmax]
            errors_gnd = cum_distribution_gnd[argmax]
            fidelity = cum_distribution_diff[argmax] / nshots
            assignment_fidelity = 1 - (errors_exc + errors_gnd) / nshots / 2
            # assignment_fidelity = 1/2 + (cum_distribution_exc[argmax] - cum_distribution_gnd[argmax])/nshots/2
            results[qubit] = ((rotation_angle * 360 / (2 * np.pi)) % 360, threshold, fidelity, assignment_fidelity)
        return results
