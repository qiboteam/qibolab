import copy

import numpy as np
import yaml
from qibo.config import log, raise_error

from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence, PulseType
from qibolab.result import ExecutionResults
from qibolab.sweeper import Parameter


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
        self.repetition_duration = self.settings["settings"]["repetition_duration"]

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
        self.qb_port[qubit].current = bias

    def get_attenuation(self, qubit):
        return self.ro_port[qubit].attenuation

    def get_bias(self, qubit):
        return self.qb_port[qubit].current

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

    def execute_pulse_sequence(self, sequence: PulseSequence, nshots=None):
        if not self.is_connected:
            raise_error(RuntimeError, "Execution failed because instruments are not connected.")
        if nshots is None:
            nshots = self.hardware_avg

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
        ro_pulses = {}
        changed = {}
        data = {}
        for name in self.instruments:
            roles[name] = self.settings["instruments"][name]["roles"]
            if "control" in roles[name] or "readout" in roles[name]:
                instrument_pulses[name] = sequence.get_channel_pulses(*self.instruments[name].channels)
                # Change pulses frequency to if and correct lo accordingly (before was done in qibolab)

                if "readout" in roles[name]:
                    for pulse in instrument_pulses[name]:
                        ro_pulses[pulse.serial] = pulse
                        if abs(pulse.frequency) > self.instruments[name].FREQUENCY_LIMIT:
                            # TODO: implement algorithm to find correct LO
                            if_frequency = self.native_gates["single_qubit"][pulse.qubit]["MZ"]["if_frequency"]
                            self.set_lo_readout_frequency(pulse.qubit, pulse.frequency - if_frequency)
                            pulse.frequency = if_frequency
                            changed[pulse.serial] = True
                elif "control" in roles[name]:
                    for pulse in instrument_pulses[name]:
                        if abs(pulse.frequency) > self.instruments[name].FREQUENCY_LIMIT:
                            # TODO: implement algorithm to find correct LO
                            if_frequency = self.native_gates["single_qubit"][pulse.qubit]["RX"]["if_frequency"]
                            self.set_lo_drive_frequency(pulse.qubit, pulse.frequency - if_frequency)
                            pulse.frequency = if_frequency
                            changed[pulse.serial] = True

                self.instruments[name].process_pulse_sequence(instrument_pulses[name], nshots, self.repetition_duration)
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
                        if all([pulse.serial in changed for pulse in instrument_pulses[name].ro_pulses]):
                            # FIXME: for precision sweep in resonator spectroscopy
                            # change necessary to perform precision sweep
                            # TODO: move this to instruments (ask Alvaro)
                            # TODO: check if this will work with multiplex
                            for sequencers in self.instruments[name]._sequencers.values():
                                for sequencer in sequencers:
                                    sequencer.pulses = instrument_pulses[name].ro_pulses
                        results = self.instruments[name].acquire()
                        existing_keys = set(acquisition_results.keys()) & set(results.keys())
                        for key, value in results.items():
                            if key in existing_keys:
                                acquisition_results[key].update(value)
                            else:
                                acquisition_results[key] = value

        # change back the frequency of the pulses
        for name in self.instruments:
            roles[name] = self.settings["instruments"][name]["roles"]
            if "readout" in roles[name]:
                instrument_pulses[name] = sequence.get_channel_pulses(*self.instruments[name].channels)
                for pulse in instrument_pulses[name]:
                    if pulse.serial in changed:
                        pippo = acquisition_results[pulse.serial]
                        # if abs(pulse.frequency) > 300e6:
                        pulse.frequency += self.get_lo_readout_frequency(pulse.qubit)
                        acquisition_results[pulse.serial] = pippo
            if "control" in roles[name]:
                instrument_pulses[name] = sequence.get_channel_pulses(*self.instruments[name].channels)
                for pulse in instrument_pulses[name]:
                    if pulse.serial in changed:
                        pulse.frequency += self.get_lo_drive_frequency(pulse.qubit)

        for ro_pulse in ro_pulses.values():
            data[ro_pulse.serial] = ExecutionResults.from_components(*acquisition_results[ro_pulse.serial])
            data[ro_pulse.qubit] = copy.copy(data[ro_pulse.serial])
        return data

    def sweep(self, sequence, *sweepers, nshots=1024, average=True, relaxation_time=None):
        results = {}
        sweeper_pulses = {}

        # create copy of the sequence
        copy_sequence = copy.deepcopy(sequence)
        map_original_shifted = {pulse: pulse.serial for pulse in copy.deepcopy(copy_sequence).ro_pulses}

        # create dictionary containing pulses for each sweeper that point to the same original sequence
        # which is copy_sequence
        for sweeper in sweepers:
            if sweeper.pulses is not None:
                sweeper_pulses[sweeper.parameter] = {
                    pulse.serial: pulse for pulse in copy_sequence if pulse in sweeper.pulses
                }

        # perform sweeping recursively
        self._sweep_recursion(
            copy_sequence,
            copy.deepcopy(sequence),
            *sweepers,
            nshots=nshots,
            average=average,
            relaxation_time=relaxation_time,
            results=results,
            sweeper_pulses=sweeper_pulses,
            map_original_shifted=map_original_shifted,
        )
        return results

    def _sweep_recursion(
        self,
        sequence,
        original_sequence,
        *sweepers,
        nshots=1024,
        average=True,
        relaxation_time=None,
        results=None,
        sweeper_pulses=None,
        map_original_shifted=None,
    ):
        sweeper = sweepers[0]

        # store values before starting to sweep
        if sweeper.pulses is not None:
            original_value = self._save_original_value(sweeper, sweeper_pulses)

        # perform sweep recursively
        for value in sweeper.values:
            self._update_pulse_sequence_parameters(
                sweeper, sweeper_pulses, original_sequence, map_original_shifted, value
            )
            if len(sweepers) > 1:
                self._sweep_recursion(
                    sequence,
                    original_sequence,
                    *sweepers[1:],
                    nshots=nshots,
                    average=average,
                    relaxation_time=relaxation_time,
                    results=results,
                    sweeper_pulses=sweeper_pulses,
                    map_original_shifted=map_original_shifted,
                )
            else:
                new_sequence = copy.deepcopy(sequence)
                result = self.execute_pulse_sequence(new_sequence, nshots)

                # colllect result and append to original pulse
                for original_pulse, new_serial in map_original_shifted.items():
                    acquisition = result[new_serial].compute_average() if average else result[new_serial]

                    if results:
                        results[original_pulse.serial] += acquisition
                        results[original_pulse.qubit] += acquisition
                    else:
                        results[original_pulse.serial] = acquisition
                        results[original_pulse.qubit] = copy.copy(results[original_pulse.serial])

        # restore initial value of the pul
        if sweeper.pulses is not None:
            self._restore_initial_value(sweeper, sweeper_pulses, original_value)

    def _save_original_value(self, sweeper, sweeper_pulses):
        """Helper method for _sweep_recursion"""
        original_value = {}
        pulses = sweeper_pulses[sweeper.parameter]
        # save original value of the parameter swept
        for pulse in pulses:
            if sweeper.parameter is Parameter.attenuation:
                original_value[pulse] = self.get_attenuation(pulses[pulse].qubit)
            elif sweeper.parameter is Parameter.gain:
                original_value[pulse] = self.get_gain(pulses[pulse].qubit)
            elif sweeper.parameter is Parameter.bias:
                original_value[pulse] = self.get_bias(pulses[pulse].qubit)
            else:
                original_value[pulse] = getattr(pulses[pulse], sweeper.parameter.name)

        return original_value

    def _restore_initial_value(self, sweeper, sweeper_pulses, original_value):
        """Helper method for _sweep_recursion"""
        pulses = sweeper_pulses[sweeper.parameter]
        for pulse in pulses:
            if sweeper.parameter is Parameter.attenuation:
                self.set_attenuation(pulses[pulse].qubit, original_value[pulse])
            elif sweeper.parameter is Parameter.gain:
                self.set_gain(pulses[pulse].qubit, original_value[pulse])
            elif sweeper.parameter is Parameter.bias:
                self.set_bias(pulses[pulse].qubit, original_value[pulse])
            else:
                setattr(pulses[pulse], sweeper.parameter.name, original_value[pulse])

    def _update_pulse_sequence_parameters(
        self, sweeper, sweeper_pulses, original_sequence, map_original_shifted, value
    ):
        """Helper method for _sweep_recursion"""
        if sweeper.pulses is not None:
            pulses = sweeper_pulses[sweeper.parameter]
            for pulse in pulses:
                if sweeper.parameter is Parameter.frequency:
                    if pulses[pulse].type is PulseType.READOUT:
                        value += self.qubits[pulses[pulse].qubit].readout_frequency
                    else:
                        value += self.qubits[pulses[pulse].qubit].drive_frequency
                    setattr(pulses[pulse], sweeper.parameter.name, value)
                elif sweeper.parameter is Parameter.amplitude:
                    if pulses[pulse].type is PulseType.READOUT:
                        current_amplitude = self.native_gates["single_qubit"][pulses[pulse].qubit]["MZ"]["amplitude"]
                    else:
                        current_amplitude = self.native_gates["single_qubit"][pulses[pulse].qubit]["RX"]["amplitude"]
                    setattr(pulses[pulse], sweeper.parameter.name, float(current_amplitude * value))
                if pulses[pulse].type is PulseType.READOUT:
                    to_modify = [
                        pulse1 for pulse1 in original_sequence.ro_pulses if pulse1.qubit == pulses[pulse].qubit
                    ]
                    if to_modify:
                        map_original_shifted[to_modify[0]] = pulses[pulse].serial

        if sweeper.qubits is not None:
            for qubit in sweeper.qubits:
                if sweeper.parameter is Parameter.attenuation:
                    self.set_attenuation(qubit, value)
                elif sweeper.parameter is Parameter.gain:
                    self.set_gain(qubit, value)
                elif sweeper.parameter is Parameter.bias:
                    self.set_bias(qubit, value)

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
