import copy

import numpy as np
from qibo.config import raise_error

from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence, ReadoutPulse
from qibolab.result import ExecutionResults


class MultiqubitPlatform(AbstractPlatform):
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

    def set_current(self, qubit, current):
        self.qb_port[qubit].current = current

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
            data[ro_pulse.serial] = ExecutionResults.from_components(*acquisition_results[key])
            data[ro_pulse.qubit] = copy.copy(data[ro_pulse.serial])
        return data

    def sweep(self, sequence, *sweepers, nshots=1024, average=True):
        original = copy.deepcopy(sequence)
        map_old_new_pulse = {pulse: pulse.serial for pulse in sequence.ro_pulses}
        results = {}
        if len(sweepers) == 1:
            # single sweeper
            sweeper = sweepers[0]
            initial_pulses = sweeper.pulses
            # Remove initial pulses
            for pulse in sweeper.pulses:
                sequence.remove(pulse)
            for value in sweeper.values:
                for pulse in copy.deepcopy(sweeper.pulses):
                    shifted_pulses = []
                    if sweeper.parameter == "frequency":
                        setattr(pulse, sweeper.parameter, getattr(original[pulse.qubit], sweeper.parameter) + value)
                    elif sweeper.parameter == "amplitude":
                        if max(sweeper.values) > 1:
                            self.set_attenuation(pulse.qubit, value)
                        else:
                            setattr(pulse, sweeper.parameter, value)
                    elif sweeper.paramter == "gain":
                        self.set_gain(pulse.qubit, value)
                    else:
                        setattr(pulse, sweeper.parameter, value)
                    if isinstance(pulse, ReadoutPulse):
                        map_old_new_pulse[original[pulse.qubit]] = pulse.serial

                    # Add pulse with parameter shifted
                    sequence.add(pulse)
                    shifted_pulses.append(pulse)

                result = self.execute_pulse_sequence(sequence, nshots)

                # remove shifted pulses from sequence
                for shifted_pulse in shifted_pulses:
                    sequence.remove(shifted_pulse)

                # colllect result and append to original pulse
                for old, new_serial in map_old_new_pulse.items():
                    result[new_serial].i = result[new_serial].i.mean()
                    result[new_serial].q = result[new_serial].q.mean()
                    if old.serial in results:
                        results[old.serial] += result[new_serial]
                    else:
                        results[old.serial] = result[new_serial]
                        results[old.qubit] = copy.copy(results[old.serial])

            for pulse in initial_pulses:
                sequence.add(pulse)
        elif len(sweepers) == 2:
            # 2 sweepers simultaneously
            initial_pulses = sweepers[0].pulses + sweepers[1].pulses
            for pulse in initial_pulses:
                sequence.remove(pulse)
            for value1 in sweepers[0].values:
                for value2 in sweepers[1].values:
                    for sweeper in sweepers:
                        for pulse in copy.deepcopy(sweeper.pulses):
                            shifted_pulses = []
                            value = value1 if sweeper == sweepers[0] else value2
                            if sweeper.parameter == "frequency":
                                setattr(
                                    pulse, sweeper.parameter, getattr(original[pulse.qubit], sweeper.parameter) + value
                                )
                            elif sweeper.parameter == "amplitude":
                                if max(sweeper.values) > 1:
                                    self.set_attenuation(pulse.qubit, value)
                                else:
                                    setattr(pulse, sweeper.parameter, value)
                            elif sweeper.paramter == "gain":
                                self.set_gain(pulse.qubit, value)
                            else:
                                setattr(pulse, sweeper.parameter, value)
                            if isinstance(pulse, ReadoutPulse):
                                map_old_new_pulse[original[pulse.qubit]] = pulse.serial

                            # Add pulse with parameter shifted
                            sequence.add(pulse)
                            shifted_pulses.append(pulse)

                    result = self.execute_pulse_sequence(sequence, nshots)

                    # remove shifted pulses from sequence
                    for shifted_pulse in shifted_pulses:
                        sequence.remove(shifted_pulse)
                    for old, new_serial in map_old_new_pulse.items():
                        result[new_serial].i = result[new_serial].i.mean()
                        result[new_serial].q = result[new_serial].q.mean()
                        if old.serial in results:
                            results[old.serial] += result[new_serial]
                        else:
                            results[old.serial] = result[new_serial]
                            results[old.qubit] = copy.copy(results[old.serial])

            for pulse in initial_pulses:
                sequence.add(pulse)
        else:
            raise_error("Qblox platform supports can support up to 2 sweepers.")

        return results

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
