from qibo.config import raise_error

from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence


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

    def get_attenuation(self, qubit):
        return self.ro_port[qubit].attenuation

    def get_current(self, qubit):
        return self.qb_port[qubit].current

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
        self.ro_channel = {}
        self.qd_channel = {}
        self.qf_channel = {}
        self.qrm = {}
        self.qcm = {}
        self.qbm = {}
        self.ro_port = {}
        self.qd_port = {}
        self.qf_port = {}
        for qubit in self.qubit_channel_map:
            self.ro_channel[qubit] = self.qubit_channel_map[qubit][0]
            self.qd_channel[qubit] = self.qubit_channel_map[qubit][1]
            self.qf_channel[qubit] = self.qubit_channel_map[qubit][2]

            if not self.qubit_instrument_map[qubit][0] is None:
                self.qrm[qubit] = self.instruments[self.qubit_instrument_map[qubit][0]]
                self.ro_port[qubit] = self.qrm[qubit].ports[
                    self.qrm[qubit].channel_port_map[self.qubit_channel_map[qubit][0]]
                ]
            if not self.qubit_instrument_map[qubit][1] is None:
                self.qcm[qubit] = self.instruments[self.qubit_instrument_map[qubit][1]]
                self.qd_port[qubit] = self.qcm[qubit].ports[
                    self.qcm[qubit].channel_port_map[self.qubit_channel_map[qubit][1]]
                ]
            if not self.qubit_instrument_map[qubit][2] is None:
                self.qbm[qubit] = self.instruments[self.qubit_instrument_map[qubit][2]]
                self.qf_port[qubit] = self.qbm[qubit].dacs[self.qubit_channel_map[qubit][2]]

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

        changed = {}
        for name in self.instruments:
            roles[name] = self.settings["instruments"][name]["roles"]
            if "control" in roles[name] or "readout" in roles[name]:
                instrument_pulses[name] = sequence.get_channel_pulses(*self.instruments[name].channels)
                # Change pulses frequency to if and correct lo accordingly (before was done in qibolab)

                if "readout" in roles[name]:
                    for pulse in instrument_pulses[name]:
                        if abs(pulse.frequency) > 300e6:
                            # TODO: implement algorithm to find correct LO
                            if_frequency = self.native_gates["single_qubit"][pulse.qubit]["MZ"]["frequency"]
                            self.set_lo_readout_frequency(pulse.qubit, pulse.frequency - if_frequency)
                            pulse.frequency = if_frequency
                            changed[pulse.serial] = True
                elif "control" in roles[name]:
                    for pulse in instrument_pulses[name]:
                        if abs(pulse.frequency) > 500e6:
                            # TODO: implement algorithm to find correct LO
                            if_frequency = self.native_gates["single_qubit"][pulse.qubit]["RX"]["frequency"]
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
                        # if abs(pulse.frequency) > 300e6:
                        pulse.frequency += self.get_lo_readout_frequency(pulse.qubit)
            if "control" in roles[name]:
                instrument_pulses[name] = sequence.get_channel_pulses(*self.instruments[name].channels)
                for pulse in instrument_pulses[name]:
                    if pulse.serial in changed:
                        pulse.frequency += self.get_lo_drive_frequency(pulse.qubit)

        return acquisition_results

    def measure_fidelity(self, qubits=None, nshots=None):
        import numpy as np

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
