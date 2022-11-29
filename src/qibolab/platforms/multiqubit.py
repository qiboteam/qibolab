from qibo.config import raise_error

from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence


class MultiqubitPlatform(AbstractPlatform):
    def run_calibration(self):
        raise_error(NotImplementedError)

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
        for name in self.instruments:
            roles[name] = self.settings["instruments"][name]["roles"]
            if "control" in roles[name] or "readout" in roles[name]:
                instrument_pulses[name] = sequence.get_channel_pulses(*self.instruments[name].channels)
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
                        acquisition_results.update(self.instruments[name].acquire())

        return acquisition_results
    
    def measure_fidelity(self, nshots=None):
        import numpy as np
        self.reload_settings()
        results = {}
        for qubit in self.qubits:
            self.qrm[qubit].ports['i1'].hardware_demod_en = True # required for binning
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

            real_values_exc = [x.real for x in iq_exc_rotated]
            real_values_gnd = [x.real for x in iq_gnd_rotated]
            real_values_combined = real_values_exc + real_values_gnd
            real_values_combined.sort()

            cum_distribution_exc = [sum(map(lambda x : x.real >= real_value, real_values_exc)) for real_value in real_values_combined]
            cum_distribution_gnd = [sum(map(lambda x : x.real >= real_value, real_values_gnd)) for real_value in real_values_combined]
            cum_distribution_diff = np.abs(np.array(cum_distribution_exc) - np.array(cum_distribution_gnd))
            argmax = np.argmax(cum_distribution_diff)
            threshold = real_values_combined[argmax]
            errors_exc = nshots - cum_distribution_exc[argmax]
            errors_gnd = cum_distribution_gnd[argmax]
            fidelity = cum_distribution_diff[argmax]/nshots
            assignment_fidelity = 1 - (errors_exc + errors_gnd)/nshots/2
            #assignment_fidelity = 1/2 + (cum_distribution_exc[argmax] - cum_distribution_gnd[argmax])/nshots/2
            results[qubit] = (rotation_angle, threshold, fidelity, assignment_fidelity)
        return results
