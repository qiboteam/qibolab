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
        # sequence.plot()

        # Process Pulse Sequence. Assign pulses to instruments and generate waveforms & program
        instrument_pulses = {}
        roles = {}
        for name in self.instruments:
            roles[name] = self.settings["instruments"][name]["roles"]
            if "readout" in roles[name] or "control" in roles[name]:
                instrument_pulses[name] = sequence.get_channel_pulses(*self.instruments[name].channels)
                self.instruments[name].process_pulse_sequence(instrument_pulses[name], nshots, self.repetition_duration)
                self.instruments[name].upload()

        for name in self.instruments:
            if "control" in roles[name]:
                if not instrument_pulses[name].is_empty:
                    self.instruments[name].play_sequence()

        acquisition_results = {}
        for name in self.instruments:
            if "readout" in roles[name]:
                if not instrument_pulses[name].is_empty:
                    if not instrument_pulses[name].ro_pulses.is_empty:
                        acquisition_results.update(self.instruments[name].play_sequence_and_acquire())
                    else:
                        self.instruments[name].play_sequence()
        return acquisition_results
