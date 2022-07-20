from qibo.config import raise_error
from qibolab.platforms.abstract import AbstractPlatform


class MultiqubitPlatform(AbstractPlatform):

    def run_calibration(self):
        raise_error(NotImplementedError)

    def execute_pulse_sequence(self, sequence, nshots=None):
        if not self.is_connected:
            raise_error(RuntimeError, "Execution failed because instruments are not connected.")
        if nshots is None:
            nshots = self.hardware_avg

        # PreProcess Pulse Sequence
        # Sort by pulse start
        pulse_sequence = sequence.pulses
        pulse_sequence.sort(key=lambda pulse: pulse.start)
        if len(pulse_sequence) > 0:
            pulse_sequence_duration = pulse_sequence[-1].start + pulse_sequence[-1].duration


        # Process Pulse Sequence. Assign pulses to channels
        channel_pulses = {}
        for channel in self.channels:
            channel_pulses[channel] = []
        for pulse in pulse_sequence:
            if pulse.channel in self.channels:
                channel_pulses[pulse.channel].append(pulse)
            else:
                raise_error(RuntimeError, f"{self.name} does not have channel {pulse.channel}, only:\n{self.channels}.")

        # Process Pulse Sequence. Assign pulses to instruments and generate waveforms & program
        instrument_pulses = {}
        for name in self.instruments:
            instrument_pulses[name] = {}
            if 'QCM' in self.settings['instruments'][name]['class'] or 'QRM' in self.settings['instruments'][name]['class']:
                for channel in self.instruments[name].channel_port_map:
                    if channel in self.channels:
                        instrument_pulses[name][channel] = channel_pulses[channel]
                self.instruments[name].process_pulse_sequence(instrument_pulses[name], nshots)
                self.instruments[name].upload()

        for name in self.instruments:
            if instrument_pulses[name] is not None:
                if 'QCM' in self.settings['instruments'][name]['class']:
                    self.instruments[name].play_sequence()

        acquisition_results = None
        for name in self.instruments:
            if instrument_pulses[name] is not None:
                if 'QRM' in self.settings['instruments'][name]['class']:
                    if 'ro' in [p.type for ch in self.instruments[name].channel_port_map for p in instrument_pulses[name][ch]]:
                        acquisition_results = self.instruments[name].play_sequence_and_acquire()
                    else:
                        self.instruments[name].play_sequence()

        return acquisition_results
