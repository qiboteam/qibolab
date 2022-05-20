from qibo.config import raise_error
from qibolab.platforms.abstract import AbstractPlatform
import importlib

class MultiqubitPlatform(AbstractPlatform):
    def __init__(self, name, runcard):
        super().__init__(name, runcard)
        
        self.instrument_settings = self.settings['instruments']
        self.instruments = {}

        for name in self.instrument_settings:
            lib = self.instrument_settings[name]['lib']
            i_class = self.instrument_settings[name]['class']
            ip = self.instrument_settings[name]['ip']
            InstrumentClass = getattr(importlib.import_module(f"qibolab.instruments.{lib}"), i_class)
            instance = InstrumentClass(name, ip)
            # instance.__dict__.update(self.settings['shared_settings'])
            self.instruments[name] = instance
            setattr(self, name, instance)    

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
        pulse_sequence.sortsort(key=lambda pulse: pulse.start) 
        pulse_sequence_duration = pulse_sequence[-1].start + pulse_sequence[-1].duration

        # Process Pulse Sequence. Generate Waveforms and Program
        channel_pulses = {}
        for channel in self.channels:
            channel_pulses[channel] = []
        for pulse in pulse_sequence:
            if pulse.channel in self.channels:
                channel_pulses[pulse.channel].append(pulse)
            else:
                raise_error(RuntimeError, f"{self.name} does not have channel {pulse.channel}, only:\n{self.channels}.")

        instrument_pulses = {}
        for name in self.instruments:
            instrument_pulses[name] = {}
            if self.instrument_settings[name]['class'] in ['ClusterQRM', 'ClusterQCM']:
                for channel in self.instruments[name].channel_port_map.keys():
                    if channel in self.channels:
                        instrument_pulses[name][channel] = channel_pulses[channel]
                self.instruments[name].process_pulse_sequence(instrument_pulses[name], nshots)
                self.instruments[name].upload()

        for name in self.instruments:
            if instrument_pulses[name] is not None:
                if self.instrument_settings[name]['class'] in ['ClusterQCM']:
                    self.instruments[name].play_sequence()

        for name in self.instruments:
            if instrument_pulses[name] is not None:
                if self.instrument_settings[name]['class'] in ['ClusterQRM']:
                    acquisition_results = self.instruments[name].play_sequence_and_acquire()

        return acquisition_results
