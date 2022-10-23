from qibolab.instruments.abstract import AbstractInstrument
from zhinst.ziPython import ziDAQServer
from zhinst.toolkit import Session, SHFQAChannelMode, Waveforms


class ReadoutSequence:

    def __init__(self, pulses, nshots):
        self.nshots = nshots
        self.channel = pulses[0].channel

        # TODO: Handle more than one pulses here
        self.waveforms = Waveforms()
        pulse = pulses[0]
        wave = pulse.modulated_waveform_i.data + 1j * pulse.modulated_waveform_q.data
        self.waveforms.assign_waveform(
            slot=0,
            wave1=wave
        )

        # TODO: Use hardware averages here
        self.program = """
            repeat(%d) {{
                repeat(1) {{
                    waitDigTrigger(1);
                    startQA(QA_GEN_ALL, QA_INT_ALL, true, 0, 0x0);
                }}
            }}
        """ % nshots


class SHFQC_QA(AbstractInstrument):

    def __init__(self, name, address):
        super().__init__(name, address)
        self.device = None

        self.channels = None
        self.lo_frequency = None # in units of Hz
        self.gain = None
        self.input_range = None # in units of dBm
        self.output_range = None # in units of dBm

        self._latest_sequence = None

    def connect(self):
        session = Session('localhost')
        self.device = session.connect_device(self.address)

    def setup(self, **kwargs):
        self.channels = kwargs.get("channels")
        self.gain = kwargs.get("gain")
        self.lo_frequency = kwargs.get("lo_frequency")
        self.input_range = kwargs.get("input_range")
        self.output_range = kwargs.get("output_range")

    def start(self):
        pass

    def stop(self):
        # TODO: Remember to stop sequencer here
        pass

    def disconnect(self):
        session = Session('localhost')
        session.disconnect_device(self.address)

    def process_pulse_sequence(self, instrument_pulses, nshots, repetition_duration):
        # configure inputs and outputs
        #NUM_AVERAGES = 100
        #MODE_AVERAGES = 0 # 0: cyclic; 1: sequential;
        #INTEGRATION_TIME = PULSE_DURATION # in units of second
        self._latest_sequence = ReadoutSequence(instrument_pulses, nshots)

    def upload(self):
        channel = self.device.qachannels[self._latest_sequence.channel]
        channel.configure_channel(
            center_frequency=self.lo_frequency,
            input_range=self.input_range,
            output_range=self.output_range,
            mode=SHFQAChannelMode.READOUT, # READOUT or SPECTROSCOPY
        )
        # TODO: Perhaps we need to set gain somewhere (?)
        #print(daq.getDouble("/dev12146/qachannels/0/oscs/0/gain"))
        #print(daq.getDouble("/dev12146/qachannels/0/oscs/0/freq") / 1e9)

        channel.input.on(1)
        channel.output.on(1)

        # upload readout pulses and integration weights to waveform memory
        waveforms = self._latest_sequence.waveforms
        channel.generator.clearwave() # clear all readout waveforms
        channel.generator.write_to_waveform_memory(waveforms)
        #channel.readout.integration.clearweight() # clear all integration weights
        #channel.readout.write_integration_weights(
        #    weights=weights,
            # compensation for the delay between generator output and input of the integration unit
        #    integration_delay=220e-9
        #)

        # configure sequencer
        channel.generator.configure_sequencer_triggering(
            aux_trigger="software_trigger0", # chanNtriginM, chanNseqtrigM, chanNrod
            play_pulse_delay=0, # 0s delay between startQA trigger and the readout pulse
        )

        channel.generator.load_sequencer_program(self._latest_sequence.program)

    def play_sequence_and_acquire(self):
        channel = self.device.qachannels[self._latest_sequence.channel]
        channel.readout.run() # enable QA Result Logger
        channel.generator.enable_sequencer(single=True)
        self.device.start_continuous_sw_trigger(
            num_triggers=self._latest_sequence.nshots, wait_time=2e-3
        )
        return channel.readout.read()
