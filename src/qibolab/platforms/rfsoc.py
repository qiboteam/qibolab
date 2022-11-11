from qick import AveragerProgram, QickSoc

from qibolab.platforms.abstract import AbstractPlatform


class Program(AveragerProgram):
    def __init__(self, soc, cfg, sequence):
        super().__init__(soc, cfg)
        self.sequence = sequence
        # TODO: Move all cfg declarations in __init__

    def initialize(self):
        ro_channel = self.cfg["resonator_channel"]
        qd_channel = self.cfg["qubit_channel"]

        self.declare_gen(ch=ro_channel, nqz=1)  # Readout
        self.declare_gen(ch=qd_channel, nqz=2)  # Qubit

        # assume one drive and one ro pulse
        qd_pulse = self.sequence.qd_pulses[0]
        ro_pulse = self.sequence.ro_pulses[0]
        assert qd_pulse.channel == qd_channel
        assert ro_pulse.channel == ro_channel

        # conver pulse lengths to clock ticks
        ro_length = self.soc.us2cycles(ro_pulse.duration * 1e-3, gen_ch=ro_channel)
        qd_length = self.soc.us2cycles(qd_pulse.duration * 1e-3, gen_ch=qd_channel)

        for ch in [0, 1]:  # configure the readout lengths and downconversion frequencies
            self.declare_readout(ch=ch, length=ro_length, freq=ro_pulse.frequency * 1e-6, gen_ch=ro_channel)

        # convert frequencies to dac register value
        # TODO: Why are frequencies converted after declaring the readout?
        ro_frequency = self.freq2reg(ro_pulse.frequency * 1e-6, gen_ch=ro_channel, ro_ch=0)
        qd_frequency = self.freq2reg(qd_pulse.frequency * 1e-6, gen_ch=qd_channel)

        # calculate pulse gain from amplitude
        max_gain = self.cfg["max_gain"]
        qd_gain = qd_pulse.amplitude * max_gain
        ro_gain = ro_pulse.amplitude * max_gain

        # add qubit and readout pulses to respective channels
        # TODO: Register proper shapes and phases to pulses
        self.set_pulse_registers(
            ch=qd_channel,
            style="const",
            freq=qd_frequency,
            phase=qd_pulse.phase,
            gain=qd_gain,
            length=qd_pulse.duration,
        )
        self.set_pulse_registers(
            ch=ro_channel,
            style="const",
            freq=ro_frequency,
            phase=ro_pulse.phase,
            gain=ro_gain,
            length=ro_pulse.duration,
        )

        self.synci(200)

    def body(self):
        ro_channel = self.cfg["resonator_channel"]
        qd_channel = self.cfg["qubit_channel"]
        delay_before_readout = self.cfg["delay_before_readout"]
        delay_before_readout = self.us2cycles(delay_before_readout * 1e-3)

        # play drive pulse
        self.pulse(ch=qd_channel)
        # align channels and wait some time (defined in the runcard)
        self.sync_all(delay_before_readout)

        # trigger measurement, play measurement pulse, wait for qubit to relax
        syncdelay = self.us2cycles(self.cfg["relax_delay"] * 1e-3)
        self.measure(pulse_ch=self.cfg["resonator_channel"], adcs=[0, 1], wait=True, syncdelay=syncdelay)


class RFSocPlatform(AbstractPlatform):
    def __init__(self, name, runcard):
        log.info(f"Loading platform {name} from runcard {runcard}")
        self.name = name
        self.runcard = runcard
        self.is_connected = False
        # Load platform settings
        with open(runcard) as file:
            self.settings = yaml.safe_load(file)

        self.nqubits = self.settings.get("nqubits")
        if self.nqubits == 1:
            self.resonator_type = "3D"
        else:
            self.resonator_type = "2D"

        self.resonator_channel = self.settings.get("hardware_config").get("resonator_channel")
        self.qubit_channel = self.settings.get("hardware_config").get("qubit_channel")

        self.soc = QickSoc()
        self.cfg = {"reps": self.settings.get("reps")}
        self.cfg.update(self.settings.get("hardware_config"))
        self.cfg.update(self.settings.get("readout_config"))
        self.cfg.update(self.settings.get("qubit_config"))
        # self.cfg["sigma"] = self.soc.us2cycles(0.025, gen_ch=qubit_channel)

    def reload_settings(self):
        raise NotImplementedError

    def run_calibration(self, show_plots=False):  # pragma: no cover
        raise NotImplementedError

    def connect(self):
        raise NotImplementedError

    def setup(self):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def disconnect(self):
        raise NotImplementedError

    def execute_pulse_sequence(self, sequence, nshots=None):
        program = Program(self.soc, self.cfg, sequence)
        # TODO: Pass optional values: ``threshold``
        avgi, avgq = program.acquire(self.soc, load_pulses=True, progress=False, debug=False)
        return avgi, avgq
