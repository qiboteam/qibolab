""" RFSoC fpga driver.

Supports the following FPGA:
    RFSoC 4x2
"""
import numpy as np
from qick import AveragerProgram, QickSoc

from qibolab.instruments.abstract import AbstractInstrument, InstrumentException
from qibolab.pulses import (
    Drag,
    Gaussian,
    Pulse,
    PulseSequence,
    PulseShape,
    PulseType,
    ReadoutPulse,
    Rectangular,
)
from qibolab.result import ExecutionResults
from qibolab.sweeper import Parameter, Sweeper


class ExecutePulseSequence(AveragerProgram):
    """This qick AveragerProgram handles a qibo sequence of pulse"""

    def __init__(self, soc, cfg, sequence):
        """In this function the most important settings are defined and the sequence is transpiled.

        * set the conversion coefficients to be used for frequency and time values
        * max_gain, adc_trig_offset, max_sampling_rate are imported from cfg (runcard settings)
        * connections are defined (drive and readout channel for each qubit)

        * pulse_sequence, readouts, channels are defined to be filled by convert_sequence()

        * cfg["reps"] is set from hardware_avg
        * super.__init__
        """

        # conversion coefficients
        self.MHz = 0.000001
        self.mu_s = 0.001

        # settings
        self.max_gain = cfg["max_gain"]  # TODO redundancy
        self.adc_trig_offset = cfg["adc_trig_offset"]
        self.max_sampling_rate = cfg["sampling_rate"]
        self.relax_delay = cfg["repetition_duration"]

        # connections  (for every qubit here are defined drive and readout lines
        self.connections = {
            "0": {"drive": 1, "readout": 0, "adc_ch": 0},
        }

        self.pulse_sequence = {}
        self.readouts = []
        self.channels = []

        # fill the self.pulse_sequence and the self.readout_pulses oject
        self.soc = soc
        self.soccfg = soc  # No need for a different soc config object since qick is on board
        self.convert_sequence(sequence["pulses"])

        cfg["reps"] = sequence["nshots"]
        super().__init__(soc, cfg)

    def convert_sequence(self, sequence):
        """In this function we transpile the sequence of pulses in a form better suited for qick

        * Note that all the conversions (in different "standard" units and in registers valued are done here

        Three object are of touched by this function:
        * self.pulse_sequence is a dictionary that contains all the pulse with information regarding the pulse itself.
          To be used in initialize (set_pulse_registers) and in body (executing)

        * self.readouts is a list that contains all the readout information.
          To be used in initialize for declare_readout

        * self.channels is a list that contain channel number and nyquist zone of initialization


        Templates:
        self.pulse_sequence = {
            serial: {
                "channel":
                "type":
                "freq":
                "length":
                "phase":
                "time":
                "gain"
                "waveform":     # readout as a default value

                "shape":    # these only if drive
                "sigma":
                "name":

                "delta":    # these only if drag
                "alpha":

                "adc_trig_offset":      # these only if readout
                "wait": False
                "syncdelay": 100
            }
        }
        self.readouts = {
            0: {
                "adc_ch":
                "gen_ch":
                "length":
                "freq":
            }
        }
        self.channels = [(channel, generation), ...]

        """

        for _, pulse in sequence.items():
            pulse_dic = {}

            pulse_dic["type"] = pulse["type"]
            pulse_dic["time"] = pulse["start"]

            gen_ch, adc_ch = self.from_qubit_to_ch(
                pulse["qubit"], pulse["type"]  # if drive pulse return only gen_ch, otherwise both
            )
            pulse_dic["freq"] = self.soc.freq2reg(
                pulse["frequency"] * self.MHz,  # TODO maybe differentiate between drive and readout
                gen_ch=gen_ch,
                ro_ch=adc_ch,
            )

            length = pulse["duration"] * self.mu_s
            pulse_dic["length"] = self.soc.us2cycles(length)  # uses tproc clock now
            pulse_dic["phase"] = self.deg2reg(
                pulse["relative_phase"], gen_ch=gen_ch  # TODO maybe differentiate between drive and readout
            )

            pulse_dic["gain"] = int(pulse["amplitude"] * self.max_gain)

            if pulse_dic["type"] == "qd":
                pulse_dic["ch"] = gen_ch

                pulse_dic["waveform"] = pulse["shape"]  # TODO redundancy
                pulse_dic["shape"] = pulse["shape"]
                pulse_dic["name"] = pulse["shape"]
                pulse_dic["style"] = "arb"

                sigma = length / pulse["rel_sigma"]
                pulse_dic["sigma"] = self.soc.us2cycles(sigma)

                if pulse_dic["shape"] == "Drag":
                    pulse_dic["delta"] = pulse_dic["sigma"]  # TODO redundancy
                    pulse_dic["alpha"] = pulse["beta"]

            elif pulse_dic["type"] == "ro":
                pulse_dic["ch"] = gen_ch

                # pulse_dic["waveform"] = None  # this could be unsupported
                pulse_dic["adc_trig_offset"] = self.adc_trig_offset
                pulse_dic["wait"] = False
                pulse_dic["syncdelay"] = 200  # clock ticks
                pulse_dic["style"] = "const"
                pulse_dic["adc_ch"] = adc_ch

                # prepare readout declaration values
                readout = {}
                readout["adc_ch"] = adc_ch
                readout["gen_ch"] = gen_ch
                readout["length"] = self.soc.us2cycles(
                    length
                )  # TODO not sure it should be the same as the pulse! This is the window for the adc
                readout["freq"] = pulse["frequency"] * self.MHz  # this need the MHz value!

                self.readouts.append(readout)

            self.pulse_sequence[pulse["serial"]] = pulse_dic  # TODO check if deep copy

            if pulse["frequency"] < self.max_sampling_rate / 2:
                zone = 1
            else:
                zone = 2
            self.channels.append((gen_ch, zone))

    def from_qubit_to_ch(self, qubit, pulse_type):
        """Helper function for retrieving channel numbers from qubits"""

        drive_ch = self.connections[str(qubit)]["drive"]
        readout_ch = self.connections[str(qubit)]["readout"]
        adc_ch = self.connections[str(qubit)]["adc_ch"]

        if pulse_type == "qd":
            return drive_ch, None
        elif pulse_type == "ro":
            return readout_ch, adc_ch

    def initialize(self):
        """This function gets called automatically by qick super.__init__, it contains:

        * declaration of channels and nyquist zones
        * declaration of readouts (just one per channel, otherwise ignores it)
        * for element in sequence calls the add_pulse_to_register function
          (if first pulse for channel, otherwise it will be done in the body)

        """

        # declare nyquist zones for all used channels
        for channel in self.channels:
            self.declare_gen(ch=channel[0], nqz=channel[1])

        # declare readouts
        channel_already_declared = []
        for readout in self.readouts:
            if readout["adc_ch"] not in channel_already_declared:
                channel_already_declared.append(readout["adc_ch"])
            else:
                print(f"Avoided redecalaration of channel {readout['ch']}")  # TODO raise warning
                continue
            self.declare_readout(
                ch=readout["adc_ch"], length=readout["length"], freq=readout["freq"], gen_ch=readout["gen_ch"]
            )

        # list of channels where a pulse is already been registered
        first_pulse_registered = []

        for serial, pulse in self.pulse_sequence.items():
            if pulse["ch"] not in first_pulse_registered:
                first_pulse_registered.append(pulse["ch"])
            else:
                continue

            self.add_pulse_to_register(pulse)

        self.synci(200)

    def add_pulse_to_register(self, pulse):
        """The task of this function is to call the set_pulse_registers function"""

        if pulse["type"] == "qd":
            if pulse["shape"] == "Gaussian":
                self.add_gauss(ch=pulse["ch"], name=pulse["name"], sigma=pulse["sigma"], length=pulse["length"])

            elif pulse["shape"] == "Drag":
                self.add_DRAG(
                    ch=pulse["ch"],
                    name=pulse["name"],
                    sigma=pulse["sigma"],
                    delta=pulse["delta"],
                    alpha=pulse["alpha"],
                    length=pulse["length"],
                )

            else:
                raise Exception(f'Pulse shape {pulse["shape"]} not recognized!')

            self.set_pulse_registers(
                ch=pulse["ch"],
                style=pulse["style"],
                freq=pulse["freq"],
                phase=pulse["phase"],
                gain=pulse["gain"],
                waveform=pulse["waveform"],
            )

        elif pulse["type"] == "ro":
            self.set_pulse_registers(
                ch=pulse["ch"],
                style=pulse["style"],
                freq=pulse["freq"],
                phase=pulse["phase"],
                gain=pulse["gain"],
                length=pulse["length"],
                # waveform=pulse["waveform"],
            )
        else:
            raise Exception(f'Pulse type {pulse["type"]} not recognized!')

    def body(self):
        """Execute sequence of pulses.

        If the pulse is already loaded it just launches it,
        otherwise first calls the add_pulse_to_register function.

        If readout pulse it does a measurment with an adc trigger, in general does not wait.

        At the end of the pulse wait for clock.
        """

        # list of channels where a pulse is already been executed
        first_pulse_executed = []

        for serial, pulse in self.pulse_sequence.items():
            if pulse["ch"] in first_pulse_executed:
                self.add_pulse_to_register(pulse)
            else:
                first_pulse_executed.append(pulse["ch"])

            if pulse["type"] == "qd":
                self.pulse(ch=pulse["ch"], t=pulse["time"])
            elif pulse["type"] == "ro":
                self.measure(
                    pulse_ch=pulse["ch"],
                    adcs=[pulse["adc_ch"]],
                    adc_trig_offset=pulse["adc_trig_offset"],
                    t=pulse["time"],
                    wait=pulse["wait"],
                    syncdelay=pulse["syncdelay"],
                )
        self.wait_all()
        self.sync_all(self.relax_delay)


class TII_RFSOC4x2(AbstractInstrument):
    """Instrument object for controlling the RFSoC4x2 FPGA.

    The connection requires the FPGA to have a server currently listening.
    The ``connect`` and the ``setup`` functions must be called before playing pulses with
    ``play`` (for arbitrary qibolab ``PulseSequence``) or ``sweep``.

    Args:
        name (str): Name of the instrument instance.
        address (str): IP address and port for connecting to the FPGA.
    """

    def __init__(self, name: str, address: str):
        super().__init__(name, address)
        self.cfg: dict = {}
        self.is_connected = True
        self.soc = QickSoc()

    def connect(self):
        """Connects to the FPGA instrument."""
        pass

    def setup(self, qubits, sampling_rate, repetition_duration, adc_trig_offset, max_gain, **kwargs):
        """Configures the instrument.

        A connection to the instrument needs to be established beforehand.
        Args: Settings taken from runcard
            qubits: parameter not used
            repetition_duration (int): delay before readout (ms)
            adc_trig_offset (int):
            max_gain (int): defined in dac units so that amplitudes can be relative

        Raises:
            Exception = If attempting to set a parameter without a connection to the instrument.
        """
        if self.is_connected:
            # Load needed settings
            self.cfg = {
                "sampling_rate": sampling_rate,
                "repetition_duration": repetition_duration,
                "adc_trig_offset": adc_trig_offset,
                "max_gain": max_gain,
            }

        else:
            raise Exception("The instrument cannot be set up, there is no connection")

    def play(self, qubits, sequence, relaxation_time, nshots=1000):
        """Executes the sequence of instructions and retrieves the readout results.

        Each readout pulse generates a separate acquisition.

        Args:
            qubits: parameter not used
            sequence (PulseSequence): arbitary qibolab pulse sequence to execute
            nshots (int): number of shots
            relaxation_time (int): delay before readout (ms)

        Returns:
            A dictionary mapping the readout pulse serial to am ExecutionResults object
        """

        data = {"nshots": nshots, "relaxation_time": relaxation_time, "pulses": {}}
        for i, pulse in enumerate(sequence):
            data["pulses"][str(i)] = self.convert_pulse_to_dic(pulse)

        program = ExecutePulseSequence(self.soc, self.cfg, data)
        avgi, avgq = program.acquire(self.soc, load_pulses=True, progress=False, debug=False)

        avgi = np.array(avgi[0][0])
        avgq = np.array(avgq[0][0])

        serial = sequence.ro_pulses[0].serial
        return {serial: ExecutionResults.from_components(avgi, avgq)}

    def sweep(self, qubits, sequence, *sweepers, relaxation_time, nshots=1000, average=True):
        """Play a pulse sequence while sweeping one or more parameters.

        Args:
            qubits: parameter not used
            sequence (PulseSequence): arbitary qibolab pulse sequence to execute
            *sweepers (list): A list of qibolab Sweepers objects
            nshots (int): number of shots
            relaxation_time (int): delay before readout (ms)
            average: parameter not used

        Returns:
            A dictionary mapping the readout pulse serial to am ExecutionResults object

        Raises:
            Exception = If attempting to use more than one Sweeper.
            Exception = If average is set to False

        """

        raise NotImplementedError("Sweepers are not yet implemented")

        if average is False:
            raise NotImplementedError("Only averaged results are supported")
        if len(sweepers) > 1:
            raise NotImplementedError("Only one sweeper is supported.")

        #  Parsing the sweeper to dictionary and after to a json file
        sweeper = sweepers[0]

        json_dic = {}
        json_dic["nshots"] = nshots
        json_dic["relaxation_time"] = relaxation_time

        json_dic["parameter"] = str(sweeper.parameter)
        start = sweeper.values[0].item()
        expt = len(sweeper.values)
        step = (sweeper.values[1] - sweeper.values[0]).item()
        json_dic["range"] = {"start": start, "step": step, "expt": expt}

        pulses_dic = {}
        for i, pulse in enumerate(sequence.pulses):  # convert pulses to dictionary
            pulses_dic[str(i)] = self.convert_pulse_to_dic(pulse)
        json_dic["pulses"] = pulses_dic

        json_dic["opCode"] = "sweep"  # opCode parameter for server

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            received = bytearray()
            # connect to server
            sock.connect((self.host, self.port))
            # send data
            sock.sendall(json.dumps(json_dic).encode())
            # receive data back from the server
            # wait for packets until the server is sending them
            while 1:
                tmp = sock.recv(4096)
                if not tmp:
                    break
                received.extend(tmp)
            avg = json.loads(received)

            pulse_serial = avg["serial"]
            avgi = np.array(avg["avg_di"])
            avgq = np.array(avg["avg_dq"])

        return {pulse_serial: ExecutionResults.from_components(avgi, avgq)}

    def convert_pulse_to_dic(self, pulse):
        """Funtion to convert pulse object attributes to a dictionary"""
        pulse: Pulse
        pulse_shape: PulseShape

        if pulse.type == PulseType.DRIVE:
            pulse_shape = pulse.shape
            if type(pulse_shape) is Drag:
                shape = "Drag"
                style = "arb"
                rel_sigma = pulse_shape.rel_sigma
                beta = pulse_shape.beta
            elif type(pulse_shape) is Gaussian:
                shape = "Gaussian"
                style = "arb"
                rel_sigma = pulse_shape.rel_sigma
                beta = 0
            elif type(pulse_shape) is Rectangular:
                shape = "Rectangular"
                style = "const"
                rel_sigma = 0
                beta = 0
            pulse_dictionary = {
                "start": pulse.start,
                "duration": pulse.duration,
                "amplitude": pulse.amplitude,
                "frequency": pulse.frequency,
                "relative_phase": pulse.relative_phase,
                "shape": shape,
                "style": style,
                "rel_sigma": rel_sigma,
                "beta": beta,
                "type": "qd",
                "channel": 1,
                "qubit": pulse.qubit,
                "serial": pulse.serial,  # TODO remove redundancy
            }

        elif pulse.type == PulseType.READOUT:
            pulse_dictionary = {
                "start": pulse.start,
                "duration": pulse.duration,
                "amplitude": pulse.amplitude,
                "frequency": pulse.frequency,
                "relative_phase": pulse.relative_phase,
                "shape": "const",
                "type": "ro",
                "channel": 0,
                "qubit": pulse.qubit,
                "serial": pulse.serial,  # TODO remove redundancy
            }

        return pulse_dictionary

    def start(self):
        """Empty method to comply with AbstractInstrument interface."""
        pass

    def stop(self):
        """Empty method to comply with AbstractInstrument interface."""
        pass

    def disconnect(self):
        """Closes the connection to the instrument."""
        self.is_connected = False
