""" RFSoC FPGA driver.

This driver needs the library Qick installed

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
    """This qick AveragerProgram handles a qibo sequence of pulses"""

    def __init__(self, soc, cfg, sequence):
        """In this function we define the most important settings and the sequence is transpiled.

        In detail:
            * set the conversion coefficients to be used for frequency and time values
            * max_gain, adc_trig_offset, max_sampling_rate are imported from cfg (runcard settings)
            * syncdelay (for each measurement) is defined explicitly
            * connections are defined (drive, readout channel and adc_ch for each qubit)

            * pulse_sequence, readouts, channels are defined to be filled by convert_sequence()

            * cfg["reps"] is set from hardware_avg
            * super.__init__
        """

        # fill the self.pulse_sequence and the self.readout_pulses oject
        self.soc = soc
        self.soccfg = soc  # No need for a different soc config object since qick is on board
        self.sequence = sequence

        # conversion coefficients (in runcard we have Hz and ns)
        self.MHz = 0.000001
        self.us = 0.001

        # settings
        self.max_gain = cfg["max_gain"]  # TODO redundancy
        self.adc_trig_offset = cfg["adc_trig_offset"]
        self.max_sampling_rate = cfg["sampling_rate"]
        self.relax_delay = cfg["repetition_duration"]
        self.syncdelay = self.us2cycles(1.0)  # TODO maybe better in runcard
        cfg["reps"] = cfg["nshots"]

        # connections  (for every qubit here are defined drive and readout lines)
        self.connections = {
            "0": {"drive": 1, "readout": 0, "adc_ch": 0},
        }

        super().__init__(soc, cfg)

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
        ch_already_declared = []
        for pulse in self.sequence:
            # TODO remove function
            if pulse.type == PulseType.DRIVE:
                gen_ch, adc_ch = self.from_qubit_to_ch(pulse.qubit, "qd")
            else:
                gen_ch, adc_ch = self.from_qubit_to_ch(pulse.qubit, "ro")
            if gen_ch not in ch_already_declared:
                ch_already_declared.append(gen_ch)

                if pulse.frequency < self.max_sampling_rate / 2:
                    zone = 1
                else:
                    zone = 2
                self.declare_gen(gen_ch, nqz=zone)
            else:
                print(f"Avoided redecalaration of channel {gen_ch}")  # TODO

        # declare readouts
        ro_ch_already_declared = []
        for readout_pulse in self.sequence.ro_pulses:
            # TODO remove function
            gen_ch, adc_ch = self.from_qubit_to_ch(readout_pulse.qubit, "ro")
            if adc_ch not in ro_ch_already_declared:
                ro_ch_already_declared.append(adc_ch)
                length = self.soc.us2cycles(readout_pulse.duration * self.us)
                freq = readout_pulse.frequency * self.MHz

                self.declare_readout(ch=adc_ch, length=length, freq=freq, gen_ch=gen_ch)
            else:
                print(f"Avoided redecalaration of channel {adc_ch}")  # TODO

        # list of channels where a pulse is already been registered
        first_pulse_registered = []

        for pulse in self.sequence:
            # TODO remove function
            if pulse.type == PulseType.DRIVE:
                gen_ch, adc_ch = self.from_qubit_to_ch(pulse.qubit, "qd")
            else:
                gen_ch, adc_ch = self.from_qubit_to_ch(pulse.qubit, "ro")

            if gen_ch not in first_pulse_registered:
                first_pulse_registered.append(gen_ch)
                self.add_pulse_to_register(pulse)

        self.synci(200)

    def add_pulse_to_register(self, pulse):
        """The task of this function is to call the set_pulse_registers function"""

        # TODO remove function
        if pulse.type == PulseType.DRIVE:
            gen_ch, adc_ch = self.from_qubit_to_ch(pulse.qubit, "qd")
        else:
            gen_ch, adc_ch = self.from_qubit_to_ch(pulse.qubit, "ro")

        time = self.soc.us2cycles(pulse.start * self.us)
        gain = int(pulse.amplitude * self.max_gain)
        phase = self.deg2reg(pulse.relative_phase, gen_ch=gen_ch)

        us_length = pulse.duration * self.us
        soc_length = self.soc.us2cycles(us_length)

        if pulse.type == PulseType.DRIVE:
            name = pulse.shape.name
            sigma = us_length / pulse.shape.rel_sigma

            freq = self.soc.freq2reg(pulse.frequency * self.MHz, gen_ch=gen_ch)

            if isinstance(pulse.shape, Gaussian):
                self.add_gauss(ch=gen_ch, name=name, sigma=sigma, length=soc_length)

            elif isinstance(pulse.shape, Drag):
                self.add_DRAG(
                    ch=gen_ch,
                    name=name,
                    sigma=sigma,
                    delta=sigma,  # TODO: check if correct
                    alpha=pulse.beta,
                    length=soc_length,
                )

            else:
                raise NotImplementedError(f"Pulse shape {pulse.shape} not supported!")

            self.set_pulse_registers(
                ch=gen_ch,
                style="arb",
                freq=freq,
                phase=phase,
                gain=gain,
                waveform=name,
            )

        elif pulse.type == PulseType.READOUT:
            if not isinstance(pulse.shape, Rectangular):
                raise NotImplementedError("Only Rectangular readout pulses are supported")

            freq = self.soc.freq2reg(pulse.frequency * self.MHz, gen_ch=gen_ch, ro_ch=adc_ch)

            self.set_pulse_registers(ch=gen_ch, style="const", freq=freq, phase=phase, gain=gain, length=soc_length)
        else:
            raise Exception(f"Pulse type {pulse.type} not recognized!")

    def body(self):
        """Execute sequence of pulses.

        If the pulse is already loaded it just launches it,
        otherwise first calls the add_pulse_to_register function.

        If readout pulse it does a measurment with an adc trigger, in general does not wait.

        At the end of the pulse wait for clock.
        """

        # list of channels where a pulse is already been executed
        first_pulse_executed = []

        for pulse in self.sequence:
            time = self.soc.us2cycles(pulse.start * self.us)

            # TODO remove function
            if pulse.type == PulseType.DRIVE:
                gen_ch, adc_ch = self.from_qubit_to_ch(pulse.qubit, "qd")
            else:
                gen_ch, adc_ch = self.from_qubit_to_ch(pulse.qubit, "ro")

            if gen_ch in first_pulse_executed:
                self.add_pulse_to_register(pulse)
            else:
                first_pulse_executed.append(gen_ch)

            if pulse.type == PulseType.DRIVE:
                self.pulse(ch=gen_ch, t=time)
            elif pulse.type == PulseType.READOUT:
                self.measure(
                    pulse_ch=gen_ch,
                    adcs=[adc_ch],
                    adc_trig_offset=self.adc_trig_offset,
                    t=time,
                    wait=False,  # TODO maybe better not hardcoded
                    syncdelay=self.syncdelay,
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

        self.cfg["nshots"] = nshots
        if relaxation_time is not None:
            self.cfg["repetition_duration"] = relaxation_time

        program = ExecutePulseSequence(self.soc, self.cfg, sequence)
        avgi, avgq = program.acquire(
            self.soc, readouts_per_experiment=len(sequence.ro_pulses), load_pulses=True, progress=False, debug=False
        )

        results = {}
        for i, ro_pulse in enumerate(sequence.ro_pulses):
            i_pulse = np.array(avgi[0][i])
            q_pulse = np.array(avgq[0][i])

            serial = ro_pulse.serial
            results[serial] = ExecutionResults.from_components(i_pulse, q_pulse)

        return results

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

        """
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
        """

    def start(self):
        """Empty method to comply with AbstractInstrument interface."""
        pass

    def stop(self):
        """Empty method to comply with AbstractInstrument interface."""
        pass

    def disconnect(self):
        """Closes the connection to the instrument."""
        self.is_connected = False
