""" RFSoC FPGA driver.

This driver needs the library Qick installed

Supports the following FPGAs:
   RFSoC 4x2
   ZCU111

In page 0, register 13-14 are used for internal counters.
Registers 15-21 are to be used by sweepers
"""

import numpy as np
from qick import AveragerProgram, QickSoc, RAveragerProgram
from qick.qick_asm import QickRegisterManagerMixin

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

    def __init__(self, soc, cfg, sequence, qubits):
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
        self.qubits = qubits

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

        super().__init__(soc, cfg)

    def initialize(self):
        """This function gets called automatically by qick super.__init__, it contains:

        * declaration of channels and nyquist zones
        * declaration of readouts (just one per channel, otherwise ignores it)
        * for element in sequence calls the add_pulse_to_register function
          (if first pulse for channel, otherwise it will be done in the body)

        """

        # declare nyquist zones for all used channels
        ch_already_declared = []

        for pulse in self.sequence.qd_pulses:
            qd_ch = self.qubits[pulse.qubit].drive.ports[0][1]
            adc_ch = self.qubits[pulse.qubit].feedback.ports[0][1]
            ro_ch = self.qubits[pulse.qubit].readout.ports[0][1]
            gen_ch = qd_ch if pulse.type == PulseType.DRIVE else ro_ch

            if gen_ch not in ch_already_declared:
                ch_already_declared.append(gen_ch)

                if pulse.frequency < self.max_sampling_rate / 2:
                    zone = 1
                else:
                    zone = 2
                self.declare_gen(gen_ch, nqz=zone)
            else:
                print(f"Avoided redecalaration of channel {gen_ch}")  # TODO 

        mux_freqs  = []
        mux_gains = []
        for pulse in self.sequence.ro_pulses:
            qd_ch = self.qubits[pulse.qubit].drive.ports[0][1]
            adc_ch = self.qubits[pulse.qubit].feedback.ports[0][1]
            ro_ch = self.qubits[pulse.qubit].readout.ports[0][1]
            gen_ch = qd_ch if pulse.type == PulseType.DRIVE else ro_ch

            if pulse.frequency < self.max_sampling_rate / 2:
                zone = 1
            else:
                zone = 2
            mux_gains.append(pulse.amplitude)
            mux_freqs.append((pulse.frequency - self.cfg["mixer_freq"] - self.cfg["LO_freq"])*self.MHz)

        self.declare_gen(ch=ro_ch, nqz=zone,
                        mixer_freq=self.cfg["mixer_freq"],
                        mux_freqs=mux_freqs,
                        mux_gains=mux_gains,
                        ro_ch=adc_ch)

        # declare readouts
        ro_ch_already_declared = []
        for readout_pulse in self.sequence.ro_pulses:
            adc_ch = self.qubits[readout_pulse.qubit].feedback.ports[0][1]
            ro_ch = self.qubits[readout_pulse.qubit].readout.ports[0][1]
            if adc_ch not in ro_ch_already_declared:
                ro_ch_already_declared.append(ro_ch)
                length = self.soc.us2cycles(readout_pulse.duration * self.us)
                freq = readout_pulse.frequency * self.MHz

                self.declare_readout(ch=adc_ch, length=length, freq=freq, gen_ch=ro_ch)
            else:
                print(f"Avoided redecalaration of channel {adc_ch}")  # TODO

        # list of channels where a pulse is already been registered
        first_pulse_registered = []

        for pulse in self.sequence:
            qd_ch = self.qubits[pulse.qubit].drive.ports[0][1]
            adc_ch = self.qubits[pulse.qubit].feedback.ports[0][1]
            ro_ch = self.qubits[pulse.qubit].readout.ports[0][1]

            gen_ch = qd_ch if pulse.type == PulseType.DRIVE else ro_ch

            if gen_ch not in first_pulse_registered:
                first_pulse_registered.append(gen_ch)
                self.add_pulse_to_register(pulse)

        self.synci(200)

    def add_pulse_to_register(self, pulse):
        """The task of this function is to call the set_pulse_registers function"""

        qd_ch = self.qubits[pulse.qubit].drive.ports[0][1]
        adc_ch = self.qubits[pulse.qubit].feedback.ports[0][1]
        ro_ch = self.qubits[pulse.qubit].readout.ports[0][1]
        gen_ch = qd_ch if pulse.type == PulseType.DRIVE else ro_ch

        time = self.soc.us2cycles(pulse.start * self.us)
        gain = int(pulse.amplitude * self.max_gain)
        if pulse.amplitude > 1:
            raise Exception("Relative amplitude higher than 1!")
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

            #freq = self.soc.freq2reg(pulse.frequency * self.MHz, gen_ch=gen_ch, ro_ch=adc_ch)

            self.set_pulse_registers(ch=gen_ch, style="const", length=soc_length,
                                     mask=[pulse.qubit])
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

            qd_ch = self.qubits[pulse.qubit].drive.ports[0][1]
            adc_ch = self.qubits[pulse.qubit].feedback.ports[0][1]
            ro_ch = self.qubits[pulse.qubit].readout.ports[0][1]
            gen_ch = qd_ch if pulse.type == PulseType.DRIVE else ro_ch

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


class ExecuteSingleSweep(RAveragerProgram):
    """This qick AveragerProgram handles a qibo sequence of pulses"""

    def __init__(self, soc, cfg, sequence, qubits, sweeper):
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
        self.qubits = qubits

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

        # sweeper Settings
        self.sweeper = sweeper
        cfg["start"] = sweeper.values[0]
        cfg["step"] = sweeper.values[1] - sweeper.values[0]
        cfg["expts"] = len(sweeper.values)

        super().__init__(soc, cfg)

    def initialize(self):
        """This function gets called automatically by qick super.__init__, it contains:

        * declaration of channels and nyquist zones
        * declaration of readouts (just one per channel, otherwise ignores it)
        * for element in sequence calls the add_pulse_to_register function
          (if first pulse for channel, otherwise it will be done in the body)

        """

        # find page and register of sweeper

        pulse = self.sweeper.pulses[0]
        qd_ch = self.qubits[pulse.qubit].drive.ports[0][1]
        adc_ch = self.qubits[pulse.qubit].feedback.ports[0][1]
        ro_ch = self.qubits[pulse.qubit].readout.ports[0][1]
        gen_ch = qd_ch if pulse.type == PulseType.DRIVE else ro_ch

        self.sweeper_page = self.ch_page(gen_ch)
        if self.sweeper.parameter == Parameter.frequency:
            self.sweeper_reg = self.sreg(gen_ch, "freq")
            self.cfg["start"] = self.soc.freq2reg(self.cfg["start"] * self.MHz, gen_ch)
            self.cfg["step"] = self.soc.freq2reg(self.cfg["step"] * self.MHz, gen_ch)

        elif self.sweeper.parameter == Parameter.amplitude:
            self.sweeper_reg = self.sreg(gen_ch, "gain")
            self.cfg["start"] = self.cfg["start"] * self.max_gain
            self.cfg["step"] = self.cfg["step"] * self.max_gain

            if self.cfg["start"] + self.cfg["step"] * self.cfg["expts"] > self.max_gain:
                raise Exception("Amplitude higher than maximum!")

        # declare nyquist zones for all used channels
        ch_already_declared = []
        for pulse in self.sequence:
            qd_ch = self.qubits[pulse.qubit].drive.ports[0][1]
            adc_ch = self.qubits[pulse.qubit].feedback.ports[0][1]
            ro_ch = self.qubits[pulse.qubit].readout.ports[0][1]
            gen_ch = qd_ch if pulse.type == PulseType.DRIVE else ro_ch

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
            adc_ch = self.qubits[pulse.qubit].feedback.ports[0][1]
            ro_ch = self.qubits[pulse.qubit].readout.ports[0][1]
            if adc_ch not in ro_ch_already_declared:
                ro_ch_already_declared.append(adc_ch)
                length = self.soc.us2cycles(readout_pulse.duration * self.us)
                freq = readout_pulse.frequency * self.MHz

                self.declare_readout(ch=adc_ch, length=length, freq=freq, gen_ch=ro_ch)
            else:
                print(f"Avoided redecalaration of channel {adc_ch}")  # TODO

        # list of channels where a pulse is already been registered
        first_pulse_registered = []

        for pulse in self.sequence:
            qd_ch = self.qubits[pulse.qubit].drive.ports[0][1]
            adc_ch = self.qubits[pulse.qubit].feedback.ports[0][1]
            ro_ch = self.qubits[pulse.qubit].readout.ports[0][1]
            gen_ch = qd_ch if pulse.type == PulseType.DRIVE else ro_ch

            if gen_ch not in first_pulse_registered:
                first_pulse_registered.append(gen_ch)
                self.add_pulse_to_register(pulse)

        self.synci(200)

    def add_pulse_to_register(self, pulse):
        """The task of this function is to call the set_pulse_registers function"""

        qd_ch = self.qubits[pulse.qubit].drive.ports[0][1]
        adc_ch = self.qubits[pulse.qubit].feedback.ports[0][1]
        ro_ch = self.qubits[pulse.qubit].readout.ports[0][1]
        gen_ch = qd_ch if pulse.type == PulseType.DRIVE else ro_ch

        time = self.soc.us2cycles(pulse.start * self.us)
        gain = int(pulse.amplitude * self.max_gain)
        if pulse.amplitude > 1:
            raise Exception("Relative amplitude higher than 1!")
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

    def update(self):
        self.mathi(self.sweeper_page, self.sweeper_reg, self.sweeper_reg, "+", self.cfg["step"])

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

            qd_ch = self.qubits[pulse.qubit].drive.ports[0][1]
            adc_ch = self.qubits[pulse.qubit].feedback.ports[0][1]
            ro_ch = self.qubits[pulse.qubit].readout.ports[0][1]
            gen_ch = qd_ch if pulse.type == PulseType.DRIVE else ro_ch

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
                "mixer_freq": mixer_freq,
                "LO_freq": LO_freq,
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

        program = ExecutePulseSequence(self.soc, self.cfg, sequence, qubits)
        avgi, avgq = program.acquire(
            self.soc, readouts_per_experiment=len(sequence.ro_pulses), load_pulses=True, progress=False, debug=False
        )

class TII_RFSOC_ZCU111(AbstractInstrument):
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
        self.soc = QickSoc(bitfile="/home/xilinx/jupyter_notebooks/qick_111_rfbv1_mux.bit")

    def connect(self):
        """Connects to the FPGA instrument."""
        pass

    def setup(self, qubits, sampling_rate, repetition_duration, adc_trig_offset, max_gain, mixer_freq, LO_freq, **kwargs):
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
                "mixer_freq": mixer_freq,
                "LO_freq": LO_freq,
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

        program = ExecutePulseSequence(self.soc, self.cfg, sequence, qubits)
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

    def recursive_python_sweep(self, qubits, sequence, *sweepers):
        if len(sweepers) == 0:
            program = ExecutePulseSequence(self.soc, self.cfg, sequence, qubits)
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
        else:
            sweep_results = []
            sweeper = sweepers[0]
            if len(sweeper.pulses) > 1:
                raise NotImplementedError("Only one pulse per sweep is supported")
            for idx, pulse in enumerate(sequence):
                # identify index of sweeped pulse
                if pulse.serial == sweeper.pulses[0].serial:
                    idx_pulse = idx
                    break

            for val in sweeper.values:
                if sweeper.parameter == Parameter.frequency:
                    sequence[idx_pulse].frequency = val
                elif sweeper.parameter == Parameter.amplitude:
                    # TODO: understand how this is relative!!
                    sequence[idx_pulse].amplitude = val
                else:
                    raise NotImplementedError("Parameter type not implemented")
                if len(sweepers) == 0:
                    if not self.get_if_python_sweep(sequence, *sweepers):
                        program = ExecuteSingleSweep(self.soc, self.cfg, sequence, qubits, sweepers[0])
                        values, avgi, avgq = program.acquire(
                            self.soc,
                            readouts_per_experiment=len(sequence.ro_pulses),
                            load_pulses=True,
                            progress=False,
                            debug=False,
                        )
                        res = self.convert_sweep_results(sweepers[0], sequence, avgi, avgq)
                    else:
                        res = self.recursive_python_sweep(qubits, sequence, *sweepers[1:])
                else:
                    res = self.recursive_python_sweep(qubits, sequence, *sweepers[1:])
                sweep_results.append(res)
        return sweep_results

    def get_if_python_sweep(self, sequence, *sweepers):
        python_sweep = True

        is_amp = sweepers[0].parameter == Parameter.amplitude
        is_freq = sweepers[0].parameter == Parameter.frequency

        # if there is only a sweeper
        if len(sweepers) == 1:
            is_ro = sweepers[0].pulses[0].type == PulseType.READOUT
            # if it's not a sweep on the readout freq
            if not (is_freq and is_ro):
                # if the sweep is on the first pulse of the channel
                for pulse in sequence:
                    same_qubit = pulse.qubit == sweepers[0].pulses[0].qubit
                    same_pulse = pulse.serial == sweepers[0].pulses[0].serial
                    if same_qubit and same_pulse:
                        python_sweep = False
                    elif same_qubit and not same_pulse:
                        break
        return python_sweep

    def convert_sweep_results(self, sweeper, sequence, avgi, avgq):
        sweep_results = []
        for j, val in enumerate(sweeper.values):
            results = {}

            for i, ro_pulse in enumerate(sequence.ro_pulses):
                i_pulse = np.array(avgi[0][i][j])
                q_pulse = np.array(avgq[0][i][j])

                serial = ro_pulse.serial  # TODO this can change during
                results[serial] = ExecutionResults.from_components(i_pulse, q_pulse)
            sweep_results.append(results)
        return sweep_results

    def sweep(self, qubits, sequence, *sweepers, relaxation_time, nshots=1000, average=True):
        self.cfg["nshots"] = nshots
        if relaxation_time is not None:
            self.cfg["repetition_duration"] = relaxation_time

        return self.recursive_python_sweep(qubits, sequence, *sweepers)

    def start(self):
        """Empty method to comply with AbstractInstrument interface."""
        pass

    def stop(self):
        """Empty method to comply with AbstractInstrument interface."""
        pass

    def disconnect(self):
        """Closes the connection to the instrument."""
        self.is_connected = False
