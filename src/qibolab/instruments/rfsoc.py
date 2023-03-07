""" RFSoC FPGA driver.

This driver needs the library Qick installed

Supports the following FPGA:
 A   RFSoC 4x2

"""

from copy import deepcopy
from typing import List

import numpy as np
from qick import AveragerProgram, QickSoc, RAveragerProgram
from qick.qick_asm import QickRegisterManagerMixin

from qibolab.instruments.abstract import AbstractInstrument, InstrumentException
from qibolab.platforms.abstract import Qubit
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
from qibolab.result import AveragedResults, ExecutionResults
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
            adc_ch = self.qubits[readout_pulse.qubit].feedback.ports[0][1]
            ro_ch = self.qubits[readout_pulse.qubit].readout.ports[0][1]
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
        # cfg["start"] = sweeper.values[0]
        # cfg["step"] = sweeper.values[1] - sweeper.values[0]
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
        start = self.sweeper.values[0]
        step = self.sweeper.values[1] - self.sweeper.values[0]
        if self.sweeper.parameter == Parameter.frequency:
            self.sweeper_reg = self.sreg(gen_ch, "freq")
            self.cfg["start"] = self.soc.freq2reg(start * self.MHz, gen_ch)
            self.cfg["step"] = self.soc.freq2reg(step * self.MHz, gen_ch)

        elif self.sweeper.parameter == Parameter.amplitude:
            self.sweeper_reg = self.sreg(gen_ch, "gain")
            self.cfg["start"] = int(start * self.max_gain)
            self.cfg["step"] = int(step * self.max_gain)

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

    Playing pulses requires first the execution of the ``setup`` function.
    The two way of executing pulses are with ``play`` (for arbitrary qibolab ``PulseSequence``) or
    with ``sweep`` that execute a ``PulseSequence`` object with one or more ``Sweeper``.

    Args:
        name (str): Name of the instrument instance.

    Attributes:
        cfg (dict): Configuration dictionary required for pulse execution.
        soc (QickSoc): ``Qick`` object needed to access system blocks.
    """

    def __init__(self, name: str, address: str = None):
        # The address parameter should be None since qibolab is executed directly on-board
        super().__init__(name, address)
        self.cfg: dict = {}
        self.soc = QickSoc()

    def connect(self):
        """Empty method to comply with AbstractInstrument interface."""
        pass

    def start(self):
        """Empty method to comply with AbstractInstrument interface."""
        pass

    def stop(self):
        """Empty method to comply with AbstractInstrument interface."""
        pass

    def disconnect(self):
        """Empty method to comply with AbstractInstrument interface."""
        pass

    def setup(
        self,
        qubits: List[Qubit],
        sampling_rate: int,
        repetition_duration: int,
        adc_trig_offset: int,
        max_gain: int,
        **kwargs,
    ):
        """Configures the instrument.

        Args: Settings taken from runcard
            qubits (list): list of :class:`qibolab.platforms.abstract.Qubit`, parameter not used here.
            sampling_rate (int): sampling rate of the RFSoC (Hz).
            repetition_duration (int): delay before readout (ms).
            adc_trig_offset (int): single offset for all adc triggers (clock ticks).
            max_gain (int): maximum output power of the DAC (DAC units).

            **kwargs: no additional arguments are expected and used
        """

        # Load settings needed for QickPrograms
        self.cfg = {
            "sampling_rate": sampling_rate,
            "repetition_duration": repetition_duration,
            "adc_trig_offset": adc_trig_offset,
            "max_gain": max_gain,
        }

    def play(self, qubits: List[Qubit], sequence: PulseSequence, relaxation_time: int, nshots: int = 1000) -> dict:
        """Executes the sequence of instructions and retrieves the readout results.

        Each readout pulse generates a separate acquisition.
        The relaxation_time and the number of shots have default values.

        Args:
            qubits (list): List of :class:`qibolab.platforms.utils.Qubit` objects passed from the platform.
            sequence (:class:`qibolab.pulses.PulseSequence`). Pulse sequence to play.
            nshots (int): Number of repetitions (shots) of the experiment.
            relaxation_time (int): Time to wait for the qubit to relax to its ground state between shots in ns.
        Returns:
            A dictionary mapping the readout pulses serial to `qibolab.ExecutionResults` objects
        """

        # TODO: simmetries beetween repetition_duration Vs relaxation_time and nshots Vs relaxation_time
        # nshots gets added to the dictionary
        self.cfg["nshots"] = nshots
        # if new value is passed, relaxation_time is updated in the dictionary
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
            # results[serial] = ExecutionResults.from_components(i_pulse, q_pulse)
            results[serial] = AveragedResults(i_pulse, q_pulse)

        return results

    def recursive_python_sweep(
        self, qubits: List[Qubit], sequence: PulseSequence, or_sequence: PulseSequence, *sweepers: Sweeper
    ) -> dict:
        """Execute a sweep of an arbitrary number of Sweepers via recursion.

        Args:
            qubits (list): List of `qibolab.platforms.utils.Qubit` objects passed from the platform.
            sequence (`qibolab.pulses.PulseSequence`). Pulse sequence to play.
            *sweepers (`qibolab.Sweeper`): Sweeper objects.
        Returns:
            A dictionary mapping the readout pulses serial to `qibolab.ExecutionResults` objects
        Raises:
            NotImplementedError: if a sweep refers to more than one pulse.
            NotImplementedError: if a sweep refers to a parameter different from frequency or amplitude.
        """
        sequence = deepcopy(sequence)
        original_ro = [ro.serial for ro in or_sequence.ro_pulses]
        # If there are no sweepers run ExecutePulseSequence acquisition. Last layer for recursion.
        if len(sweepers) == 0:
            program = ExecutePulseSequence(self.soc, self.cfg, sequence, qubits)
            avgi, avgq = program.acquire(
                self.soc, readouts_per_experiment=len(sequence.ro_pulses), load_pulses=True, progress=False, debug=False
            )
            # results parsing
            results = {}
            for i, serial in enumerate(original_ro):
                i_pulse = np.array([avgi[0][i]], np.float64)
                q_pulse = np.array([avgq[0][i]], np.float64)

                # results[serial] = ExecutionResults.from_components(i_pulse, q_pulse)
                results[serial] = AveragedResults(i_pulse, q_pulse)
            return results
        else:  # If sweepers are still in queue
            sweep_results = {}
            # check that the first (outest) sweeper is supported
            sweeper = sweepers[0]
            if len(sweeper.pulses) > 1:
                raise NotImplementedError("Only one pulse per sweep is supported")
            is_amp = sweeper.parameter == Parameter.amplitude
            is_freq = sweeper.parameter == Parameter.frequency
            if not (is_amp or is_freq):
                raise NotImplementedError("Parameter type not implemented")

            # if there is only one sweeper and is supported by qick than use hardware sweep
            if len(sweepers) == 1 and not self.get_if_python_sweep(sequence, *sweepers):
                program = ExecuteSingleSweep(self.soc, self.cfg, sequence, qubits, sweepers[0])
                values, avgi, avgq = program.acquire(
                    self.soc,
                    readouts_per_experiment=len(sequence.ro_pulses),
                    load_pulses=True,
                    progress=False,
                    debug=False,
                )
                # convert results from qick output to qibolab results
                res = self.convert_sweep_results(sweepers[0], sequence, original_ro, avgi, avgq)
                sweep_results = self.merge_sweep_results(sweep_results, res)
            else:  # if it's not possible to execute qick sweep re-call function
                # identify index of sweeped pulse TODO: there is a better way
                idx_pulse = or_sequence.index(sweeper.pulses[0])
                for val in sweeper.values:
                    if is_freq:
                        f0 = qubits[sequence[idx_pulse].qubit].readout_frequency
                        # TODO relative frequency?
                        sequence[idx_pulse].frequency = val + f0
                    elif is_amp:
                        sequence[idx_pulse].amplitude = val
                    res = self.recursive_python_sweep(qubits, sequence, or_sequence, *sweepers[1:])
                    sweep_results = self.merge_sweep_results(sweep_results, res)
        return sweep_results

    def merge_sweep_results(self, old_dict, addition):
        for serial in addition:
            if serial in old_dict:
                old_dict[serial] = old_dict[serial] + addition[serial]
            else:
                old_dict[serial] = addition[serial]
        return old_dict

    def get_if_python_sweep(self, sequence: PulseSequence, *sweepers: Sweeper) -> bool:
        """Check if a sweeper must be run with python loop or on hardware.

        To be run on qick internal loop a sweep must:
            * not be on the readout frequency
            * be the first pulse of a channel
            * be just one sweeper

        Args:
            sequence (:class:`qibolab.pulses.PulseSequence`). Pulse sequence to play.
            *sweepers (`qibolab.Sweeper`): Sweeper objects.
        Returns:
            A boolean value true if the sweeper must be executed by python loop, false otherwise
        """
        # TODO: maybe improve this function and check the channel and not the qubit (multiplexing)
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

    def convert_sweep_results(
        self, sweeper: Sweeper, sequence: PulseSequence, original_ro: list, avgi: list, avgq: list
    ) -> dict:
        """Convert Qick RAveragerProgram results to qibolab dictionary results"""
        sweep_results = {}
        # original_ro = [ro.serial for ro in sequence.ro_pulses]
        for j, val in enumerate(sweeper.values):
            results = {}

            for i, serial in enumerate(original_ro):
                i_pulse = np.array([avgi[0][i][j]], np.float64)
                q_pulse = np.array([avgq[0][i][j]], np.float64)

                results[serial] = ExecutionResults.from_components(i_pulse, q_pulse)
            sweep_results = self.merge_sweep_results(sweep_results, results)
        return sweep_results

    def sweep(
        self,
        qubits: List[Qubit],
        sequence: PulseSequence,
        *sweepers: Sweeper,
        relaxation_time: int,
        nshots: int = 1000,
        average: bool = True,
    ) -> dict:
        """Executes the sweep and retrieves the readout results.

        Each readout pulse generates a separate acquisition.
        The relaxation_time and the number of shots have default values.

        Args:
            qubits (list): List of :class:`qibolab.platforms.utils.Qubit` objects passed from the platform.
            sequence (:class:`qibolab.pulses.PulseSequence`). Pulse sequence to play.
            *sweepers (`qibolab.Sweeper`): Sweeper objects.
            nshots (int): Number of repetitions (shots) of the experiment.
            relaxation_time (int): Time to wait for the qubit to relax to its ground state between shots in ns.
        Returns:
            A dictionary mapping the readout pulses serial to `qibolab.ExecutionResults` objects
        """
        self.cfg["nshots"] = nshots
        if relaxation_time is not None:
            self.cfg["repetition_duration"] = relaxation_time

        # TODO: bug?
        # sweep_sequence = sequence.deep_copy()
        # added for "clarity"
        or_sequence = sequence

        return self.recursive_python_sweep(qubits, sequence, or_sequence, *sweepers)
