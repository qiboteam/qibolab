""" RFSoC FPGA driver.

This driver needs the library Qick installed

Supports the following FPGA:
 *   RFSoC 4x2

"""

from typing import Dict, List, Tuple, Union

import numpy as np
from qick import AveragerProgram, QickSoc, RAveragerProgram

from qibolab.instruments.abstract import AbstractInstrument
from qibolab.platforms.abstract import Qubit
from qibolab.pulses import (
    Drag,
    Gaussian,
    Pulse,
    PulseSequence,
    PulseType,
    ReadoutPulse,
    Rectangular,
)
from qibolab.result import AveragedResults, ExecutionResults
from qibolab.sweeper import Parameter, Sweeper


class ExecutePulseSequence(AveragerProgram):
    """This qick AveragerProgram handles a qibo sequence of pulses"""

    def __init__(self, soc: QickSoc, cfg: dict, sequence: PulseSequence, qubits: List[Qubit]):
        """In this function we define the most important settings.

        In detail:
            * set the conversion coefficients to be used for frequency and
              time values
            * max_gain, adc_trig_offset, max_sampling_rate are imported from
              cfg (runcard settings)

            * relaxdelay (for each execution) is taken from cfg (runcard)
            * syncdelay (for each measurement) is defined explicitly
            * wait_initialize is defined explicitly

            * super.__init__
        """

        self.soc = soc
        # No need for a different soc config object since qick is on board
        self.soccfg = soc
        # fill the self.pulse_sequence and the self.readout_pulses oject
        self.sequence = sequence
        self.qubits = qubits

        # conversion coefficients (in runcard we have Hz and ns)
        self.MHz = 0.000001
        self.us = 0.001

        # general settings
        self.max_gain = cfg["max_gain"]
        self.adc_trig_offset = cfg["adc_trig_offset"]
        self.max_sampling_rate = cfg["sampling_rate"]

        # TODO maybe better elsewhere
        # relax_delay is the time waited at the end of the program (for ADC)
        # syncdelay is the time waited at the end of every measure (overall t)
        # wait_initialize is the time waited at the end of initialize
        # all of these are converted using tproc CLK
        self.relax_delay = self.us2cycles(cfg["repetition_duration"] * self.us)
        self.syncdelay = self.us2cycles(0)
        self.wait_initialize = self.us2cycles(2.0)

        super().__init__(soc, cfg)

    def acquire(
        self,
        soc: QickSoc,
        readouts_per_experiment: int = 1,
        load_pulses: bool = True,
        progress: bool = False,
        debug: bool = False,
        average: bool = False,
    ) -> Tuple[List[float], List[float]]:
        """Calls the super() acquire function.

        Args:
            readouts_per_experiment (int): relevant for internal acquisition
            load_pulse, progress, debug (bool): internal Qick parameters
            average (bool): if true return averaged res, otherwise single shots
        """
        if average:
            return super().acquire(
                soc,
                readouts_per_experiment=readouts_per_experiment,
                load_pulses=load_pulses,
                progress=progress,
                debug=debug,
            )
        else:
            # super().acquire function fill buffers used in collect_shots
            super().acquire(
                soc,
                readouts_per_experiment=readouts_per_experiment,
                load_pulses=load_pulses,
                progress=progress,
                debug=debug,
            )
            return self.collect_shots()

    def collect_shots(self) -> Tuple[List[float], List[float]]:
        """Reads the internal buffers and returns single shots (i,q)"""
        tot_i = []
        tot_q = []

        adcs = []  # list of adcs per readouts (not unique values)
        lengths = []  # length of readouts (only one per adcs)
        for pulse in self.sequence.ro_pulses:
            adc_ch = self.qubits[pulse.qubit].feedback.ports[0][1]
            ro_ch = self.qubits[pulse.qubit].readout.ports[0][1]
            if adc_ch not in adcs:
                lengths.append(self.soc.us2cycles(pulse.duration * self.us, gen_ch=ro_ch))
            adcs.append(adc_ch)

        adcs, adc_count = np.unique(adcs, return_counts=True)

        for idx, adc_ch in enumerate(adcs):
            count = adc_count[adc_ch]
            i_val = self.di_buf[idx].reshape((count, self.cfg["reps"])) / lengths[idx]
            q_val = self.dq_buf[idx].reshape((count, self.cfg["reps"])) / lengths[idx]

            tot_i.append(i_val)
            tot_q.append(q_val)
        return tot_i, tot_q

    def initialize(self):
        """This function gets called automatically by qick super.__init__,

        it contains:
        * declaration of channels and nyquist zones
        * declaration of readouts (just one per channel, otherwise ignores it)
        * for element in sequence calls the add_pulse_to_register function
          (if first pulse for channel, otherwise it will be done in the body)

        """

        # declare nyquist zones for all used channels
        ch_already_declared = []
        for pulse in self.sequence:
            qd_ch = self.qubits[pulse.qubit].drive.ports[0][1]
            ro_ch = self.qubits[pulse.qubit].readout.ports[0][1]
            gen_ch = qd_ch if pulse.type == PulseType.DRIVE else ro_ch

            if gen_ch not in ch_already_declared:
                ch_already_declared.append(gen_ch)

                if pulse.frequency < self.max_sampling_rate / 2:
                    self.declare_gen(gen_ch, nqz=1)
                else:
                    self.declare_gen(gen_ch, nqz=2)

        # declare readouts
        ro_ch_already_declared = []
        for readout_pulse in self.sequence.ro_pulses:
            adc_ch = self.qubits[readout_pulse.qubit].feedback.ports[0][1]
            ro_ch = self.qubits[readout_pulse.qubit].readout.ports[0][1]
            if adc_ch not in ro_ch_already_declared:
                ro_ch_already_declared.append(adc_ch)
                length = self.soc.us2cycles(readout_pulse.duration * self.us, gen_ch=ro_ch)
                freq = readout_pulse.frequency * self.MHz
                # in declare_readout frequency in MHz
                self.declare_readout(ch=adc_ch, length=length, freq=freq, gen_ch=ro_ch)

        # register first pulses of all channels
        first_pulse_registered = []
        for pulse in self.sequence:
            qd_ch = self.qubits[pulse.qubit].drive.ports[0][1]
            ro_ch = self.qubits[pulse.qubit].readout.ports[0][1]

            gen_ch = qd_ch if pulse.type == PulseType.DRIVE else ro_ch

            if gen_ch not in first_pulse_registered:
                first_pulse_registered.append(gen_ch)
                self.add_pulse_to_register(pulse)

        # sync all channels and wait some time
        self.sync_all(self.wait_initialize)

    def add_pulse_to_register(self, pulse: Pulse):
        """This function calls the set_pulse_registers function"""

        # find channels relevant for this pulse
        qd_ch = self.qubits[pulse.qubit].drive.ports[0][1]
        adc_ch = self.qubits[pulse.qubit].feedback.ports[0][1]
        ro_ch = self.qubits[pulse.qubit].readout.ports[0][1]
        gen_ch = qd_ch if pulse.type == PulseType.DRIVE else ro_ch

        # convert amplitude in gain and check is valid
        gain = int(pulse.amplitude * self.max_gain)
        if abs(gain) > self.max_gain:
            raise Exception("Amp must be in [-1,1], was: {pulse.amplitude}")

        # phase converted from rad (qibolab) to deg (qick) and then to reg vals
        phase = self.deg2reg(np.degrees(pulse.relative_phase), gen_ch=gen_ch)

        # pulse length converted with DAC CLK
        us_length = pulse.duration * self.us
        soc_length = self.soc.us2cycles(us_length, gen_ch=gen_ch)

        is_drag = isinstance(pulse.shape, Drag)
        is_gaus = isinstance(pulse.shape, Gaussian)
        is_rect = isinstance(pulse.shape, Rectangular)

        # pulse freq converted with frequency matching
        if pulse.type == PulseType.DRIVE:
            freq = self.soc.freq2reg(pulse.frequency * self.MHz, gen_ch=gen_ch)
        elif pulse.type == PulseType.READOUT:
            freq = self.soc.freq2reg(pulse.frequency * self.MHz, gen_ch=gen_ch, ro_ch=adc_ch)
        else:
            raise Exception(f"Pulse type {pulse.type} not recognized!")

        # if pulse is drag or gauss first define the i-q shape and then set reg
        if is_drag or is_gaus:
            name = pulse.serial
            sigma = us_length / pulse.shape.rel_sigma
            sigma = self.soc.us2cycles(
                us_length / pulse.shape.rel_sigma, gen_ch=gen_ch
            )  # TODO probably conversion is linear

            if is_gaus:
                self.add_gauss(ch=gen_ch, name=name, sigma=sigma, length=soc_length)

            elif is_drag:
                self.add_DRAG(
                    ch=gen_ch,
                    name=name,
                    sigma=sigma,
                    delta=sigma,  # TODO: check if correct
                    alpha=pulse.beta,
                    length=soc_length,
                )

            self.set_pulse_registers(
                ch=gen_ch,
                style="arb",
                freq=freq,
                phase=phase,
                gain=gain,
                waveform=name,
            )

        # if pulse is rectangular set directly register
        elif is_rect:
            self.set_pulse_registers(ch=gen_ch, style="const", freq=freq, phase=phase, gain=gain, length=soc_length)

        else:
            raise NotImplementedError(f"Shape {pulse.shape} not supported!")

    def body(self):
        """Execute sequence of pulses.

        If the pulse is already loaded in the register just launch it,
        otherwise first calls the add_pulse_to_register function.

        If readout it does a measurement with an adc trigger, it does not wait.

        At the end of the pulse wait for clock.
        """

        # list of channels where a pulse is already been executed
        first_pulse_executed = []

        for pulse in self.sequence:
            # time follows tproc CLK
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
                    adc_trig_offset=time + self.adc_trig_offset,
                    t=time,
                    wait=False,
                    syncdelay=self.syncdelay,
                )
        self.wait_all()
        self.sync_all(self.relax_delay)


class ExecuteSingleSweep(RAveragerProgram):
    """This qick AveragerProgram handles a qibo sequence of pulses"""

    def __init__(self, soc: QickSoc, cfg: dict, sequence: PulseSequence, qubits: List[Qubit], sweeper: Sweeper):
        """In this function we define the most important settings.

        In detail:
            * set the conversion coefficients to be used for frequency and time
            * max_gain, adc_trig_offset, max_sampling_rate are imported from
              cfg (runcard settings)

            * relaxdelay (for each execution) is taken from cfg (runcard )
            * syncdelay (for each measurement) is defined explicitly
            * wait_initialize is defined explicitly

            * the cfg["expts"] (number of sweeped values) is set

            * super.__init__
        """

        self.soc = soc
        # No need for a different soc config object since qick is on board
        self.soccfg = soc
        # fill the self.pulse_sequence and the self.readout_pulses oject
        self.sequence = sequence
        self.qubits = qubits

        # conversion coefficients (in runcard we have Hz and ns)
        self.MHz = 0.000001
        self.us = 0.001

        # settings
        self.max_gain = cfg["max_gain"]
        self.adc_trig_offset = cfg["adc_trig_offset"]
        self.max_sampling_rate = cfg["sampling_rate"]

        # TODO maybe better elsewhere
        # relax_delay is the time waited at the end of the program (for ADC)
        # syncdelay is the time waited at the end of every measure
        # wait_initialize is the time waited at the end of initialize
        # all of these are converted using tproc CLK
        self.relax_delay = self.us2cycles(cfg["repetition_duration"] * self.us)
        self.syncdelay = self.us2cycles(0)
        self.wait_initialize = self.us2cycles(2.0)

        # sweeper Settings
        self.sweeper = sweeper
        self.sweeper_reg = None
        self.sweeper_page = None
        cfg["expts"] = len(sweeper.values)

        super().__init__(soc, cfg)

    def acquire(
        self,
        soc: QickSoc,
        readouts_per_experiment: int = 1,
        load_pulses: bool = True,
        progress: bool = False,
        debug: bool = False,
        average: bool = False,
    ) -> Tuple[List[float], List[float]]:
        """Calls the super() acquire function.

        Args:
            readouts_per_experiment (int): relevant for internal acquisition
            load_pulse, progress, debug (bool): internal Qick parameters
            average (bool): if true return averaged res, otherwise single shots
        """
        if average:
            _, i_val, q_val = super().acquire(
                soc,
                readouts_per_experiment=readouts_per_experiment,
                load_pulses=load_pulses,
                progress=progress,
                debug=debug,
            )
            return i_val, q_val
        else:
            # super().acquire function fill buffers used in collect_shots
            super().acquire(
                soc,
                readouts_per_experiment=readouts_per_experiment,
                load_pulses=load_pulses,
                progress=progress,
                debug=debug,
            )
            return self.collect_shots()

    def collect_shots(self) -> Tuple[List[float], List[float]]:
        """Reads the internal buffers and returns single shots (i,q)"""
        tot_i = []
        tot_q = []

        adcs = []  # list of adcs per readouts (not unique values)
        lengths = []  # length of readouts (only one per adcs)
        for pulse in self.sequence.ro_pulses:
            adc_ch = self.qubits[pulse.qubit].feedback.ports[0][1]
            ro_ch = self.qubits[pulse.qubit].readout.ports[0][1]
            if adc_ch not in adcs:
                lengths.append(self.soc.us2cycles(pulse.duration * self.us, gen_ch=ro_ch))
            adcs.append(adc_ch)

        adcs, adc_count = np.unique(adcs, return_counts=True)

        for idx, adc_ch in enumerate(adcs):
            count = adc_count[adc_ch]
            i_val = self.di_buf[idx].reshape((count, self.cfg["expts"], self.cfg["reps"])) / lengths[idx]
            q_val = self.dq_buf[idx].reshape((count, self.cfg["expts"], self.cfg["reps"])) / lengths[idx]

            tot_i.append(i_val)
            tot_q.append(q_val)
        return tot_i, tot_q

    def initialize(self):
        """This function gets called automatically by qick super.__init__,

        it contains:
        * declaration of sweeper register settings
        * declaration of channels and nyquist zones
        * declaration of readouts (just one per channel, otherwise ignores it)
        * for element in sequence calls the add_pulse_to_register function
          (if first pulse for channel, otherwise it will be done in the body)

        """

        # find channels of sweeper pulse
        pulse = self.sweeper.pulses[0]
        qd_ch = self.qubits[pulse.qubit].drive.ports[0][1]
        adc_ch = self.qubits[pulse.qubit].feedback.ports[0][1]
        ro_ch = self.qubits[pulse.qubit].readout.ports[0][1]
        gen_ch = qd_ch if pulse.type == PulseType.DRIVE else ro_ch

        # find page of sweeper pulse channel
        self.sweeper_page = self.ch_page(gen_ch)

        # define start and step values
        start = self.sweeper.values[0]
        step = self.sweeper.values[1] - self.sweeper.values[0]

        # find register of sweeped parameter and assign start and step
        if self.sweeper.parameter == Parameter.frequency:
            self.sweeper_reg = self.sreg(gen_ch, "freq")
            self.cfg["start"] = self.soc.freq2reg(start * self.MHz, gen_ch)
            self.cfg["step"] = self.soc.freq2reg(step * self.MHz, gen_ch)

            # TODO: should stop if nyquist zone changes in the sweep

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
            ro_ch = self.qubits[pulse.qubit].readout.ports[0][1]
            gen_ch = qd_ch if pulse.type == PulseType.DRIVE else ro_ch

            if gen_ch not in ch_already_declared:
                ch_already_declared.append(gen_ch)

                if pulse.frequency < self.max_sampling_rate / 2:
                    self.declare_gen(gen_ch, nqz=1)
                else:
                    self.declare_gen(gen_ch, nqz=2)

        # declare readouts
        ro_ch_already_declared = []
        for readout_pulse in self.sequence.ro_pulses:
            adc_ch = self.qubits[pulse.qubit].feedback.ports[0][1]
            ro_ch = self.qubits[pulse.qubit].readout.ports[0][1]
            if adc_ch not in ro_ch_already_declared:
                ro_ch_already_declared.append(adc_ch)
                length = self.soc.us2cycles(readout_pulse.duration * self.us, gen_ch=ro_ch)
                freq = readout_pulse.frequency * self.MHz
                # for declare_readout freqs in MHz and not in register values
                self.declare_readout(ch=adc_ch, length=length, freq=freq, gen_ch=ro_ch)

        # register first pulses of all channels
        first_pulse_registered = []
        for pulse in self.sequence:
            qd_ch = self.qubits[pulse.qubit].drive.ports[0][1]
            ro_ch = self.qubits[pulse.qubit].readout.ports[0][1]
            gen_ch = qd_ch if pulse.type == PulseType.DRIVE else ro_ch

            if gen_ch not in first_pulse_registered:
                first_pulse_registered.append(gen_ch)
                self.add_pulse_to_register(pulse)

        # sync all channels and wait some time
        self.sync_all(self.wait_initialize)

    def add_pulse_to_register(self, pulse):
        """This function calls the set_pulse_registers function"""

        is_sweeped = self.sweeper.pulses[0] == pulse

        # find channels relevant for this pulse
        qd_ch = self.qubits[pulse.qubit].drive.ports[0][1]
        adc_ch = self.qubits[pulse.qubit].feedback.ports[0][1]
        ro_ch = self.qubits[pulse.qubit].readout.ports[0][1]
        gen_ch = qd_ch if pulse.type == PulseType.DRIVE else ro_ch

        # assign gain parameter
        if is_sweeped and self.sweeper.parameter == Parameter.amplitude:
            gain = self.cfg["start"]
        else:
            gain = int(pulse.amplitude * self.max_gain)

        if abs(gain) > self.max_gain:
            raise Exception("Amp must be in [-1,1], was: {pulse.amplitude}")

        # phase converted from rad (qibolab) to deg (qick) and to register vals
        phase = self.deg2reg(np.degrees(pulse.relative_phase), gen_ch=gen_ch)

        # pulse length converted with DAC CLK
        us_length = pulse.duration * self.us
        soc_length = self.soc.us2cycles(us_length, gen_ch=gen_ch)

        is_drag = isinstance(pulse.shape, Drag)
        is_gaus = isinstance(pulse.shape, Gaussian)
        is_rect = isinstance(pulse.shape, Rectangular)

        # pulse freq converted with frequency matching
        if pulse.type == PulseType.DRIVE:
            if is_sweeped and self.sweeper.parameter == Parameter.frequency:
                freq = self.cfg["start"]
            else:
                freq = self.soc.freq2reg(pulse.frequency * self.MHz, gen_ch=gen_ch)

        elif pulse.type == PulseType.READOUT:
            freq = self.soc.freq2reg(pulse.frequency * self.MHz, gen_ch=gen_ch, ro_ch=adc_ch)
        else:
            raise Exception(f"Pulse type {pulse.type} not recognized!")

        # if pulse is drag or gaus first define the i-q shape and then set regs
        if is_drag or is_gaus:
            name = pulse.serial
            sigma = us_length / pulse.shape.rel_sigma
            sigma = self.soc.us2cycles(
                us_length / pulse.shape.rel_sigma, gen_ch=gen_ch
            )  # TODO probably conversion is linear

            if is_gaus:
                self.add_gauss(ch=gen_ch, name=name, sigma=sigma, length=soc_length)

            elif is_drag:
                self.add_DRAG(
                    ch=gen_ch,
                    name=name,
                    sigma=sigma,
                    delta=sigma,  # TODO: check if correct
                    alpha=pulse.beta,
                    length=soc_length,
                )

            self.set_pulse_registers(
                ch=gen_ch,
                style="arb",
                freq=freq,
                phase=phase,
                gain=gain,
                waveform=name,
            )

        # if pulse is rectangular set directly register
        elif is_rect:
            self.set_pulse_registers(ch=gen_ch, style="const", freq=freq, phase=phase, gain=gain, length=soc_length)

        else:
            raise NotImplementedError(f"Shape {pulse.shape} not supported!")

    def update(self):
        """Update function for sweeper"""
        self.mathi(self.sweeper_page, self.sweeper_reg, self.sweeper_reg, "+", self.cfg["step"])

    def body(self):
        """Execute sequence of pulses.

        If the pulse is already loaded in the register just launch it,
        otherwise first calls the add_pulse_to_register function.

        If readout it does a measurement with an adc trigger, it does not wait.

        At the end of the pulse wait for clock and call update function.
        """

        # list of channels where a pulse is already been executed
        first_pulse_executed = []

        for pulse in self.sequence:
            # time follows tproc CLK
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
                    adc_trig_offset=time + self.adc_trig_offset,
                    t=time,
                    wait=False,
                    syncdelay=self.syncdelay,
                )
        self.wait_all()
        self.sync_all(self.relax_delay)


class TII_RFSOC4x2(AbstractInstrument):
    """Instrument object for controlling the RFSoC4x2 FPGA.

    Playing pulses requires first the execution of the ``setup`` function.
    The two way of executing pulses are with ``play`` (for arbitrary
    qibolab ``PulseSequence``) or with ``sweep`` that execute a
    ``PulseSequence`` object with one or more ``Sweeper``.

    Args:
        name (str): Name of the instrument instance.

    Attributes:
        cfg (dict): Configuration dictionary required for pulse execution.
        soc (QickSoc): ``Qick`` object needed to access system blocks.
    """

    def __init__(self, name: str):
        # address is None since qibolab is on board
        super().__init__(name, address=None)
        self.cfg: dict = {}  # dictionary with runcard, filled in setup()
        self.soc = QickSoc()  # QickSoc object
        self.states_calibrated = False

    def connect(self):
        """Empty method to comply with AbstractInstrument interface."""

    def start(self):
        """Empty method to comply with AbstractInstrument interface."""

    def stop(self):
        """Empty method to comply with AbstractInstrument interface."""

    def disconnect(self):
        """Empty method to comply with AbstractInstrument interface."""

    def setup(
        self,
        qubits: List[Qubit],
        sampling_rate: int,
        repetition_duration: int,
        adc_trig_offset: int,
        max_gain: int,
        calibrate: bool = True,
        **kwargs,
    ):
        """Configures the instrument.

        Args: Settings taken from runcard (except calibrate argument)
            calibrate(bool): if true runs the calibrate_state routine
                             getting treshold and angle
            qubits (list): list of `qibolab.platforms.abstract.Qubit`
            sampling_rate (int): sampling rate of the RFSoC (Hz).
            repetition_duration (int): delay before readout (ns).
            adc_trig_offset (int): single offset for all adc triggers
                                   (tproc CLK ticks).
            max_gain (int): maximum output power of the DAC (DAC units).

            **kwargs: no additional arguments are expected and used
        """

        self.cfg = {
            "sampling_rate": sampling_rate,
            "repetition_duration": repetition_duration,
            "adc_trig_offset": adc_trig_offset,
            "max_gain": max_gain,
        }

        if calibrate:
            self.calibration_cfg = {}
            self.calibrate_states(qubits)
            self.states_calibrated = True

    def calibrate_states(self, qubits: List[Qubit]):
        """Runs a calibration and sets threshold and angle paramters"""
        # TODO maybe this could be moved to create_tii_rfsoc4x2() to
        #      use create_RX_pulse and create_MZ_pulse
        # TODO integration with qibocal routines

        for qubit in qubits:
            self.calibration_cfg[qubit] = {}

            # definition of MZ pulse
            ro_duration = 2000
            ro_frequency = qubits[qubit].readout_frequency
            ro_amplitude = 0.046
            ro_shape = Rectangular()
            ro_channel = qubits[qubit].readout.name
            # definition of RX pulse
            qd_duration = 28
            qd_frequency = qubits[qubit].drive_frequency
            qd_amplitude = 0.09
            qd_shape = Rectangular()
            qd_channel = qubits[qubit].drive.name
            # sequence that should measure 0
            sequence0 = PulseSequence()
            ro_pulse = ReadoutPulse(0, ro_duration, ro_amplitude, ro_frequency, 0, ro_shape, ro_channel, qubit=qubit)

            sequence0.add(ro_pulse)
            res0 = self.play(qubits, sequence0, nshots=2000)[ro_pulse.serial]

            # sequence that should measure 1
            sequence1 = PulseSequence()
            qd_pulse = Pulse(0, qd_duration, qd_amplitude, qd_frequency, 0, qd_shape, qd_channel, qubit=qubit)
            ro_pulse = ReadoutPulse(
                qd_pulse.finish, ro_duration, ro_amplitude, ro_frequency, 0, ro_shape, ro_channel, qubit=qubit
            )

            sequence1.add(qd_pulse)
            sequence1.add(ro_pulse)
            res1 = self.play(qubits, sequence1, nshots=2000)[ro_pulse.serial]

            #
            i_zero = res0.i
            q_zero = res0.q
            i_one = res1.i
            q_one = res1.q

            x_zero, y_zero = np.median(i_zero), np.median(q_zero)
            x_one, y_one = np.median(i_one), np.median(q_one)

            theta = -np.arctan2((y_one - y_zero), (x_one - x_zero))

            ig_new = i_zero * np.cos(theta) - q_zero * np.sin(theta)
            ie_new = i_one * np.cos(theta) - q_one * np.sin(theta)

            numbins = 200
            n_zero, binsg = np.histogram(ig_new, bins=numbins)
            n_one, binse = np.histogram(ie_new, bins=numbins)

            contrast = np.abs((np.cumsum(n_zero) - np.cumsum(n_one)) / (0.5 * n_zero.sum() + 0.5 * n_one.sum()))
            tind = contrast.argmax()
            threshold = binsg[tind]

            self.calibration_cfg[qubit]["threshold"] = threshold
            self.calibration_cfg[qubit]["rotation_angle"] = theta

    def play(
        self, qubits: List[Qubit], sequence: PulseSequence, relaxation_time: int = None, nshots: int = 1000
    ) -> Dict[str, ExecutionResults]:
        """Executes the sequence of instructions and retrieves readout results.

        Each readout pulse generates a separate acquisition.
        The relaxation_time and the number of shots have default values.

        Args:
            qubits (list): List of `qibolab.platforms.utils.Qubit` objects
                           passed from the platform.
            sequence (`qibolab.pulses.PulseSequence`). Pulse sequence to play.
            nshots (int): Number of repetitions (shots) of the experiment.
            relaxation_time (int): Time to wait for the qubit to relax to its
                                   ground state between shots in ns.
        Returns:
            A dictionary mapping the readout pulses serial to
            `qibolab.ExecutionResults` objects
        """

        # TODO: repetition_duration Vs relaxation_time
        # TODO: nshots Vs relaxation_time

        # nshots gets added to configuration dictionary
        self.cfg["reps"] = nshots
        # if new value is passed, relaxation_time is updated in the dictionary
        # TODO: actually this changes the actual dictionary, maybe better not
        if relaxation_time is not None:
            self.cfg["repetition_duration"] = relaxation_time

        program = ExecutePulseSequence(self.soc, self.cfg, sequence, qubits)
        average = False
        toti, totq = program.acquire(
            self.soc,
            readouts_per_experiment=len(sequence.ro_pulses),
            load_pulses=True,
            progress=False,
            debug=False,
            average=average,
        )

        results = {}
        adcs = np.unique([qubits[p.qubit].feedback.ports[0][1] for p in sequence.ro_pulses])
        for j in range(len(adcs)):
            for i, ro_pulse in enumerate(sequence.ro_pulses):
                i_pulse = np.array(toti[j][i])
                q_pulse = np.array(totq[j][i])

                serial = ro_pulse.serial

                shots = self.classify_shots(i_pulse, q_pulse, qubits[ro_pulse.qubit])
                results[serial] = ExecutionResults.from_components(i_pulse, q_pulse, shots)

        return results

    def classify_shots(self, i_values: List[float], q_values: List[float], qubit: Qubit) -> List[float]:
        """Classify IQ values using qubit threshold and rotation_angle"""
        if self.states_calibrated:
            angle = self.calibration_cfg[qubit.name]["rotation_angle"]
            threshold = self.calibration_cfg[qubit.name]["threshold"]
        else:
            if qubit.rotation_angle is None or qubit.threshold is None:
                return None
            angle = np.radians(qubit.rotation_angle)
            threshold = qubit.threshold

        rotated = np.cos(angle) * np.array(i_values) - np.sin(angle) * np.array(q_values)
        shots = np.heaviside(np.array(rotated) - threshold, 0)
        return shots

    def recursive_python_sweep(
        self,
        qubits: List[Qubit],
        sequence: PulseSequence,
        or_sequence: PulseSequence,
        *sweepers: Sweeper,
        average: bool,
    ) -> Dict[str, Union[AveragedResults, ExecutionResults]]:
        """Execute a sweep of an arbitrary number of Sweepers via recursion.

        Args:
            qubits (list): List of `qibolab.platforms.utils.Qubit` objects
                    passed from the platform.
            sequence (`qibolab.pulses.PulseSequence`): Pulse sequence to play.
                    This object is a deep copy of the original
                    sequence and gets modified.
            or_sequence (`qibolab.pulses.PulseSequence`): Reference to original
                    sequence to not modify.
            *sweepers (`qibolab.Sweeper`): Sweeper objects.
        Returns:
            A dictionary mapping the readout pulses serial to qibolab
            results objects
        Raises:
            NotImplementedError: if a sweep refers to more than one pulse.
            NotImplementedError: if a sweep refers to a parameter different
                                 from frequency or amplitude.
        """
        # gets a list containing the original sequence output serials
        original_ro = [ro.serial for ro in or_sequence.ro_pulses]

        # If there are no sweepers run ExecutePulseSequence acquisition.
        # Last layer for recursion.
        if len(sweepers) == 0:
            program = ExecutePulseSequence(self.soc, self.cfg, sequence, qubits)
            toti, totq = program.acquire(
                self.soc,
                readouts_per_experiment=len(sequence.ro_pulses),
                load_pulses=True,
                progress=False,
                debug=False,
                average=average,
            )
            results = {}
            adcs = np.unique([qubits[p.qubit].feedback.ports[0][1] for p in sequence.ro_pulses])
            for j in range(len(adcs)):
                for i, serial in enumerate(original_ro):
                    i_pulse = np.array(toti[j][i])
                    q_pulse = np.array(totq[j][i])

                    if average:
                        results[serial] = AveragedResults(i_pulse, q_pulse)
                    else:
                        qubit = qubits[or_sequence.ro_pulses[i].qubit]
                        shots = self.classify_shots(i_pulse, q_pulse, qubit)
                        results[serial] = ExecutionResults.from_components(i_pulse, q_pulse, shots)
            return results

        # If sweepers are still in queue
        else:
            # check that the first (outest) sweeper is supported
            sweeper = sweepers[0]
            if len(sweeper.pulses) > 1:
                raise NotImplementedError("Only one pulse per sweep supported")
            is_amp = sweeper.parameter == Parameter.amplitude
            is_freq = sweeper.parameter == Parameter.frequency
            if not (is_amp or is_freq):
                raise NotImplementedError("Parameter type not implemented")

            # if there is one sweeper supported by qick than use hardware sweep
            if len(sweepers) == 1 and not self.get_if_python_sweep(sequence, qubits, *sweepers):
                program = ExecuteSingleSweep(self.soc, self.cfg, sequence, qubits, sweepers[0])
                toti, totq = program.acquire(
                    self.soc,
                    readouts_per_experiment=len(sequence.ro_pulses),
                    load_pulses=True,
                    progress=False,
                    debug=False,
                    average=average,
                )
                if average:
                    # convert averaged results
                    res = self.convert_av_sweep_results(sweepers[0], original_ro, toti, totq)
                else:
                    # convert not averaged results
                    res = self.convert_nav_sweep_results(sweepers[0], original_ro, sequence, qubits, toti, totq)
                return res

            # if it's not possible to execute qick sweep re-call function
            else:
                sweep_results = {}
                idx_pulse = or_sequence.index(sweeper.pulses[0])
                for val in sweeper.values:
                    if is_freq:
                        sequence[idx_pulse].frequency = val
                    elif is_amp:
                        sequence[idx_pulse].amplitude = val
                    res = self.recursive_python_sweep(qubits, sequence, or_sequence, *sweepers[1:], average=average)
                    # merge the dictionary obtained with the one already saved
                    sweep_results = self.merge_sweep_results(sweep_results, res)
        return sweep_results

    def merge_sweep_results(
        self,
        dict_a: Dict[str, Union[AveragedResults, ExecutionResults]],
        dict_b: Dict[str, Union[AveragedResults, ExecutionResults]],
    ) -> Dict[str, Union[AveragedResults, ExecutionResults]]:
        """Merge two dictionary mapping pulse serial to Results object.
        If dict_b has a key (serial) that dict_a does not have, simply add it,
        otherwise sum the two results (`qibolab.result.ExecutionResults`
        or `qibolab.result.AveragedResults`)
        Args:
            dict_a (dict): dict mapping ro pulses serial to qibolab res objects
            dict_b (dict): dict mapping ro pulses serial to qibolab res objects
        Returns:
            A dict mapping the readout pulses serial to qibolab results objects
        """
        for serial in dict_b:
            if serial in dict_a:
                dict_a[serial] = dict_a[serial] + dict_b[serial]
            else:
                dict_a[serial] = dict_b[serial]
        return dict_a

    def get_if_python_sweep(self, sequence: PulseSequence, qubits: List[Qubit], *sweepers: Sweeper) -> bool:
        """Check if a sweeper must be run with python loop or on hardware.

        To be run on qick internal loop a sweep must:
            * not be on the readout frequency
            * be the first pulse of a channel
            * be just one sweeper

        Args:
            sequence (`qibolab.pulses.PulseSequence`). Pulse sequence to play.
            qubits (list): List of `qibolab.platforms.utils.Qubit` objects
                           passed from the platform.
            *sweepers (`qibolab.Sweeper`): Sweeper objects.
        Returns:
            A boolean value true if the sweeper must be executed by python
            loop, false otherwise
        """

        # is_amp = sweepers[0].parameter == Parameter.amplitude
        is_freq = sweepers[0].parameter == Parameter.frequency

        # if there isn't only a sweeper do a python sweep
        if len(sweepers) != 1:
            return True

        is_ro = sweepers[0].pulses[0].type == PulseType.READOUT
        # if it's a sweep on the readout freq do a python sweep
        if is_freq and is_ro:
            return True

        # check if the sweeped pulse is the first on the DAC channel
        for pulse in sequence:
            pulse_q = qubits[pulse.qubit]
            sweep_q = qubits[sweepers[0].pulses[0].qubit]
            pulse_ch = pulse_q.feedback[0][1] if is_ro else pulse_q.drive.ports[0][1]
            sweep_ch = sweep_q.feedback[0][1] if is_ro else sweep_q.drive.ports[0][1]
            is_same_ch = pulse_ch == sweep_ch
            is_same_pulse = pulse.serial == sweepers[0].pulses[0].serial
            # if channels are equal and pulses are equal we can hardware sweep
            if is_same_ch and is_same_pulse:
                return False
            elif is_same_ch and not is_same_pulse:
                return True

        # this return should not be reachable, here for safety
        return True

    def convert_av_sweep_results(
        self, sweeper: Sweeper, original_ro: List[str], avgi: List[float], avgq: List[float]
    ) -> Dict[str, AveragedResults]:
        """Convert sweep results from acquire(average=True) to qibolab dict res
        Args:
            *sweepers (`qibolab.Sweeper`): Sweeper objects.
            original_ro (list): list of the ro serials of the original sequence
            avgi (list): averaged i vals obtained with `acquire(average=True)`
            avgq (list): averaged q vals obtained with `acquire(average=True)`
        Returns:
            A dict mapping the readout pulses serial to qibolab results objects
        """
        # TODO extend to readout on multiple adcs
        sweep_results = {}
        # add a result for every value of the sweep
        for j in range(len(sweeper.values)):
            results = {}
            # add a result for every readouts pulse
            for i, serial in enumerate(original_ro):
                i_pulse = np.array([avgi[0][i][j]])
                q_pulse = np.array([avgq[0][i][j]])
                results[serial] = AveragedResults(i_pulse, q_pulse)
            # merge new result with already saved ones
            sweep_results = self.merge_sweep_results(sweep_results, results)
        return sweep_results

    def convert_nav_sweep_results(
        self,
        sweeper: Sweeper,
        original_ro: List[str],
        sequence: PulseSequence,
        qubits: List[Qubit],
        toti: List[float],
        totq: List[float],
    ) -> Dict[str, ExecutionResults]:
        """Convert sweep res from acquire(average=False) to qibolab dict res
        Args:
            *sweepers (`qibolab.Sweeper`): Sweeper objects.
            original_ro (list): list of ro serials of the original sequence
            sequence (`qibolab.pulses.PulseSequence`). Pulse sequence to play.
            qubits (list): List of `qibolab.platforms.utils.Qubit` objects
                 passed from the platform.
            toti (list): i values obtained with `acquire(average=True)`
            totq (list): q values obtained with `acquire(average=True)`
        Returns:
            A dict mapping the readout pulses serial to qibolab results objects
        """
        sweep_results = {}

        adcs = np.unique([qubits[p.qubit].feedback.ports[0][1] for p in sequence.ro_pulses])
        for k in range(len(adcs)):
            for j in range(len(sweeper.values)):
                results = {}
                # add a result for every readouts pulse
                for i, serial in enumerate(original_ro):
                    i_pulse = np.array(toti[k][i][j])
                    q_pulse = np.array(totq[k][i][j])

                    qubit = qubits[sequence.ro_pulses[i].qubit]
                    shots = self.classify_shots(i_pulse, q_pulse, qubit)
                    results[serial] = ExecutionResults.from_components(i_pulse, q_pulse, shots)
                # merge new result with already saved ones
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
    ) -> Dict[str, Union[AveragedResults, ExecutionResults]]:
        """Executes the sweep and retrieves the readout results.

        Each readout pulse generates a separate acquisition.
        The relaxation_time and the number of shots have default values.

        Args:
            qubits (list): List of `qibolab.platforms.utils.Qubit` objects
                           passed from the platform.
            sequence (`qibolab.pulses.PulseSequence`). Pulse sequence to play.
            *sweepers (`qibolab.Sweeper`): Sweeper objects.
            relaxation_time (int): Time to wait for the qubit to relax to its
                                   ground state between shots in ns.
            nshots (int): Number of repetitions (shots) of the experiment.
            average (bool): if False returns single shot measurements
        Returns:
            A dict mapping the readout pulses serial to qibolab results objects
        """

        # nshots gets added to configuration dictionary
        self.cfg["reps"] = nshots
        # if new value is passed, relaxation_time is updated in the dictionary
        # TODO: actually this changes the actual dictionary, maybe better not
        if relaxation_time is not None:
            self.cfg["repetition_duration"] = relaxation_time

        # sweepers.values are modified to reflect actual sweeped values
        for sweeper in sweepers:
            if sweeper.parameter == Parameter.frequency:
                sweeper.values += sweeper.pulses[0].frequency
            elif sweeper.parameter == Parameter.amplitude:
                continue  # amp does not need modification, here for clarity

        # deep copy of the sequence that can be modified without harm
        sweepsequence = sequence.copy()

        results = self.recursive_python_sweep(qubits, sweepsequence, sequence, *sweepers, average=average)

        # sweepers.values are converted back to original relative values
        for sweeper in sweepers:
            if sweeper.parameter == Parameter.frequency:
                sweeper.values -= sweeper.pulses[0].frequency
            elif sweeper.parameter == Parameter.amplitude:
                continue

        return results
