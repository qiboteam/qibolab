# -*- coding: utf-8 -*-
""" Qblox instruments driver.

Supports the following Instruments:
    Cluster
    Cluster QRM-RF
    Cluster QCM-RF
    compatible with qblox-instruments driver 0.6.1 (23/5/2022)

https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/
"""

import json

import numpy as np
from qblox_instruments.qcodes_drivers.cluster import Cluster as QbloxCluster
from qblox_instruments.qcodes_drivers.qcm_qrm import QcmQrm as QbloxQrmQcm
from qblox_instruments.qcodes_drivers.sequencer import Sequencer as QbloxSequencer

from qibolab.instruments.abstract import AbstractInstrument, InstrumentException
from qibolab.pulses import Pulse, PulseSequence, PulseShape, PulseType, Waveform


class WaveformsBuffer:
    """A class to represent a buffer that holds the unique waveforms used by a sequencer.

    Attributes:
        unique_waveforms: list = A list of unique Waveform objects
        available_memory: int = The amount of memory available expressed in numbers of samples
    """

    SIZE: int = 16383

    class NotEnoughMemory(Exception):
        """An error raised when there is not enough memory left to add more waveforms."""

        pass

    def __init__(self):
        """Initialises the buffer with an empty list of unique waveforms."""
        self.unique_waveforms: list = []  # Waveform
        self.available_memory: int = WaveformsBuffer.SIZE

    def add_waveforms(self, waveform_i: Waveform, waveform_q: Waveform):
        """Adds a pair of i and q waveforms to the list of unique waveforms.

        Waveforms are added to the list if they were not there before.
        Each of the waveforms (i and q) is processed individually.

        Args:
            waveform_i: Waveform = A Waveform object containing the samples of the real component
                of the pulse wave.
            waveform_q: Waveform = A Waveform object containing the samples of the imaginary component
                of the pulse wave.

        Raises:
            NotEnoughMemory: If the memory needed to store the waveforms in more than
                the avalible memory.
        """
        if waveform_i not in self.unique_waveforms or waveform_q not in self.unique_waveforms:
            memory_needed = 0
            if not waveform_i in self.unique_waveforms:
                memory_needed += len(waveform_i)
            if not waveform_q in self.unique_waveforms:
                memory_needed += len(waveform_q)

            if self.available_memory >= memory_needed:
                if not waveform_i in self.unique_waveforms:
                    self.unique_waveforms.append(waveform_i)
                if not waveform_q in self.unique_waveforms:
                    self.unique_waveforms.append(waveform_q)
            else:
                raise WaveformsBuffer.NotEnoughMemory


class Sequencer:
    """A class to extend the functionality of qblox_instruments Sequencer.

    A sequencer is a hardware component sinthesised in the instrument FPGA responsible for
    fetching waveforms from memory, preprocessing them and sending them to the ADC/DAC.
    https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/documentation/sequencer.html

    This class extends the sequencer functionality by holding additional data required when
    processing a pulse sequence:
        the sequencer number,
        the sequence of pulses to be played,
        a buffer of unique waveforms, and
        the four components of the sequence file:
            waveforms dictionary
            acquisition dictionary
            weights dictionary
            program

    Attributes:
        TODO add a description of the attributes once they have decided to be public or private
    """

    def __init__(self, number):
        """Initialises the sequencer.

        The reference to the underlying qblox_instruments object is saved and empty
        WaveformsBuffer and PulseSequence instances.
        """
        # TODO initialise self.device

        self.number: int = number
        self.device: QbloxSequencer = None
        self.pulses: PulseSequence = PulseSequence()
        self.waveforms_buffer: WaveformsBuffer = WaveformsBuffer()
        self.waveforms: dict = {}
        self.acquisitions: dict = {}
        self.weights: dict = {}
        self.program: str = ""


class Cluster(AbstractInstrument):
    """A class to extend the functionality of qblox_instruments Cluster.

    The class exposes the attribute `reference_clock_source` that enables the
    selection of an internal or external clock source.

    The class inherits from AbstractInstrument and implements its interface methods:
        __init__()
        connect()
        setup()
        start()
        stop()
        disconnect()

    Attributes:
        reference_clock_source: str = ('internal', 'external') Instructs the cluster to use the
            internal clock source or an external source.
        device: QbloxCluster = A reference to the underlying
            `qblox_instruments.qcodes_drivers.cluster.Cluster` object. It can be used to access other
            features not directly exposed by this wrapper.
            https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/api_reference/cluster.html
    """

    property_wrapper = lambda parent, *parameter: property(
        lambda self: parent.device.get(parameter[0]),
        lambda self, x: parent._set_device_parameter(parent.device, *parameter, value=x),
    )

    def __init__(self, name, address):
        """Initialises the instrument storing its name and address."""
        super().__init__(name, address)
        self.device: QbloxCluster = None
        # self.reference_clock_source: str

        self._device_parameters = {}

    def connect(self):
        """Connects to the instrument.

        If the connection is successful, it saves a reference to the underlying object
        in the attribute `device`.
        The device is reset on each connection.
        """
        global cluster
        if not self.is_connected:
            for attempt in range(3):
                try:
                    QbloxCluster.close_all()
                    self.device = QbloxCluster(self.name, self.address)
                    self.device.reset()
                    cluster = self.device
                    self.is_connected = True
                    # DEBUG: Print Cluster Status
                    # print(self.device.get_system_status())
                    setattr(
                        self.__class__,
                        "reference_clock_source",
                        self.property_wrapper("reference_source"),
                    )
                    break
                except Exception as exc:
                    print(f"Unable to connect:\n{str(exc)}\nRetrying...")
            if not self.is_connected:
                raise InstrumentException(self, f"Unable to connect to {self.name}")
        else:
            pass

    def _set_device_parameter(self, target, *parameters, value):
        """Sets a parameter of the instrument, if it changed from the last stored in the cache.

        Args:
            target = an instance of qblox_instruments.qcodes_drivers.qcm_qrm.QcmQrm or
                                    qblox_instruments.qcodes_drivers.sequencer.Sequencer
            *parameters: list = A list of parameters to be cached and set.
            value = The value to set the paramters.
        Raises:
            Exception = If attempting to set a parameter without a connection to the instrument.
        """
        if self.is_connected:
            key = target.name + "." + parameters[0]
            if not key in self._device_parameters:
                for parameter in parameters:
                    if not hasattr(target, parameter):
                        raise Exception(f"The instrument {self.name} does not have parameters {parameter}")
                    target.set(parameter, value)
                self._device_parameters[key] = value
            elif self._device_parameters[key] != value:
                for parameter in parameters:
                    target.set(parameter, value)
                self._device_parameters[key] = value
        else:
            raise Exception("There is no connection to the instrument {self.name}")

    def _erase_device_parameters_cache(self):
        """Erases the cache of instrument parameters."""
        self._device_parameters = {}

    def setup(self, **kwargs):
        """Configures the instrument with the settings saved in the runcard.

        Args:
            **kwargs: A dictionary with the settings:
                reference_clock_source
        """
        if self.is_connected:
            self.reference_clock_source = kwargs["reference_clock_source"]
        else:
            raise Exception("The instrument cannot be set up, there is no connection")

    def start(self):
        """Empty method to comply with AbstractInstrument interface."""
        pass

    def stop(self):
        """Empty method to comply with AbstractInstrument interface."""
        pass

    def disconnect(self):
        """Closes the connection to the instrument."""
        self.is_connected = False
        self.device.close()
        global cluster
        cluster = None
        # TODO: set modules is_connected to False


cluster: QbloxCluster = None
# TODO: replace the global variable with a collection of clusters implemented
# as a static property of the class Cluster


class ClusterQRM_RF(AbstractInstrument):
    """Qblox Cluster Qubit Readout Module RF driver.

    Qubit Readout Module RF (QRM-RF) is an instrument that integrates an arbitratry
    wave generator, a digitizer, a local oscillator and a mixer. It has one output
    and one input ports. Each port has a path0 and path1 for the i(in-phase) and q(quadrature)
    components of the RF signal. The sampling rate of its ADC/DAC is 1 GSPS.
    https://www.qblox.com/cluster

    The class aims to simplify the configuration of the instrument, exposing only
    those parameters most frequencly used and hiding other more complex components.

    A reference to the underlying `qblox_instruments.qcodes_drivers.qcm_qrm.QRM_QCM`
    object is provided via the attribute `device`, allowing the advanced user to gain
    access to the features that are not exposed directly by the class.

    In order to accelerate the execution, the instrument settings are cached, so that
    the communication with the instrument only happens when the parameters change.
    This caching is done with the method `_set_device_parameter(target, *parameters, value)`.

        ports:
            o1:                                             # output port settings
                attenuation                 : 30                # (dB) 0 to 60, must be multiple of 2
                lo_enabled                  : true
                lo_frequency                : 6_000_000_000     # (Hz) from 2e9 to 18e9
                gain                        : 1                 # for path0 and path1 -1.0<=v<=1.0
                hardware_mod_en             : false
            i1:                                             # input port settings
                hardware_demod_en           : false

            acquisition_hold_off        : 130                   # minimum 4ns
            acquisition_duration        : 1800

            channel_port_map:                                   # Refrigerator Channel : Instrument port
                10: o1
                1: i1

    The class inherits from AbstractInstrument and implements its interface methods:
        __init__()
        connect()
        setup()
        start()
        stop()
        disconnect()

    Attributes:
        name: str = A unique name given to the instrument.
        address: str = IP_address:module_number (the IP address of the cluster and
            the module number)
        ports = A dictionary giving access to the input and output ports objects.
            ports['o1']
            ports['i1']

            ports['o1'].attenuation: int = (mapped to qrm.out0_att) Controls the attenuation applied to
                the output port. Must be a multiple of 2
            ports['o1'].lo_enabled: bool = (mapped to qrm.out0_in0_lo_en) Enables or disables the
                local oscillator.
            ports['o1'].lo_frequency: int = (mapped to qrm.out0_in0_lo_freq) Sets the frequency of the
                local oscillator.
            ports['o1'].gain: float = (mapped to qrm.sequencers[0].gain_awg_path0 and qrm.sequencers[0].gain_awg_path1)
                Sets the gain on both paths of the output port.
            ports['o1'].hardware_mod_en: bool = (mapped to qrm.sequencers[0].mod_en_awg) Enables pulse
                modulation in hardware. When set to False, pulse modulation is done at the host computer
                and a modulated pulse waveform should be uploaded to the instrument. When set to True,
                the envelope of the pulse should be uploaded to the instrument and it modulates it in
                real time by its FPGA using the sequencer nco (numerically controlled oscillator).
            ports['o1'].nco_freq: int = (mapped to qrm.sequencers[0].nco_freq)        # TODO mapped, but not configurable from the runcard
            ports['o1'].nco_phase_offs = (mapped to qrm.sequencers[0].nco_phase_offs) # TODO mapped, but not configurable from the runcard

            ports['i1'].hardware_demod_en: bool = (mapped to qrm.sequencers[0].demod_en_acq) Enables demodulation and integration of the acquired
                pulses in hardware. When set to False, the filtration, demodulation and integration of the
                acquired pulses is done at the host computer. When set to True, the demodulation, integration
                and discretization of the pulse is done in real time at the FPGA of the instrument.

            acquisition_hold_off: int = Delay between the start of playing a readout pulse and the start of
                the acquisition, in ns. Must be > 0 and multiple of 4.
            acquisition_duration: int = (mapped to qrm.sequencers[0].integration_length_acq) Duration
                of the pulse acquisition, in ns. Must be > 0 and multiple of 4.
            discretization_threshold_acq: float = (mapped to qrm.sequencers[0].discretization_threshold_acq)    # TODO mapped, but not configurable from the runcard
            phase_rotation_acq: float = (mapped to qrm.sequencers[0].phase_rotation_acq)                        # TODO mapped, but not configurable from the runcard

            channel_port_map:                                           # Refrigerator Channel : Instrument port
                10: o1 # IQ Port = out0 & out1
                1: i1

        Sequencer 0 is used always for acquisitions and it is the first sequencer used to synthesise pulses.
        Sequencer 1 to 6 are used as needed to sinthesise simultaneous pulses on the same channel
        or when the memory of the default sequencers rans out.

    """

    DEFAULT_SEQUENCERS: dict = {"o1": 0, "i1": 0}
    SAMPLING_RATE: int = 1e9  # 1 GSPS

    property_wrapper = lambda parent, *parameter: property(
        lambda self: parent.device.get(parameter[0]),
        lambda self, x: parent._set_device_parameter(parent.device, *parameter, value=x),
    )
    sequencer_property_wrapper = lambda parent, sequencer, *parameter: property(
        lambda self: parent.device.sequencers[sequencer].get(parameter[0]),
        lambda self, x: parent._set_device_parameter(parent.device.sequencers[sequencer], *parameter, value=x),
    )

    def __init__(self, name: str, address: str):
        super().__init__(name, address)
        self.device: QbloxQrmQcm = None
        self.ports: dict = {}
        self.acquisition_hold_off: int
        self.acquisition_duration: int
        self.discretization_threshold_acq: float
        self.phase_rotation_acq: float
        self.channel_port_map: dict = {}
        self.channels: list = []

        self._cluster: QbloxCluster = None
        self._input_ports_keys = ["i1"]
        self._output_ports_keys = ["o1"]
        self._sequencers: dict[Sequencer] = {"o1": []}
        self._port_channel_map: dict = {}
        self._last_pulsequence_hash: int = 0
        self._current_pulsesequence_hash: int
        self._device_parameters = {}
        self._device_num_output_ports = 1
        self._device_num_sequencers: int
        self._free_sequencers_numbers: list[int] = []
        self._used_sequencers_numbers: list[int] = []
        self._unused_sequencers_numbers: list[int] = []

    def connect(self):
        """Connects to the instrument using the instrument settings in the runcard.

        Once connected, it creates port classes with properties mapped to various instrument
        parameters.
        """
        global cluster
        if not self.is_connected:
            if cluster:
                self.device: QbloxQrmQcm = cluster.modules[int(self.address.split(":")[1]) - 1]

                self.ports["o1"] = type(
                    f"port_o1",
                    (),
                    {
                        "attenuation": self.property_wrapper("out0_att"),
                        "lo_enabled": self.property_wrapper("out0_in0_lo_en"),
                        "lo_frequency": self.property_wrapper("out0_in0_lo_freq"),
                        "gain": self.sequencer_property_wrapper(
                            self.DEFAULT_SEQUENCERS["o1"], "gain_awg_path0", "gain_awg_path1"
                        ),
                        "hardware_mod_en": self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS["o1"], "mod_en_awg"),
                        "nco_freq": self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS["o1"], "nco_freq"),
                        "nco_phase_offs": self.sequencer_property_wrapper(
                            self.DEFAULT_SEQUENCERS["o1"], "nco_phase_offs"
                        ),
                    },
                )()

                self.ports["i1"] = type(
                    f"port_i1",
                    (),
                    {
                        "hardware_demod_en": self.sequencer_property_wrapper(
                            self.DEFAULT_SEQUENCERS["i1"], "demod_en_acq"
                        )
                    },
                )()

                setattr(
                    self.__class__,
                    "acquisition_duration",
                    self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS["o1"], "integration_length_acq"),
                )
                setattr(
                    self.__class__,
                    "discretization_threshold_acq",
                    self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS["o1"], "discretization_threshold_acq"),
                )
                setattr(
                    self.__class__,
                    "phase_rotation_acq",
                    self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS["o1"], "phase_rotation_acq"),
                )

                self._cluster = cluster
                self.is_connected = True

                self._set_device_parameter(self.device, "in0_att", value=0)
                self._set_device_parameter(
                    self.device, "out0_offset_path0", "out0_offset_path1", value=0
                )  # Default after reboot = 7.625
                self._set_device_parameter(
                    self.device, "scope_acq_avg_mode_en_path0", "scope_acq_avg_mode_en_path1", value=True
                )
                self._set_device_parameter(
                    self.device, "scope_acq_sequencer_select", value=self.DEFAULT_SEQUENCERS["o1"]
                )
                self._set_device_parameter(
                    self.device, "scope_acq_trigger_level_path0", "scope_acq_trigger_level_path1", value=0
                )
                self._set_device_parameter(
                    self.device, "scope_acq_trigger_mode_path0", "scope_acq_trigger_mode_path1", value="sequencer"
                )

                target = self.device.sequencers[self.DEFAULT_SEQUENCERS["o1"]]

                self._set_device_parameter(target, "channel_map_path0_out0_en", value=True)
                self._set_device_parameter(target, "channel_map_path1_out1_en", value=True)
                self._set_device_parameter(target, "cont_mode_en_awg_path0", "cont_mode_en_awg_path1", value=False)
                self._set_device_parameter(
                    target, "cont_mode_waveform_idx_awg_path0", "cont_mode_waveform_idx_awg_path1", value=0
                )
                self._set_device_parameter(target, "marker_ovr_en", value=True)  # Default after reboot = False
                self._set_device_parameter(target, "marker_ovr_value", value=15)  # Default after reboot = 0
                self._set_device_parameter(target, "mixer_corr_gain_ratio", value=1)
                self._set_device_parameter(target, "mixer_corr_phase_offset_degree", value=0)
                self._set_device_parameter(target, "offset_awg_path0", value=0)
                self._set_device_parameter(target, "offset_awg_path1", value=0)
                self._set_device_parameter(target, "sync_en", value=True)  # Default after reboot = False
                self._set_device_parameter(target, "upsample_rate_awg_path0", "upsample_rate_awg_path1", value=0)

                self._set_device_parameter(target, "channel_map_path0_out0_en", "channel_map_path1_out1_en", value=True)

                self._device_num_sequencers = len(self.device.sequencers)
                for sequencer in range(1, self._device_num_sequencers):
                    self._set_device_parameter(
                        self.device.sequencers[sequencer],
                        "channel_map_path0_out0_en",
                        "channel_map_path1_out1_en",
                        value=False,
                    )  # Default after reboot = True

    def _set_device_parameter(self, target, *parameters, value):
        """Sets a parameter of the instrument, if it changed from the last stored in the cache.

        Args:
            target = an instance of qblox_instruments.qcodes_drivers.qcm_qrm.QcmQrm or
                                    qblox_instruments.qcodes_drivers.sequencer.Sequencer
            *parameters: list = A list of parameters to be cached and set.
            value = The value to set the paramters.
        Raises:
            Exception = If attempting to set a parameter without a connection to the instrument.
        """
        if self.is_connected:
            key = target.name + "." + parameters[0]
            if not key in self._device_parameters:
                for parameter in parameters:
                    if not hasattr(target, parameter):
                        raise Exception(f"The instrument {self.name} does not have parameters {parameter}")
                    target.set(parameter, value)
                self._device_parameters[key] = value
            elif self._device_parameters[key] != value:
                for parameter in parameters:
                    target.set(parameter, value)
                self._device_parameters[key] = value
        else:
            raise Exception("There is no connection to the instrument {self.name}")

    def _erase_device_parameters_cache(self):
        """Erases the cache of instrument parameters."""
        self._device_parameters = {}

    def setup(self, **kwargs):
        """Configures the instrument.

        A connection to the instrument needs to be established beforehand.
        Args:
            **kwargs: dict = A dictionary of settings loaded from the runcard:
                kwargs['channel_port_map']
                kwargs['ports']['o1']['attenuation']
                kwargs['ports']['o1']['lo_enabled']
                kwargs['ports']['o1']['lo_frequency']
                kwargs['ports']['o1']['gain']
                kwargs['ports']['o1']['hardware_mod_en']
                kwargs['ports']['i1']['hardware_demod_en']
                kwargs['acquisition_hold_off']
                kwargs['acquisition_duration']
        Raises:
            Exception = If attempting to set a parameter without a connection to the instrument.
        """

        if self.is_connected:
            # Load settings
            self.channel_port_map = kwargs["channel_port_map"]
            self._port_channel_map = {v: k for k, v in self.channel_port_map.items()}
            self.channels = list(self.channel_port_map.keys())

            self.ports["o1"].attenuation = kwargs["ports"]["o1"]["attenuation"]  # Default after reboot = 7
            self.ports["o1"].lo_enabled = kwargs["ports"]["o1"]["lo_enabled"]  # Default after reboot = True
            self.ports["o1"].lo_frequency = kwargs["ports"]["o1"][
                "lo_frequency"
            ]  # Default after reboot = 6_000_000_000
            self.ports["o1"].gain = kwargs["ports"]["o1"]["gain"]  # Default after reboot = 1
            self.ports["o1"].hardware_mod_en = kwargs["ports"]["o1"]["hardware_mod_en"]  # Default after reboot = False

            self.ports["o1"].nco_freq = 0  # Default after reboot = 1
            self.ports["o1"].nco_phase_offs = 0  # Default after reboot = 1

            self.ports["i1"].hardware_demod_en = kwargs["ports"]["i1"][
                "hardware_demod_en"
            ]  # Default after reboot = False

            self.acquisition_hold_off = kwargs["acquisition_hold_off"]
            self.acquisition_duration = kwargs["acquisition_duration"]
            self.discretization_threshold_acq = 0  # Default after reboot = 1
            self.phase_rotation_acq = 0  # Default after reboot = 1

        else:
            raise Exception("The instrument cannot be set up, there is no connection")

    def process_pulse_sequence(self, instrument_pulses: PulseSequence, nshots: int, repetition_duration: int):
        """Processes a list of pulses, generating the waveforms and sequence program required by
        the instrument to synthesise them.

        The output of the process is a list of sequencers used for each port, configured with the information
        required to play the sequence.
        The following features are supported:
            - Hardware modulation and demodulation.
            - Multiplexed readout.
            - Sequencer memory optimisation.
            - Extended waveform memory with the use of multiple sequencers.
            - Overlapping pulses.
            - Waveform and Sequence Program cache.
            - Pulses of up to 8192 pairs of i, q samples

        Args:
        instrument_pulses: PulseSequence = A collection of Pulse objects to be played by the instrument.
        nshots: int = The number of times the sequence of pulses should be repeated.
        repetition_duration: int = The total duration of every pulse sequence execution and reset time.
        """

        # Save the hash of the current sequence of pulses.
        self._current_pulsesequence_hash = hash(
            (
                instrument_pulses,
                nshots,
                repetition_duration,
                self.ports["o1"].hardware_mod_en,
                self.ports["i1"].hardware_demod_en,
            )
        )

        # Check if the sequence to be processed is the same as the last one.
        # If so, there is no need to generate new waveforms and program
        # Except if hardware demodulation is activated (to force a memory reset until qblox fix the issue)
        if self.ports["i1"].hardware_demod_en or self._current_pulsesequence_hash != self._last_pulsequence_hash:
            port = "o1"
            # initialise the list of free sequencer numbers to include the default for each port {'o1': 0}
            self._free_sequencers_numbers = [self.DEFAULT_SEQUENCERS[port]] + [1, 2, 3, 4, 5]

            # split the collection of instruments pulses by ports
            port_pulses: PulseSequence = instrument_pulses.get_channel_pulses(self._port_channel_map[port])

            # initialise the list of sequencers required by the port
            self._sequencers[port] = []

            if not port_pulses.is_empty:
                # split the collection of port pulses in non overlapping pulses
                non_overlapping_pulses: PulseSequence
                for non_overlapping_pulses in port_pulses.separate_overlapping_pulses():
                    # TODO: for non_overlapping_same_frequency_pulses in non_overlapping_pulses.separate_different_frequency_pulses():

                    # each set of not overlapping pulses will be played by a separate sequencer
                    # check sequencer availability
                    if len(self._free_sequencers_numbers) == 0:
                        raise Exception(
                            f"The number of sequencers requried to play the sequence exceeds the number available {self._device_num_sequencers}."
                        )

                    # select a new sequencer and configure it as required
                    next_sequencer_number = self._free_sequencers_numbers.pop(0)
                    if next_sequencer_number != self.DEFAULT_SEQUENCERS[port]:
                        for parameter in self.device.sequencers[self.DEFAULT_SEQUENCERS[port]].parameters:
                            value = self.device.sequencers[self.DEFAULT_SEQUENCERS[port]].get(param_name=parameter)
                            if not value is None:
                                target = self.device.sequencers[next_sequencer_number]
                                self._set_device_parameter(target, parameter, value=value)
                    if self.ports["i1"].hardware_demod_en or self.ports["o1"].hardware_mod_en:
                        self._set_device_parameter(
                            self.device.sequencers[next_sequencer_number],
                            "nco_freq",
                            value=non_overlapping_pulses[0].frequency,
                        )  # TODO: for non_overlapping_same_frequency_pulses[0].frequency
                    sequencer = Sequencer(next_sequencer_number)

                    # add the sequencer to the list of sequencers required by the port
                    self._sequencers[port].append(sequencer)

                    # make a temporary copy of the pulses to be processed
                    pulses_to_be_processed = non_overlapping_pulses.shallow_copy()
                    while not pulses_to_be_processed.is_empty:
                        pulse: Pulse = pulses_to_be_processed[0]
                        # select between envelope or modulated waveforms depending on hardware modulation setting
                        if self.ports[port].hardware_mod_en:
                            pulse.waveform_i, pulse.waveform_q = pulse.envelope_waveforms
                        else:
                            pulse.waveform_i, pulse.waveform_q = pulse.modulated_waveforms

                        # attempt to save the waveforms to the sequencer waveforms buffer
                        try:
                            sequencer.waveforms_buffer.add_waveforms(pulse.waveform_i, pulse.waveform_q)
                            sequencer.pulses.add(pulse)
                            pulses_to_be_processed.remove(pulse)

                        # if there is not enough memory in the current sequencer, use another one
                        except WaveformsBuffer.NotEnoughMemory:
                            if len(pulse.waveform_i) + len(pulse.waveform_q) > WaveformsBuffer.SIZE:
                                raise NotImplementedError(
                                    f"Pulses with waveforms longer than the memory of a sequencer ({WaveformsBuffer.SIZE // 2}) are not supported."
                                )
                            if len(self._free_sequencers_numbers) == 0:
                                raise Exception(
                                    f"The number of sequencers requried to play the sequence exceeds the number available {self._device_num_sequencers}."
                                )

                            # select a new sequencer and configure it as required
                            next_sequencer_number = self._free_sequencers_numbers.pop(0)
                            for parameter in self.device.sequencers[self.DEFAULT_SEQUENCERS[port]].parameters:
                                value = self.device.sequencers[self.DEFAULT_SEQUENCERS[port]].get(param_name=parameter)
                                if not value is None:
                                    target = self.device.sequencers[next_sequencer_number]
                                    self._set_device_parameter(target, parameter, value=value)
                            if self.ports["i1"].hardware_demod_en or self.ports["o1"].hardware_mod_en:
                                self._set_device_parameter(
                                    self.device.sequencers[next_sequencer_number], "nco_freq", value=pulse.frequency
                                )
                            sequencer = Sequencer(next_sequencer_number)

                            # add the sequencer to the list of sequencers required by the port
                            self._sequencers[port].append(sequencer)

            # update the lists of used and unused sequencers that will be needed later on
            self._used_sequencers_numbers = []
            for port in self._output_ports_keys:
                for sequencer in self._sequencers[port]:
                    self._used_sequencers_numbers.append(sequencer.number)
            self._unused_sequencers_numbers = []
            for n in range(self._device_num_sequencers):
                if not n in self._used_sequencers_numbers:
                    self._unused_sequencers_numbers.append(n)

            # generate and store the Waveforms dictionary, the Acquisitions dictionary, the Weights and the Program
            for port in self._output_ports_keys:
                for sequencer in self._sequencers[port]:

                    # Waveforms
                    for index, waveform in enumerate(sequencer.waveforms_buffer.unique_waveforms):
                        sequencer.waveforms[waveform.serial] = {"data": waveform.data.tolist(), "index": index}

                    # Acquisitions
                    for acquisition_index, pulse in enumerate(sequencer.pulses.ro_pulses):
                        sequencer.acquisitions[pulse.serial] = {"num_bins": nshots, "index": acquisition_index}

                    # Program
                    minimum_delay_between_instructions = 4
                    wait_loop_step: int = 1000

                    pulses = sequencer.pulses
                    sequence_total_duration = (
                        pulses.start + pulses.duration + minimum_delay_between_instructions
                    )  # the minimum delay between instructions is 4ns
                    time_between_repetitions = repetition_duration - sequence_total_duration
                    assert time_between_repetitions > 0

                    wait_time = time_between_repetitions
                    extra_wait = wait_time % wait_loop_step
                    while wait_time > 0 and extra_wait < 4:
                        wait_loop_step += 1
                        extra_wait = wait_time % wait_loop_step
                    num_wait_loops = (wait_time - extra_wait) // wait_loop_step

                    header = f"""
                    move 0,R0                # loop iterator (nshots)
                    nop
                    wait_sync {minimum_delay_between_instructions}
                    loop:"""
                    if self.ports["i1"].hardware_demod_en or self.ports["o1"].hardware_mod_en:
                        header += "\n" + "                    reset_ph"
                    body = ""

                    footer = f"""
                    # wait {wait_time} ns"""
                    if num_wait_loops > 0:
                        footer += f"""
                    move {num_wait_loops},R2
                    nop
                    waitloop2:
                        wait {wait_loop_step}
                        loop R2,@waitloop2"""
                    if extra_wait > 0:
                        footer += f"""
                    wait {extra_wait}"""
                    else:
                        footer += f"""
                    # wait 0"""

                    footer += f"""
                    add R0,1,R0              # increment iterator
                    nop                      # wait a cycle for R0 to be available
                    jlt R0,{nshots},@loop        # nshots
                    stop
                    """

                    # Add an initial wait instruction for the first pulse of the sequence
                    wait_time = pulses[0].start
                    extra_wait = wait_time % wait_loop_step
                    while wait_time > 0 and extra_wait < 4:
                        wait_loop_step += 1
                        extra_wait = wait_time % wait_loop_step
                    num_wait_loops = (wait_time - extra_wait) // wait_loop_step

                    if wait_time > 0:
                        initial_wait_instruction = f"""
                    # wait {wait_time} ns"""
                        if num_wait_loops > 0:
                            initial_wait_instruction += f"""
                    move {num_wait_loops},R1
                    nop
                    waitloop1:
                        wait {wait_loop_step}
                        loop R1,@waitloop1"""
                        if extra_wait > 0:
                            initial_wait_instruction += f"""
                    wait {extra_wait}"""
                        else:
                            initial_wait_instruction += f"""
                    # wait 0"""
                    else:
                        initial_wait_instruction = """
                    # wait 0"""

                    body += initial_wait_instruction

                    for n in range(pulses.count):
                        if (self.ports["i1"].hardware_demod_en or self.ports["o1"].hardware_mod_en) and pulses[
                            n
                        ].relative_phase != 0:
                            # Set phase
                            p = 10
                            phase = (pulses[n].relative_phase * 360 / (2 * np.pi)) % 360
                            coarse = int(round(phase / 0.9, p))
                            fine = int(round((phase - coarse * 0.9) / 2.25e-3, p))
                            ultra_fine = int(round((phase - coarse * 0.9 - fine * 2.25e-3) / 3.6e-7, p))
                            error = abs(phase - coarse * 0.9 - fine * 2.25e-3 - ultra_fine * 3.6e-7)
                            assert error < 3.6e-7
                            set_ph_instruction = f"                    set_ph {coarse}, {fine}, {ultra_fine}"
                            set_ph_instruction += (
                                " " * (45 - len(set_ph_instruction))
                                + f"# set relative phase {pulses[n].relative_phase} rads"
                            )
                            body += "\n" + set_ph_instruction
                        if pulses[n].type == PulseType.READOUT:
                            delay_after_play = self.acquisition_hold_off

                            if len(pulses) > n + 1:
                                # If there are more pulses to be played, the delay is the time between the pulse end and the next pulse start
                                delay_after_acquire = pulses[n + 1].start - pulses[n].start - self.acquisition_hold_off
                            else:
                                delay_after_acquire = (
                                    sequence_total_duration - pulses[n].start - self.acquisition_hold_off
                                )

                            if delay_after_acquire < minimum_delay_between_instructions:
                                raise Exception(
                                    f"The minimum delay after starting acquisition is {minimum_delay_between_instructions}ns."
                                )

                            # Prepare play instruction: play arg0, arg1, arg2.
                            #   arg0 is the index of the I waveform
                            #   arg1 is the index of the Q waveform
                            #   arg2 is the delay between starting the instruction and the next instruction
                            play_instruction = f"                    play {sequencer.waveforms_buffer.unique_waveforms.index(pulses[n].waveform_i)},{sequencer.waveforms_buffer.unique_waveforms.index(pulses[n].waveform_q)},{delay_after_play}"
                            # Add the serial of the pulse as a comment
                            play_instruction += " " * (45 - len(play_instruction)) + f"# play waveforms {pulses[n]}"
                            body += "\n" + play_instruction

                            # Prepare acquire instruction: acquire arg0, arg1, arg2.
                            #   arg0 is the index of the acquisition
                            #   arg1 is the index of the data bin
                            #   arg2 is the delay between starting the instruction and the next instruction
                            acquire_instruction = f"                    acquire {pulses.ro_pulses.index(pulses[n])},R0,{delay_after_acquire}"
                            # Add the serial of the pulse as a comment
                            body += "\n" + acquire_instruction

                        else:
                            # Calculate the delay_after_play that is to be used as an argument to the play instruction
                            if len(pulses) > n + 1:
                                # If there are more pulses to be played, the delay is the time between the pulse end and the next pulse start
                                delay_after_play = pulses[n + 1].start - pulses[n].start
                            else:
                                delay_after_play = sequence_total_duration - pulses[n].start

                            if delay_after_play < minimum_delay_between_instructions:
                                raise Exception(
                                    f"The minimum delay between pulses is {minimum_delay_between_instructions}ns."
                                )

                            # Prepare play instruction: play arg0, arg1, arg2.
                            #   arg0 is the index of the I waveform
                            #   arg1 is the index of the Q waveform
                            #   arg2 is the delay between starting the instruction and the next instruction
                            play_instruction = f"                    play {sequencer.waveforms_buffer.unique_waveforms.index(pulses[n].waveform_i)},{sequencer.waveforms_buffer.unique_waveforms.index(pulses[n].waveform_q)},{delay_after_play}"
                            # Add the serial of the pulse as a comment
                            play_instruction += " " * (45 - len(play_instruction)) + f"# play waveforms {pulses[n]}"
                            body += "\n" + play_instruction

                    sequencer.program = header + body + footer

    def upload(self):
        """Uploads waveforms and programs of all sequencers and arms them in preparation for execution.

        This method should be called after `process_pulse_sequence()`.
        It configures certain parameters of the instrument based on the needs of resources determined
        while processing the pulse sequence.
        """
        if self.ports["i1"].hardware_demod_en or self._current_pulsesequence_hash != self._last_pulsequence_hash:
            self._last_pulsequence_hash = self._current_pulsesequence_hash

            # Setup
            for sequencer_number in self._used_sequencers_numbers:
                target = self.device.sequencers[sequencer_number]
                self._set_device_parameter(target, "sync_en", value=True)
                self._set_device_parameter(target, "marker_ovr_en", value=True)  # Default after reboot = False
                self._set_device_parameter(target, "marker_ovr_value", value=15)  # Default after reboot = 0

            for sequencer_number in self._unused_sequencers_numbers:
                target = self.device.sequencers[sequencer_number]
                self._set_device_parameter(target, "sync_en", value=False)
                self._set_device_parameter(target, "marker_ovr_en", value=False)  # Default after reboot = False
                self._set_device_parameter(target, "marker_ovr_value", value=0)  # Default after reboot = 0
                # self._set_device_parameter(target, 'channel_map_path0_out0_en', value = False)
                # self._set_device_parameter(target, 'channel_map_path1_out1_en', value = False)

            # Upload waveforms and program
            qblox_dict = {}
            sequencer: Sequencer
            for port in self._output_ports_keys:
                for sequencer in self._sequencers[port]:
                    # Add sequence program and waveforms to single dictionary and write to JSON file
                    filename = f"{self.name}_sequencer{sequencer.number}_sequence.json"
                    qblox_dict[sequencer] = {
                        "waveforms": sequencer.waveforms,
                        "weights": sequencer.weights,
                        "acquisitions": sequencer.acquisitions,
                        "program": sequencer.program,
                    }
                    with open(self.data_folder / filename, "w", encoding="utf-8") as file:
                        json.dump(qblox_dict[sequencer], file, indent=4)

                    # Upload json file to the device sequencers
                    self.device.sequencers[sequencer.number].sequence(str(self.data_folder / filename))

        # Arm sequencers
        for sequencer_number in self._used_sequencers_numbers:
            self.device.arm_sequencer(sequencer_number)

        # DEBUG: QRM Print Readable Snapshot
        # print(self.name)
        # self.device.print_readable_snapshot(update=True)

    def play_sequence(self):
        """Plays the sequence of pulses ignoring acquisitions.

        This could be useful when thq QRM is used as a QCM (qubit control module)."""
        # Start used sequencers
        for sequencer_number in self._used_sequencers_numbers:
            self.device.start_sequencer(sequencer_number)

    def play_sequence_and_acquire(self):
        """Executes the sequence of instructions and retrieves the readout results.

        Each readout pulse generates a separate acquisition.
        Returns:
            A dictionary of {Readout Pulse Serial: Acquisition Results} where Acquisition Results is
            a list of [amplitude[V], phase[rad], i[V], q[V]]

        """
        # Start playing sequences in all sequencers
        for sequencer_number in self._used_sequencers_numbers:
            self.device.start_sequencer(sequencer_number)

        # Retrieve data
        acquisition_results = {}
        for port in self._output_ports_keys:
            # wait until sequencers have stopped
            for sequencer in self._sequencers[port]:
                self.device.get_sequencer_state(sequencer_number, timeout=1)
                self.device.get_acquisition_state(sequencer_number, timeout=1)
            # store scope acquisition data with the first acquisition
            sequencer = self._sequencers[port][0]
            sequencer_number = sequencer.number
            assert (
                sequencer_number == self.DEFAULT_SEQUENCERS[port]
            )  # The first sequencer should always be the default sequencer
            scope_acquisition_name = next(
                iter(sequencer.acquisitions)
            )  # returns the first item in sequencer.acquisitions dict
            self.device.store_scope_acquisition(sequencer_number, scope_acquisition_name)
            #
            for sequencer in self._sequencers[port]:
                raw_results = self.device.get_acquisitions(sequencer_number)
                if not self.ports["i1"].hardware_demod_en:
                    acquisition_results["averaged_integrated"] = {}
                    acquisition_results["averaged_raw"] = {}
                    for pulse in sequencer.pulses.ro_pulses:
                        i, q = self._process_acquisition_results(
                            raw_results[scope_acquisition_name], pulse, demodulate=True
                        )
                        acquisition_results[pulse.serial] = np.sqrt(i**2 + q**2), np.arctan2(q, i), i, q
                        acquisition_results[pulse.qubit] = acquisition_results[pulse.serial]

                        acquisition_results["averaged_integrated"][pulse.serial] = acquisition_results[pulse.serial]
                        acquisition_results["averaged_integrated"][pulse.qubit] = acquisition_results[pulse.qubit]

                        acquisition_results["averaged_raw"][pulse.serial] = (
                            raw_results[scope_acquisition_name]["acquisition"]["scope"]["path0"]["data"][
                                0 : self.acquisition_duration
                            ],
                            raw_results[scope_acquisition_name]["acquisition"]["scope"]["path1"]["data"][
                                0 : self.acquisition_duration
                            ],
                        )
                        acquisition_results["averaged_raw"][pulse.qubit] = acquisition_results["averaged_raw"][
                            pulse.serial
                        ]
                else:
                    acquisition_results["averaged_integrated"] = {}
                    acquisition_results["binned_integrated"] = {}
                    acquisition_results["binned_classified"] = {}
                    acquisition_results["probability"] = {}
                    for pulse in sequencer.pulses.ro_pulses:
                        acquisition_name = pulse.serial
                        i, q = self._process_acquisition_results(raw_results[acquisition_name], pulse, demodulate=False)
                        acquisition_results[pulse.serial] = np.sqrt(i**2 + q**2), np.arctan2(q, i), i, q
                        acquisition_results[pulse.qubit] = acquisition_results[pulse.serial]

                        acquisition_results["averaged_integrated"][pulse.serial] = acquisition_results[
                            pulse.serial
                        ]  # integrated_averaged
                        acquisition_results["averaged_integrated"][pulse.qubit] = acquisition_results[pulse.qubit]

                        # Save individual shots
                        shots_i = np.array(
                            [
                                (val / self.acquisition_duration)
                                for val in raw_results[acquisition_name]["acquisition"]["bins"]["integration"]["path0"]
                            ]
                        )
                        shots_q = np.array(
                            [
                                (val / self.acquisition_duration)
                                for val in raw_results[acquisition_name]["acquisition"]["bins"]["integration"]["path1"]
                            ]
                        )
                        acquisition_results["binned_integrated"][pulse.serial] = [
                            (msr, phase, i, q)
                            for msr, phase, i, q in zip(
                                np.sqrt(shots_i**2 + shots_q**2), np.arctan2(shots_q, shots_i), shots_i, shots_q
                            )
                        ]
                        acquisition_results["binned_integrated"][pulse.qubit] = acquisition_results[
                            "binned_integrated"
                        ][pulse.serial]

                        acquisition_results["binned_classified"][pulse.serial] = raw_results[acquisition_name][
                            "acquisition"
                        ]["bins"]["threshold"]
                        acquisition_results["binned_classified"][pulse.qubit] = acquisition_results[
                            "binned_classified"
                        ][pulse.serial]

                        acquisition_results["probability"][pulse.serial] = np.mean(
                            acquisition_results["binned_classified"][pulse.serial]
                        )
                        acquisition_results["probability"][pulse.qubit] = acquisition_results["probability"][
                            pulse.serial
                        ]

                        # DEBUG: QRM Plot Incomming Pulses
                        # import qibolab.instruments.debug.incomming_pulse_plotting as pp
                        # pp.plot(raw_results)
        return acquisition_results

    def _process_acquisition_results(self, acquisition_results, readout_pulse: Pulse, demodulate=True):
        """Processes the results of the acquisition.

        If hardware demodulation is disabled, it demodulates and integrates the acquired pulse. If enabled,
        if processes the results are required by qblox.
        """
        if demodulate:
            acquisition_frequency = readout_pulse.frequency

            # DOWN Conversion
            n0 = 0
            n1 = self.acquisition_duration
            input_vec_I = np.array(acquisition_results["acquisition"]["scope"]["path0"]["data"][n0:n1])
            input_vec_Q = np.array(acquisition_results["acquisition"]["scope"]["path1"]["data"][n0:n1])
            input_vec_I -= np.mean(input_vec_I)  # qblox does not remove the offsets in hardware
            input_vec_Q -= np.mean(input_vec_Q)

            modulated_i = input_vec_I
            modulated_q = input_vec_Q

            num_samples = modulated_i.shape[0]
            time = np.arange(num_samples) / PulseShape.SAMPLING_RATE

            cosalpha = np.cos(2 * np.pi * acquisition_frequency * time)
            sinalpha = np.sin(2 * np.pi * acquisition_frequency * time)
            demod_matrix = np.sqrt(2) * np.array([[cosalpha, sinalpha], [-sinalpha, cosalpha]])
            result = []
            for it, t, ii, qq in zip(np.arange(modulated_i.shape[0]), time, modulated_i, modulated_q):
                result.append(demod_matrix[:, :, it] @ np.array([ii, qq]))
            demodulated_signal = np.array(result)
            integrated_signal = np.mean(demodulated_signal, axis=0)

            # import matplotlib.pyplot as plt
            # plt.plot(input_vec_I[:400])
            # plt.plot(list(map(list, zip(*demodulated_signal)))[0][:400])
            # plt.show()
        else:
            i = np.mean(
                [
                    (val / self.acquisition_duration)
                    for val in acquisition_results["acquisition"]["bins"]["integration"]["path0"]
                ]
            )
            q = np.mean(
                [
                    (val / self.acquisition_duration)
                    for val in acquisition_results["acquisition"]["bins"]["integration"]["path1"]
                ]
            )
            integrated_signal = i, q
        return integrated_signal

    def start(self):
        """Empty method to comply with AbstractInstrument interface."""
        pass

    def stop(self):
        """Stops all sequencers"""
        try:
            self.device.stop_sequencer()
        except:
            pass

    def disconnect(self):
        """Empty method to comply with AbstractInstrument interface."""
        self._cluster = None
        self.is_connected = False


class ClusterQCM_RF(AbstractInstrument):
    """Qblox Cluster Qubit Control Module RF driver.

    Qubit Control Module RF (QCM-RF) is an instrument that integrates an arbitratry
    wave generator, with a local oscillator and a mixer. It has two output ports
    Each port has a path0 and path1 for the i(in-phase) and q(quadrature) components
    of the RF signal. The sampling rate of its ADC/DAC is 1 GSPS.
    https://www.qblox.com/cluster

    The class aims to simplify the configuration of the instrument, exposing only
    those parameters most frequencly used and hiding other more complex components.

    A reference to the underlying `qblox_instruments.qcodes_drivers.qcm_qrm.QRM_QCM`
    object is provided via the attribute `device`, allowing the advanced user to gain
    access to the features that are not exposed directly by the class.

    In order to accelerate the execution, the instrument settings are cached, so that
    the communication with the instrument only happens when the parameters change.
    This caching is done with the method `_set_device_parameter(target, *parameters, value)`.

        ports:
            o1:
                attenuation                  : 24 # (dB)
                lo_enabled                   : true
                lo_frequency                 : 4_042_590_000 # (Hz) from 2e9 to 18e9
                gain                         : 0.17 # for path0 and path1 -1.0<=v<=1.0
                hardware_mod_en              : false
            o2:
                attenuation                  : 24 # (dB)
                lo_enabled                   : true
                lo_frequency                 : 5_091_155_529 # (Hz) from 2e9 to 18e9
                gain                         : 0.28 # for path0 and path1 -1.0<=v<=1.0
                hardware_mod_en              : false

        channel_port_map:
            21: o1 # IQ Port = out0 & out1
            22: o2 # IQ Port = out2 & out3

    The class inherits from AbstractInstrument and implements its interface methods:
        __init__()
        connect()
        setup()
        start()
        stop()
        disconnect()

    Attributes:
        name: str = A unique name given to the instrument.
        address: str = IP_address:module_number (the IP address of the cluster and
            the module number)
        ports = A dictionary giving access to the output ports objects.
            ports['o1']
            ports['o2']

            ports['o1'].attenuation: int = (mapped to qrm.out0_att) Controls the attenuation applied to
                the output port. Must be a multiple of 2
            ports['o1'].lo_enabled: bool = (mapped to qrm.out0_in0_lo_en) Enables or disables the
                local oscillator.
            ports['o1'].lo_frequency: int = (mapped to qrm.out0_in0_lo_freq) Sets the frequency of the
                local oscillator.
            ports['o1'].gain: float = (mapped to qrm.sequencers[0].gain_awg_path0 and qrm.sequencers[0].gain_awg_path1)
                Sets the gain on both paths of the output port.
            ports['o1'].hardware_mod_en: bool = (mapped to qrm.sequencers[0].mod_en_awg) Enables pulse
                modulation in hardware. When set to False, pulse modulation is done at the host computer
                and a modulated pulse waveform should be uploaded to the instrument. When set to True,
                the envelope of the pulse should be uploaded to the instrument and it modulates it in
                real time by its FPGA using the sequencer nco (numerically controlled oscillator).
            ports['o1'].nco_freq: int = (mapped to qrm.sequencers[0].nco_freq)        # TODO mapped, but not configurable from the runcard
            ports['o1'].nco_phase_offs = (mapped to qrm.sequencers[0].nco_phase_offs) # TODO mapped, but not configurable from the runcard

            ports['o2'].attenuation: int = (mapped to qrm.out1_att) Controls the attenuation applied to
                the output port. Must be a multiple of 2
            ports['o2'].lo_enabled: bool = (mapped to qrm.out1_lo_en) Enables or disables the
                local oscillator.
            ports['o2'].lo_frequency: int = (mapped to qrm.out1_lo_freq) Sets the frequency of the
                local oscillator.
            ports['o2'].gain: float = (mapped to qrm.sequencers[1].gain_awg_path0 and qrm.sequencers[0].gain_awg_path1)
                Sets the gain on both paths of the output port.
            ports['o2'].hardware_mod_en: bool = (mapped to qrm.sequencers[1].mod_en_awg) Enables pulse
                modulation in hardware. When set to False, pulse modulation is done at the host computer
                and a modulated pulse waveform should be uploaded to the instrument. When set to True,
                the envelope of the pulse should be uploaded to the instrument and it modulates it in
                real time by its FPGA using the sequencer nco (numerically controlled oscillator).
            ports['o2'].nco_freq: int = (mapped to qrm.sequencers[1].nco_freq)        # TODO mapped, but not configurable from the runcard
            ports['o2'].nco_phase_offs = (mapped to qrm.sequencers[1].nco_phase_offs) # TODO mapped, but not configurable from the runcard

            channel_port_map:                                           # Refrigerator Channel : Instrument port
                21: o1 # IQ Port = out0 & out1
                22: o2 # IQ Port = out2 & out3

        Sequencer 0 is used always the first sequencer used to synthesise pulses on port o1.
        Sequencer 1 is used always the first sequencer used to synthesise pulses on port o2.
        Sequencer 2 to 6 are used as needed to sinthesise simultaneous pulses on the same channel
        or when the memory of the default sequencers rans out.

    """

    DEFAULT_SEQUENCERS = {"o1": 0, "o2": 1}
    SAMPLING_RATE: int = 1e9  # 1 GSPS

    property_wrapper = lambda parent, *parameter: property(
        lambda self: parent.device.get(parameter[0]),
        lambda self, x: parent._set_device_parameter(parent.device, *parameter, value=x),
    )
    sequencer_property_wrapper = lambda parent, sequencer, *parameter: property(
        lambda self: parent.device.sequencers[sequencer].get(parameter[0]),
        lambda self, x: parent._set_device_parameter(parent.device.sequencers[sequencer], *parameter, value=x),
    )

    def __init__(self, name: str, address: str):
        super().__init__(name, address)
        self.device: QbloxQrmQcm = None
        self.ports: dict = {}
        self.channel_port_map: dict = {}
        self.channels: list = []

        self._cluster: QbloxCluster = None
        self._output_ports_keys = ["o1", "o2"]
        self._sequencers: dict[Sequencer] = {"o1": [], "o2": []}
        self._port_channel_map: dict = {}
        self._last_pulsequence_hash: int = 0
        self._current_pulsesequence_hash: int
        self._device_parameters = {}
        self._device_num_output_ports = 2
        self._device_num_sequencers: int
        self._free_sequencers_numbers: list[int] = []
        self._used_sequencers_numbers: list[int] = []
        self._unused_sequencers_numbers: list[int] = []

    def connect(self):
        """Connects to the instrument using the instrument settings in the runcard.

        Once connected, it creates port classes with properties mapped to various instrument
        parameters.
        """
        global cluster
        if not self.is_connected:
            if cluster:
                self.device = cluster.modules[int(self.address.split(":")[1]) - 1]

                self.ports["o1"] = type(
                    f"port_o1",
                    (),
                    {
                        "attenuation": self.property_wrapper("out0_att"),
                        "lo_enabled": self.property_wrapper("out0_lo_en"),
                        "lo_frequency": self.property_wrapper("out0_lo_freq"),
                        "gain": self.sequencer_property_wrapper(
                            self.DEFAULT_SEQUENCERS["o1"], "gain_awg_path0", "gain_awg_path1"
                        ),
                        "hardware_mod_en": self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS["o1"], "mod_en_awg"),
                        "nco_freq": self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS["o1"], "nco_freq"),
                        "nco_phase_offs": self.sequencer_property_wrapper(
                            self.DEFAULT_SEQUENCERS["o1"], "nco_phase_offs"
                        ),
                    },
                )()

                self.ports["o2"] = type(
                    f"port_o1",
                    (),
                    {
                        "attenuation": self.property_wrapper("out1_att"),
                        "lo_enabled": self.property_wrapper("out1_lo_en"),
                        "lo_frequency": self.property_wrapper("out1_lo_freq"),
                        "gain": self.sequencer_property_wrapper(
                            self.DEFAULT_SEQUENCERS["o2"], "gain_awg_path0", "gain_awg_path1"
                        ),
                        "hardware_mod_en": self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS["o2"], "mod_en_awg"),
                        "nco_freq": self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS["o2"], "nco_freq"),
                        "nco_phase_offs": self.sequencer_property_wrapper(
                            self.DEFAULT_SEQUENCERS["o2"], "nco_phase_offs"
                        ),
                    },
                )()

                self._cluster = cluster
                self.is_connected = True
                self._set_device_parameter(
                    self.device, "out0_offset_path0", "out0_offset_path1", value=0
                )  # Default after reboot = 7.625
                self._set_device_parameter(
                    self.device, "out1_offset_path0", "out1_offset_path1", value=0
                )  # Default after reboot = 7.625

                for target in [
                    self.device.sequencers[self.DEFAULT_SEQUENCERS["o1"]],
                    self.device.sequencers[self.DEFAULT_SEQUENCERS["o2"]],
                ]:

                    self._set_device_parameter(target, "cont_mode_en_awg_path0", "cont_mode_en_awg_path1", value=False)
                    self._set_device_parameter(
                        target, "cont_mode_waveform_idx_awg_path0", "cont_mode_waveform_idx_awg_path1", value=0
                    )
                    self._set_device_parameter(target, "marker_ovr_en", value=True)  # Default after reboot = False
                    self._set_device_parameter(target, "marker_ovr_value", value=15)  # Default after reboot = 0
                    self._set_device_parameter(target, "mixer_corr_gain_ratio", value=1)
                    self._set_device_parameter(target, "mixer_corr_phase_offset_degree", value=0)
                    self._set_device_parameter(target, "offset_awg_path0", value=0)
                    self._set_device_parameter(target, "offset_awg_path1", value=0)
                    self._set_device_parameter(target, "sync_en", value=True)  # Default after reboot = False
                    self._set_device_parameter(target, "upsample_rate_awg_path0", "upsample_rate_awg_path1", value=0)

                self._set_device_parameter(
                    self.device.sequencers[self.DEFAULT_SEQUENCERS["o1"]],
                    "channel_map_path0_out0_en",
                    "channel_map_path1_out1_en",
                    value=True,
                )
                self._set_device_parameter(
                    self.device.sequencers[self.DEFAULT_SEQUENCERS["o1"]],
                    "channel_map_path0_out2_en",
                    "channel_map_path1_out3_en",
                    value=False,
                )
                self._set_device_parameter(
                    self.device.sequencers[self.DEFAULT_SEQUENCERS["o2"]],
                    "channel_map_path0_out0_en",
                    "channel_map_path1_out1_en",
                    value=False,
                )
                self._set_device_parameter(
                    self.device.sequencers[self.DEFAULT_SEQUENCERS["o2"]],
                    "channel_map_path0_out2_en",
                    "channel_map_path1_out3_en",
                    value=True,
                )
                self.device_num_sequencers = len(self.device.sequencers)
                for sequencer in range(2, self.device_num_sequencers):
                    self._set_device_parameter(
                        self.device.sequencers[sequencer],
                        "channel_map_path0_out0_en",
                        "channel_map_path1_out1_en",
                        value=False,
                    )  # Default after reboot = True
                    self._set_device_parameter(
                        self.device.sequencers[sequencer],
                        "channel_map_path0_out2_en",
                        "channel_map_path1_out3_en",
                        value=False,
                    )  # Default after reboot = True

    def _set_device_parameter(self, target, *parameters, value):
        """Sets a parameter of the instrument, if it changed from the last stored in the cache.

        Args:
            target = an instance of qblox_instruments.qcodes_drivers.qcm_qrm.QcmQrm or
                                    qblox_instruments.qcodes_drivers.sequencer.Sequencer
            *parameters: list = A list of parameters to be cached and set.
            value = The value to set the paramters.
        Raises:
            Exception = If attempting to set a parameter without a connection to the instrument.
        """
        if self.is_connected:
            key = target.name + "." + parameters[0]
            if not key in self._device_parameters:
                for parameter in parameters:
                    if not hasattr(target, parameter):
                        raise Exception(f"The instrument {self.name} does not have parameters {parameter}")
                    target.set(parameter, value)
                self._device_parameters[key] = value
            elif self._device_parameters[key] != value:
                for parameter in parameters:
                    target.set(parameter, value)
                self._device_parameters[key] = value
        else:
            raise Exception("There is no connection to the instrument {self.name}")

    def _erase_device_parameters_cache(self):
        """Erases the cache of instrument parameters."""
        self._device_parameters = {}

    def setup(self, **kwargs):
        """Configures the instrument.

        A connection to the instrument needs to be established beforehand.
        Args:
            **kwargs: dict = A dictionary of settings loaded from the runcard:
                kwargs['channel_port_map']
                kwargs['ports']['o1']['attenuation']
                kwargs['ports']['o1']['lo_enabled']
                kwargs['ports']['o1']['lo_frequency']
                kwargs['ports']['o1']['gain']
                kwargs['ports']['o1']['hardware_mod_en']

                kwargs['ports']['o2']['attenuation']
                kwargs['ports']['o2']['lo_enabled']
                kwargs['ports']['o2']['lo_frequency']
                kwargs['ports']['o2']['gain']
                kwargs['ports']['o2']['hardware_mod_en']

                kwargs['channel_port_map']
        Raises:
            Exception = If attempting to set a parameter without a connection to the instrument.
        """

        if self.is_connected:
            # Load settings
            self.channel_port_map = kwargs["channel_port_map"]
            self._port_channel_map = {v: k for k, v in self.channel_port_map.items()}
            self.channels = list(self.channel_port_map.keys())

            self.ports["o1"].attenuation = kwargs["ports"]["o1"]["attenuation"]  # Default after reboot = 7
            self.ports["o1"].lo_enabled = kwargs["ports"]["o1"]["lo_enabled"]  # Default after reboot = True
            self.ports["o1"].lo_frequency = kwargs["ports"]["o1"][
                "lo_frequency"
            ]  # Default after reboot = 6_000_000_000
            self.ports["o1"].gain = kwargs["ports"]["o1"]["gain"]  # Default after reboot = 1
            self.ports["o1"].hardware_mod_en = kwargs["ports"]["o1"]["hardware_mod_en"]  # Default after reboot = False
            self.ports["o1"].nco_freq = 0  # Default after reboot = 1
            self.ports["o1"].nco_phase_offs = 0  # Default after reboot = 1

            self.ports["o2"].attenuation = kwargs["ports"]["o2"]["attenuation"]  # Default after reboot = 7
            self.ports["o2"].lo_enabled = kwargs["ports"]["o2"]["lo_enabled"]  # Default after reboot = True
            self.ports["o2"].lo_frequency = kwargs["ports"]["o2"][
                "lo_frequency"
            ]  # Default after reboot = 6_000_000_000
            self.ports["o2"].gain = kwargs["ports"]["o2"]["gain"]  # Default after reboot = 1
            self.ports["o2"].hardware_mod_en = kwargs["ports"]["o2"]["hardware_mod_en"]  # Default after reboot = False
            self.ports["o2"].nco_freq = 0  # Default after reboot = 1
            self.ports["o2"].nco_phase_offs = 0  # Default after reboot = 1

        else:
            raise Exception("The instrument cannot be set up, there is no connection")

    def process_pulse_sequence(self, instrument_pulses: PulseSequence, nshots: int, repetition_duration: int):
        """Processes a list of pulses, generating the waveforms and sequence program required by
        the instrument to synthesise them.

        The output of the process is a list of sequencers used for each port, configured with the information
        required to play the sequence.
        The following features are supported:
            - Hardware modulation and demodulation.
            - Multiplexed readout.
            - Sequencer memory optimisation.
            - Extended waveform memory with the use of multiple sequencers.
            - Overlapping pulses.
            - Waveform and Sequence Program cache.
            - Pulses of up to 8192 pairs of i, q samples

        Args:
        instrument_pulses: PulseSequence = A collection of Pulse objects to be played by the instrument.
        nshots: int = The number of times the sequence of pulses should be repeated.
        repetition_duration: int = The total duration of every pulse sequence execution and reset time.
        """

        # Save the hash of the current sequence of pulses.
        self._current_pulsesequence_hash = hash(
            (
                instrument_pulses,
                nshots,
                repetition_duration,
                self.ports["o1"].hardware_mod_en,
                self.ports["o2"].hardware_mod_en,
            )
        )

        # Check if the sequence to be processed is the same as the last one.
        # If so, there is no need to generate new waveforms and program
        if self._current_pulsesequence_hash != self._last_pulsequence_hash:
            self._free_sequencers_numbers = [2, 3, 4, 5]

            # process the pulses for every port
            for port in self.ports:
                # split the collection of instruments pulses by ports
                port_pulses: PulseSequence = instrument_pulses.get_channel_pulses(self._port_channel_map[port])

                # initialise the list of sequencers required by the port
                self._sequencers[port] = []

                if not port_pulses.is_empty:
                    # initialise the list of free sequencer numbers to include the default for each port {'o1': 0, 'o2': 1}
                    self._free_sequencers_numbers = [self.DEFAULT_SEQUENCERS[port]] + self._free_sequencers_numbers

                    # split the collection of port pulses in non overlapping pulses
                    non_overlapping_pulses: PulseSequence
                    for non_overlapping_pulses in port_pulses.separate_overlapping_pulses():
                        # TODO: for non_overlapping_same_frequency_pulses in non_overlapping_pulses.separate_different_frequency_pulses():

                        # each set of not overlapping pulses will be played by a separate sequencer
                        # check sequencer availability
                        if len(self._free_sequencers_numbers) == 0:
                            raise Exception(
                                f"The number of sequencers requried to play the sequence exceeds the number available {self.device_num_sequencers}."
                            )

                        # select a new sequencer and configure it as required
                        next_sequencer_number = self._free_sequencers_numbers.pop(0)
                        if next_sequencer_number != self.DEFAULT_SEQUENCERS[port]:
                            for parameter in self.device.sequencers[self.DEFAULT_SEQUENCERS[port]].parameters:
                                value = self.device.sequencers[self.DEFAULT_SEQUENCERS[port]].get(param_name=parameter)
                                if not value is None:
                                    target = self.device.sequencers[next_sequencer_number]
                                    self._set_device_parameter(target, parameter, value=value)
                        if self.ports[port].hardware_mod_en:
                            self._set_device_parameter(
                                self.device.sequencers[next_sequencer_number],
                                "nco_freq",
                                value=non_overlapping_pulses[0].frequency,
                            )  # TODO: for non_overlapping_same_frequency_pulses[0].frequency
                        sequencer = Sequencer(next_sequencer_number)

                        # add the sequencer to the list of sequencers required by the port
                        self._sequencers[port].append(sequencer)

                        # make a temporary copy of the pulses to be processed
                        pulses_to_be_processed = non_overlapping_pulses.shallow_copy()
                        while not pulses_to_be_processed.is_empty:
                            pulse: Pulse = pulses_to_be_processed[0]
                            # select between envelope or modulated waveforms depending on hardware modulation setting
                            if self.ports[port].hardware_mod_en:
                                pulse.waveform_i, pulse.waveform_q = pulse.envelope_waveforms
                            else:
                                pulse.waveform_i, pulse.waveform_q = pulse.modulated_waveforms

                            # attempt to save the waveforms to the sequencer waveforms buffer
                            try:
                                sequencer.waveforms_buffer.add_waveforms(pulse.waveform_i, pulse.waveform_q)
                                sequencer.pulses.add(pulse)
                                pulses_to_be_processed.remove(pulse)

                            # if there is not enough memory in the current sequencer, use another one
                            except WaveformsBuffer.NotEnoughMemory:
                                if len(pulse.waveform_i) + len(pulse.waveform_q) > WaveformsBuffer.SIZE:
                                    raise NotImplementedError(
                                        f"Pulses with waveforms longer than the memory of a sequencer ({WaveformsBuffer.SIZE // 2}) are not supported."
                                    )
                                if len(self._free_sequencers_numbers) == 0:
                                    raise Exception(
                                        f"The number of sequencers requried to play the sequence exceeds the number available {self.device_num_sequencers}."
                                    )

                                # select a new sequencer and configure it as required
                                next_sequencer_number = self._free_sequencers_numbers.pop(0)
                                for parameter in self.device.sequencers[self.DEFAULT_SEQUENCERS[port]].parameters:
                                    value = self.device.sequencers[self.DEFAULT_SEQUENCERS[port]].get(
                                        param_name=parameter
                                    )
                                    if not value is None:
                                        target = self.device.sequencers[next_sequencer_number]
                                        self._set_device_parameter(target, parameter, value=value)
                                self._set_device_parameter(
                                    self.device.sequencers[next_sequencer_number], "nco_freq", value=pulse.frequency
                                )
                                sequencer = Sequencer(next_sequencer_number)

                                # add the sequencer to the list of sequencers required by the port
                                self._sequencers[port].append(sequencer)

            # update the lists of used and unused sequencers that will be needed later on
            self._used_sequencers_numbers = []
            for port in self._output_ports_keys:
                for sequencer in self._sequencers[port]:
                    self._used_sequencers_numbers.append(sequencer.number)
            self._unused_sequencers_numbers = []
            for n in range(self.device_num_sequencers):
                if not n in self._used_sequencers_numbers:
                    self._unused_sequencers_numbers.append(n)

            # generate and store the Waveforms dictionary, the Acquisitions dictionary, the Weights and the Program
            for port in self._output_ports_keys:
                for sequencer in self._sequencers[port]:

                    # Waveforms
                    for index, waveform in enumerate(sequencer.waveforms_buffer.unique_waveforms):
                        sequencer.waveforms[waveform.serial] = {"data": waveform.data.tolist(), "index": index}

                    # Acquisitions
                    for acquisition_index, pulse in enumerate(sequencer.pulses.ro_pulses):
                        sequencer.acquisitions[pulse.serial] = {"num_bins": 1, "index": acquisition_index}

                    # Program
                    minimum_delay_between_instructions = 4
                    wait_loop_step: int = 1000

                    pulses = sequencer.pulses
                    sequence_total_duration = (
                        pulses.start + pulses.duration + minimum_delay_between_instructions
                    )  # the minimum delay between instructions is 4ns
                    time_between_repetitions = repetition_duration - sequence_total_duration
                    assert time_between_repetitions > 0

                    wait_time = time_between_repetitions
                    extra_wait = wait_time % wait_loop_step
                    while wait_time > 0 and extra_wait < 4:
                        wait_loop_step += 1
                        extra_wait = wait_time % wait_loop_step
                    num_wait_loops = (wait_time - extra_wait) // wait_loop_step

                    header = f"""
                    move {nshots},R0             # nshots
                    nop
                    wait_sync {minimum_delay_between_instructions}
                    loop:"""
                    if self.ports[port].hardware_mod_en:
                        header += "\n" + "                    reset_ph"
                    body = ""

                    footer = f"""
                    # wait {wait_time} ns"""
                    if num_wait_loops > 0:
                        footer += f"""
                    move {num_wait_loops},R2
                    nop
                    waitloop2:
                        wait {wait_loop_step}
                        loop R2,@waitloop2"""
                    if extra_wait > 0:
                        footer += f"""
                    wait {extra_wait}"""
                    else:
                        footer += f"""
                    # wait 0"""

                    footer += f"""
                    loop R0,@loop
                    stop
                    """

                    # Add an initial wait instruction for the first pulse of the sequence
                    wait_time = pulses[0].start
                    extra_wait = wait_time % wait_loop_step
                    while wait_time > 0 and extra_wait < 4:
                        wait_loop_step += 1
                        extra_wait = wait_time % wait_loop_step
                    num_wait_loops = (wait_time - extra_wait) // wait_loop_step

                    if wait_time > 0:
                        initial_wait_instruction = f"""
                    # wait {wait_time} ns"""
                        if num_wait_loops > 0:
                            initial_wait_instruction += f"""
                    move {num_wait_loops},R1
                    nop
                    waitloop1:
                        wait {wait_loop_step}
                        loop R1,@waitloop1"""
                        if extra_wait > 0:
                            initial_wait_instruction += f"""
                    wait {extra_wait}"""
                        else:
                            initial_wait_instruction += f"""
                    # wait 0"""
                    else:
                        initial_wait_instruction = """
                    # wait 0"""

                    body += initial_wait_instruction

                    for n in range(pulses.count):
                        # Calculate the delay_after_play that is to be used as an argument to the play instruction
                        if len(pulses) > n + 1:
                            # If there are more pulses to be played, the delay is the time between the pulse end and the next pulse start
                            delay_after_play = pulses[n + 1].start - pulses[n].start
                        else:
                            delay_after_play = sequence_total_duration - pulses[n].start

                        if delay_after_play < minimum_delay_between_instructions:
                            raise Exception(
                                f"The minimum delay between pulses is {minimum_delay_between_instructions}ns."
                            )
                        if self.ports[port].hardware_mod_en and pulses[n].relative_phase != 0:
                            # Set phase
                            p = 10
                            phase = (pulses[n].relative_phase * 360 / (2 * np.pi)) % 360
                            coarse = int(round(phase / 0.9, p))
                            fine = int(round((phase - coarse * 0.9) / 2.25e-3, p))
                            ultra_fine = int(round((phase - coarse * 0.9 - fine * 2.25e-3) / 3.6e-7, p))
                            error = abs(phase - coarse * 0.9 - fine * 2.25e-3 - ultra_fine * 3.6e-7)
                            assert error < 3.6e-7
                            set_ph_instruction = f"                    set_ph {coarse}, {fine}, {ultra_fine}"
                            set_ph_instruction += (
                                " " * (45 - len(set_ph_instruction))
                                + f"# set relative phase {pulses[n].relative_phase} rads"
                            )
                            body += "\n" + set_ph_instruction
                        # Prepare play instruction: play arg0, arg1, arg2.
                        #   arg0 is the index of the I waveform
                        #   arg1 is the index of the Q waveform
                        #   arg2 is the delay between starting the instruction and the next instruction
                        play_instruction = f"                    play {sequencer.waveforms_buffer.unique_waveforms.index(pulses[n].waveform_i)},{sequencer.waveforms_buffer.unique_waveforms.index(pulses[n].waveform_q)},{delay_after_play}"
                        # Add the serial of the pulse as a comment
                        play_instruction += " " * (45 - len(play_instruction)) + f"# play waveforms {pulses[n]}"
                        body += "\n" + play_instruction

                    sequencer.program = header + body + footer

    def upload(self):
        """Uploads waveforms and programs of all sequencers and arms them in preparation for execution.

        This method should be called after `process_pulse_sequence()`.
        It configures certain parameters of the instrument based on the needs of resources determined
        while processing the pulse sequence.
        """
        if self._current_pulsesequence_hash != self._last_pulsequence_hash:
            self._last_pulsequence_hash = self._current_pulsesequence_hash

            # Setup
            for sequencer_number in self._used_sequencers_numbers:
                target = self.device.sequencers[sequencer_number]
                self._set_device_parameter(target, "sync_en", value=True)
                self._set_device_parameter(target, "marker_ovr_en", value=True)  # Default after reboot = False
                self._set_device_parameter(target, "marker_ovr_value", value=15)  # Default after reboot = 0

            for sequencer_number in self._unused_sequencers_numbers:
                target = self.device.sequencers[sequencer_number]
                self._set_device_parameter(target, "sync_en", value=False)
                self._set_device_parameter(target, "marker_ovr_en", value=True)  # Default after reboot = False
                self._set_device_parameter(target, "marker_ovr_value", value=0)  # Default after reboot = 0
                # self._set_device_parameter(target, 'channel_map_path0_out0_en', value = False)
                # self._set_device_parameter(target, 'channel_map_path0_out2_en', value = False)
                # self._set_device_parameter(target, 'channel_map_path1_out1_en', value = False)
                # self._set_device_parameter(target, 'channel_map_path1_out3_en', value = False)

            # Upload waveforms and program
            qblox_dict = {}
            sequencer: Sequencer
            for port in self._output_ports_keys:
                for sequencer in self._sequencers[port]:
                    # Add sequence program and waveforms to single dictionary and write to JSON file
                    filename = f"{self.name}_sequencer{sequencer.number}_sequence.json"
                    qblox_dict[sequencer] = {
                        "waveforms": sequencer.waveforms,
                        "weights": sequencer.weights,
                        "acquisitions": sequencer.acquisitions,
                        "program": sequencer.program,
                    }
                    with open(self.data_folder / filename, "w", encoding="utf-8") as file:
                        json.dump(qblox_dict[sequencer], file, indent=4)

                    # Upload json file to the device sequencers
                    self.device.sequencers[sequencer.number].sequence(str(self.data_folder / filename))

        # Arm sequencers
        for sequencer_number in self._used_sequencers_numbers:
            self.device.arm_sequencer(sequencer_number)

        # DEBUG: QRM Print Readable Snapshot
        # print(self.name)
        # self.device.print_readable_snapshot(update=True)

    def play_sequence(self):
        """Executes the sequence of instructions."""
        for sequencer_number in self._used_sequencers_numbers:
            # Start used sequencers
            self.device.start_sequencer(sequencer_number)

    def start(self):
        """Empty method to comply with AbstractInstrument interface."""
        pass

    def stop(self):
        """Stops all sequencers"""
        try:
            self.device.stop_sequencer()
        except:
            pass

    def disconnect(self):
        """Empty method to comply with AbstractInstrument interface."""
        self._cluster = None
        self.is_connected = False
