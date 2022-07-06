"""Enum classes"""
from enum import Enum


class Category(Enum):
    """Category of settings.

    Args:
        enum (str): Available types of settings cattegories:
        * platform
        * qubit
        * awg
        * signal_generator
        * buses
        * bus
        * schema
        * resonator
        * node
    """

    PLATFORM = "platform"
    QUBIT = "qubit"
    AWG = "awg"
    SIGNAL_GENERATOR = "signal_generator"
    SCHEMA = "schema"
    RESONATOR = "resonator"
    BUSES = "buses"
    BUS = "bus"
    SYSTEM_CONTROL = "system_control"
    EXPERIMENT = "experiment"
    ATTENUATOR = "attenuator"
    DC_SOURCE = "dc_source"
    CHIP = "chip"
    NODE = "node"


class Instrument(Enum):
    """Instrument.

    Args:
        enum (str): Available types of instruments:
        * platform
        * awg
        * signal_generator
        * system_control
        * attenuator
    """

    PLATFORM = "platform"
    AWG = "awg"
    SIGNAL_GENERATOR = "signal_generator"
    SYSTEM_CONTROL = "system_control"
    ATTENUATOR = "attenuator"


class ReferenceClock(Enum):
    """Qblox reference clock.

    Args:
        enum (str): Available types of reference clock:
        * Internal
        * External
    """

    INTERNAL = "internal"
    EXTERNAL = "external"


class AcquireTriggerMode(Enum):
    """Qblox acquire trigger mode.

    Args:
        enum (str): Available types of trigger modes:
        * sequencer
        * level
    """

    SEQUENCER = "sequencer"
    LEVEL = "level"


class IntegrationMode(Enum):
    """Qblox integration mode.

    Args:
        enum (str): Available types of integration modes:
        * ssb
    """

    SSB = "ssb"


class GateName(Enum):
    """Gate names.

    Args:
        enum (str): Available types of gate names:
        * I
        * X
        * Y
        * M
        * RX
        * RY
        * XY
    """

    I = "I"  # noqa: E741
    X = "X"
    RX = "RX"
    Y = "Y"
    RY = "RY"
    XY = "XY"
    M = "M"


class AcquisitionName(Enum):
    """Acquisition names.

    Args:
        enum (str): Available types of acquisition names:
        * single
    """

    SINGLE = "single"
    LARGE = "large"


class SchemaDrawOptions(Enum):
    """Schema draw options.

    Args:
        enum (str): Available types of schema draw options:
        * print
        * file
    """

    PRINT = "print"
    FILE = "file"


class PulseName(Enum):
    """Pulse names.

    Args:
        Enum (str): Available types of Pulse names:
        * pulse
        * readout_pulse
    """

    PULSE = "pulse"
    READOUT_PULSE = "readout_pulse"


class PulseShapeName(Enum):
    """Pulse shape options.

    Args:
        Enum (str): Available types of PulseShape options:
        * gaussian
    """

    GAUSSIAN = "gaussian"
    DRAG = "drag"
    RECTANGULAR = "rectangular"


class BusSubcategory(Enum):
    """Bus types.

    Args:
        enum (str): Available types of Bus:
        * control
        * readout
    """

    CONTROL = "control"
    READOUT = "readout"


class SystemControlSubcategory(Enum):
    """Bus element names. Contains names of bus elements that are not instruments.

    Args:
        enum (str): Available bus element names:
        * mixer_based_system_control
        * simulated_system_control
    """

    MIXER_BASED_SYSTEM_CONTROL = "mixer_based_system_control"
    SIMULATED_SYSTEM_CONTROL = "simulated_system_control"


class NodeName(Enum):
    """Node names.

    Args:
        enum (str): Available node names:
        * qubit
        * resonator
        * coupler
    """

    QUBIT = "qubit"
    RESONATOR = "resonator"
    COUPLER = "coupler"
    PORT = "port"


class InstrumentName(Enum):
    """Instrument names.

    Args:
        enum (str): Available bus element names:
        * qblox_qcm
        * qblox_qrm
        * rohde_schwarz
        * mini_circuits
        * mixer_based_system_control
        * integrated_system_control
        * simulated_system_control
    """

    QBLOX_QCM = "qblox_qcm"
    QBLOX_QRM = "qblox_qrm"
    ROHDE_SCHWARZ = "rohde_schwarz"
    INTEGRATED_SYSTEM_CONTROL = "integrated_system_control"
    MINI_CIRCUITS = "mini_circuits"  # step attenuator
    KEITHLEY2600 = "keithley_2600"


class Parameter(Enum):
    """Parameter names."""

    FREQUENCY = "frequency"
    GAIN = "gain"
    DURATION = "duration"
    AMPLITUDE = "amplitude"
    PHASE = "phase"
    DELAY_BETWEEN_PULSES = "delay_between_pulses"
    DELAY_BEFORE_READOUT = "delay_before_readout"
    GATE_DURATION = "gate_duration"
    NUM_SIGMAS = "num_sigmas"
    DRAG_COEFFICIENT = "drag_coefficient"
    REFERENCE_CLOCK = "reference_clock"
    SEQUENCER = "sequencer"
    SYNC_ENABLED = "sync_enabled"
    POWER = "power"
    EPSILON = "epsilon"
    DELTA = "delta"
    OFFSET_I = "offset_i"
    OFFSET_Q = "offset_q"
    SAMPLING_RATE = "sampling_rate"
    INTEGRATION = "integration"
    INTEGRATION_LENGTH = "integration_length"
    ACQUISITION_DELAY_TIME = "acquisition_delay_time"
    ATTENUATION = "attenuation"
    REPETITION_DURATION = "repetition_duration"
    HARDWARE_AVERAGE = "hardware_average"
    SOFTWARE_AVERAGE = "software_average"
    NUM_BINS = "num_bins"


class ResultName(Enum):
    """Result names.

    Args:
        enum (str): Available bus element names:
        * qblox
        * simulator
    """

    QBLOX = "qblox"
    SIMULATOR = "simulator"
