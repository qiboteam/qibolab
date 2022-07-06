"""Constants"""

# Environment variables
DATA = "DATA"  # variable containing the path where data is saved
RUNCARDS = "RUNCARDS"  # variable containing the runcard's path

RESULTS_FILENAME = "results.yml"
EXPERIMENT_FILENAME = "experiment.yml"

DEFAULT_PLATFORM_NAME = "galadriel"


# TODO: Distribute constants over different classes
class RUNCARD:
    """YAML constants."""

    ID = "id_"
    NAME = "name"
    ALIAS = "alias"
    CATEGORY = "category"
    SUBCATEGORY = "subcategory"
    INSTRUMENT = "instrument"
    ELEMENTS = "elements"
    READOUT = "readout"
    SETTINGS = "settings"
    PLATFORM = "platform"
    SCHEMA = "schema"
    AWG = "awg"
    SIGNAL_GENERATOR = "signal_generator"
    ATTENUATOR = "attenuator"
    SYSTEM_CONTROL = "system_control"
    IP = "ip"
    FIRMWARE = "firmware"


class SIGNALGENERATOR:
    """SignalGenerator attribute names."""

    FREQUENCY = "frequency"


class PLATFORM:
    """Platform constants."""

    PULSES = "pulses"


class EXPERIMENT:
    """Experiment constants."""

    HARDWARE_AVERAGE = "hardware_average"
    SOFTWARE_AVERAGE = "software_average"
    REPETITION_DURATION = "repetition_duration"
    SHAPE = "shape"
    RESULTS = "results"
    NUM_SEQUENCES = "num_sequences"
    SEQUENCES = "sequences"


class SCHEMA:
    """Schema constants."""

    INSTRUMENTS = "instruments"
    BUSES = "buses"
    CHIP = "chip"


class BUS:
    """Bus constants."""

    PORT = "port"
    SYSTEM_CONTROL = "system_control"
    ATTENUATOR = "attenuator"
    SEQUENCES = "sequences"
    NUM_SEQUENCES = "num_sequences"
    SHAPE = "shape"  # shape of the results
    RESULTS = "results"


class LOOP:
    """Loop class and attribute names."""

    LOOP = "loop"
    PARAMETER = "parameter"
    START = "start"
    STOP = "stop"
    NUM = "num"
    STEP = "step"


class PULSESEQUENCES:
    """PulseSequenes attribute names."""

    ELEMENTS = "elements"


class PULSESEQUENCE:
    """PulseSequence attribute names."""

    PULSES = "pulses"
    PORT = "port"


class PULSE:
    """Pulse attribute names."""

    NAME = "name"
    AMPLITUDE = "amplitude"
    FREQUENCY = "frequency"
    PHASE = "phase"
    DURATION = "duration"
    PORT = "port"
    PULSE_SHAPE = "pulse_shape"
    START_TIME = "start_time"


UNITS = {"frequency": "Hz"}

UNIT_PREFIX = {
    1e-24: "y",  # yocto
    1e-21: "z",  # zepto
    1e-18: "a",  # atto
    1e-15: "f",  # femto
    1e-12: "p",  # pico
    1e-9: "n",  # nano
    1e-6: "u",  # micro
    1e-3: "m",  # mili
    1e-2: "c",  # centi
    1e-1: "d",  # deci
    1e3: "k",  # kilo
    1e6: "M",  # mega
    1e9: "G",  # giga
    1e12: "T",  # tera
    1e15: "P",  # peta
    1e18: "E",  # exa
    1e21: "Z",  # zetta
    1e24: "Y",  # yotta
}
