from dataclasses import dataclass, field, fields
from typing import ClassVar, Dict, Literal, Optional, Union

DIGITAL_DELAY = 57
DIGITAL_BUFFER = 18
"""Default calibration parameters for digital pulses.

https://docs.quantum-machines.co/1.1.7/qm-qua-sdk/docs/Guides/octave/#calibrating-the-digital-pulse

Digital markers are used for LO triggering.
"""


@dataclass
class QMPort:
    """Abstract representation of Quantum Machine instrument ports.

    Contains the ports settings for each device.
    """

    device: str
    """Name of the device holding this port."""
    number: int
    """Number of this port in the device."""

    key: ClassVar[Optional[str]] = None
    """Key corresponding to this port type in the Quantum Machines config.

    Used in :meth:`qibolab.instruments.qm.config.QMConfig.register_port`.
    """

    @property
    def pair(self):
        """Representation of the port in the Quantum Machines config."""
        return (self.device, self.number)

    def setup(self, **kwargs):
        """Updates port settings."""
        for name, value in kwargs.items():
            if not hasattr(self, name):
                raise KeyError(f"Unknown port setting {name}.")
            setattr(self, name, value)

    @property
    def settings(self):
        """Serialization of the port settings to dump in the runcard YAML.

        Only fields that provide the ``metadata['settings']`` flag are dumped
        in the serialization.
        """
        return {
            fld.name: getattr(self, fld.name)
            for fld in fields(self)
            if fld.metadata.get("settings", False)
        }

    @property
    def config(self):
        """Port settings in the format of the Quantum Machines config.

        Field ``metadata['config']`` are used to translate qibolab port setting names
        to the corresponding Quantum Machine config properties.
        """
        data = {}
        for fld in fields(self):
            if "config" in fld.metadata:
                data[fld.metadata["config"]] = getattr(self, fld.name)
        return {self.number: data}


class QMOutput(QMPort):
    """Abstract Quantum Machines output port."""

    @property
    def name(self):
        """Name of the port when dumping instrument settings on the runcard
        YAML."""
        return f"o{self.number}"


class QMInput(QMPort):
    """Abstract Quantum Machines input port."""

    @property
    def name(self):
        """Name of the port when dumping instrument settings on the runcard
        YAML."""
        return f"i{self.number}"


@dataclass
class OPXOutput(QMOutput):
    key: ClassVar[str] = "analog_outputs"

    offset: float = field(default=0.0, metadata={"config": "offset"})
    """Constant voltage to be applied on the output."""
    filter: Dict[str, float] = field(
        default_factory=dict, metadata={"config": "filter", "settings": True}
    )
    """FIR and IIR filters to be applied to correct signal distortions."""

    @property
    def settings(self):
        """OPX+ output settings to be dumped to the runcard YAML.

        Filter is removed if empty to simplify the runcard.
        """
        data = super().settings
        if len(self.filter) == 0:
            del data["filter"]
        return data


@dataclass
class OPXInput(QMInput):
    key: ClassVar[str] = "analog_inputs"

    offset: float = field(default=0.0, metadata={"config": "offset"})
    """Constant voltage to be applied on the output."""
    gain: int = field(default=0, metadata={"config": "gain_db", "settings": True})
    """Gain applied to amplify the input."""


@dataclass
class OPXIQ:
    """Pair of I-Q ports."""

    i: Union[OPXOutput, OPXInput]
    """Port implementing the I-component of the signal."""
    q: Union[OPXOutput, OPXInput]
    """Port implementing the Q-component of the signal."""


@dataclass
class FEMOutput(OPXOutput):
    fem_number: int = 0
    fem_type: Literal["LF", "MF"] = "LF"

    @property
    def name(self):
        return f"{self.fem_number}/o{self.number}"

    @property
    def pair(self):
        return (self.device, self.fem_number, self.number)


@dataclass
class FEMInput(OPXInput):
    fem_number: int = 0
    fem_type: Literal["LF", "MF"] = "LF"

    @property
    def name(self):
        return f"{self.fem_number}/i{self.number}"

    @property
    def pair(self):
        return (self.device, self.fem_number, self.number)


@dataclass
class OctaveOutput(QMOutput):
    key: ClassVar[str] = "RF_outputs"

    lo_frequency: float = field(
        default=0.0, metadata={"config": "LO_frequency", "settings": True}
    )
    """Local oscillator frequency."""
    gain: int = field(default=0, metadata={"config": "gain", "settings": True})
    """Local oscillator gain.

    Can be in the range [-20 : 0.5 : 20] dB.
    """
    lo_source: str = field(default="internal", metadata={"config": "LO_source"})
    """Local oscillator clock source.

    Can be external or internal.
    """
    output_mode: str = field(default="triggered", metadata={"config": "output_mode"})
    """Can be: "always_on" / "always_off"/ "triggered" / "triggered_reversed"."""
    digital_delay: int = DIGITAL_DELAY
    """Delay for digital output channel."""
    digital_buffer: int = DIGITAL_BUFFER
    """Buffer for digital output channel."""

    opx_port: Optional[OPXOutput] = None
    """OPX+ port that is connected to the Octave port."""

    @property
    def digital_inputs(self):
        """Generates `digitalInputs` entry for elements in QM config.

        Digital markers are used to switch LOs on in triggered mode.
        """
        opx_port = self.opx_port.i
        if isinstance(opx_port, (FEMOutput, FEMInput)):
            port = (opx_port.device, opx_port.fem_number, opx_port.number)
        else:
            port = (opx_port.device, opx_port.number)
        return {
            "output_switch": {
                "port": port,
                "delay": self.digital_delay,
                "buffer": self.digital_buffer,
            }
        }


@dataclass
class OctaveInput(QMInput):
    key: ClassVar[str] = "RF_inputs"

    lo_frequency: float = field(
        default=0.0, metadata={"config": "LO_frequency", "settings": True}
    )
    """Local oscillator frequency."""
    lo_source: str = field(default="internal", metadata={"config": "LO_source"})
    """Local oscillator clock source.

    Can be external or internal.
    """
    IF_mode_I: str = field(default="direct", metadata={"config": "IF_mode_I"})
    IF_mode_Q: str = field(default="direct", metadata={"config": "IF_mode_Q"})

    opx_port: Optional[OPXIQ] = None
    """OPX+ port that is connected to the Octave port."""
