from dataclasses import dataclass, field


@dataclass(frozen=True)
class OutputSwitch:
    port: tuple[str, int]
    delay: int = 57
    buffer: int = 18


def output_switch(opx: str, port: int):
    """Create output switch section."""
    return {"output_switch": OutputSwitch((opx, 2 * port - 1))}


@dataclass
class DcElement:
    singleInput: dict[str, tuple[str, int]]
    intermediate_frequency: int = 0
    operations: dict[str, str] = field(default_factory=dict)


@dataclass
class RfOctaveElement:
    RF_inputs: dict[str, tuple[str, int]]
    digitalInputs: dict[str, OutputSwitch]
    intermediate_frequency: int
    operations: dict[str, str] = field(default_factory=dict)


@dataclass
class AcquireOctaveElement:
    RF_inputs: dict[str, tuple[str, int]]
    RF_outputs: dict[str, tuple[str, int]]
    digitalInputs: dict[str, OutputSwitch]
    intermediate_frequency: int
    time_of_flight: int = 24
    smearing: int = 0
    operations: dict[str, str] = field(default_factory=dict)


Element = DcElement | RfElement | AcquireElement
