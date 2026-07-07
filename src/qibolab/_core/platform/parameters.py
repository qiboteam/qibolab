import os

from ..components import (
    AcquisitionChannel,
    AcquisitionConfig,
    Channel,
    Config,
    DcChannel,
    DcConfig,
    IqChannel,
    IqConfig,
    IqMixerConfig,
    OscillatorConfig,
)
from ..identifier import ChannelId
from ..native import Native, NativeContainer, SingleQubitNatives, TwoQubitNatives
from ..parameters import NativeGates, Parameters, Settings
from ..pulses import Acquisition, Pulse, Readout, Rectangular
from ..qubits import Qubit, QubitMap
from .components import Hardware
from .load import evaluate_path, load_platform
from .platform import PARAMETERS

__all__ = ["initialize_parameters"]


def _gate_channel(qubit: Qubit, gate: str) -> ChannelId | None:
    """Default channel that a native gate plays on."""
    if gate in ("RX", "RX90", "CNOT"):
        return qubit.drive
    if gate == "RX12":
        return qubit.drive_extra[(1, 2)]
    if gate == "MZ":
        return qubit.acquisition
    if gate in ("CP", "CZ", "iSWAP"):
        return qubit.flux


def _gate_sequence(qubit: Qubit, gate: str) -> Native:
    """Default sequence corresponding to a native gate."""
    channel = _gate_channel(qubit, gate)
    pulse = Pulse(duration=0, amplitude=0, envelope=Rectangular())
    if gate != "MZ":
        assert channel is not None
        return Native([(channel, pulse)])

    assert channel is not None
    return Native(
        [(channel, Readout(acquisition=Acquisition(duration=0), probe=pulse))]
    )


def _pair_to_qubit(pair: str, qubits: QubitMap) -> Qubit:
    """Get first qubit of a pair given in ``{q0}-{q1}`` format."""
    q = tuple(pair.split("-"))[0]
    try:
        return qubits[q]
    except KeyError:
        return qubits[int(q)]


def _native_builder(cls, qubit: Qubit, natives: set[str]) -> NativeContainer:
    """Build default native gates for a given qubit or pair.

    In case of pair, ``qubit`` is assumed to be the first qubit of the pair,
    and a default pulse is added on that qubit, because at this stage we don't
    know which qubit is the high frequency one.
    """
    return cls(
        **{
            gate: _gate_sequence(qubit, gate)
            for gate in cls.model_fields.keys() & natives
        }
    )


def _channel_config(id: ChannelId, channel: Channel) -> dict[ChannelId, Config]:
    """Default configs correspondign to a channel."""
    if isinstance(channel, DcChannel):
        return {id: DcConfig(offset=0)}
    if isinstance(channel, AcquisitionChannel):
        return {id: AcquisitionConfig(delay=0, smearing=0)}
    if isinstance(channel, IqChannel):
        configs = {id: IqConfig(frequency=0)}
        if channel.lo is not None:
            configs[channel.lo] = OscillatorConfig(frequency=0, power=0)
        if channel.mixer is not None:
            configs[channel.mixer] = IqMixerConfig()
        return configs
    return {id: Config()}


def initialize_parameters(
    hardware: Hardware,
    natives: set[str] | None = None,
    pairs: list[str] | None = None,
) -> Parameters:
    """Generates default ``Parameters`` for a given hardware configuration."""
    natives = set(natives if natives is not None else ())
    configs = {}
    for instrument in hardware.instruments.values():
        if hasattr(instrument, "channels"):
            for id, channel in instrument.channels.items():
                configs |= _channel_config(id, channel)

    single_qubit = {
        q: _native_builder(SingleQubitNatives, qubit, natives - {"CP"})
        for q, qubit in hardware.qubits.items()
    }
    coupler = {
        q: _native_builder(SingleQubitNatives, qubit, natives & {"CP"})
        for q, qubit in hardware.couplers.items()
    }
    two_qubit = {
        pair: _native_builder(
            TwoQubitNatives, _pair_to_qubit(pair, hardware.qubits), natives
        )
        for pair in (pairs if pairs is not None else ())
    }

    native_gates = NativeGates(
        single_qubit=single_qubit, coupler=coupler, two_qubit=two_qubit
    )

    return Parameters(settings=Settings(), configs=configs, native_gates=native_gates)


def reset_parameters(
    name: str | os.PathLike[str],
    natives: set[str] | None = None,
    pairs: list[str] | None = None,
) -> None:
    """Reset parameters to default values."""
    hardware = load_platform(evaluate_path(name))
    assert isinstance(hardware, Hardware)
    parameters = initialize_parameters(hardware, natives=natives, pairs=pairs)
    parameters_path = evaluate_path(name) / PARAMETERS
    parameters_path.write_text(parameters.model_dump_json(indent=2))
