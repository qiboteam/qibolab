from collections import defaultdict
from dataclasses import dataclass, field, fields, replace
from enum import Flag, auto
from typing import List, Optional, Union

from qibo import gates

from qibolab.pulses import FluxPulse, PulseConstructor, PulseSequence, PulseType


class NativeType(Flag):
    """Define available types of native gates.

    Should have the same names with qibo gates.
    """

    M = auto()
    Z = auto()
    RZ = auto()
    GPI2 = auto()
    CZ = auto()
    iSWAP = auto()

    @classmethod
    def from_gate(cls, gate: gates.Gate):
        try:
            return getattr(cls, gate.__class__.__name__)
        except AttributeError:
            raise ValueError(f"Gate {gate} cannot be used as native.")


@dataclass
class NativePulse:
    """Container with parameters required to generate a pulse implementing a native gate."""

    name: str
    """Name of the gate that the pulse implements."""
    duration: int
    amplitude: float
    shape: str
    pulse_type: PulseType
    qubit: "qubits.Qubit"
    frequency: int = 0
    relative_start: int = 0
    """Relative start is relevant for two-qubit gate operations which correspond to a pulse sequence."""

    # used for qblox
    if_frequency: Optional[int] = None
    # TODO: Note sure if the following parameters are useful to be in the runcard
    start: int = 0
    phase: float = 0.0

    @classmethod
    def from_dict(cls, name, pulse, qubit):
        """Parse the dictionary provided by the runcard.

        Args:
            name (str): Name of the native gate (dictionary key).
            pulse (dict): Dictionary containing the parameters of the pulse implementing
                the gate, as loaded from the runcard.
            qubits (:class:`qibolab.platforms.abstract.Qubit`): Qubit that the
                pulse is acting on
        """
        kwargs = pulse.copy()
        kwargs["pulse_type"] = PulseType(kwargs.pop("type"))
        kwargs["qubit"] = qubit
        return cls(name, **kwargs)

    @property
    def raw(self):
        data = {fld.name: getattr(self, fld.name) for fld in fields(self) if getattr(self, fld.name) is not None}
        del data["name"]
        del data["start"]
        if self.pulse_type is PulseType.FLUX:
            del data["frequency"]
            del data["phase"]
        data["qubit"] = self.qubit.name
        data["type"] = data.pop("pulse_type").value
        return data

    def pulse(self, start, relative_phase=0.0):
        """Construct the :class:`qibolab.pulses.Pulse` object implementing the gate.

        Args:
            start (int): Start time of the pulse in the sequence.
            relative_phase (float): Relative phase of the pulse.

        Returns:
            A :class:`qibolab.pulses.DrivePulse` or :class:`qibolab.pulses.DrivePulse`
            or :class:`qibolab.pulses.FluxPulse` with the pulse parameters of the gate.
        """
        if self.pulse_type is PulseType.FLUX:
            return FluxPulse(
                start + self.relative_start,
                self.duration,
                self.amplitude,
                self.shape,
                channel=self.qubit.flux.name,
                qubit=self.qubit.name,
            )

        pulse_cls = PulseConstructor[self.pulse_type.name].value
        channel = getattr(self.qubit, self.pulse_type.name.lower()).name
        return pulse_cls(
            start + self.relative_start,
            self.duration,
            self.amplitude,
            self.frequency,
            relative_phase,
            self.shape,
            channel,
            qubit=self.qubit.name,
        )


@dataclass
class VirtualZPulse:
    """Container with parameters required to add a virtual Z phase in a pulse sequence."""

    phase: float
    qubit: "qubits.Qubit"

    @property
    def raw(self):
        return {"type": "virtual_z", "phase": self.phase, "qubit": self.qubit.name}


@dataclass
class NativeSequence:
    """List of :class:`qibolab.platforms.native.NativePulse` objects implementing a gate.

    Relevant for two-qubit gates, which usually require a sequence of pulses to be implemented.
    These pulses may act on qubits different than the qubits the gate is targeting.
    """

    name: str
    pulses: List[Union[NativePulse, VirtualZPulse]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, name, sequence, qubits):
        """Constructs the native sequence from the dictionaries provided in the runcard.

        Args:
            name (str): Name of the gate the sequence is applying.
            sequence (dict): Dictionary describing the sequence as provided in the runcard.
            qubits (list): List of :class:`qibolab.platforms.abstract.Qubit` object for all
                qubits in the platform. All qubits are required because the sequence may be
                acting on qubits that the implemented gate is not targeting.
        """
        pulses = []

        # If sequence contains only one pulse dictionary, convert it into a list that can be iterated below
        if isinstance(sequence, dict):
            sequence = [sequence]

        for i, pulse in enumerate(sequence):
            pulse = pulse.copy()
            qubit = qubits[pulse.pop("qubit")]
            pulse_type = pulse.pop("type")
            if pulse_type == "virtual_z":
                phase = pulse["phase"]
                pulses.append(VirtualZPulse(phase, qubit))
            else:
                pulses.append(NativePulse(f"{name}{i}", **pulse, pulse_type=PulseType(pulse_type), qubit=qubit))
        return cls(name, pulses)

    @property
    def raw(self):
        return [pulse.raw for pulse in self.pulses]

    def sequence(self, start=0):
        """Creates a :class:`qibolab.pulses.PulseSequence` object implementing the sequence."""
        sequence = PulseSequence()
        virtual_z_phases = defaultdict(int)

        for pulse in self.pulses:
            if isinstance(pulse, NativePulse):
                sequence.add(pulse.pulse(start=start))
            else:
                virtual_z_phases[pulse.qubit.name] += pulse.phase

        # TODO: Maybe ``virtual_z_phases`` should be an attribute of ``PulseSequence``
        return sequence, virtual_z_phases


@dataclass
class SingleQubitNatives:
    """Container with the native single-qubit gates acting on a specific qubit."""

    RX: Optional[NativePulse] = None
    MZ: Optional[NativePulse] = None

    @property
    def RX90(self) -> NativePulse:
        """RX90 native pulse is inferred from RX by halving its amplitude."""
        return replace(self.RX, name="RX90", amplitude=self.RX.amplitude / 2.0)

    @classmethod
    def from_dict(cls, qubit, native_gates):
        """Parse native gates of the qubit from the runcard.

        Args:
            qubit (:class:`qibolab.platforms.abstract.Qubit`): Qubit object that the
                native gates are acting on.
            native_gates (dict): Dictionary with native gate pulse parameters as loaded
                from the runcard.
        """
        pulses = {n: NativePulse.from_dict(n, pulse, qubit=qubit) for n, pulse in native_gates.items()}
        return cls(**pulses)

    @property
    def raw(self):
        """Serialize native gate pulses. ``None`` gates are not included."""
        data = {}
        for fld in fields(self):
            attr = getattr(self, fld.name)
            if attr is not None:
                data[fld.name] = attr.raw
                del data[fld.name]["qubit"]
        return data


@dataclass
class TwoQubitNatives:
    """Container with the native two-qubit gates acting on a specific pair of qubits."""

    CZ: Optional[NativeSequence] = None
    iSWAP: Optional[NativeSequence] = None

    @classmethod
    def from_dict(cls, qubits, native_gates):
        sequences = {n: NativeSequence.from_dict(n, seq, qubits) for n, seq in native_gates.items()}
        return cls(**sequences)

    @property
    def raw(self):
        data = {}
        for fld in fields(self):
            gate = getattr(self, fld.name)
            if gate is not None:
                data[fld.name] = gate.raw
        return data

    @property
    def types(self):
        gate_types = NativeType(0)
        for fld in fields(self):
            gate = fld.name
            if getattr(self, gate) is not None:
                gate_types |= NativeType[gate]
        return gate_types
