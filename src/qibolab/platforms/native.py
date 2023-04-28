from collections import defaultdict
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Union

from qibo.config import raise_error

from qibolab.pulses import FluxPulse, PulseConstructor, PulseSequence, PulseType


@dataclass
class NativePulse:
    """Container with parameters required to generate a pulse implementing a native gate."""

    name: str
    duration: int
    amplitude: float
    shape: str
    pulse_type: PulseType
    qubit: "Qubit"
    frequency: int = 0
    relative_start: int = 0
    """Relative start is relevant for two-qubit gate operations which correspond to a pulse sequence."""

    @classmethod
    def from_dict(cls, name, **kwargs):
        """Parse the dictionary provided by the runcard.

        Args:
            name (str): Name of the native gate (dictionary key).
            kwargs (dict): Dictionary containing the parameters of the pulse
                implementing the gate.
        """
        field_names = {field.name for field in fields(cls)}
        new_kwargs = {k: v for k, v in kwargs.items() if k in field_names}
        new_kwargs["pulse_type"] = PulseType(kwargs["type"])
        return cls(name, **new_kwargs)

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

        else:
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
class VirtualPulse:
    """Container with parameters required to add a virtual Z phase in a pulse sequence."""

    phase: float
    qubit: "Qubit"


@dataclass
class NativeSequence:
    """List of :class:`qibolab.platforms.native.NativePulse` objects implementing a gate.

    Relevant for two-qubit gates, which usually require a sequence of pulses to be implemented.
    These pulses may act on qubits different than the qubits the gate is targeting.
    """

    name: str
    pulses: List[Union[NativePulse, VirtualPulse]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, name, qubits, sequence):
        """Constructs the native sequence from the dictionaries provided in the runcard.

        Args:
            name (str): Name of the gate the sequence is applying.
            qubits (list): List of :class:`qibolab.platforms.abstract.Qubit` object for all
                qubits in the platform. All qubits are required because the sequence may be
                acting on qubits that the implemented gate is not targeting.
            sequence (dict): Dictionary describing the sequence as provided in the runcard.
        """
        pulses = []

        # If sequence contains only one pulse dictionary, convert it into a list that can be iterated below
        if isinstance(sequence, dict):
            sequence = [sequence]

        for i, pulse in enumerate(sequence):
            pulse_type = pulse["type"]
            kwargs = dict(pulse)
            qubit = qubits[kwargs.pop("qubit")]
            if pulse_type == "virtual_z":
                phase = kwargs["phase"]
                pulses.append(VirtualPulse(phase, qubit))
            else:
                pulses.append(NativePulse.from_dict(f"{name}{i}", **kwargs, qubit=qubit))
        return cls(name, pulses)

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
class NativeSingleQubitGates:
    """Container with the native single qubit gates acting on a specific qubit."""

    MZ: NativePulse
    RX: NativePulse
    RX90: NativePulse

    @classmethod
    def from_dict(cls, qubit, native_gates):
        """Parse native gates of the qubit from the runcard.

        Args:
            qubit (:class:`qibolab.platforms.abstract.Qubit`): Qubit object that the
                native gates are acting on.
            native_gates (dict): Dictionary with native gate pulse parameters as loaded
                from the runcard.
        """
        pulses = {n: NativePulse.from_dict(n, **native_gates[n], qubit=qubit) for n in ["MZ", "RX"]}
        pulses["RX90"] = NativePulse.from_dict("RX90", **native_gates["RX"], qubit=qubit)
        pulses["RX90"].amplitude /= 2.0
        return cls(**pulses)
