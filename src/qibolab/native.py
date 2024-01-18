from collections import defaultdict
from dataclasses import dataclass, field, fields, replace
from typing import List, Optional, Union

from qibolab.pulses import Pulse, PulseSequence, PulseType


@dataclass
class NativePulse:
    """Container with parameters required to generate a pulse implementing a
    native gate."""

    name: str
    """Name of the gate that the pulse implements."""
    duration: int
    amplitude: float
    shape: str
    pulse_type: PulseType
    qubit: "qubits.Qubit"
    frequency: int = 0
    relative_start: int = 0
    """Relative start is relevant for two-qubit gate operations which
    correspond to a pulse sequence."""

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
        data = {
            fld.name: getattr(self, fld.name)
            for fld in fields(self)
            if getattr(self, fld.name) is not None
        }
        del data["name"]
        del data["start"]
        if self.pulse_type is PulseType.FLUX:
            del data["frequency"]
            del data["phase"]
        data["qubit"] = self.qubit.name
        data["type"] = data.pop("pulse_type").value
        return data

    def pulse(self, start, relative_phase=0.0):
        """Construct the :class:`qibolab.pulses.Pulse` object implementing the
        gate.

        Args:
            start (int): Start time of the pulse in the sequence.
            relative_phase (float): Relative phase of the pulse.

        Returns:
            A :class:`qibolab.pulses.DrivePulse` or :class:`qibolab.pulses.DrivePulse`
            or :class:`qibolab.pulses.FluxPulse` with the pulse parameters of the gate.
        """
        if self.pulse_type is PulseType.FLUX:
            return Pulse.flux(
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
    """Container with parameters required to add a virtual Z phase in a pulse
    sequence."""

    phase: float
    qubit: "qubits.Qubit"

    @property
    def raw(self):
        return {"type": "virtual_z", "phase": self.phase, "qubit": self.qubit.name}


@dataclass
class CouplerPulse:
    """Container with parameters required to add a coupler pulse in a pulse
    sequence."""

    duration: int
    amplitude: float
    shape: str
    coupler: "couplers.Coupler"
    relative_start: int = 0

    @classmethod
    def from_dict(cls, pulse, coupler):
        """Parse the dictionary provided by the runcard.

        Args:
            name (str): Name of the native gate (dictionary key).
            pulse (dict): Dictionary containing the parameters of the pulse implementing
                the gate, as loaded from the runcard.
            coupler (:class:`qibolab.platforms.abstract.Coupler`): Coupler that the
                pulse is acting on
        """
        kwargs = pulse.copy()
        kwargs["coupler"] = coupler
        kwargs.pop("type")
        return cls(**kwargs)

    @property
    def raw(self):
        return {
            "type": "coupler",
            "duration": self.duration,
            "amplitude": self.amplitude,
            "shape": self.shape,
            "coupler": self.coupler.name,
            "relative_start": self.relative_start,
        }

    def pulse(self, start):
        """Construct the :class:`qibolab.pulses.Pulse` object implementing the
        gate.

        Args:
            start (int): Start time of the pulse in the sequence.

        Returns:
            A :class:`qibolab.pulses.FluxPulse` with the pulse parameters of the gate.
        """
        return CouplerFluxPulse(
            start + self.relative_start,
            self.duration,
            self.amplitude,
            self.shape,
            channel=self.coupler.flux.name,
            qubit=self.coupler.name,
        )


@dataclass
class NativeSequence:
    """List of :class:`qibolab.platforms.native.NativePulse` objects
    implementing a gate.

    Relevant for two-qubit gates, which usually require a sequence of
    pulses to be implemented. These pulses may act on qubits different
    than the qubits the gate is targeting.
    """

    name: str
    pulses: List[Union[NativePulse, VirtualZPulse]] = field(default_factory=list)
    coupler_pulses: List[CouplerPulse] = field(default_factory=list)

    @classmethod
    def from_dict(cls, name, sequence, qubits, couplers):
        """Constructs the native sequence from the dictionaries provided in the
        runcard.

        Args:
            name (str): Name of the gate the sequence is applying.
            sequence (dict): Dictionary describing the sequence as provided in the runcard.
            qubits (list): List of :class:`qibolab.qubits.Qubit` object for all
                qubits in the platform. All qubits are required because the sequence may be
                acting on qubits that the implemented gate is not targeting.
            couplers (list): List of :class:`qibolab.couplers.Coupler` object for all
                couplers in the platform. All couplers are required because the sequence may be
                acting on couplers that the implemented gate is not targeting.
        """
        pulses = []
        coupler_pulses = []

        # If sequence contains only one pulse dictionary, convert it into a list that can be iterated below
        if isinstance(sequence, dict):
            sequence = [sequence]

        for i, pulse in enumerate(sequence):
            pulse = pulse.copy()
            pulse_type = pulse.pop("type")
            if pulse_type == "coupler":
                pulse["coupler"] = couplers[pulse.pop("coupler")]
                coupler_pulses.append(CouplerPulse(**pulse))
            else:
                qubit = qubits[pulse.pop("qubit")]
                if pulse_type == "virtual_z":
                    phase = pulse["phase"]
                    pulses.append(VirtualZPulse(phase, qubit))
                else:
                    pulses.append(
                        NativePulse(
                            f"{name}{i}",
                            **pulse,
                            pulse_type=PulseType(pulse_type),
                            qubit=qubit,
                        )
                    )
        return cls(name, pulses, coupler_pulses)

    @property
    def raw(self):
        pulses = [pulse.raw for pulse in self.pulses]
        coupler_pulses = [pulse.raw for pulse in self.coupler_pulses]
        return pulses + coupler_pulses

    def sequence(self, start=0):
        """Creates a :class:`qibolab.pulses.PulseSequence` object implementing
        the sequence."""
        sequence = PulseSequence()
        virtual_z_phases = defaultdict(int)

        for pulse in self.pulses:
            if isinstance(pulse, NativePulse):
                sequence.append(pulse.pulse(start=start))
            else:
                virtual_z_phases[pulse.qubit.name] += pulse.phase

        for coupler_pulse in self.coupler_pulses:
            sequence.append(coupler_pulse.pulse(start=start))
        # TODO: Maybe ``virtual_z_phases`` should be an attribute of ``PulseSequence``
        return sequence, virtual_z_phases


@dataclass
class SingleQubitNatives:
    """Container with the native single-qubit gates acting on a specific
    qubit."""

    RX: Optional[NativePulse] = None
    """Pulse to drive the qubit from state 0 to state 1."""
    RX12: Optional[NativePulse] = None
    """Pulse to drive to qubit from state 1 to state 2."""
    MZ: Optional[NativePulse] = None
    """Measurement pulse."""

    @property
    def RX90(self) -> NativePulse:
        """RX90 native pulse is inferred from RX by halving its amplitude."""
        return replace(self.RX, name="RX90", amplitude=self.RX.amplitude / 2.0)

    @classmethod
    def from_dict(cls, qubit, native_gates):
        """Parse native gates of the qubit from the runcard.

        Args:
            qubit (:class:`qibolab.qubits.Qubit`): Qubit object that the
                native gates are acting on.
            native_gates (dict): Dictionary with native gate pulse parameters as loaded
                from the runcard.
        """
        pulses = {
            n: NativePulse.from_dict(n, pulse, qubit=qubit)
            for n, pulse in native_gates.items()
        }
        return cls(**pulses)

    @property
    def raw(self):
        """Serialize native gate pulses.

        ``None`` gates are not included.
        """
        data = {}
        for fld in fields(self):
            attr = getattr(self, fld.name)
            if attr is not None:
                data[fld.name] = attr.raw
                del data[fld.name]["qubit"]
        return data


@dataclass
class CouplerNatives:
    """Container with the native single-qubit gates acting on a specific
    qubit."""

    CP: Optional[NativePulse] = None
    """Pulse to activate the coupler."""

    @classmethod
    def from_dict(cls, coupler, native_gates):
        """Parse coupler native gates from the runcard.

        Args:
            coupler (:class:`qibolab.couplers.Coupler`): Coupler object that the
                native pulses are acting on.
            native_gates (dict): Dictionary with native gate pulse parameters as loaded
                from the runcard [Reusing the dict from qubits].
        """
        pulses = {
            n: CouplerPulse.from_dict(pulse, coupler=coupler)
            for n, pulse in native_gates.items()
        }
        return cls(**pulses)

    @property
    def raw(self):
        """Serialize native gate pulses.

        ``None`` gates are not included.
        """
        data = {}
        for fld in fields(self):
            attr = getattr(self, fld.name)
            if attr is not None:
                data[fld.name] = attr.raw
        return data


@dataclass
class TwoQubitNatives:
    """Container with the native two-qubit gates acting on a specific pair of
    qubits."""

    CZ: Optional[NativeSequence] = field(default=None, metadata={"symmetric": True})
    CNOT: Optional[NativeSequence] = field(default=None, metadata={"symmetric": False})
    iSWAP: Optional[NativeSequence] = field(default=None, metadata={"symmetric": True})

    @property
    def symmetric(self):
        """Check if the defined two-qubit gates are symmetric between target
        and control qubits."""
        return all(
            fld.metadata["symmetric"] or getattr(self, fld.name) is None
            for fld in fields(self)
        )

    @classmethod
    def from_dict(cls, qubits, couplers, native_gates):
        sequences = {
            n: NativeSequence.from_dict(n, seq, qubits, couplers)
            for n, seq in native_gates.items()
        }
        return cls(**sequences)

    @property
    def raw(self):
        data = {}
        for fld in fields(self):
            gate = getattr(self, fld.name)
            if gate is not None:
                data[fld.name] = gate.raw
        return data
