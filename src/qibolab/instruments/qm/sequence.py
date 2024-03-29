import collections
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union

import numpy as np
from numpy import typing as npt
from qm import qua
from qm.qua._dsl import _Variable  # for type declaration only
from qualang_tools.bakery import baking
from qualang_tools.bakery.bakery import Baking

from qibolab.instruments.qm.acquisition import Acquisition
from qibolab.pulses import Pulse, PulseType

from .config import SAMPLING_RATE, QMConfig

DurationsType = Union[List[int], npt.NDArray[int]]
"""Type of values that can be accepted in a duration sweeper."""


@dataclass
class QMPulse:
    """Wrapper around :class:`qibolab.pulses.Pulse` for easier translation to
    QUA program.

    These pulses are defined when :meth:`qibolab.instruments.qm.QMOPX.play` is called
    and hold attributes for the ``element`` and ``operation`` that corresponds to each pulse,
    as defined in the QM config.
    """

    pulse: Pulse
    """:class:`qibolab.pulses.Pulse` corresponding to the ``QMPulse``."""
    element: Optional[str] = None
    """Element that the pulse will be played on, as defined in the QM
    config."""
    operation: Optional[str] = None
    """Name of the operation that is implementing the pulse in the QM
    config."""
    relative_phase: Optional[float] = None
    """Relative phase of the pulse normalized to follow QM convention.

    May be overwritten when sweeping phase.
    """
    wait_time: int = 0
    """Time (in clock cycles) to wait before playing this pulse.

    Calculated and assigned by
    :meth: `qibolab.instruments.qm.Sequence.add`.
    """
    wait_time_variable: Optional[_Variable] = None
    """Time (in clock cycles) to wait before playing this pulse when we are
    sweeping start."""
    swept_duration: Optional[_Variable] = None
    """Pulse duration when sweeping it."""

    acquisition: Optional[Acquisition] = None
    """Data class containing the variables required for data acquisition for
    the instrument."""

    next_: Set["QMPulse"] = field(default_factory=set)
    """Pulses that will be played after the current pulse.

    These pulses need to be re-aligned if we are sweeping the start or
    duration.
    """
    elements_to_align: Set[str] = field(default_factory=set)

    def __post_init__(self):
        pulse_type = self.pulse.type.name.lower()
        amplitude = format(self.pulse.amplitude, ".6f").rstrip("0").rstrip(".")
        self.element: str = f"{pulse_type}{self.pulse.qubit}"
        self.operation: str = (
            f"{pulse_type}({self.pulse.duration}, {amplitude}, {self.pulse.shape})"
        )
        self.relative_phase: float = self.pulse.relative_phase / (2 * np.pi)
        self.elements_to_align.add(self.element)

    def __hash__(self):
        return hash(self.pulse)

    @property
    def duration(self):
        """Duration of the pulse as defined in the
        :class:`qibolab.pulses.PulseSequence`.

        Remains constant even when we are sweeping the duration of this
        pulse.
        """
        return self.pulse.duration

    @property
    def wait_cycles(self):
        """Instrument clock cycles (1 cycle = 4ns) to wait before playing the
        pulse.

        This property will be used in the QUA ``wait`` command, so that it is compatible
        with and without start sweepers.
        """
        if self.wait_time_variable is not None:
            return self.wait_time_variable + self.wait_time
        if self.wait_time >= 4:
            return self.wait_time
        return None

    def play(self):
        """Play the pulse.

        Relevant only in the context of a QUA program.
        """
        qua.play(self.operation, self.element, duration=self.swept_duration)


@dataclass
class BakedPulse(QMPulse):
    """Baking allows 1ns resolution in the pulse waveforms."""

    segments: List[Baking] = field(default_factory=list)
    """Baked segments implementing the pulse."""
    amplitude: Optional[float] = None
    """Amplitude of the baked pulse.

    Relevant only when sweeping amplitude.
    """
    durations: Optional[DurationsType] = None

    def __hash__(self):
        return super().__hash__()

    @property
    def duration(self):
        return self.segments[-1].get_op_length()

    @staticmethod
    def calculate_waveform(original_waveform, t):
        if t == 0:  # Otherwise, the baking will be empty and will not be created
            return [0.0] * 16

        expanded_waveform = list(original_waveform)
        for i in range(t // len(original_waveform)):
            expanded_waveform.extend(original_waveform)
        return expanded_waveform[:t]

    def bake(self, config: QMConfig, durations: DurationsType):
        self.segments = []
        self.durations = durations
        for t in durations:
            with baking(config.__dict__, padding_method="right") as segment:
                if self.pulse.type is PulseType.FLUX:
                    waveform = self.pulse.envelope_waveform_i(
                        SAMPLING_RATE
                    ).data.tolist()
                    waveform = self.calculate_waveform(waveform, t)
                else:
                    waveform_i = self.pulse.envelope_waveform_i(
                        SAMPLING_RATE
                    ).data.tolist()
                    waveform_q = self.pulse.envelope_waveform_q(
                        SAMPLING_RATE
                    ).data.tolist()
                    waveform = [
                        self.calculate_waveform(waveform_i, t),
                        self.calculate_waveform(waveform_q, t),
                    ]
                segment.add_op(self.operation, self.element, waveform)
                segment.play(self.operation, self.element)
            self.segments.append(segment)

    @property
    def amplitude_array(self):
        if self.amplitude is None:
            return None
        return [(self.element, self.amplitude)]

    def play(self):
        if self.swept_duration is not None:
            with qua.switch_(self.swept_duration):
                for dur, segment in zip(self.durations, self.segments):
                    with qua.case_(dur):
                        segment.run(amp_array=self.amplitude_array)
        else:
            segment = self.segments[0]
            segment.run(amp_array=self.amplitude_array)


@dataclass
class Sequence:
    """Pulse sequence containing QM specific pulses (``qmpulse``).

    Defined in :meth:`qibolab.instruments.qm.QMOPX.play`.
    Holds attributes for the ``element`` and ``operation`` that
    corresponds to each pulse, as defined in the QM config.
    """

    qmpulses: List[QMPulse] = field(default_factory=list)
    """List of :class:`qibolab.instruments.qm.QMPulse` objects corresponding to
    the original pulses."""
    pulse_to_qmpulse: Dict[Pulse, QMPulse] = field(default_factory=dict)
    """Map from qibolab pulses to QMPulses (useful when sweeping)."""
    clock: Dict[str, int] = field(default_factory=lambda: collections.defaultdict(int))
    """Dictionary used to keep track of times of each element, in order to
    calculate wait times."""
    pulse_finish: Dict[int, List[QMPulse]] = field(
        default_factory=lambda: collections.defaultdict(list)
    )
    """Map to find all pulses that finish at a given time (useful for
    ``_find_previous``)."""

    def _find_previous(self, pulse):
        for finish in reversed(sorted(self.pulse_finish.keys())):
            if finish <= pulse.start:
                # first try to find a previous pulse targeting the same qubit
                last_pulses = self.pulse_finish[finish]
                for previous in reversed(last_pulses):
                    if previous.pulse.qubit == pulse.qubit:
                        return previous
                # otherwise
                if finish == pulse.start:
                    return last_pulses[-1]
        return None

    def add(self, qmpulse: QMPulse):
        pulse = qmpulse.pulse
        self.pulse_to_qmpulse[pulse.id] = qmpulse

        previous = self._find_previous(pulse)
        if previous is not None:
            previous.next_.add(qmpulse)

        wait_time = pulse.start - self.clock[qmpulse.element]
        if wait_time >= 12:
            qmpulse.wait_time = wait_time // 4 + 1
            self.clock[qmpulse.element] += 4 * qmpulse.wait_time
        self.clock[qmpulse.element] += qmpulse.duration

        self.pulse_finish[pulse.finish].append(qmpulse)
        self.qmpulses.append(qmpulse)

    def shift(self):
        """Shift all pulses that come after a ``BakedPulse`` a bit to avoid
        overlapping pulses."""
        to_shift = collections.deque()
        for qmpulse in self.qmpulses:
            if isinstance(qmpulse, BakedPulse):
                to_shift.extend(qmpulse.next_)
        while to_shift:
            qmpulse = to_shift.popleft()
            qmpulse.wait_time += 2
            to_shift.extend(qmpulse.next_)

    def play(self, relaxation_time=0):
        """Part of QUA program that plays an arbitrary pulse sequence.

        Should be used inside a ``program()`` context.
        """
        needs_reset = False
        qua.align()
        for qmpulse in self.qmpulses:
            pulse = qmpulse.pulse
            if qmpulse.wait_cycles is not None:
                qua.wait(qmpulse.wait_cycles, qmpulse.element)
            if pulse.type is PulseType.READOUT:
                qmpulse.acquisition.measure(qmpulse.operation, qmpulse.element)
            else:
                if (
                    not isinstance(qmpulse.relative_phase, float)
                    or qmpulse.relative_phase != 0
                ):
                    qua.frame_rotation_2pi(qmpulse.relative_phase, qmpulse.element)
                    needs_reset = True
                qmpulse.play()
                if needs_reset:
                    qua.reset_frame(qmpulse.element)
                    needs_reset = False
                if len(qmpulse.elements_to_align) > 1:
                    qua.align(*qmpulse.elements_to_align)

        if relaxation_time > 0:
            qua.wait(relaxation_time // 4)
