from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Set, Union

import numpy as np
from numpy import typing as npt
from qm import qua
from qualang_tools.bakery import baking
from qualang_tools.bakery.bakery import Baking

from qibolab.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab.pulses import Pulse, PulseType
from qibolab.qubits import Qubit

from .acquisition import ACQUISITION_TYPES, Acquisition
from .config import SAMPLING_RATE, QMConfig, float_serial

DurationsType = Union[List[int], npt.NDArray[int]]
"""Type of values that can be accepted in a duration sweeper."""


@dataclass(frozen=True)
class Align:
    elements: tuple = field(default_factory=tuple)

    def __call__(self):
        qua.align(*self.elements)


@dataclass(frozen=True)
class Wait:
    elements: tuple
    duration: int
    """Time (in clock cycles) to wait before playing this pulse.

    Calculated and assigned by :meth: `qibolab.instruments.qm.instructions.Instructions.append`.
    """

    def __call__(self, additional_wait=None):
        """``additional_wait`` is given when we are sweeping start."""
        if additional_wait is not None:
            qua.wait(additional_wait + self.duration, *self.elements)
        elif self.duration >= 4:
            qua.wait(self.duration, *self.elements)


@dataclass(frozen=True)
class Play:
    """Wrapper around :class:`qibolab.pulses.Pulse` for easier translation to
    QUA program.

    These pulses are defined when :meth:`qibolab.instruments.qm.QMOPX.play` is called
    and hold attributes for the ``element`` and ``operation`` that corresponds to each pulse,
    as defined in the QM config.
    """

    element: str
    """QM config element that the instruction will be played on."""
    amplitude: float
    duration: int
    relative_phase: float
    """Relative phase of the pulse normalized to follow QM convention."""
    shape: str

    @classmethod
    def from_pulse(cls, pulse):
        element = f"{pulse.type.name.lower()}{pulse.qubit}"
        phase = (pulse.relative_phase % (2 * np.pi)) / (2 * np.pi)
        shape = str(pulse.shape)
        return cls(element, pulse.amplitude, pulse.duration, phase, shape)

    @property
    def operation(self):
        """Name of the operation implementing the pulse in the QM config."""
        amplitude = float_serial(self.amplitude)
        phase = float_serial(self.relative_phase)
        return f"{self.element}({self.duration}, {amplitude}, {phase}, {self.shape})"

    def __call__(self, duration=None, amplitude=None, phase=None):
        """Play the pulse.

        Args:
            duration: QUA variable when sweeping duration.
            phase: QUA variable when sweeping phase.
        """
        if phase is not None:
            qua.frame_rotation_2pi(phase, self.element)
        if amplitude is not None:
            operation = self.operation * amplitude
        else:
            operation = self.operation

        qua.play(operation, self.element, duration=duration)

        if phase is not None:
            qua.reset_frame(self.element)


@dataclass(frozen=True)
class Measure(Play):

    def __call__(self, acquisition: Acquisition):
        """Play the measurement instruction.

        Args:
            Dataclass containing the variables required for data acquisition
            from the instrument.
        """
        acquisition.measure(self.operation, self.element)


def calculate_waveform(original_waveform, t):
    """Calculate waveform array for baked pulses."""
    if t == 0:
        # Otherwise, baking will be empty and will not be created
        return [0.0] * 16

    expanded_waveform = list(original_waveform)
    for i in range(t // len(original_waveform)):
        expanded_waveform.extend(original_waveform)
    return expanded_waveform[:t]


@dataclass(frozen=True)
class Bake(Play):
    """Baking allows 1ns resolution in the pulse waveforms."""

    segments: List[Baking] = field(default_factory=list)
    """Baked segments implementing the pulse."""
    durations: Optional[DurationsType] = None

    @classmethod
    def from_pulse(cls, pulse, config: QMConfig, durations: DurationsType):
        instruction = super().from_pulse(pulse)
        operation = instruction.operation
        element = instruction.element

        segments = []
        for t in durations:
            with baking(config.__dict__, padding_method="right") as segment:
                if pulse.type is PulseType.FLUX:
                    waveform = pulse.envelope_waveform_i(SAMPLING_RATE).data.tolist()
                    waveform = calculate_waveform(waveform, t)
                else:
                    waveform_i = pulse.envelope_waveform_i(SAMPLING_RATE).data.tolist()
                    waveform_q = pulse.envelope_waveform_q(SAMPLING_RATE).data.tolist()
                    waveform = [
                        calculate_waveform(waveform_i, t),
                        calculate_waveform(waveform_q, t),
                    ]
                segment.add_op(operation, element, waveform)
                segment.play(operation, element)
            segments.append(segment)

        kwargs = asdict(instruction)
        kwargs.update(
            dict(
                segments=segments,
                duration=segments[-1].get_op_length(),
                durations=durations,
            )
        )
        return cls(**kwargs)

    def __call__(self, duration=None, amplitude=None):
        """Plays the baked pulse.

        Args:
            duration: QUA variable when sweeping duration.
            amplitude: Amplitude of the baked pulse.
                Relevant only when sweeping amplitude.
        """
        if amplitude is None:
            amp_array = None
        else:
            amp_array = [(self.element, amplitude)]

        if duration is not None:
            with qua.switch_(duration):
                for dur, segment in zip(self.durations, self.segments):
                    with qua.case_(dur):
                        segment.run(amp_array=amp_array)
        else:
            segment = self.segments[0]
            segment.run(amp_array=amp_array)


# wait_time_variable: Optional[_Variable] = None
# """Time (in clock cycles) to wait before playing this pulse when we are
# sweeping start."""
# next_: Set["QMPulse"] = field(default_factory=set)
# """Pulses that will be played after the current pulse.
#
# These pulses need to be re-aligned if we are sweeping the start or
# duration.
# """
#
# elements_to_align: Set[str] = field(default_factory=set)
# self.elements_to_align.add(self.element)


def create_acquisition(
    instruction: Measure, qubit: Qubit, options: ExecutionParameters
):
    """Create container for the variables used for saving acquisition in the
    QUA program.

    Args:
        instruction ...
        qubit (dict): :class:`qibolab.qubits.Qubit` object that is measured.
        options (:class:`qibolab.execution_parameters.ExecutionParameters`): Execution
            options containing acquisition type and averaging mode.

    Returns:
        :class:`qibolab.instruments.qm.acquisition.Acquisition` object containing acquisition variables.
    """
    average = options.averaging_mode is AveragingMode.CYCLIC
    kwargs = {}
    if options.acquisition_type is AcquisitionType.DISCRIMINATION:
        kwargs = {"threshold": qubit.threshold, "angle": qubit.iq_angle}
    acquisition = ACQUISITION_TYPES[options.acquisition_type](
        instruction.operation, qubit.name, average, **kwargs
    )
    return acquisition


InstructionType = Union[Align, Wait, Play]


@dataclass
class Instructions:
    config: QMConfig
    to_bake: Set[Pulse]

    instructions: List[InstructionType] = field(default_factory=list)
    """List of instructions to deploy to the instruments."""
    acquisitions: Dict[Measure, Acquisition] = field(default_factory=dict)
    """Map from measurement instructions to acquisition objects."""
    kwargs: Dict[InstructionType, dict] = field(
        default_factory=lambda: defaultdict(dict)
    )

    pulse_to_instruction: Dict[str, InstructionType] = field(default_factory=dict)
    """Map from qibolab pulses to instructions (useful for measurements and
    when sweeping)."""

    clock: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    """Dictionary used to keep track of times of each element, in order to
    calculate wait times."""
    pulse_finish: Dict[int, List[InstructionType]] = field(
        default_factory=lambda: defaultdict(list)
    )
    """Map to find all pulses that finish at a given time (useful for
    ``_find_previous``)."""

    # def _find_previous(self, pulse):
    #    for finish in reversed(sorted(self.pulse_finish.keys())):
    #        if finish <= pulse.start:
    #            # first try to find a previous pulse targeting the same qubit
    #            last_pulses = self.pulse_finish[finish]
    #            for previous in reversed(last_pulses):
    #                if previous.pulse.qubit == pulse.qubit:
    #                    return previous
    #            # otherwise
    #            if finish == pulse.start:
    #                return last_pulses[-1]
    #    return None

    def append(self, qubit, pulse, options):
        if (
            pulse.duration % 4 != 0
            or pulse.duration < 16
            or pulse.serial in self.to_bake
        ):
            instruction = Bake.from_pulse(pulse, self.config, [pulse.duration])
        else:
            if pulse.type is PulseType.READOUT:
                instruction = Measure.from_pulse(pulse)
                if instruction not in self.acquisitions:
                    self.acquisitions[instruction] = create_acquisition(
                        instruction, qubit, options
                    )
                    self.update_kwargs(
                        instruction, acquisition=self.acquisitions[instruction]
                    )
                self.acquisitions[instruction].keys.append(pulse.serial)
            else:
                instruction = Play.from_pulse(pulse)

            operation = instruction.operation
            if operation not in self.config.pulses:
                self.config.register_pulse(pulse, operation, instruction.element, qubit)

        self.pulse_to_instruction[pulse.serial] = instruction

        # previous = self._find_previous(pulse)
        # if previous is not None:
        #    previous.next_.add(qmpulse)

        element = instruction.element
        wait_time = pulse.start - self.clock[instruction.element]
        if wait_time >= 12:
            delay = Wait((element,), duration=wait_time // 4 + 1)
            self.instructions.append(delay)
            self.clock[element] += 4 * delay.duration
        self.clock[element] += instruction.duration

        self.pulse_finish[pulse.finish].append(instruction)
        self.instructions.append(instruction)

    def update_kwargs(self, instruction, **kwargs):
        self.kwargs[instruction].update(kwargs)

    # def shift(self):
    #    """Shift all pulses that come after a ``BakedPulse`` a bit to avoid
    #    overlapping pulses."""
    #    to_shift = collections.deque()
    #    for qmpulse in self.qmpulses:
    #        if isinstance(qmpulse, BakedPulse):
    #            to_shift.extend(qmpulse.next_)
    #    while to_shift:
    #        qmpulse = to_shift.popleft()
    #        qmpulse.wait_time += 2
    #        to_shift.extend(qmpulse.next_)

    def __iter__(self):
        return self.instructions

    def play(self, relaxation_time=0):
        """Part of QUA program that plays an arbitrary pulse sequence.

        Should be used inside a ``program()`` context.
        """
        qua.align()
        for instruction in self.instructions:
            kwargs = self.kwargs[instruction]
            instruction(**kwargs)
            # if len(qmpulse.elements_to_align) > 1:
            #     qua.align(*qmpulse.elements_to_align)

        if relaxation_time > 0:
            qua.wait(relaxation_time // 4)


class UniqueInstructions(dict):

    def __getitem__(self, key):
        if key not in self:
            self[key] = len(self)
        return super().__getitem__(key)
