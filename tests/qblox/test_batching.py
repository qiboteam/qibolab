import pytest
from pydantic import ValidationError

from qibolab._core.execution_parameters import AveragingMode, ExecutionParameters
from qibolab._core.identifier import ChannelId
from qibolab._core.instruments.qblox import batching
from qibolab._core.instruments.qblox.q1asm.ast_ import Register
from qibolab._core.instruments.qblox.sequence.sequence import compile
from qibolab._core.pulses import (
    Acquisition,
    Delay,
    Gaussian,
    Pulse,
    PulseLike,
    Readout,
    Rectangular,
    VirtualZ,
)
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import (
    ParallelSweepers,
    Parameter,
    Sweeper,
    normalize_sweepers,
)


def _sequence() -> PulseSequence:
    return PulseSequence(
        [
            ("0/drive", Delay(duration=4)),
            ("0/acquisition", Acquisition(duration=8)),
        ]
    )


def test_batch_sequences_by_cluster_memory_limits(monkeypatch):
    """Sequences are merged into batches respecting QCM line limits."""
    # the starting number of lines is 50 and each sequence adds 1.6 lines so with
    # the qcm_lines limit of 54, only 2 sequences can be merged together.
    monkeypatch.setattr(batching, "per_shot_memory", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(
        batching,
        "cluster_memory_limits",
        {
            "acq_memory": 1000,
            "acq_number": 1000,
            "qcm_lines": 54,
            "qrm_lines": 1000,
        },
    )

    merged_sequences = batching.batch_sequences_by_cluster_memory_limits(
        sequences=[_sequence(), _sequence(), _sequence()],
        sweepers=[],
        options=ExecutionParameters(relaxation_time=100),
        qcm_channels={"0/drive"},
        qrm_channels={"0/acquisition"},
    )

    assert len(merged_sequences) == 2
    assert len(merged_sequences[0].acquisitions) == 2
    assert len(merged_sequences[1].acquisitions) == 1


def test_batch_sequences_by_cluster_memory_limits_oversize_sequence_error_raise(
    monkeypatch,
):
    """Raises ValueError when a single sequence exceeds cluster memory limits."""
    # with the acq_memory limit of 5 and each sequence having a memory of 6, the
    # individual sequence already exceeds the cluster memory limit, so a ValueError
    # should be raised.
    monkeypatch.setattr(batching, "per_shot_memory", lambda *_args, **_kwargs: 6)
    monkeypatch.setattr(
        batching,
        "cluster_memory_limits",
        {
            "acq_memory": 5,
            "acq_number": 1000,
            "qcm_lines": 1000,
            "qrm_lines": 1000,
        },
    )

    with pytest.raises(
        ValueError,
        match="An individual sequence exceeds Qblox cluster memory limits",
    ):
        batching.batch_sequences_by_cluster_memory_limits(
            sequences=[_sequence()],
            sweepers=[],
            options=ExecutionParameters(relaxation_time=100),
            qcm_channels={"0/drive"},
            qrm_channels={"0/acquisition"},
        )


def test_batch_sequences_by_register_limit():
    """Swept pulses are batched to fit within the sequencer register limit."""
    pulses: list[PulseLike] = [
        Pulse(duration=20, amplitude=0.1, envelope=Rectangular()) for _ in range(30)
    ]
    sequences = [PulseSequence([("0/drive", p)]) for p in pulses]
    sweeper = Sweeper(parameter=Parameter.amplitude, range=(0, 1, 0.1), pulses=pulses)

    batches = batching._split_sequences_by_memory_limits(
        sequences,
        [[sweeper]],
        ExecutionParameters(nshots=10, relaxation_time=100),
        qcm_channels={"0/drive"},
        qrm_channels=set(),
    )

    # 6 fixed registers + 1 loop counter leave 57 registers for the swept pulses
    assert sum(len(b) for b in batches) == 30
    assert [len(b) for b in batches] == [30]

    many = sequences * 3
    batches = batching._split_sequences_by_memory_limits(
        many,
        [[sweeper]],
        ExecutionParameters(nshots=10, relaxation_time=100),
        qcm_channels={"0/drive"},
        qrm_channels=set(),
    )
    assert [len(b) for b in batches] == [90]


def test_batch_registers_split_distinct_swept_pulses():
    """Distinct swept pulses each consume a register; batches split at the limit."""
    pulses: list[PulseLike] = [
        Pulse(duration=20, amplitude=0.1, envelope=Rectangular()) for _ in range(90)
    ]
    sequences = [PulseSequence([("0/drive", p)]) for p in pulses]
    sweeper = Sweeper(parameter=Parameter.amplitude, range=(0, 1, 0.1), pulses=pulses)

    batches = batching._split_sequences_by_memory_limits(
        sequences * 2,
        [[sweeper]],
        ExecutionParameters(nshots=10, relaxation_time=100),
        qcm_channels={"0/drive"},
        qrm_channels=set(),
    )

    # Splitting resets the pulse ID union; each batch can hold 57 distinct IDs.
    assert [len(b) for b in batches] == [57, 57, 57, 9]


@pytest.mark.parametrize("parameter", [Parameter.amplitude, Parameter.duration])
@pytest.mark.parametrize("envelope", [Rectangular(), Gaussian(rel_sigma=5)])
def test_batch_registers_deduplicate_repeated_pulse(parameter, envelope):
    """Repeated identical pulses share a single register across sequences."""
    channel = "0/drive"
    pulse = Pulse(duration=20, amplitude=0.1, envelope=envelope)
    sequence = PulseSequence([(channel, pulse)] * 58)
    sweeper = Sweeper(
        parameter=parameter,
        range=(20, 40, 1) if parameter is Parameter.duration else (0, 1, 0.1),
        pulses=[pulse],
    )

    batches = batching._split_sequences_by_memory_limits(
        [sequence, sequence],
        [[sweeper]],
        ExecutionParameters(nshots=10, relaxation_time=100),
        qcm_channels={channel},
        qrm_channels=set(),
    )

    assert [len(b) for b in batches] == [2]


def test_batch_registers_deduplicate_multiple_parameters():
    """Gaussian pulses under a duration sweep each need 3 registers."""
    pulses: list[PulseLike] = [
        Pulse(duration=20, amplitude=0.1, envelope=Gaussian(rel_sigma=5))
        for _ in range(20)
    ]
    sequences = [PulseSequence([("0/drive", p)] * 2) for p in pulses]
    sweepers = normalize_sweepers(
        [[Sweeper(parameter=Parameter.duration, range=(20, 40, 1), pulses=pulses)]]
    )

    batches = batching._split_sequences_by_memory_limits(
        sequences * 2,
        sweepers,
        ExecutionParameters(nshots=10, relaxation_time=100),
        qcm_channels={"0/drive"},
        qrm_channels=set(),
    )

    # Each unique Gaussian pulse ID needs three duration-sweep registers.
    assert [len(b) for b in batches] == [19, 19, 2]


def test_batch_registers_are_counted_per_channel():
    """Pulse register budget is per sequencer: 100 pulses split across 2 channels fit
    because each channel gets its own 57-register allowance."""
    channels = ["0/drive", "1/drive"]
    pulses: list[PulseLike] = [
        Pulse(duration=20, amplitude=0.1, envelope=Rectangular()) for _ in range(100)
    ]
    sequences = [PulseSequence([(channels[i % 2], p)]) for i, p in enumerate(pulses)]
    sweeper = Sweeper(parameter=Parameter.amplitude, range=(0, 1, 0.1), pulses=pulses)

    batches = batching._split_sequences_by_memory_limits(
        sequences,
        [[sweeper]],
        ExecutionParameters(nshots=10, relaxation_time=100),
        qcm_channels=set(channels),
        qrm_channels=set(),
    )

    # 7 fixed registers leave 57 per sequencer: 100 pulses fit once spread on 2 channels
    assert [len(b) for b in batches] == [100]


def _channel_sweep_setup(nchannels: int, npulses: int):
    """Build channels, rectangular pulses, and sweepers: one amplitude sweeper on all
    pulses plus one frequency sweeper per channel."""
    channels = [f"{i}/drive" for i in range(nchannels)]
    pulses: list[PulseLike] = [
        Pulse(duration=20, amplitude=0.1, envelope=Rectangular())
        for _ in range(npulses)
    ]
    sweepers = normalize_sweepers(
        [
            [
                Sweeper(
                    parameter=Parameter.amplitude, range=(0.1, 1, 0.1), pulses=pulses
                )
            ],
            [
                Sweeper(
                    parameter=Parameter.frequency,
                    range=(1e6, 2e6, 1e5),
                    channels=[c],
                )
                for c in channels
            ],
        ]
    )
    return channels, pulses, sweepers


def test_batch_registers_ignore_other_channels_sweepers():
    """Sweeper loop registers are scoped to their own channel: frequency sweepers on
    24 other channels don't add loop overhead to 0/drive's budget."""
    channels, pulses, sweepers = _channel_sweep_setup(25, 40)
    options = ExecutionParameters(nshots=10, relaxation_time=100)

    # one sequence: 6 fixed + 2 loops + 1 own frequency + 40 amplitudes
    batches = batching._split_sequences_by_memory_limits(
        [PulseSequence([("0/drive", p) for p in pulses])],
        sweepers,
        options,
        qcm_channels=set(channels),
        qrm_channels=set(),
    )
    assert [len(b) for b in batches] == [1]

    # frequency sweeps on the other channels do not take registers on 0/drive
    batches = batching._split_sequences_by_memory_limits(
        [PulseSequence([("0/drive", p)]) for p in pulses],
        sweepers,
        options,
        qcm_channels=set(channels),
        qrm_channels=set(),
    )
    assert [len(b) for b in batches] == [40]


def test_batch_registers_count_own_channel_sweepers():
    """Sweepers on the same channel do consume its register budget."""
    channels, pulses, sweepers = _channel_sweep_setup(1, 60)

    # 8 fixed + 1 own frequency + 60 amplitudes exceeds the 64 registers
    with pytest.raises(ValueError, match="exceeds the Q1 sequencer registers"):
        batching._split_sequences_by_memory_limits(
            [PulseSequence([("0/drive", p) for p in pulses])],
            sweepers,
            ExecutionParameters(nshots=10, relaxation_time=100),
            qcm_channels=set(channels),
            qrm_channels=set(),
        )


@pytest.mark.parametrize("nparameters", [1, 57, 58])
def test_batch_registers_count_sweeper_only_channels(nparameters):
    """Channel-parameter sweepers on a channel with no pulses still count registers."""
    sequence = PulseSequence([("0/drive", Delay(duration=4))])
    sweepers = normalize_sweepers(
        [
            [
                Sweeper(
                    parameter=Parameter.frequency,
                    range=(1e6, 2e6, 1e5),
                    channels=["1/drive"],
                )
                for _ in range(nparameters)
            ]
        ]
    )
    options = ExecutionParameters(nshots=1, relaxation_time=100)

    # Six fixed registers and one loop leave 57 channel-parameter registers.
    if nparameters > 57:
        with pytest.raises(
            ValueError,
            match="channel 1/drive: requested=65, limit=64",
        ):
            batching._split_sequences_by_memory_limits(
                [sequence],
                sweepers,
                options,
                qcm_channels={"0/drive", "1/drive"},
                qrm_channels=set(),
            )
    else:
        batches = batching._split_sequences_by_memory_limits(
            [sequence, sequence],
            sweepers,
            options,
            qcm_channels={"0/drive", "1/drive"},
            qrm_channels=set(),
        )
        assert [len(batch) for batch in batches] == [2]


def test_batch_registers_duration_sweeper_mixed_envelopes():
    """Duration sweepers count registers per envelope type (Gaussian=3, Rectangular=1)."""
    pulses: list[PulseLike] = [
        Pulse(duration=20, amplitude=0.1, envelope=Rectangular()) for _ in range(20)
    ]
    pulses.append(Pulse(duration=20, amplitude=0.1, envelope=Gaussian(rel_sigma=5)))
    sweepers = normalize_sweepers(
        [[Sweeper(parameter=Parameter.duration, range=(20, 40, 1), pulses=pulses)]]
    )

    # only the Gaussian needs 3 registers, the rectangular pulses need 1 each
    counts = batching._pulse_sweep_register_counts(sweepers)
    assert sum(counts.values()) == 20 + 3

    batches = batching._split_sequences_by_memory_limits(
        [PulseSequence([("0/drive", p) for p in pulses])],
        sweepers,
        ExecutionParameters(nshots=3, relaxation_time=100),
        qcm_channels={"0/drive"},
        qrm_channels=set(),
    )
    assert [len(b) for b in batches] == [1]


def _assert_register_budget_matches_compiler(
    sequence: PulseSequence,
    sweepers: list[ParallelSweepers],
    options: ExecutionParameters,
) -> dict[ChannelId, int]:
    """Assert batching's predicted register counts match the compiler's allocation."""
    sequence = sequence.align_to_delays()
    pulse_counts = batching._pulse_sweep_register_counts(sweepers)
    channel_counts = batching._channel_sweep_register_counts(sweepers)
    pulse_ids = batching._swept_pulse_ids_per_channel(
        sequence, pulse_counts.keys(), channel_counts.keys()
    )
    predicted = batching._base_registers_per_channel(
        sequence,
        batching.Registers.first_available() + len(sweepers),
        channel_counts,
    )
    for channel, ids in pulse_ids.items():
        predicted[channel] += sum(pulse_counts[pid] for pid in ids)

    compiled = compile(sequence, sweepers, options, sampling_rate=1.0, merged_vzs=False)
    # Inspect the emitted operands rather than duplicating the compiler's allocation.
    actual = {
        channel: 1
        + max(
            operand.number
            for line in sequencer.program.lines
            for field in type(line.instruction).model_fields
            if isinstance(operand := getattr(line.instruction, field), Register)
        )
        for channel, sequencer in compiled.items()
    }
    assert predicted == actual
    return actual


@pytest.mark.parametrize("averaging_mode", list(AveragingMode))
@pytest.mark.parametrize("ndimensions", [0, 1, 2, 3])
def test_register_budget_matches_compiler(averaging_mode, ndimensions):
    """Catch drift between batching's register estimates and compiler allocation."""
    gaussian = Pulse(duration=20, amplitude=0.1, envelope=Gaussian(rel_sigma=5))
    rectangular = Pulse(duration=20, amplitude=0.1, envelope=Rectangular())
    short = Pulse(duration=4, amplitude=0.1, envelope=Rectangular())
    delay = Delay(duration=20)
    vz = VirtualZ(phase=0.1)
    sequence = PulseSequence(
        [
            ("0/drive", gaussian),
            ("0/drive", gaussian),
            ("0/drive", rectangular),
            ("0/drive", short),
            ("0/drive", vz),
            ("1/drive", gaussian),
            ("1/drive", delay),
            ("0/acquisition", Readout.from_probe(gaussian)),
        ]
    )
    sweepers = normalize_sweepers(
        [
            [
                Sweeper(
                    parameter=Parameter.duration,
                    range=(20, 28, 4),
                    pulses=[gaussian, rectangular, delay],
                ),
                Sweeper(
                    parameter=Parameter.duration,
                    range=(4, 12, 4),
                    pulses=[short],
                ),
            ],
            [
                Sweeper(
                    parameter=Parameter.amplitude,
                    range=(0.1, 0.3, 0.1),
                    pulses=[gaussian, rectangular],
                ),
                Sweeper(parameter=Parameter.phase, range=(0, 0.2, 0.1), pulses=[vz]),
            ],
            [
                Sweeper(
                    parameter=Parameter.frequency,
                    range=(1e6, 1.2e6, 1e5),
                    channels=[
                        "0/drive",
                        "1/drive",
                        "0/acquisition",
                        "0/probe",
                        "2/drive",
                    ],
                ),
                Sweeper(
                    parameter=Parameter.offset,
                    range=(0, 0.2, 0.1),
                    channels=["1/drive"],
                ),
            ],
        ][:ndimensions]
    )
    _assert_register_budget_matches_compiler(
        sequence,
        sweepers,
        ExecutionParameters(
            nshots=2, relaxation_time=100, averaging_mode=averaging_mode
        ),
    )


@pytest.mark.parametrize(
    ("parameter", "envelope", "capacity"),
    [
        (Parameter.amplitude, Rectangular(), 57),
        (Parameter.duration, Rectangular(), 57),
        (Parameter.duration, Gaussian(rel_sigma=5), 19),
    ],
)
def test_register_limited_batches_match_compiler(parameter, envelope, capacity):
    """Prevent unnecessary splitting or register overflow at the compiler's limit."""
    pulses: list[PulseLike] = [
        Pulse(duration=20, amplitude=0.1, envelope=envelope) for _ in range(60)
    ]
    channels = {"0/drive", "1/drive"}
    sequences = [
        PulseSequence([(ch, pulse) for ch in sorted(channels)] * 2) for pulse in pulses
    ]
    sweepers = normalize_sweepers(
        [
            [
                Sweeper(
                    parameter=parameter,
                    range=(20, 28, 4)
                    if parameter is Parameter.duration
                    else (0.1, 0.3, 0.1),
                    pulses=pulses,
                )
            ]
        ]
    )
    options = ExecutionParameters(nshots=2, relaxation_time=100)
    batches = batching.batch_sequences_by_cluster_memory_limits(
        sequences, sweepers, options, qcm_channels=channels, qrm_channels=set()
    )

    assert len(batches) == (len(sequences) + capacity - 1) // capacity
    budgets = [
        _assert_register_budget_matches_compiler(batch, sweepers, options)
        for batch in batches
    ]
    assert budgets[0] == {ch: 64 for ch in channels}

    oversized = PulseSequence(
        [(ch, pulse) for pulse in pulses[: capacity + 1] for ch in sorted(channels)]
    )
    with pytest.raises(ValueError, match="exceeds the Q1 sequencer registers"):
        batching.batch_sequences_by_cluster_memory_limits(
            [oversized], sweepers, options, qcm_channels=channels, qrm_channels=set()
        )
    with pytest.raises(ValidationError) as error:
        compile(oversized, sweepers, options, sampling_rate=1.0, merged_vzs=False)
    assert error.value.errors()[0]["input"] == {"number": 64}
