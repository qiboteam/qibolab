import pytest

from qibolab._core.execution_parameters import ExecutionParameters
from qibolab._core.instruments.qblox import batching
from qibolab._core.pulses import Acquisition, Delay
from qibolab._core.sequence import PulseSequence


def _sequence() -> PulseSequence:
    return PulseSequence(
        [
            ("0/drive", Delay(duration=4)),
            ("0/acquisition", Acquisition(duration=8)),
        ]
    )


def test_batch_sequences_by_cluster_memory_limits(monkeypatch):
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
    from qibolab._core.pulses import Pulse, Rectangular
    from qibolab._core.sweeper import Parameter, Sweeper

    pulses = [
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
    assert [len(b) for b in batches] == [57, 33]


def test_batch_registers_are_counted_per_channel():
    from qibolab._core.pulses import Pulse, Rectangular
    from qibolab._core.sweeper import Parameter, Sweeper

    channels = ["0/drive", "1/drive"]
    pulses = [
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
