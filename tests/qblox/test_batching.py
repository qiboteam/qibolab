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
