from uuid import uuid4

import pytest

from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.instruments.qblox import batching, cluster
from qibolab._core.instruments.qblox.sequence.acquisition import (
    Acquisition as Q1Acquisition,
)
from qibolab._core.instruments.qblox.sequence.sequence import Q1Sequence
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


def test_raw_acquisition_sequences_are_not_batched(monkeypatch):
    monkeypatch.setattr(
        cluster,
        "_add_time_of_flight",
        lambda sequence, _configs: sequence,
    )
    monkeypatch.setattr(
        cluster,
        "batch_sequences_by_cluster_memory_limits",
        lambda *_args, **_kwargs: pytest.fail("RAW sequences must not be batched"),
    )

    sequences = [_sequence(), _sequence()]
    unbatched = cluster._batch_sequences(
        sequences,
        sweepers=[],
        options=ExecutionParameters(
            acquisition_type=AcquisitionType.RAW,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        qcm_channels={"0/drive"},
        qrm_channels={"0/acquisition"},
        configs={},
    )

    assert len(unbatched) == len(sequences)


def test_raw_acquisition_allows_at_most_one_pulse_per_sequencer():
    acquisition = Q1Acquisition(num_bins=1, index=0)
    one_acquisition = Q1Sequence.empty().model_copy(
        update={"acquisitions": {uuid4(): acquisition}}
    )
    two_acquisitions = Q1Sequence.empty().model_copy(
        update={"acquisitions": {uuid4(): acquisition, uuid4(): acquisition}}
    )

    cluster._validate_raw_acquisitions({"0/acquisition": one_acquisition})
    # The limit applies independently to each sequencer, not across the module.
    cluster._validate_raw_acquisitions(
        {
            "0/acquisition": one_acquisition,
            "1/acquisition": one_acquisition,
        }
    )
    with pytest.raises(ValueError, match="at most one acquisition pulse per sequencer"):
        cluster._validate_raw_acquisitions({"0/acquisition": two_acquisitions})


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
