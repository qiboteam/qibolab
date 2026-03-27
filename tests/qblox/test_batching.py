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
        drive_channels={"0/drive"},
        acquisition_channels={"0/acquisition"},
    )

    assert len(merged_sequences) == 2
    assert len(merged_sequences[0].acquisitions) == 2
    assert len(merged_sequences[1].acquisitions) == 1
