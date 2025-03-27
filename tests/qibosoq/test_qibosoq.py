from pathlib import Path

import pytest
import qibosoq.client

from qibolab.execution_parameters import ExecutionParameters
from qibolab.platform import Platform, create_platform
from qibolab.pulses import PulseSequence


@pytest.fixture
def platform(monkeypatch):
    monkeypatch.setenv("QIBOLAB_PLATFORMS", Path(__file__).parent)
    return create_platform("1q")


def test_client_commands(platform: Platform, monkeypatch):
    commands = []

    def register(server_commands: dict, host: str, port: int):
        commands.append(server_commands)
        measurements = sum(
            1 for x in server_commands["sequence"] if x["type"] == "readout"
        )
        n = measurements * server_commands["cfg"]["reps"]
        return [[0] * n], [[0] * n]

    monkeypatch.setattr(qibosoq.client, "connect", register)

    seq = PulseSequence()
    rx = platform.create_RX_pulse(0)
    seq.add(rx)
    seq.add(platform.create_MZ_pulse(0, start=rx.finish))
    platform.execute_pulse_sequence(seq, ExecutionParameters(100))
