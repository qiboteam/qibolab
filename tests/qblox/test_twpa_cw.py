"""Tests for TWPA pump support in the Qblox driver (CW and swept modes)."""

import numpy as np

from qibolab._core.components import IqChannel, OscillatorConfig
from qibolab._core.execution_parameters import AcquisitionType, ExecutionParameters
from qibolab._core.instruments.qblox.cluster import Cluster
from qibolab._core.instruments.qblox.config.port import PortAddress
from qibolab._core.instruments.qblox.config.sequencer import SequencerConfig
from qibolab._core.instruments.qblox.q1asm.ast_ import (
    Loop,
    SetAwgOffs,
    SetFreq,
    Stop,
    UpdParam,
    Wait,
    WaitSync,
)
from qibolab._core.instruments.qblox.sequence import Q1Sequence, compile
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import Parameter, Sweeper

# ---------------------------------------------------------------------------
# `Q1Sequence` CW and Swept TWPA
# ---------------------------------------------------------------------------


def test_q1sequence_cw():
    seq = Q1Sequence.cw()
    assert seq.is_cw is True
    assert len(seq.waveforms) == 1
    assert len(seq.acquisitions) == 0
    assert len(seq.program.elements) == 1
    assert isinstance(seq.program.elements[0].instruction, Stop)


def test_q1sequence_from_twpa_frequency_swept():
    options = ExecutionParameters(nshots=100, relaxation_time=50_000)
    sweeper = Sweeper(
        parameter=Parameter.frequency,
        values=np.array([6.0e9, 6.1e9, 6.2e9]),
        channels=["twpa_ch"],
    )
    seq = Q1Sequence.from_twpa(
        options=options,
        sweepers=[[sweeper]],
        sampling_rate=1.0,
        channel="twpa_ch",
        duration=1000.0,
    )

    assert seq.is_cw is False
    assert len(seq.waveforms) == 0
    assert len(seq.acquisitions) == 0

    instructions = [line.instruction for line in seq.program.elements]
    assert any(isinstance(ins, UpdParam) for ins in instructions)
    assert any(isinstance(ins, WaitSync) for ins in instructions)
    assert any(isinstance(ins, Wait) and ins.duration == 1000 for ins in instructions)
    # Check no relaxation wait (duration 50_000) is present
    assert not any(
        isinstance(ins, Wait) and ins.duration == 50_000 for ins in instructions
    )
    # Check loop logic and SetFreq are present
    assert any(isinstance(ins, SetFreq) for ins in instructions)
    assert any(isinstance(ins, Loop) for ins in instructions)


def test_q1sequence_from_twpa_offset_swept():
    options = ExecutionParameters(nshots=50, relaxation_time=100_000)
    sweeper = Sweeper(
        parameter=Parameter.offset,
        values=np.array([0.1, 0.2, 0.3]),
        channels=["twpa_ch"],
    )
    seq = Q1Sequence.from_twpa(
        options=options,
        sweepers=[[sweeper]],
        sampling_rate=1.0,
        channel="twpa_ch",
        duration=500.0,
    )

    assert seq.is_cw is False
    instructions = [line.instruction for line in seq.program.elements]
    assert any(isinstance(ins, UpdParam) for ins in instructions)
    assert any(isinstance(ins, WaitSync) for ins in instructions)
    assert any(isinstance(ins, Wait) and ins.duration == 500 for ins in instructions)
    assert not any(
        isinstance(ins, Wait) and ins.duration == 100_000 for ins in instructions
    )
    assert any(isinstance(ins, SetAwgOffs) for ins in instructions)
    assert any(isinstance(ins, Loop) for ins in instructions)


# ---------------------------------------------------------------------------
# `compile` with TWPA
# ---------------------------------------------------------------------------


def test_compile_twpa_cw():
    ps = PulseSequence()
    options = ExecutionParameters(nshots=10, relaxation_time=1000)
    seqs = compile(
        sequence=ps,
        sweepers=[],
        options=options,
        sampling_rate=1.0,
        merged_vzs=True,
        twpas={"twpa_ch": "twpa_pump"},
    )
    assert "twpa_ch" in seqs
    assert seqs["twpa_ch"].is_cw is True


def test_compile_twpa_swept():
    ps = PulseSequence()
    options = ExecutionParameters(nshots=10, relaxation_time=1000)
    sweeper = Sweeper(
        parameter=Parameter.frequency,
        values=np.array([6.0e9, 6.1e9]),
        channels=["twpa_ch"],
    )
    seqs = compile(
        sequence=ps,
        sweepers=[[sweeper]],
        options=options,
        sampling_rate=1.0,
        merged_vzs=True,
        twpas={"twpa_ch": "twpa_pump"},
    )
    assert "twpa_ch" in seqs
    assert seqs["twpa_ch"].is_cw is False


# ---------------------------------------------------------------------------
# `SequencerConfig.build` for TWPA
# ---------------------------------------------------------------------------


def test_sequencer_config_build_twpa_cw():
    address = PortAddress.from_path("8/o1")
    channels = {"twpa_ch": IqChannel(path="8/o1", lo="lo1")}
    configs = {
        "twpa_pump": OscillatorConfig(frequency=6.5e9, power=10.0),
        "lo1": OscillatorConfig(frequency=5.0e9, power=10.0),
    }

    seq = Q1Sequence.cw()
    cfg = SequencerConfig.build(
        address=address,
        channel_id="twpa_ch",
        channels=channels,
        configs=configs,
        acquisition=AcquisitionType.INTEGRATION,
        rf=True,
        sequence=seq,
        twpa="twpa_pump",
    )

    assert cfg.sync_en is False
    assert cfg.cont_mode_en_awg_path0 is True
    assert cfg.cont_mode_en_awg_path1 is True
    assert cfg.offset_awg_path0 == 1.0
    assert cfg.gain_awg_path0 == 1.0
    assert cfg.gain_awg_path1 == 1.0
    assert cfg.nco_freq == 1_500_000_000


def test_sequencer_config_build_twpa_swept():
    address = PortAddress.from_path("8/o1")
    channels = {"twpa_ch": IqChannel(path="8/o1", lo="lo1")}
    configs = {
        "twpa_pump": OscillatorConfig(frequency=6.5e9, power=10.0),
        "lo1": OscillatorConfig(frequency=5.0e9, power=10.0),
    }

    options = ExecutionParameters(nshots=10, relaxation_time=1000)
    sweeper = Sweeper(
        parameter=Parameter.frequency,
        values=np.array([6.0e9, 6.1e9]),
        channels=["twpa_ch"],
    )
    seq = Q1Sequence.from_twpa(options, [[sweeper]], 1.0, "twpa_ch", 100.0)

    cfg = SequencerConfig.build(
        address=address,
        channel_id="twpa_ch",
        channels=channels,
        configs=configs,
        acquisition=AcquisitionType.INTEGRATION,
        rf=True,
        sequence=seq,
        twpa="twpa_pump",
    )

    assert cfg.sync_en is True
    assert cfg.cont_mode_en_awg_path0 is None
    assert cfg.offset_awg_path0 == 0.0
    assert cfg.gain_awg_path0 is None
    assert cfg.gain_awg_path1 is None
    assert cfg.nco_freq == 1_500_000_000


# ---------------------------------------------------------------------------
# `Cluster` attributes and resolution helpers
# ---------------------------------------------------------------------------


def test_cluster_twpa_fields_defaults():
    c = Cluster(address="addr", name="my_cluster")
    assert c.twpas == {}


def test_cluster_twpa_custom():
    c = Cluster(address="addr", name="my_cluster", twpas={"twpa_ch": "twpa_pump"})
    assert c.twpas == {"twpa_ch": "twpa_pump"}


def test_cluster_twpa_resolution():
    c = Cluster(
        address="addr",
        name="my_cluster",
        channels={
            "ch1": IqChannel(path="8/o1"),
            "ch2": IqChannel(path="7/o2"),
        },
        twpas={
            "ch1": "pump1",
            "7/o2": "pump2",
        },
    )
    assert c._twpa_channels == {"ch1", "ch2", "7/o2"}
    assert c._twpa_config("ch1", PortAddress.from_path("8/o1")) == "pump1"
    assert c._twpa_config("ch2", PortAddress.from_path("7/o2")) == "pump2"
    assert c._twpa_config("unknown", PortAddress.from_path("1/o1")) is None
