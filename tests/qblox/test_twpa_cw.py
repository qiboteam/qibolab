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
        twpas={"twpa": ("8/o1", None)},
    )
    assert "twpa" in seqs
    assert seqs["twpa"].is_cw is True


def test_compile_twpa_swept():
    ps = PulseSequence()
    options = ExecutionParameters(nshots=10, relaxation_time=1000)
    sweeper = Sweeper(
        parameter=Parameter.frequency,
        values=np.array([6.0e9, 6.1e9]),
        channels=["twpa"],
    )
    seqs = compile(
        sequence=ps,
        sweepers=[[sweeper]],
        options=options,
        sampling_rate=1.0,
        merged_vzs=True,
        twpas={"twpa": ("8/o1", None)},
    )
    assert "twpa" in seqs
    assert seqs["twpa"].is_cw is False


# ---------------------------------------------------------------------------
# `SequencerConfig.build` for TWPA
# ---------------------------------------------------------------------------


def test_sequencer_config_build_twpa_cw():
    c = Cluster(
        address="addr",
        name="my_cluster",
        twpas={"twpa": ("8/o1", None)},
    )
    address = PortAddress.from_path("8/o1")
    configs = {
        "twpa": OscillatorConfig(frequency=6.5e9, power=10.0),
    }

    seq = Q1Sequence.cw()
    cfg = SequencerConfig.build(
        address=address,
        channel_id="twpa",
        channels=c.all_channels,
        configs=configs,
        acquisition=AcquisitionType.INTEGRATION,
        rf=True,
        sequence=seq,
    )

    assert cfg.sync_en is False
    assert cfg.cont_mode_en_awg_path0 is True
    assert cfg.cont_mode_en_awg_path1 is True
    assert cfg.offset_awg_path0 == 1.0
    assert cfg.gain_awg_path0 == 1.0
    assert cfg.gain_awg_path1 == 1.0
    assert cfg.nco_freq == 0
    assert cfg.mixer_corr_gain_ratio is None
    assert cfg.mixer_corr_phase_offset_degree is None


def test_sequencer_config_build_twpa_with_mixer():
    from qibolab._core.components import IqMixerConfig

    c = Cluster(
        address="addr",
        name="my_cluster",
        twpas={"twpa": ("8/o1", "mixer2")},
    )
    address = PortAddress.from_path("8/o1")
    configs = {
        "twpa": OscillatorConfig(frequency=6.5e9, power=10.0),
        "mixer2": IqMixerConfig(
            scale_q=1.05, phase_q=2.5, offset_i=0.01, offset_q=-0.02
        ),
    }

    seq = Q1Sequence.cw()
    cfg = SequencerConfig.build(
        address=address,
        channel_id="twpa",
        channels=c.all_channels,
        configs=configs,
        acquisition=AcquisitionType.INTEGRATION,
        rf=True,
        sequence=seq,
    )

    assert cfg.nco_freq == 0
    assert cfg.mixer_corr_gain_ratio == 1.05
    assert cfg.mixer_corr_phase_offset_degree == 2.5


def test_sequencer_config_build_twpa_swept():
    c = Cluster(
        address="addr",
        name="my_cluster",
        twpas={"twpa": ("8/o1", None)},
    )
    address = PortAddress.from_path("8/o1")
    configs = {
        "twpa": OscillatorConfig(frequency=6.5e9, power=10.0),
    }

    options = ExecutionParameters(nshots=10, relaxation_time=1000)
    sweeper = Sweeper(
        parameter=Parameter.frequency,
        values=np.array([6.0e9, 6.1e9]),
        channels=["twpa"],
    )
    seq = Q1Sequence.from_twpa(options, [[sweeper]], 1.0, "twpa", 100.0)

    cfg = SequencerConfig.build(
        address=address,
        channel_id="twpa",
        channels=c.all_channels,
        configs=configs,
        acquisition=AcquisitionType.INTEGRATION,
        rf=True,
        sequence=seq,
    )

    assert cfg.sync_en is True
    assert cfg.cont_mode_en_awg_path0 is None
    assert cfg.offset_awg_path0 == 0.0
    assert cfg.gain_awg_path0 is None
    assert cfg.gain_awg_path1 is None
    assert cfg.nco_freq == 0


# ---------------------------------------------------------------------------
# `Cluster` attributes and resolution helpers
# ---------------------------------------------------------------------------


def test_cluster_twpa_fields_defaults():
    c = Cluster(address="addr", name="my_cluster")
    assert c.twpas == {}


def test_cluster_twpa_custom():
    c = Cluster(
        address="addr",
        name="my_cluster",
        twpas={"twpa1": ("8/o1", None), "twpa2": ("7/o1", "mixer2")},
    )
    assert c.twpas == {"twpa1": ("8/o1", None), "twpa2": ("7/o1", "mixer2")}


def test_cluster_twpa_channels_by_module():
    c = Cluster(
        address="addr",
        name="my_cluster",
        channels={"drive": IqChannel(path="8/o2")},
        twpas={"twpa": ("8/o1", None), "twpa2": ("7/o1", "mixer2")},
    )
    assert "drive" in c.all_channels
    assert "twpa" in c.all_channels
    assert "twpa2" in c.all_channels
    assert c.all_channels["twpa"].path == "8/o1"
    assert c.all_channels["twpa"].lo == "twpa"
    assert c.all_channels["twpa"].mixer is None
    assert c.all_channels["twpa2"].path == "7/o1"
    assert c.all_channels["twpa2"].lo == "twpa2"
    assert c.all_channels["twpa2"].mixer == "mixer2"

    by_mod = c._channels_by_module
    assert 8 in by_mod
    assert 7 in by_mod
    assert [ch for ch, _ in by_mod[8]] == ["drive", "twpa"]
    assert [ch for ch, _ in by_mod[7]] == ["twpa2"]
    assert c._los["twpa"] == "twpa"
    assert c._los["twpa2"] == "twpa2"
    assert "twpa" not in c._mixers
    assert c._mixers["twpa2"] == "mixer2"


# ---------------------------------------------------------------------------
# `map_ports` and `Twpa` helpers
# ---------------------------------------------------------------------------


def test_map_ports_with_twpa():
    from qibolab._core.instruments.qblox.platform import Twpa, map_ports
    from qibolab._core.qubits import Qubit

    cluster = {
        "qcm_rf0": (
            8,
            {
                "o1": [Twpa(name="twpa")],
                "o2": ["q0"],
            },
        ),
        "qcm_rf1": (
            7,
            {
                1: [Twpa(name="twpa2", mixer="mixer2")],
                2: ["q1"],
            },
        ),
        "qrm_rf0": (
            6,
            {
                "io1": ["q0", "q1"],
            },
        ),
    }

    qubits = {
        "q0": Qubit(drive="0/drive", probe="0/probe", acquisition="0/acquisition"),
        "q1": Qubit(drive="1/drive", probe="1/probe", acquisition="1/acquisition"),
    }

    channels, twpas = map_ports(cluster, qubits)

    # Channels checks
    assert "0/drive" in channels
    assert channels["0/drive"].path == "8/o2"
    assert "1/drive" in channels
    assert channels["1/drive"].path == "7/o2"
    assert "0/probe" in channels
    assert channels["0/probe"].path == "6/o1"
    assert "0/acquisition" in channels
    assert channels["0/acquisition"].path == "6/i1"
    assert channels["0/acquisition"].probe == "0/probe"

    # Twpas checks
    assert twpas == {
        "twpa": ("8/o1", None),
        "twpa2": ("7/o1", "mixer2"),
    }


def test_infer_los_and_mixers_with_twpa():
    from qibolab._core.instruments.qblox.platform import Twpa, infer_los, infer_mixers

    cluster = {
        "qcm_rf0": (
            8,
            {
                "o1": [Twpa(name="twpa")],
                "o2": ["q0"],
            },
        ),
        "qrm_rf0": (
            6,
            {
                "io1": ["q0"],
            },
        ),
    }

    los = infer_los(cluster)
    mixers = infer_mixers(cluster)

    assert los == {
        ("q0", False): "qcm_rf0/o2/lo",
        ("q0", True): "qrm_rf0/o1/lo",
    }
    assert mixers == {
        ("q0", False): "qcm_rf0/o2/mixer",
        ("q0", True): "qrm_rf0/o1/mixer",
    }
