"""Testing Qutip emulator engine helpers."""

import json
from pathlib import Path

import pytest

from qibolab._core.instruments.emulator.engine.qutip import (
    DUMP_MANIFEST_FILENAME,
    HAMILTONIAN_FILENAME,
    STATE_FILENAME,
    QutipEngine,
)


class FakeQutip:
    """Small qsave/qload stand-in used to test dump path handling."""

    def qsave(self, obj: dict, path: str) -> None:
        Path(path).with_suffix(".qu").write_text(json.dumps(obj))

    def qload(self, path: str) -> dict:
        return json.loads(Path(path).with_suffix(".qu").read_text())


@pytest.fixture
def qutip_engine() -> QutipEngine:
    engine = QutipEngine()
    engine.__dict__["engine"] = FakeQutip()
    return engine


def test_dump_results_creates_separate_run_directories(
    qutip_engine: QutipEngine, tmp_path: Path
):
    first = qutip_engine.dump_results({"hamiltonian": 1}, {"states": [1]}, tmp_path)
    second = qutip_engine.dump_results({"hamiltonian": 2}, {"states": [2]}, tmp_path)

    assert first != second
    assert first.parent == tmp_path
    assert second.parent == tmp_path
    assert first.name.startswith("run-")
    assert second.name.startswith("run-")
    assert not list(tmp_path.glob(f"{HAMILTONIAN_FILENAME}*"))
    assert not list(tmp_path.glob(f"{STATE_FILENAME}*"))


def test_dump_results_writes_manifest_and_loads_run(
    qutip_engine: QutipEngine, tmp_path: Path
):
    run_dir = qutip_engine.dump_results(
        {"hamiltonian": "saved"}, {"states": ["saved"]}, tmp_path
    )

    manifest = json.loads((run_dir / DUMP_MANIFEST_FILENAME).read_text())

    assert manifest["version"] == 1
    assert manifest["engine"] == "qutip"
    assert manifest["files"] == {
        "hamiltonian": f"{HAMILTONIAN_FILENAME}.qu",
        "states": f"{STATE_FILENAME}.qu",
    }

    dump = qutip_engine.load_results(run_dir)

    assert dump.path == run_dir
    assert dump.hamiltonian == {"hamiltonian": "saved"}
    assert dump.states == {"states": ["saved"]}
    assert dump.manifest == manifest


def test_load_results_uses_latest_run_from_dump_root(
    qutip_engine: QutipEngine, tmp_path: Path
):
    qutip_engine.dump_results({"hamiltonian": 1}, {"states": [1]}, tmp_path)
    latest = qutip_engine.dump_results({"hamiltonian": 2}, {"states": [2]}, tmp_path)

    dump = qutip_engine.load_results(tmp_path)

    assert dump.path == latest
    assert dump.hamiltonian == {"hamiltonian": 2}
    assert dump.states == {"states": [2]}
