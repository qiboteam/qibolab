"""Testing Qutip emulator engine helpers."""

import json
import re
from pathlib import Path

import pytest

from qibolab._core.instruments.emulator.engine.qutip import (
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
    assert re.fullmatch(r"run-\d{8}T\d{6}\d{6}Z", first.name)
    assert re.fullmatch(r"run-\d{8}T\d{6}\d{6}Z", second.name)
    assert not list(tmp_path.glob(f"{HAMILTONIAN_FILENAME}*"))
    assert not list(tmp_path.glob(f"{STATE_FILENAME}*"))


def test_dump_results_writes_fixed_qutip_files_and_loads_run(
    qutip_engine: QutipEngine, tmp_path: Path
):
    run_dir = qutip_engine.dump_results(
        {"hamiltonian": "saved"}, {"states": ["saved"]}, tmp_path
    )

    assert not (run_dir / "manifest.json").exists()
    assert (run_dir / f"{HAMILTONIAN_FILENAME}.qu").is_file()
    assert (run_dir / f"{STATE_FILENAME}.qu").is_file()

    dump = qutip_engine.load_results(run_dir)

    assert dump.path == run_dir
    assert dump.hamiltonian == {"hamiltonian": "saved"}
    assert dump.states == {"states": ["saved"]}
    assert not hasattr(dump, "manifest")


def test_load_results_uses_latest_run_from_dump_root(
    qutip_engine: QutipEngine, tmp_path: Path
):
    qutip_engine.dump_results({"hamiltonian": 1}, {"states": [1]}, tmp_path)
    latest = qutip_engine.dump_results({"hamiltonian": 2}, {"states": [2]}, tmp_path)

    dump = qutip_engine.load_results(tmp_path)

    assert dump.path == latest
    assert dump.hamiltonian == {"hamiltonian": 2}
    assert dump.states == {"states": [2]}
