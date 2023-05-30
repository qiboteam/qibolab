import pathlib

from qibolab.instruments.qblox.controller import QbloxController

RUNCARD = pathlib.Path(__file__).parent / "qblox.yml"


def create(runcard=RUNCARD):
    """Dummy platform using qblox cluster.

    Used in ``test_instruments_qblox.py``
    """
    return QbloxController("qblox", runcard)
