import pathlib

from qibolab.platforms.multiqubit import MultiqubitPlatform

RUNCARD = pathlib.Path(__file__).parent / "qblox.yml"


def create(runcard=RUNCARD):
    """Dummy platform using qblox cluster.

    Used in ``test_instruments_qblox.py``
    """
    return MultiqubitPlatform("qblox", runcard)
