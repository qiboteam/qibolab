__version__ = "0.0.1.dev0"


from qibo.config import log
from qibolab.platforms import TIIq
try:
    platform = TIIq() # TODO: Implement a platform selector
except AttributeError:
    platform = None
    log.warning("Cannot establish connection to TIIq instruments. Skipping...")

# TODO: Remove ``experiment`` when it is merged with ``platform``
from qibolab.experiments import Experiment
experiment = Experiment()
from qibolab.schedulers import TaskScheduler
scheduler = TaskScheduler()


def get_experiment():
    return experiment.name


def set_experiment(name):
    experiment.active_experiment = name
