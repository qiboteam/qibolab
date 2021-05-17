__version__ = '0.0.1-dev'

from qiboicarusq.experiments import Experiment
experiment = Experiment()
from qiboicarusq.schedulers import TaskScheduler
scheduler = TaskScheduler()


def get_experiment():
    return experiment.name


def set_experiment(name):
    experiment.active_experiment = name
