__version__ = "0.0.1.dev0"

from icarusq import gates


from icarusq.experiments import Experiment
experiment = Experiment()
from icarusq.schedulers import TaskScheduler
scheduler = TaskScheduler()


def get_experiment():
    return experiment.name


def set_experiment(name):
    experiment.active_experiment = name
