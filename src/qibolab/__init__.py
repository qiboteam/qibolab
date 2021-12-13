__version__ = "0.0.1.dev0"


from qibolab.platforms import TIIq
platform = TIIq() # TODO: Implement a platform selector

# TODO: Remove ``experiment`` when it is merged with ``platform``
from qibolab.experiments import Experiment
experiment = Experiment()
from qibolab.schedulers import TaskScheduler
scheduler = TaskScheduler()


def get_experiment():
    return experiment.name


def set_experiment(name):
    experiment.active_experiment = name
