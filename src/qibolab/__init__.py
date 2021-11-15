__version__ = "0.0.1.dev0"


from qibolab.experiments import Experiment
experiment = Experiment()
from qibolab.schedulers import TaskScheduler
scheduler = TaskScheduler()

from qibolab import gates, circuit


def get_experiment():
    return experiment.name


def set_experiment(name):
    experiment.active_experiment = name
