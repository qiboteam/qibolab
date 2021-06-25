from qiboicarusq.experiments.icarusq import IcarusQ
from qiboicarusq.experiments.awg import AWGSystem
from qibo.config import raise_error


class Experiment:

    available_experiments = {"icarusq": IcarusQ, "awg": AWGSystem}

    def __init__(self):
        # construct `IcarusQ` as the default experiment
        self.constructed_experiments = {"icarusq": IcarusQ()}
        self._active_experiment = self.constructed_experiments.get("icarusq")

    @property
    def active_experiment(self):
        return self._active_experiment

    @active_experiment.setter
    def active_experiment(self, name):
        if name in self.constructed_experiments:
            self._active_experiment = self.constructed_experiments.get(name)
        else:
            if name in self.available_experiments:
                new_experiment = self.available_experiments.get(name)()
                self.constructed_experiments[name] = new_experiment
                self._active_experiment = new_experiment
            else:
                raise_error(KeyError, "Unknown experiment {}.".format(name))

    def __getattr__(self, x):
        return getattr(self.active_experiment, x)
