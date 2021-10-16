from qiboicarusq.experiments.icarusq import IcarusQ
from qiboicarusq.experiments.awg import AWGSystem
from qiboicarusq.experiments.awg10q import AWGSystem10Qubits
from qiboicarusq.experiments.qblox1q import qblox1q
from qibo.config import raise_error


class Experiment:

    available_experiments = {"qblox1q": qblox1q, "icarusq": IcarusQ, "awg": AWGSystem, "awg10q": AWGSystem10Qubits}

    def __init__(self):
        self.constructed_experiments = {"qblox1q": qblox1q()}
        self._active_experiment = self.constructed_experiments.get("qblox1q")

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
