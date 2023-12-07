""" Qblox Cluster QCM driver."""

from qibolab.instruments.abstract import Instrument
from qibolab.instruments.qblox.port import QbloxInputPort, QbloxOutputPort


class ClusterModule(Instrument):
    DEFAULT_SEQUENCERS = {}

    def __init__(self, name: str, address: str):
        """
        This class defines common features shared by all Qblox modules (QCM-BB, QCM-RF, QRM-RF).
        It serves as a foundational class, unifying the behavior of the three distinct modules.
        All module-specific classes are intended to inherit from this base class.
        """

        super().__init__(name, address)
        self.ports: dict = {}

    def port(self, name: str, out: bool = True):
        def count(cls):
            return len(list(filter(lambda x: isinstance(x, cls), self.ports.values())))

        if out:
            self.ports[name] = QbloxOutputPort(
                self,
                self.DEFAULT_SEQUENCERS[f"o{count(QbloxOutputPort)+1}"],
                port_number=count(QbloxOutputPort),
                port_name=name,
            )
        else:
            self.ports[name] = QbloxInputPort(
                self,
                port_number=0,
                port_name=name,
            )
        return self.ports[name]
