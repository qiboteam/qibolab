""" Qblox Cluster QCM driver."""

from qibolab.instruments.abstract import Instrument
from qibolab.instruments.qblox.port import QbloxInputPort, QbloxOutputPort


class ClusterModule(Instrument):
    DEFAULT_SEQUENCERS = {}

    def __init__(self, name: str, address: str):
        """ """
        super().__init__(name, address)
        self.ports: dict = {}

    def port(self, name: str, out: bool = True):
        def count(cls):
            return len(list(filter(lambda x: isinstance(x, cls), self.ports)))

        if hasattr(self, "acquire"):
            print(type(self))
            if out:
                self.ports[name] = QbloxOutputPort(
                    self, self.DEFAULT_SEQUENCERS["o1"], port_number=count(QbloxOutputPort), port_name=name
                )
            else:
                self.ports[name] = QbloxInputPort(
                    self,
                    output_sequencer_number=self.DEFAULT_SEQUENCERS["o1"],
                    input_sequencer_number=self.DEFAULT_SEQUENCERS["i1"],
                    port_number=count(QbloxOutputPort),
                    port_name=name,
                )
        else:
            print(type(self))
            self.ports[name] = QbloxOutputPort(
                self, self.DEFAULT_SEQUENCERS[name], port_number=len(self.ports), port_name=name
            )
        return self.ports[name]
