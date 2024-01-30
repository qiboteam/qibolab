from qibolab.channel import NamedChannel
from qibolab.channel_config import IQChannelConfig
from qibolab.instruments.abstract_channels import IQChannel

"""The glue part of the platform shall manually define all channels according to wiring,
then for each user request appropriate channels will be collected for handling the execution.
"""


class ZIIQChannel(IQChannel, NamedChannel):
    """IQChannel using a from Zurich instruments."""

    config: IQChannelConfig

    # FIXME: add all the necessary stuff needed to define a ZI IQ channel

    def foo(self, args, kwargs): ...

    def bar(self, args): ...


# TODO: Similarly, other ZI channels can be implemented
