from qibolab._core.identifier import ChannelId
from qibolab._core.sequence import PulseSequence

__all__ = []


def assert_channels_exclusion(ps: PulseSequence, excluded: set[ChannelId]):
    """Deny group of channels.

    Mainly used to deny probe channels, since probe events are currently required to be
    bundled with acquisitions, cf. :class:`qibolab.Readout`.

    An extended discussion about the probe-acquisition merge is found at:
    https://github.com/qiboteam/qibolab/pull/1088#issuecomment-2637800857
    """
    assert not any(
        ch in excluded for ch, _ in ps
    ), "Probe channels can not be controlled independently, please use Readout events."
