"""Qblox-specific configuration types.

These extend the core component configurations with Qblox-specific
fields and are registered via :class:`qibolab.ConfigKinds` by the
platform definition (similarly to the QM driver's configs).
"""

from typing import Literal

from pydantic import Field

from qibolab._core.components.configs import Config

__all__ = ["QbloxClusterConfig"]


class QbloxClusterConfig(Config):
    """Configuration for a Qblox cluster.

    Keyed by the cluster's :attr:`qibolab.Cluster.name` in the platform
    :attr:`Parameters.configs`.

    Example (in ``parameters.json``)::

        {
            "configs": {
                "my_cluster": {
                    "kind": "qblox-cluster",
                    "twpa_sources": {
                        "o1": "twpa",
                        "o2": "twpa2"
                    }
                },
                "twpa": {
                    "kind": "oscillator",
                    "frequency": 6360000000.0,
                    "power": -10.0
                }
            }
        }

    .. note::

        The keys in :attr:`twpa_sources` are QCM-RF output port
        identifiers (e.g. ``"o1"``, ``"o2"``), **not** full path strings.
        The driver resolves the port to the appropriate module and
        sequencer automatically.
    """

    kind: Literal["qblox-cluster"] = "qblox-cluster"

    twpa_sources: dict[str, str] = Field(default_factory=dict)
    """Mapping of QCM-RF output port identifiers to TWPA oscillator config names.

    Each key is a port identifier (e.g. ``"o1"``) on a QCM-RF module that
    will generate a continuous-wave pump tone.  The value is the name of
    an :class:`OscillatorConfig` entry in :attr:`Parameters.configs`.

    The driver configures a dedicated sequencer on the module in
    continuous-waveform mode, independent of any Q1ASM sequence.

    If empty, no TWPA pumps are activated.
    """

    turn_off_on_disconnect: bool = True
    """Whether to stop the TWPA continuous-wave pumps on disconnect.

    If ``False``, the CW pumps continue playing after :meth:`Cluster.disconnect`,
    mirroring :attr:`qibolab.LocalOscillator.turn_off_on_disconnect` for
    external oscillator instruments.
    """
