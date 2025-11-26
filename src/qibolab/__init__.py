from typing import Any

from . import _core, _version, instruments
from ._core import *
from ._version import *

__all__ = []
__all__ += _core.__all__
__all__ += _version.__all__
__all__ += ["instruments"]


def __getattr__(name: str) -> Any:
    """Redirect meta-backend lookup for Qibo decoupling.

    Since it should be possible to use Qibolab without Qibo installed, especially to
    increase its compatibility with other packages (possibly incompatible with Qibo
    itself), the backend is made optional.

    However, the meta-backend mechanism looks for a variable identifier ``MetaBackend``
    in the top-most scope of the package.
    In order to preserve compatibility with this protocol, the variable is dynamically
    imported, if needed.

    .. note::

        Usage of this import is discouraged, and the import from
        :class:`qibolab.backend.MetaBackend` should be favored, for direct usage.

        This dynamic import may be phased out at a later stage. If the meta-backend
        protocol will allow for some different redirection.
    """
    if name == "MetaBackend":
        from .backend import MetaBackend

        return MetaBackend

    return globals()[name]
