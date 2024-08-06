"""Component (a.k.a. logical component) is a concept that is part of the
qibolab interface exposed to its users.

Interacting with, and controlling a quantum computer means orchestrating
its control electronics/equipment. For a qibolab user quantum computer
is just a collection of control instruments. A component represents a
piece of equipment, functionality, or configuration that can be
individually addressed, configured, and referred to in any relevant
context. It can represent a single device, part of a bigger device, or a
collection of multiple devices. Qibolab defines a handful of specific
categories of components, and each platform definition shall have only
such components, independent of what specific electronics is used, how
it is used, etc. One basic requirement for all components is that they
have unique names, and in any relevant context can be referred to by
their name.
"""

from .channels import *
from .configs import *
