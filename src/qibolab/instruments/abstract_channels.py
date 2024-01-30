from abc import ABC, abstractmethod

"""
In the instrument layer we shall try to do as many things as possible in a unified manner.
This file defines the requrements on various types of channels that instrument-specific implementations should satisfy.
E.g. there could be methods returning the memory limitation on a channel, so that a unified unrolling-batching algorithm
can be written that is not instrument-specific.

TODO: this needs proper design
"""


class IQChannel(ABC):

    @abstractmethod
    def foo(self, args, kwargs): ...

    @abstractmethod
    def bar(self, args): ...


class DCChannel(ABC):

    @abstractmethod
    def baz(self, kwargs): ...
