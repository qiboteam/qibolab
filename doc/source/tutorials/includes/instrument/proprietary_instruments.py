"""Dummy class to provide a device driver for example in instrument0.py."""

from enum import Enum


class State(Enum):
    OFF = 0
    ON = 1


class BiaserDriver:

    def __init__(self, address):
        self.address = address
        self.bias = 0
        self.state = State.OFF

    def is_connected(self):
        return True

    def set_range(self, min_value=0, max_value=65536):
        self.min_value = min_value
        self.max_value = max_value

    def on(self, bias=0):
        self.bias = bias
        self.state = State.ON

    def off(self, bias=0):
        self.bias = bias
        self.state = State.OFF


class ControllerDriver:

    def __init__(self, address):
        self.address = address
        self.bias = 0
        self.state = State.OFF

    def is_connected(self):
        return True

    def set_range(self, min_value=0, max_value=65536):
        self.min_value = min_value
        self.max_value = max_value

    def on(self, bias=0):
        self.bias = bias
        self.state = State.ON

    def off(self, bias=0):
        self.bias = bias
        self.state = State.OFF
