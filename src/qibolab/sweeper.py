# -*- coding: utf-8 -*-
import numpy as np


class Sweeper:
    def __init__(self, parameter, start, stop, count, around, pulses=None, qubits=None, wait_time=0):
        self.parameter = parameter
        self.start = start
        self.stop = stop
        self.around = around
        self.count = count
        self.wait_time = wait_time

        self.pulses = pulses
        self.qubits = qubits

        self.values = np.linspace(start, stop, count) + around

        self.pulse_type = None
        if pulses is not None:
            self.pulse_type = pulses[0].type.name.lower()
            for pulse in pulses:
                assert pulse.type.name.lower() == self.pulse_type
