class Sweeper:
    def __init__(self, parameter, values, pulses=None, qubits=None, wait_time=0):
        self.parameter = parameter
        self.values = values
        self.wait_time = wait_time

        self.pulses = pulses
        self.qubits = qubits

        self.pulse_type = None
        if pulses is not None:
            self.pulse_type = pulses[0].type.name.lower()
            for pulse in pulses:
                assert pulse.type.name.lower() == self.pulse_type
