class Sweeper:
    def __init__(self, parameter, values, pulse=None, wait_time=0):
        self.parameter = parameter
        self.values = values
        self.pulse = pulse
        self.wait_time = wait_time

        self.pulse_type = None
        if pulse is not None:
            self.pulse_type = pulse.type.name.lower()
