from qibo.config import raise_error


class Port:
    """Abstract interface for instrument parameters.

    These parameters are exposed to the user through :class:`qibolab.channels.Channel`.

    Drivers should subclass this interface and implement the getters
    and setters for all the parameters that are available for the
    corresponding instruments.
    """

    # NOTE: We can convert all these to ``@abstractmethod`` but then
    # each driver would have to implement all of them an raise errors
    # for parameters that are not supported.

    def __init__(self, name):
        self.name = name

    @property
    def offset(self):
        """DC offset that is applied to this port."""
        raise_error(NotImplementedError, "Instruments do not support offset.")

    @offset.setter
    def offset(self, value):
        raise_error(NotImplementedError, "Instruments do not support offset.")

    @property
    def lo_frequency(self):
        """Local oscillator frequency for the given port.

        Relevant only for controllers with internal local oscillators.
        """
        raise_error(NotImplementedError, "Instruments do not have internal local oscillators.")

    @lo_frequency.setter
    def lo_frequency(self, value):
        raise_error(NotImplementedError, "Instruments do not have internal local oscillators.")

    @property
    def lo_power(self):
        """Local oscillator power for the given port.

        Relevant only for controllers with internal local oscillators.
        """
        raise_error(NotImplementedError, "Instruments do not have internal local oscillators.")

    @lo_power.setter
    def lo_power(self, value):
        raise_error(NotImplementedError, "Instruments do not have internal local oscillators.")

    # TODO: Maybe gain, attenuation and power range can be unified to a single property
    @property
    def gain(self):
        """Gain that is applied to this port."""
        raise_error(NotImplementedError, "Instruments do not support gain.")

    @gain.setter
    def gain(self, value):
        raise_error(NotImplementedError, "Instruments do not support gain.")

    @property
    def attenuation(self):
        """Attenuation that is applied to this port."""
        raise_error(NotImplementedError, "Instruments do not support attenuation.")

    @attenuation.setter
    def attenuation(self, value):
        raise_error(NotImplementedError, "Instruments do not support attenuation.")

    @property
    def power_range(self):
        raise_error(NotImplementedError, "Instruments do not support power range.")

    @power_range.setter
    def power_range(self, value):
        raise_error(NotImplementedError, "Instruments do not support power range.")

    @property
    def filters(self):
        """Filters to be applied to the channel to reduce the distortions when sending flux pulses.

        Useful for two-qubit gates.
        Quantum Machines associate filters to channels but this may not be the case
        in other instruments.
        """
        raise_error(NotImplementedError, "Instruments do not support filters.")

    @filters.setter
    def filters(self, value):
        raise_error(NotImplementedError, "Instruments do not support filters.")
