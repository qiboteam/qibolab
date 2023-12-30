class Port:
    """Abstract interface for instrument parameters.

    These parameters are exposed to the user through :class:`qibolab.channels.Channel`.

    Drivers should subclass this interface and implement the getters
    and setters for all the parameters that are available for the
    corresponding instruments.

    Each port is identified by the ``name`` attribute.
    Note that the type of the identifier can be different of each port implementation.
    """

    name: str
    """Name of the port that acts as its identifier."""
    offset: float
    """DC offset that is applied to this port."""
    lo_frequency: float
    """Local oscillator frequency for the given port.

    Relevant only for controllers with internal local oscillators.
    """
    lo_power: float
    """Local oscillator power for the given port.

    Relevant only for controllers with internal local oscillators.
    """
    # TODO: Maybe gain, attenuation and power range can be unified to a single attribute
    gain: float
    """Gain that is applied to this port."""
    attenuation: float
    """Attenuation that is applied to this port."""
    power_range: int
    """Similar to attenuation (negative) and gain (positive) for (Zurich
    instruments)."""
