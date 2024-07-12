from dataclasses import dataclass


@dataclass(frozen=True)
class Octave:
    """Device handling Octaves."""

    name: str
    """Name of the device."""
    port: int
    """Network port of the Octave in the cluster configuration."""
    connectivity: str
    """OPXplus that acts as the waveform generator for the Octave."""
