"""
Abstract classes for instruments
"""

class Instrument:
    """
    General instrument class
    """
    def __init__(self, name):
        self.name = name

    def connect(self, address):
        """
        Initiates the instrument connection
        """
        pass

    def setup(self, parameters):
        """
        Setups the instrument with given parameters
        """
        pass

    def close(self):
        """
        Closes the instrument connection
        """
        pass


class DAC(Instrument):
    """
    Class for DACs/Waveform Generators
    """

    def translate(self, sequence, shots):
        """
        Translates the pulse sequence into instrument specific data
        """
        pass
    

    def upload(self, payload):
        """
        Uploads the translated pulse sequence to the instrument
        """
        pass

    def play(self):
        """
        Plays the instrument or arms it to play on receiving a trigger
        """
        pass

    def stop(self):
        """
        Disarms the instrument
        """
        pass

class ADC(Instrument):
    """
    Class for ADC/Oscilloscope/Signal Digitzers
    """

    def arm(self):
        """
        Arms the instrument to listen to a trigger
        """
        pass

    def result(self):
        """
        Returns the result waveforms
        """
        pass


class LO(Instrument):
    """
    Class for local oscillators
    """
    def start(self):
        """
        Starts the instrument
        """
        pass

    def stop(self):
        """
        Stops the instrument
        """
        pass

class Attenuator(Instrument):
    """
    Class for variable attenuators
    """

class CurrentSupply(Instrument):
    """
    Class for direct current sources
    """