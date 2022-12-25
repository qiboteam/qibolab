from types import SimpleNamespace


class Channel:
    """Representation of physical wire connection (channel).

    Name is used as a unique identifier for channels. If a channel
    with an existing name is recreated, it will refer to the existing object.
    Channel objects are created and their attributes are set during
    the platform instantiation.

    Args:
        name (str): Name of the channel as given in the platform runcard.

    Attributes:
        ports (list): List of tuples (controller (`str`), port (`int`))
            specifying the QM (I, Q) ports that the channel is connected.
        qubits (list): List of Qubit objects for the qubits connected to this channel.
        local_oscillator (:class:`qibolab.instruments.rohde_schwarz.SGS100A):
            Instrument object for the local oscillator connected to this channel.
    """

    instances = {}

    def __new__(cls, name):
        if name is None:
            return None

        if name not in cls.instances:
            new = super().__new__(cls)
            new.name = name

            new.ports = []
            new.qubits = []

            new.local_oscillator = None
            new.lo_frequency = None
            new.lo_power = None

            cls.instances[name] = new

        return cls.instances[name]

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"<Channel {self.name}>"


class Qubit:
    """Representation of a physical qubit.

    Qubit objects are instantiated during the platform initialization and
    are used to register elements in the QM config.

    Args:
        name (int): Qubit number.
        characterization (dict): Dictionary with the characterization values
            for the qubit, loaded from the runcard.
        readout (:class:`qibolab.platforms.quantum_machines.Channel`): Channel
            used to send readout pulses to the qubit.
        feedback (:class:`qibolab.platforms.quantum_machines.Channel`): Channel
            used to get readout feedback from the qubit.
        drive (:class:`qibolab.platforms.quantum_machines.Channel`): Channel
            used to send drive pulses to the qubit.
        flux (:class:`qibolab.platforms.quantum_machines.Channel`): Channel
            used to send flux pulses to the qubit.
    """

    def __init__(self, name, characterization, readout, feedback, drive, flux=None):
        self.name = name
        self.characterization = SimpleNamespace(**characterization)

        self.readout = readout
        self.feedback = feedback
        self.drive = drive
        self.flux = flux

        # register qubits to channels
        for mode in ["readout", "feedback", "drive", "flux"]:
            channel = getattr(self, mode)
            if channel is not None:
                channel.qubits.append((self, mode))

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return f"<Qubit {self.name}>"
