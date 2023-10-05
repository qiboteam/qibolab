import itertools
import pathlib

from qibolab.channels import Channel, ChannelMap
from qibolab.instruments.dummy import DummyInstrument
from qibolab.instruments.oscillator import LocalOscillator
from qibolab.platform import Platform
from qibolab.serialize import load_qubits, load_runcard, load_settings

NAME = "dummy"


def create_dummy():
    """Create a dummy platform using the dummy instrument."""
    # Create dummy controller
    instrument = DummyInstrument(NAME, 0)

    # Create local oscillator
    twpa_pump = LocalOscillator(name="twpa_pump", address=0)
    twpa_pump.frequency = 1e9
    twpa_pump.power = 10

    runcard = load_runcard(pathlib.Path(__file__).parent / "dummy.yml")
    # Create channel objects
    nqubits = runcard["nqubits"]
    channels = ChannelMap()
    channels |= Channel("readout", port=instrument["readout"])
    channels |= (Channel(f"drive-{i}", port=instrument[f"drive-{i}"]) for i in range(nqubits))
    channels |= (Channel(f"flux-{i}", port=instrument[f"flux-{i}"]) for i in range(nqubits))
    channels |= Channel("twpa", port=None)
    # FIXME: Issues with the names if they are strings maybe
    channels |= (
        Channel(f"flux_coupler-{c}", port=instrument[f"flux_coupler-{c}"])
        for c in itertools.chain(range(0, 2), range(3, 5))
    )
    channels["readout"].attenuation = 0
    channels["twpa"].local_oscillator = twpa_pump

    qubits, couplers, pairs = load_qubits(runcard)
    settings = load_settings(runcard)

    # map channels to qubits
    for q, qubit in qubits.items():
        qubit.readout = channels["readout"]
        qubit.drive = channels[f"drive-{q}"]
        qubit.flux = channels[f"flux-{q}"]
        qubit.twpa = channels["twpa"]

    # map channels to couplers
    for c, coupler in couplers.items():
        coupler.flux = channels[f"flux_coupler-{c}"]

    instruments = {instrument.name: instrument, twpa_pump.name: twpa_pump}
    instrument.sampling_rate = settings.sampling_rate * 1e-9

    return Platform(NAME, qubits, pairs, instruments, settings, resonator_type="2D", couplers=couplers)
