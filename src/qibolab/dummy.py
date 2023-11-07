import itertools
import pathlib

from qibolab.channels import Channel, ChannelMap
from qibolab.instruments.dummy import DummyInstrument
from qibolab.instruments.oscillator import LocalOscillator
from qibolab.platform import Platform
from qibolab.serialize import load_qubits, load_runcard, load_settings


def remove_couplers(runcard):
    """Remove coupler sections from runcard to create a dummy platform without couplers."""
    runcard["topology"] = list(runcard["topology"].values())
    del runcard["couplers"]
    del runcard["native_gates"]["coupler"]
    del runcard["characterization"]["coupler"]
    two_qubit = runcard["native_gates"]["two_qubit"]
    for i, gates in two_qubit.items():
        for j, gate in gates.items():
            two_qubit[i][j] = [pulse for pulse in gate if pulse["type"] != "coupler"]
    return runcard


def create_dummy(with_couplers: bool = True):
    """Create a dummy platform using the dummy instrument.

    Args:
        with_couplers (bool): Selects whether the dummy platform will have coupler qubits.
    """
    if with_couplers:
        name = "dummy_couplers"
    else:
        name = "dummy"

    # Create dummy controller
    instrument = DummyInstrument(name, 0)

    # Create local oscillator
    twpa_pump = LocalOscillator(name="twpa_pump", address=0)
    twpa_pump.frequency = 1e9
    twpa_pump.power = 10

    runcard = load_runcard(pathlib.Path(__file__).parent / "dummy.yml")
    if not with_couplers:
        runcard = remove_couplers(runcard)

    # Create channel objects
    nqubits = runcard["nqubits"]
    channels = ChannelMap()
    channels |= Channel("readout", port=instrument["readout"])
    channels |= (Channel(f"drive-{i}", port=instrument[f"drive-{i}"]) for i in range(nqubits))
    channels |= (Channel(f"flux-{i}", port=instrument[f"flux-{i}"]) for i in range(nqubits))
    channels |= Channel("twpa", port=None)
    if with_couplers:
        channels |= (
            Channel(f"flux_coupler-{c}", port=instrument[f"flux_coupler-{c}"])
            for c in itertools.chain(range(0, 2), range(3, 5))
        )
    channels["readout"].attenuation = 0
    channels["twpa"].local_oscillator = twpa_pump

    qubits, couplers, pairs = load_qubits(runcard)
    print(couplers)
    settings = load_settings(runcard)

    # map channels to qubits
    for q, qubit in qubits.items():
        qubit.readout = channels["readout"]
        qubit.drive = channels[f"drive-{q}"]
        qubit.flux = channels[f"flux-{q}"]
        qubit.twpa = channels["twpa"]

    if with_couplers:
        # map channels to couplers
        for c, coupler in couplers.items():
            coupler.flux = channels[f"flux_coupler-{c}"]

    instruments = {instrument.name: instrument, twpa_pump.name: twpa_pump}
    instrument.sampling_rate = settings.sampling_rate * 1e-9

    return Platform(name, qubits, pairs, instruments, settings, resonator_type="2D", couplers=couplers)
