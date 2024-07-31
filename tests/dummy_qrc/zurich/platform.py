import pathlib

from laboneq.dsl.device import create_connection
from laboneq.dsl.device.instruments import HDAWG, PQSC, SHFQC
from laboneq.simple import DeviceSetup

from qibolab import Platform
from qibolab.components import AcquireChannel, DcChannel, IqChannel, OscillatorConfig
from qibolab.instruments.zhinst import (
    ZiAcquisitionConfig,
    ZiChannel,
    ZiDcConfig,
    ZiIqConfig,
    Zurich,
)
from qibolab.kernels import Kernels
from qibolab.serialize import (
    load_instrument_settings,
    load_qubits,
    load_runcard,
    load_settings,
)

FOLDER = pathlib.Path(__file__).parent
QUBITS = [0, 1, 2, 3, 4]
COUPLERS = [0, 1, 3, 4]


def create():
    """A platform representing a chip with star topology, featuring 5 qubits
    and 4 couplers, controlled by Zurich Instruments SHFQC, HDAWGs and PQSC."""

    device_setup = DeviceSetup("device setup")
    device_setup.add_dataserver(host="localhost", port=8004)
    device_setup.add_instruments(
        HDAWG("device_hdawg", address="DEV8660"),
        HDAWG("device_hdawg2", address="DEV8673"),
        PQSC("device_pqsc", address="DEV10055", reference_clock_source="internal"),
        SHFQC("device_shfqc", address="DEV12146"),
    )
    device_setup.add_connections(
        "device_pqsc",
        create_connection(to_instrument="device_hdawg2", ports="ZSYNCS/1"),
        create_connection(to_instrument="device_hdawg", ports="ZSYNCS/0"),
        create_connection(to_instrument="device_shfqc", ports="ZSYNCS/2"),
    )

    runcard = load_runcard(FOLDER)
    kernels = Kernels.load(FOLDER)
    qubits, couplers, pairs = load_qubits(runcard, kernels)
    settings = load_settings(runcard)

    configs = {}
    component_params = runcard["components"]
    readout_lo = "readout/lo"
    drive_los = {
        0: "qubit_0_1/drive/lo",
        1: "qubit_0_1/drive/lo",
        2: "qubit_2_3/drive/lo",
        3: "qubit_2_3/drive/lo",
        4: "qubit_4/drive/lo",
    }
    configs[readout_lo] = OscillatorConfig(**component_params[readout_lo])
    zi_channels = []
    for q in QUBITS:
        probe_name = f"qubit_{q}/probe"
        acquisition_name = f"qubit_{q}/acquire"
        configs[probe_name] = ZiIqConfig(**component_params[probe_name])
        qubits[q].probe = IqChannel(
            name=probe_name, lo=readout_lo, mixer=None, acquisition=acquisition_name
        )
        zi_channels.append(
            ZiChannel(
                qubits[q].probe, device="device_shfqc", path="QACHANNELS/0/OUTPUT"
            )
        )

        configs[acquisition_name] = ZiAcquisitionConfig(
            **component_params[acquisition_name]
        )
        qubits[q].acquisition = AcquireChannel(
            name=acquisition_name,
            twpa_pump=None,
            probe=probe_name,
        )
        zi_channels.append(
            ZiChannel(
                qubits[q].acquisition, device="device_shfqc", path="QACHANNELS/0/INPUT"
            )
        )

        drive_name = f"qubit_{q}/drive"
        configs[drive_los[q]] = OscillatorConfig(**component_params[drive_los[q]])
        configs[drive_name] = ZiIqConfig(**component_params[drive_name])
        qubits[q].drive = IqChannel(
            name=drive_name,
            mixer=None,
            lo=drive_los[q],
        )
        zi_channels.append(
            ZiChannel(
                qubits[q].drive, device="device_shfqc", path=f"SGCHANNELS/{q}/OUTPUT"
            )
        )

        flux_name = f"qubit_{q}/flux"
        configs[flux_name] = ZiDcConfig(**component_params[flux_name])
        qubits[q].flux = DcChannel(
            name=flux_name,
        )
        zi_channels.append(
            ZiChannel(qubits[q].flux, device="device_hdawg", path=f"SIGOUTS/{q}")
        )

    for i, c in enumerate(COUPLERS):
        flux_name = f"coupler_{c}/flux"
        configs[flux_name] = ZiDcConfig(**component_params[flux_name])
        couplers[c].flux = DcChannel(name=flux_name)
        zi_channels.append(
            ZiChannel(couplers[c].flux, device="device_hdawg2", path=f"SIGOUTS/{i}")
        )

    controller = Zurich(
        "EL_ZURO",
        device_setup=device_setup,
        channels=zi_channels,
        time_of_flight=75,
        smearing=50,
    )

    instruments = {controller.name: controller}
    instruments = load_instrument_settings(runcard, instruments)
    return Platform(
        str(FOLDER),
        qubits,
        pairs,
        configs,
        instruments,
        settings,
        resonator_type="3D",
        couplers=couplers,
    )
