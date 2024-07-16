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
    load_component_config,
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

    components = {}
    measure_lo = "measure/lo"
    drive_los = {
        0: "qubit_0_1/drive/lo",
        1: "qubit_0_1/drive/lo",
        2: "qubit_2_3/drive/lo",
        3: "qubit_2_3/drive/lo",
        4: "qubit_4/drive/lo",
    }
    components[measure_lo] = load_component_config(
        runcard, measure_lo, OscillatorConfig
    )
    zi_channels = []
    for q in QUBITS:
        measure_name = f"qubit_{q}/measure"
        acquisition_name = f"qubit_{q}/acquire"
        components[measure_name] = load_component_config(
            runcard, measure_name, ZiIqConfig
        )
        qubits[q].measure = IqChannel(
            name=measure_name, lo=measure_lo, mixer=None, acquisition=acquisition_name
        )
        zi_channels.append(
            ZiChannel(
                qubits[q].measure, device="device_shfqc", path="QACHANNELS/0/OUTPUT"
            )
        )

        components[acquisition_name] = load_component_config(
            runcard, acquisition_name, ZiAcquisitionConfig
        )
        qubits[q].acquisition = AcquireChannel(
            name=acquisition_name,
            twpa_pump=None,
            measure=measure_name,
        )
        zi_channels.append(
            ZiChannel(
                qubits[q].acquisition, device="device_shfqc", path="QACHANNELS/0/INPUT"
            )
        )

        drive_name = f"qubit_{q}/drive"
        components[drive_los[q]] = load_component_config(
            runcard, drive_los[q], OscillatorConfig
        )
        components[drive_name] = load_component_config(runcard, drive_name, ZiIqConfig)
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
        components[flux_name] = load_component_config(runcard, flux_name, ZiDcConfig)
        qubits[q].flux = DcChannel(
            name=flux_name,
        )
        zi_channels.append(
            ZiChannel(qubits[q].flux, device="device_hdawg", path=f"SIGOUTS/{q}")
        )

    for i, c in enumerate(COUPLERS):
        flux_name = f"coupler_{c}/flux"
        components[flux_name] = load_component_config(runcard, flux_name, ZiDcConfig)
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
        components,
        instruments,
        settings,
        resonator_type="3D",
        couplers=couplers,
    )
