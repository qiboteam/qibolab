import pathlib

from laboneq.dsl.device import create_connection
from laboneq.dsl.device.instruments import HDAWG, PQSC, SHFQC
from laboneq.simple import DeviceSetup

from qibolab import Platform
from qibolab._core.components import (
    AcquisitionChannel,
    DcChannel,
    IqChannel,
    OscillatorConfig,
)
from qibolab._core.instruments.zhinst import (
    ZiAcquisitionConfig,
    ZiChannel,
    ZiDcConfig,
    ZiIqConfig,
    Zurich,
)
from qibolab._core.kernels import Kernels
from qibolab._core.parameters import Parameters

FOLDER = pathlib.Path(__file__).parent


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

    parameters = Parameters.load(FOLDER)
    kernels = Kernels.load(FOLDER)
    qubits, couplers, pairs = (
        parameters.native_gates.single_qubit,
        parameters.native_gates.coupler,
        parameters.native_gates.two_qubit,
    )

    configs = parameters.configs
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
    for q in qubits:
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
            **component_params[acquisition_name], kernel=kernels.get(q)
        )
        qubits[q].acquisition = AcquisitionChannel(
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

    for i, c in enumerate(couplers):
        flux_name = f"coupler_{c}/flux"
        configs[flux_name] = ZiDcConfig(**component_params[flux_name])
        couplers[c].flux = DcChannel(name=flux_name)
        zi_channels.append(
            ZiChannel(couplers[c].flux, device="device_hdawg2", path=f"SIGOUTS/{i}")
        )

    controller = Zurich(
        device_setup=device_setup,
        channels=zi_channels,
        time_of_flight=75,
        smearing=50,
    )

    return Platform(
        name=str(FOLDER),
        configs=configs,
        parameters=parameters,
        instruments={"EL_ZURO": controller},
        resonator_type="3D",
    )
