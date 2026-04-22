from qm_saas import QmSaas
from qibolab import create_platform
from qibolab._core.pulses import Pulse
from qibolab._core.pulses.envelope import Rectangular
from qibolab._core.sequence import PulseSequence

client = QmSaas(email="nandan24@nus.edu.sg", password="TpXa-LqRe-ZmWs-BkFy")

pulse_duration = 4e5
rate = int(3e8 / pulse_duration)
units = "Hz/nsec"

chirp_pulse = Pulse(
    amplitude=0.2,
    duration=pulse_duration,
    relative_phase=0.0,
    envelope=Rectangular(),
    chirp=(rate, units),
)

sequence = PulseSequence([("drive", chirp_pulse)])

with client.simulator("v2_6_0") as instance:
    # Load platform FIRST before connecting
    platform = create_platform("my_platform")

    # Override address BEFORE calling connect
    platform.instruments["con1"].address = f"{instance.host}:{instance.port}"

    # Also pass connection headers — this is what was missing
    platform.instruments["con1"].manager = None  # ensure fresh connection
    
    # Patch the connect method to use SaaS headers
    from qm import QuantumMachinesManager
    platform.instruments["con1"].manager = QuantumMachinesManager(
        host=instance.host,
        port=instance.port,
        connection_headers=instance.default_connection_headers,
    )

    platform.instruments["con1"].script_file_name = "chirp_program.py"
    results = platform.execute([sequence], nshots=1000)

    print(results)