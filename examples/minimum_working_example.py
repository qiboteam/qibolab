from qibolab import Platform
from qibolab.paths import qibolab_folder
from qibolab.pulses import Pulse, PulseSequence, ReadoutPulse

# Define PulseSequence
sequence = PulseSequence()
# Add some pulses to the pulse sequence


sequence.add(
    ReadoutPulse(
        start=4004,
        amplitude=0.9,
        duration=2000,
        frequency=7_060_000_000,
        relative_phase=0,
        shape="Rectangular",
        channel=0,
        qubit=0,
    )
)
sequence.add(
    ReadoutPulse(
        start=4004,
        amplitude=0.9,
        duration=2000,
        frequency=7_260_000_000,
        relative_phase=0,
        shape="Rectangular",
        channel=1,
        qubit=1,
    )
)
sequence.add(
    ReadoutPulse(
        start=4004,
        amplitude=0.9,
        duration=2000,
        frequency=7_460_000_000,
        relative_phase=0,
        shape="Rectangular",
        channel=2,
        qubit=2,
    )
)
print(qibolab_folder) 
# Define platform and load specific runcard
#runcard = "" /  "home" / "xilinx" / "qibolab" / "src" / "qibolab" / "runcards" / "tii_rfsocZCU111.yml"
runcard = "/home/xilinx/qibolab/src/qibolab/runcards/tii_rfsocZCU111.yml"
platform = Platform("tii_rfsocZCU111", runcard)

# Connects to lab instruments using the details specified in the calibration settings.
platform.connect()
# Configures instruments using the loaded calibration settings.
platform.setup()
# Turns on the local oscillators
platform.start()
# Executes a pulse sequence.
results = platform.execute_pulse_sequence(sequence, nshots=1000)
print(f"results (amplitude, phase, i, q): {results}")
# Turn off lab instruments
platform.stop()
# Disconnect from the instruments
platform.disconnect()
