from pulses0 import sequence

from qibolab import create_platform
from qibolab.execution_parameters import ExecutionParameters

# Define platform and load specific runcard
platform = create_platform("dummy")

# Connects to lab instruments using the details specified in the calibration settings.
platform.connect()

# Executes a pulse sequence.
options = ExecutionParameters(nshots=1000, relaxation_time=100)
results = platform.execute_pulse_sequence(sequence, options=options)

# Disconnect from the instruments
platform.disconnect()
