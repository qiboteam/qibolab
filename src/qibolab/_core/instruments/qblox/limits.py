import qblox_instruments

WAVEFORM_MEMORY = 2**14
"""Maximum waveform memory available.

https://docs.qblox.com/en/v0.16.0/cluster/q1_sequence_processor.html#waveform-memory
"""
WAVEFORM_NUMBER = 2**10
"""Maximum number of waveforms available.

https://docs.qblox.com/en/v0.16.0/cluster/q1_sequence_processor.html#waveform-memory
"""

QCM_INSTRUCTION_MEMORY = 2**14
QRM_INSTRUCTION_MEMORY = 12288
"""Maximum number of instructions per program for the QCM and QRM modules.

https://docs.qblox.com/en/v0.16.0/cluster/q1_sequence_processor.html#instruction-memory
"""

WEIGHT_MEMORY = 2**14
"""Maximum waveform memory available.

https://docs.qblox.com/en/v0.16.0/cluster/q1_sequence_processor.html#integrator
"""
WEIGHT_NUMBER = 2**5
"""Maximum number of weights available.

https://docs.qblox.com/en/v0.16.0/cluster/q1_sequence_processor.html#weight-memory
"""

if qblox_instruments.__version__ >= "1.0.0":
    ACQUISITION_MEMORY = 3e6
else:
    ACQUISITION_MEMORY = 2**17
"""Maximum acquisition memory available.

..note::
    Here the qblox instruments version is used as a proxy for the check if the cluster
    firmware >= 0.13.0, which due to the dependency requirements is an equivalent
    condition.

https://docs.qblox.com/en/v0.16.0/cluster/q1_sequence_processor.html
https://docs.qblox.com/en/main/releases.html#new-features
"""

ACQUISITION_NUMBER = 2**5
"""Maximum number of acquisitions available.

https://docs.qblox.com/en/v0.16.0/cluster/q1_sequence_processor.html#:~:text=Square%20Weight%20Acquisitions
"""

SCOPE_MEMORY = 2**14
"""Number of samples in a scope acquisition trace.

https://docs.qblox.com/en/v2026.08.0/products/qblox_instruments/tutorials/QRM/030_scope_acquisition.html
"""

SEQUENCER_REGISTERS = 64
"""Number of registers available in the Q1 sequencer.

https://docs.qblox.com/en/v2026.08.0/products/architecture/sequencers/sequencer.html#registers
"""
