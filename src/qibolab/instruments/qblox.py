from pulsar_qcm.pulsar_qcm import pulsar_qcm
from pulsar_qrm.pulsar_qrm import pulsar_qrm


class PulsarQRM(pulsar_qrm):
    """Class for interfacing with Pulsar QRM."""

    def __init__(self, label, ip,
                 ref_clock="external", sequencer=0,
                 sync_en=True, hardware_avg_en=True,
                 acq_trigger_mode="sequencer"):
        # Instantiate base object from qblox library and connect to it
        super().__init__(label, ip)

        # Reset and configure
        self.reset()
        self.reference_source(ref_clock)
        self.scope_acq_sequencer_select(sequencer)
        self.scope_acq_avg_mode_en_path0(hardware_avg_en)
        self.scope_acq_avg_mode_en_path1(hardware_avg_en)

        self.scope_acq_trigger_mode_path0(acq_trigger_mode)
        self.scope_acq_trigger_mode_path1(acq_trigger_mode)

        self.sequencer = sequencer
        if sequencer == 1:
            self.sequencer1_sync_en(sync_en)
        else:
            self.sequencer0_sync_en(sync_en)

    def setup(self, gain):
        if self.sequencer == 1:
            self.sequencer1_gain_awg_path0(gain)
            self.sequencer1_gain_awg_path1(gain)
        else:
            self.sequencer0_gain_awg_path0(gain)
            self.sequencer0_gain_awg_path1(gain)
