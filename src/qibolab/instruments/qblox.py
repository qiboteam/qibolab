from pulsar_qcm.pulsar_qcm import pulsar_qcm
from pulsar_qrm.pulsar_qrm import pulsar_qrm


class Pulsar_QRM(pulsar_qrm):
    """Class for interfacing with Pulsar QRM."""

    def __init__(self, label, ip,
                 ref_clock="external", sequencer=0,
                 sync_en=True, hardware_avg_en=True,
                 acq_trigger_mode="sequencer"):
        # Instantiate base object from qblox library and connect to it
        super().__init__(label, ip)

        # Reset and configure
        qrm.reset()
        qrm.reference_source(ref_clock)
        qrm.scope_acq_sequencer_select(sequencer)
        qrm.scope_acq_avg_mode_en_path0(hardware_avg_en)
        qrm.scope_acq_avg_mode_en_path1(hardware_avg_en)

        qrm.scope_acq_trigger_mode_path0(acq_trigger_mode)
        qrm.scope_acq_trigger_mode_path1(acq_trigger_mode)

        if sequencer == 1:
            qrm.sequencer1_sync_en(sync_en)
        else:
            qrm.sequencer0_sync_en(sync_en)

        self._qrm = qrm

    def setup(self, gain, sequencer):
        sequencer = self._settings['sequencer']
        if sequencer == 1:
            self._qrm.sequencer1_gain_awg_path0(gain))
            self._qrm.sequencer1_gain_awg_path1(settings['gain'])
        else:
            self._qrm.sequencer0_gain_awg_path0(settings['gain'])
            self._qrm.sequencer0_gain_awg_path1(settings['gain'])
