import pathlib
from qibolab.paths import qibolab_folder
import json
import numpy as np
from abc import ABC, abstractmethod
from qibo.config import raise_error
from qibolab.instruments.abstract import AbstractInstrument, InstrumentException

import logging
logger = logging.getLogger(__name__)  # TODO: Consider using a global logger

data_folder = qibolab_folder / "instruments" / "data"
data_folder.mkdir(parents=True, exist_ok=True)

class GenericPulsar(AbstractInstrument, ABC):

    def __init__(self, label, ip, sequencer, ref_clock, sync_en, is_cluster):
        super().__init__(label, ip)
        self.label = label
        # TODO When updating to the new firmware, use a sequencer mapping instead of setting a single sequencer
        self.sequencer = sequencer
        self.ref_clock = ref_clock
        self.sync_en = sync_en
        self.is_cluster = is_cluster
        self._connected = False
        self.Device = None
        self.device = None
        # To be defined in each instrument
        self.name = None
        # To be defined during setup
        self.hardware_avg = None
        self.initial_delay = None
        self.repetition_duration = None
        # hardcoded values used in ``generate_program``
        # same value is used for all readout pulses (?)
        self.delay_before_readout = 4
        self.wait_loop_step = 1000
        # maximum length of a waveform in number of samples (defined by the device memory).
        self.duration_base = 16380 
        # hardcoded values used in ``upload``
        # TODO QCM shouldn't have acquisitions
        self.acquisitions = {"single": {"num_bins": 1, "index":0}}
        self.weights = {}

    def connect(self):
        """Connects to the instruments."""
        if not self._connected:
            import socket
            try:
                self.device = self.Device(self.label, self.ip) # pylint: disable=E1102
                logger.info(f"{self.name} connection established.")
                self._connected = True
            except socket.timeout:
                # Use warning instead of exception when instruments are
                # not available so that we can run tests on different devices
                logger.warning("Could not connect to QRM. Skipping...")
            except Exception as exc:  # pragma: no cover
                raise InstrumentException(self, str(exc))
        else:
            raise_error(RuntimeError, "QRM is already connected.")
        
    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, gain):
        self._gain = gain
        if self._connected:
            if self.sequencer == 1:
                self.device.sequencer1_gain_awg_path0(gain)
                self.device.sequencer1_gain_awg_path1(gain)
            else:
                self.device.sequencer0_gain_awg_path0(gain)
                self.device.sequencer0_gain_awg_path1(gain)
        else:
            logger.warning("Cannot set gain because device is not connected.")

    def setup(self, **kwargs):
        """Sets calibration setting to QBlox instruments.
        Args:
            gain (float):
            initial_delay (float):
            repetition_duration (float):
        """
        self.gain = kwargs['gain']
        if 'initial_delay' in kwargs:
            self.initial_delay = kwargs['initial_delay']
        self.repetition_duration = kwargs['repetition_duration']

    def _translate_single_pulse(self, pulse):
        """Translates a single pulse to the instrument waveform format.
        Helper method for :meth:`qibolab.instruments.qblox.GenericPulsar.generate_waveforms`.
        Args:
            pulse (:class:`qibolab.pulses.Pulse`): Pulse object to translate.
        Returns:
            Dictionary containing the waveform corresponding to the pulse.
        """
        # Use the envelope to modulate a sinusoldal signal of frequency freq_if
        envelope_i = pulse.compile()
        # TODO: if ``envelope_q`` is not always 0 we need to find how to
        # calculate it
        envelope_q  = np.zeros(int(pulse.duration))
        envelopes   = np.array([envelope_i, envelope_q])
        time        = np.arange(pulse.duration) * self.sampling_rate
        cosalpha    = np.cos(2 * np.pi * pulse.frequency * time + pulse.phase)
        sinalpha    = np.sin(2 * np.pi * pulse.frequency * time + pulse.phase)
        mod_matrix  = np.array([[ cosalpha, sinalpha],
                                [-sinalpha, cosalpha]])
        mod_signals = np.einsum("abt,bt->ta", mod_matrix, envelopes)

        # add offsets to compensate mixer leakage
        waveform = {
            "modI": {"data": mod_signals[:, 0] + pulse.offset_i, "index": 0},
            "modQ": {"data": mod_signals[:, 1] + pulse.offset_q, "index": 1}
        }
        return waveform

    def generate_waveforms(self, pulses):
        """
        Translates a list of pulses to the instrument waveform format.

        Args:
            pulses (list): List of :class:`qibolab.pulses.Pulse` objects.

        Returns:
            Dictionary containing waveforms corresponding to all pulses.
        """
        if not pulses:
            raise_error(ValueError, "Cannot translate empty pulse sequence.")
        name = self.name

        waveforms = {}
        waveform_sequence = []  # list of waveform idx applied

        unique_pulses = []
        idx = 0

        # TODO: no es necesario comprobar si son pulsos unicos. Ya lo hacemos en el platform.execute() antes 
        # antes de llamara al translate() 
        for pulse in pulses:
            if pulse not in unique_pulses:
                unique_pulses.append(pulse)
                waveform_sequence.append(idx)
                waveform = self._translate_single_pulse(pulse)
                for mod in ['I', 'Q']:
                    new_waveform = {f"{pulse.serial}_mod{mod}": {
                        "data": waveform[f"mod{mod}"]["data"], "index": idx}}
                    waveforms.update(new_waveform)
                    idx += 1
            else:
                waveform_sequence.append(unique_pulses.index(pulse)*2)

        # # Fixing 0s addded to the qrm waveform. Needs to be improved, but working well on TIIq
        # for pulse in pulses:
        #     if(pulse.channel == "qrm"):
        #         waveforms[f"{pulse.serial}_I"]["data"] = waveforms[f"{pulse.serial}_I"]["data"][pulse.start:]
        #         waveforms[f"{pulse.serial}_Q"]["data"] = waveforms[f"{pulse.serial}_Q"]["data"][pulse.start:]

        return waveforms, waveform_sequence

    def generate_program(self, hardware_avg, initial_delay, delay_before_readout, acquire_instruction, wait_time, pulses, waveform_sequence):
        """Generates the program to be uploaded to instruments."""

        # This calculation was moved to `PulsarQCM` and `PulsarQRM`
        # if ro_pulse is not None:
        #    acquire_instruction = "acquire   0,0,4"
        #    wait_time = self.duration_base - start_last_pulse - delay_before_read_out

        if initial_delay != 0:
            initial_wait_instruction = f"wait      {initial_delay}"
        else:
            initial_wait_instruction = ""

        program = f"""
        move    {hardware_avg},R0
            nop
            wait_sync 4          # Synchronize sequencers over multiple instruments
        loop:
            {initial_wait_instruction}"""
        for i, (idx, pulse) in enumerate(zip(waveform_sequence, pulses)):
            if i < len(pulses) - 1:
                pulse_wait = pulses[i + 1].start - pulse.start
            else:
                pulse_wait = delay_before_readout

            program += f"""
            play      {idx}, {idx+1}, {pulse_wait}"""

        # Tiempo de inicio del pulso superior a 16383 ns (self.duration_base < pulses[-1].start)
        # Comenzamos nuevo ciclo 
        while (wait_time <= 0):
            wait_time = wait_time + self.duration_base

        program += f"""
            {acquire_instruction}
            wait      {wait_time}
        """
        if self.repetition_duration > self.duration_base:
            extra_duration = self.repetition_duration - self.duration_base
            extra_wait = extra_duration % self.wait_loop_step
            num_wait_loops = (extra_duration - extra_wait) // self. wait_loop_step
            program += f"""
                move      {num_wait_loops},R1
                nop
                repeatloop:
                    wait      {self.wait_loop_step}
                    loop      R1,@repeatloop
                wait      {extra_wait}
            """
        program += """
            loop    R0,@loop
            stop
        """
        return program


    @abstractmethod
    def translate(self, sequence, nshots): # pragma: no cover
        """Translates an abstract pulse sequence to QBlox format.

        Args:
            sequence (:class:`qibolab.pulses.PulseSequence`): Pulse sequence.

        Returns:
            The waveforms (dict) and program (str) required to execute the
            pulse sequence on QBlox instruments.
        """
        raise_error(NotImplementedError)

    def upload(self, waveforms, program, data_f):
        """Uploads waveforms and programs to QBlox sequencer to prepare execution."""
        import os
        # Upload waveforms and program
        # Reformat waveforms to lists
        for name, waveform in waveforms.items():
            if isinstance(waveform["data"], np.ndarray):
                waveforms[name]["data"] = waveforms[name]["data"].tolist()  # JSON only supports lists

        # Add sequence program and waveforms to single dictionary and write to JSON file
        filename = f"{self.name}_sequence.json"
        program_dict = {
            "waveforms": waveforms,
            "weights": self.weights,
            "acquisitions": self.acquisitions,
            "program": program
            }

        with open(data_folder / filename, "w", encoding="utf-8") as file:
            json.dump(program_dict, file, indent=4)

        # Upload json file to the device
        if self.sequencer == 1:
            self.device.sequencer1_waveforms_and_program(str(data_folder / filename))
        else:
            self.device.sequencer0_waveforms_and_program(str(data_folder / filename))

    def play_sequence(self):
        """Executes the uploaded instructions."""
        # arm sequencer and start playing sequence
        self.device.arm_sequencer()
        self.device.start_sequencer()

    def stop(self):
        """Stops the QBlox sequencer from sending pulses."""
        self.device.stop_sequencer()

    def close(self):
        """Disconnects from the instrument."""
        if self._connected:
            self.stop()
            self.device.close()
            self._connected = False
    def start(self):
        pass

    def disconnect(self):
        pass


class PulsarQRM(GenericPulsar):
    """Class for interfacing with Pulsar QRM."""

    def __init__(self, label, ip, ref_clock="external", sequencer=0, sync_en=True,
                 hardware_avg_en=True, acq_trigger_mode="sequencer", is_cluster=True):
        super().__init__(label, ip, sequencer, ref_clock, sync_en, is_cluster)
        # Instantiate base object from qblox library and connect to it
        self.name = "qrm"
        if self.is_cluster:
            from cluster.cluster import cluster_qrm
            self.Device = cluster_qrm
        else:
            from pulsar_qrm.pulsar_qrm import pulsar_qrm
            self.Device = pulsar_qrm
        self.connect()
        self.sequencer = sequencer
        self.hardware_avg_en = hardware_avg_en

        # Reset and configure
        self.device.reset()
        self.device.reference_source(ref_clock)
        self.device.scope_acq_sequencer_select(sequencer) # specifies which sequencer triggers the scope acquisition when using sequencer trigger mode
        self.device.scope_acq_avg_mode_en_path0(hardware_avg_en) # sets scope acquisition averaging mode enable for input path 0
        self.device.scope_acq_avg_mode_en_path1(hardware_avg_en)
        self.device.scope_acq_trigger_mode_path0(acq_trigger_mode) # sets scope acquisition trigger mode for input path 0 (‘sequencer’ = triggered by sequencer, ‘level’ = triggered by input level).
        self.device.scope_acq_trigger_mode_path1(acq_trigger_mode)
        # sync sequencer
        if self.sequencer == 1:
            self.device.sequencer1_sync_en(sync_en)
        else:
            self.device.sequencer0_sync_en(sync_en)

    def setup(self, **kwargs):
        start_sample = kwargs['start_sample']
        integration_length = kwargs['integration_length']
        sampling_rate = kwargs['sampling_rate']
        mode = kwargs['mode']
        if not 'initial_delay' in kwargs:
            kwargs['initial_delay'] = 0
        super().setup(**kwargs)
        self.start_sample = start_sample
        self.integration_length = integration_length
        self.sampling_rate = sampling_rate
        self.mode = mode

    def translate(self, sequence, delay_before_readout, nshots):
        # Allocate only readout pulses to PulsarQRM
        pulses = sequence.qrm_pulses
        waveforms, waveform_seq = self.generate_waveforms(pulses)

        # Generate program without acquire instruction
        initial_delay = pulses[0].start
        start_last_pulse = pulses[-1].start
        # Acquire waveforms over remaining duration of acquisition of input vector of length = 16380 with integration weights 0,0
        acquire_instruction = "acquire   0,0,4"
        wait_time = self.duration_base - start_last_pulse - delay_before_readout - 4
        #wait_time = self.duration_base - initial_delay - delay_before_readout - 4 # FIXME: Not sure why this hardcoded 4 is needed
        #program = self.generate_program(nshots, initial_delay, delay_before_readout, acquire_instruction, wait_time, sequence.qrm_pulses)
        program = self.generate_program(nshots, initial_delay, delay_before_readout, acquire_instruction, wait_time, pulses, waveform_seq)
        print(f'QRM: {program}')
        return waveforms, program

    def play_sequence_and_acquire(self, ro_pulse):
        """Executes the uploaded instructions and retrieves the readout results.

        Args:
            ro_pulse (:class:`qibolab.pulses.Pulse`): Readout pulse to use for
                retrieving the results.
        """
        #arm sequencer and start playing sequence
        super().play_sequence()
        #start acquisition of data
        #Wait for the sequencer to stop with a timeout period of one minute.
        self.device.get_sequencer_state(0, 1)
        #Wait for the acquisition to finish with a timeout period of one second.
        self.device.get_acquisition_state(self.sequencer, 1)
        #Move acquisition data from temporary memory to acquisition list.
        self.device.store_scope_acquisition(self.sequencer, "single")
        #Get acquisition list from instrument.
        single_acq = self.device.get_acquisitions(self.sequencer)
        i, q = self._demodulate_and_integrate(single_acq, ro_pulse)
        acquisition_results = np.sqrt(i**2 + q**2), np.arctan2(q, i), i, q
        return acquisition_results

    def _demodulate_and_integrate(self, single_acq, ro_pulse):
        #DOWN Conversion
        norm_factor = 1. / (self.integration_length)
        n0 = self.start_sample
        n1 = self.start_sample + self.integration_length
        input_vec_I = np.array(single_acq["single"]["acquisition"]["scope"]["path0"]["data"][n0: n1])
        input_vec_Q = np.array(single_acq["single"]["acquisition"]["scope"]["path1"]["data"][n0: n1])
        input_vec_I -= np.mean(input_vec_I)
        input_vec_Q -= np.mean(input_vec_Q)

        if self.mode == 'ssb':
            modulated_i = input_vec_I
            modulated_q = input_vec_Q
            time = np.arange(modulated_i.shape[0])*1e-9
            cosalpha = np.cos(2 * np.pi * ro_pulse.frequency * time)
            sinalpha = np.sin(2 * np.pi * ro_pulse.frequency * time)
            demod_matrix = 2 * np.array([[cosalpha, -sinalpha], [sinalpha, cosalpha]])
            result = []
            for it, t, ii, qq in zip(np.arange(modulated_i.shape[0]), time,modulated_i, modulated_q):
                result.append(demod_matrix[:,:,it] @ np.array([ii, qq]))
            demodulated_signal = np.array(result)
            integrated_signal = norm_factor*np.sum(demodulated_signal,axis=0)

        elif self.mode == 'optimal':  # pragma: no cover
            raise_error(NotImplementedError, "Optimal Demodulation Mode not coded yet.")
        else:  # pragma: no cover
            raise_error(NotImplementedError, "Demodulation mode not understood.")
        return integrated_signal


class PulsarQCM(GenericPulsar):

    def __init__(self, label, ip, sequencer=0, ref_clock="external", sync_en=True, is_cluster=True):
        super().__init__(label, ip, sequencer, ref_clock, sync_en, is_cluster)
        # Instantiate base object from qblox library and connect to it
        self.name = "qcm"
        if self.is_cluster:
            from cluster.cluster import cluster_qcm
            self.Device = cluster_qcm
        else:
            from pulsar_qcm.pulsar_qcm import pulsar_qcm
            self.Device = pulsar_qcm
        self.connect()
        self.sequencer = sequencer
        # Reset and configure
        self.device.reset()
        self.device.reference_source(ref_clock)
        if self.sequencer == 1:
            self.device.sequencer1_sync_en(sync_en)
        else:
            self.device.sequencer0_sync_en(sync_en)

    def translate(self, sequence, delay_before_read_out, nshots=None):
        # Allocate only qubit pulses to PulsarQRM
        pulses = sequence.qcm_pulses
        waveforms, waveform_seq = self.generate_waveforms(pulses)

        # Generate program without acquire instruction
        initial_delay = pulses[0].start
        start_last_pulse = pulses[-1].start
        acquire_instruction = ""
        #wait_time = self.duration_base - initial_delay - delay_before_read_out
        wait_time = self.duration_base - start_last_pulse - delay_before_read_out
        #program = self.generate_program(nshots, initial_delay, delay_before_read_out, acquire_instruction, wait_time, sequence.qcm_pulses)
        program = self.generate_program(nshots, initial_delay, delay_before_read_out, acquire_instruction, wait_time, pulses, waveform_seq)
        print(f'QCM: {program}')
        return waveforms, program

class GenericModule(AbstractInstrument, ABC):
        
    def connect(self):
        pass
    def setup(self):
        pass    
    def start(self):
        pass
    def stop(self):
        pass
    def disconnect(self):
        pass

    def __del__(self):
        self.disconnect()


class ClusterQRM(AbstractInstrument):


    def __init__(self, name, ip):
        super().__init__(name, ip)

        self.sequencer_channel_map = {} 
        self.last_pulsequence_hash = ""
        self.current_pulsesequence_hash = ""

    def connect(self):
        """
        Connects to the instrument using the IP address set in the runcard.
        """
        if not self.is_connected:
            try:
                from cluster.cluster import cluster_qrm
                self.device = cluster_qrm(self.name, self.ip)
            except Exception as exc:
                raise InstrumentException(self, str(exc))
            self.is_connected = True
        else:
            raise_error(Exception,'There is an open connection to the instrument already')

    # Device Property Wrappers
    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, gain):
        #TODO: Add support for independent control gains for each sequencer / path
        self._gain = gain
        for n in range(self.device._num_sequencers):
            self.device.set(f"sequencer{n}_gain_awg_path0", gain)
            self.device.set(f"sequencer{n}_gain_awg_path1", gain)

    def setup(self, **kwargs):
        """
        Sets up the instrument using the parameters of the runcard. 
        A connection needs to be established before calling this method.
        
        Args:
            ref_clock (str): {'internal', 'external'} the source of the instrument clock
            sync_en (bool): {True, False} syncronise with other instruments
            hardware_avg_en: true
            acq_trigger_mode: sequencer
            gain (float): {0 .. 1} the gain applied to all sequencers on their output paths
            start_sample: 130
            integration_length: 1500
            mode: ssb
            channel_port_map (dict): a dictionary of {channel (int): port (str) {'o1', 'o2', ...}}
        """
        # Load settings as class attributes
        gain = kwargs.pop('gain')
        self.__dict__.update(kwargs)

        # Hardcoded values used to generate sequence program
        self.wait_loop_step = 1000
        self.waveform_max_length = 16384//2 # maximum length of the combination of waveforms, per sequencer, in number of samples (defined by the sequencer memory).

        if self.is_connected:
            # Reset
            self.device.reset()
            self.gain = gain
            # Configure clock source
            self.device.reference_source(self.ref_clock)
            for n in range(self.device._num_sequencers):
                # Configure the sequencers synchronization.
                self.device.set(f"sequencer{n}_sync_en", False)
                # Disable all sequencer - port connections
                for out in range(0, 4):
                    if hasattr(self.device, f"sequencer{n}_channel_map_path{out%2}_out{out}_en"):
                        self.device.set(f"sequencer{n}_channel_map_path{out%2}_out{out}_en", False)
                # The mapping of sequencers to ports is done in upload() as the number of sequencers needed 
                # can only be determined after examining the pulse sequence

            self.device.scope_acq_trigger_mode_path0(self.scope_acq_trigger_mode) # sets scope acquisition trigger mode for input path 0 (‘sequencer’ = triggered by sequencer, ‘level’ = triggered by input level).
            self.device.scope_acq_trigger_mode_path1(self.scope_acq_trigger_mode)
            sequencer = 0 # TODO: move to yaml?
            self.device.scope_acq_sequencer_select(sequencer) # specifies which sequencer triggers the scope acquisition when using sequencer trigger mode
            
            self.device.scope_acq_avg_mode_en_path0(self.scope_acq_avg_mode_en) # sets scope acquisition averaging mode enable for input path 0
            self.device.scope_acq_avg_mode_en_path1(self.scope_acq_avg_mode_en)

        else:
            raise_error(Exception,'There is no connection to the instrument')

    def _get_start(self, pulse):
        # This is a helper method used to sort a list of pulses by their start time
        return pulse.start

    def process_pulse_sequence(self, channel_pulses, nshots):
        """
        Processes a list of pulses, generating the waveforms and sequence program required by the instrument to synthesise them.
        
        Args:
        channel_pulses (dict): a dictionary of {channel (int): pulses (list)}
        nshots (int): the number of times the sequence of pulses will be repeated
        """
        channels = self.channel_port_map.keys()

        # Check if the sequence to be processed is the same as the last one. If so, there is no need to generate waveforms and program
        self.current_pulsesequence_hash = ""
        for channel in channels:
            for pulse in channel_pulses[channel]:
                self.current_pulsesequence_hash += pulse.serial
        if not self.current_pulsesequence_hash == self.last_pulsequence_hash:

            # Sort pulses by their start time 
            for channel in channels:
                channel_pulses[channel].sort(key=self._get_start) 
            
            # Check if pulses on the same channel overlap
            for channel in channels:
                for m in range(1, len(channel_pulses[channel])):
                    channel_pulses[channel][m].overlaps = []
                    for n in range(m):
                        if channel_pulses[channel][m].start - channel_pulses[channel][n].start < channel_pulses[channel][n].duration:
                            channel_pulses[channel][m].overlaps.append(n)
                for m in range(1, len(channel_pulses[channel])):
                    if len(pulse.overlaps) > 0:
                        # TODO: Urgently needed in order to implement multiplexed readout
                        raise_error(NotImplementedError, "Overlaping pulses on the same channel are not yet supported.")

            # Allocate channel pulses to sequencers. 
            #   Each channel is mapped to a sequencer
            #   If the memory required for the sequence of pulses exceeds 16384/2, it raises an error

            self.sequencers = []        # a list of sequencers (int) used
            sequencer_pulses = {}
            sequencer = -1              # initialised to -1 so that in the first iteration it becomes 0, the first sequencer number
            for channel in channels:
                sequencer += 1
                self.sequencers.append(sequencer)
                self.sequencer_channel_map[sequencer]=channel
                sequencer_pulses[sequencer]=[]
                waveforms_length = 0
                wc = 0                  # unique waveform counter
                unique_pulses = {}      # a dictionary of {unique pulse IDs (str): and their I & Q indices (int)}
                
                # Iterate over the list of pulses to check if they are unique and if the overal length of the waveform exceeds the memory available
                for pulse in channel_pulses[channel]:
                    sequencer_pulses[sequencer].append(pulse)
                    pulse_serial = pulse.serial[pulse.serial.find(',',pulse.serial.find(',')+1)+2:-1] # removes the channel and start information from Pulse.serial to compare between pulses
                    if pulse_serial not in unique_pulses.keys():
                        # If the pulse is unique (it hasn't been saved before):
                        # Mark it as unique
                        pulse.is_unique = True
                        # Check if the overall length of the waveform exceeds memory size (waveform_max_length)
                        waveforms_length += pulse.duration
                        if waveforms_length <= self.waveform_max_length:
                            # If not, add the pulse to the list of pulses for the current sequencer and save the waveform indexes
                            pulse.waveform_indexes = [0 + wc, 1 + wc]
                            unique_pulses[pulse_serial] =  [0 + wc, 1 + wc]
                            wc += 2
                        else:
                            if waveforms_length > self.waveform_max_length:
                                raise_error(NotImplementedError, f"The sequence of pulses is longer than the memory available {self.waveform_max_length}. This is not supported yet.")
                    
                    else:
                        pulse.is_unique = False
                        pulse.waveform_indexes = unique_pulses[pulse_serial]

            # Generate waveforms and programs for each sequencer
            # FIXME: If pulses[sequencer] == [] skip, same for QCM
            pulses = sequencer_pulses
            self.waveforms = {}
            self.acquisitions = {}
            self.program = {}
            self.weights = {}

            for sequencer in self.sequencers:
                # Waveforms
                self.waveforms[sequencer] = {}
                for pulse in pulses[sequencer]:
                    if pulse.is_unique:
                        # If the pulse is unique, generate the waveforms and save them to the dictionary
                        I, Q = self.generate_waveforms_from_pulse(pulse)
                        self.waveforms[sequencer][f"{self.name}_{pulse}_pulse_I"] = {"data": I, "index": pulse.waveform_indexes[0]}
                        self.waveforms[sequencer][f"{self.name}_{pulse}_pulse_Q"] = {"data": Q, "index": pulse.waveform_indexes[1]}
                        # If it isn't unique, there is nothing to be done, it was already saved.
                
                # Acquisitions & weights
                self.acquisitions[sequencer] = {}
                self.weights[sequencer] = {}
                ac = 0 # Acquisition counter
                for pulse in pulses[sequencer]:
                    if pulse.type == 'ro':
                        pulse.acquisition_index = ac
                        pulse.num_bins = 1
                        self.acquisitions[sequencer][pulse.serial] = {"num_bins": pulse.num_bins, "index":pulse.acquisition_index}
                        ac += 1

                # Program
                sequence_total_duration = pulses[sequencer][-1].start + pulses[sequencer][-1].duration + self.minimum_delay_between_instructions # the minimum delay between instructions is 4ns
                time_between_repetitions = self.repetition_duration - sequence_total_duration
                assert time_between_repetitions > 0
                extra_wait = time_between_repetitions % self.wait_loop_step
                num_wait_loops = (time_between_repetitions - extra_wait) // self. wait_loop_step

                header = f"""
                move {nshots},R0 # nshots
                nop
                wait_sync {self.minimum_delay_between_instructions}
                loop:"""
                body = ""

                footer = f"""
                    # wait {time_between_repetitions} ns"""
                if num_wait_loops > 0:
                    footer += f"""
                    move {num_wait_loops},R1
                    nop
                    waitloop:
                        wait {self.wait_loop_step}
                        loop R1,@waitloop"""
                footer += f"""
                    wait {extra_wait}
                loop R0,@loop
                stop 
                """

                # Add an initial wait instruction for the first pulse of the sequence
                if pulses[sequencer][0].start != 0:
                        initial_wait_instruction = f"\t\t\twait {pulses[sequencer][0].start}"
                else:
                    initial_wait_instruction = "\t\t\t# wait 0"
                body += "\n" + initial_wait_instruction

                for n in range(len(pulses[sequencer])):
                    if pulses[sequencer][n].type == 'ro':
                        delay_after_play = self.start_sample - self.minimum_delay_between_instructions
                        
                        if len(pulses[sequencer]) > n + 1:
                            # If there are more pulses to be played, the delay is the time between the pulse end and the next pulse start
                            delay_after_acquire = pulses[sequencer][n + 1].start - pulses[sequencer][n].start - self.start_sample
                        else:
                            delay_after_acquire = sequence_total_duration - pulses[sequencer][n].start - self.start_sample
                            
                        if delay_after_acquire < self.minimum_delay_between_instructions:
                                raise_error(Exception, f"The minimum delay before starting acquisition is {self.minimum_delay_between_instructions}ns.")
                        
                        # Prepare play instruction: play arg0, arg1, arg2. 
                        #   arg0 is the index of the I waveform 
                        #   arg1 is the index of the Q waveform
                        #   arg2 is the delay between starting the instruction and the next instruction
                        play_instruction = f"\t\t\tplay {pulses[sequencer][n].waveform_indexes[0]},{pulses[sequencer][n].waveform_indexes[1]},{delay_after_play}"
                        # Add the serial of the pulse as a comment
                        play_instruction += " "*(26-len(play_instruction)) + f"# play waveforms {pulses[sequencer][n]}" 
                        body += "\n" + play_instruction

                        # Prepare acquire instruction: acquire arg0, arg1, arg2. 
                        #   arg0 is the index of the acquisition 
                        #   arg1 is the index of the data bin
                        #   arg2 is the delay between starting the instruction and the next instruction
                        acquire_instruction = f"\t\t\tacquire {pulses[sequencer][n].acquisition_index},0,{delay_after_acquire}"
                        # Add the serial of the pulse as a comment
                        body += "\n" + acquire_instruction

                    else:
                        # Calculate the delay_after_play that is to be used as an argument to the play instruction
                        if len(pulses[sequencer]) > n + 1:
                            # If there are more pulses to be played, the delay is the time between the pulse end and the next pulse start
                            delay_after_play = pulses[sequencer][n + 1].start - pulses[sequencer][n].start
                        else:
                            delay_after_play = sequence_total_duration - pulses[sequencer][n].start
                            
                        if delay_after_play < self.minimum_delay_between_instructions:
                                raise_error(Exception, f"The minimum delay between pulses is {self.minimum_delay_between_instructions}ns.")
                        
                        # Prepare play instruction: play arg0, arg1, arg2. 
                        #   arg0 is the index of the I waveform 
                        #   arg1 is the index of the Q waveform
                        #   arg2 is the delay between starting the instruction and the next instruction
                        play_instruction = f"\t\t\tplay {pulses[sequencer][n].waveform_indexes[0]},{pulses[sequencer][n].waveform_indexes[1]},{delay_after_play}"
                        # Add the serial of the pulse as a comment
                        play_instruction += " "*(26-len(play_instruction)) + f"# play waveforms {pulses[sequencer][n]}" 
                        body += "\n" + play_instruction

                self.program[sequencer] = header + body + footer

                # DEBUG:
                # print(f"{self.name} sequencer {sequencer} program:\n" + self.program[sequencer]) 


    def generate_waveforms_from_pulse(self, pulse, modulate = True):
        """
        Generates I & Q waveforms for the pulse passed as a parameter.

        Args:
        pulse (Pulse): a Pulse object
        modulate (bool): {True, False} module the waveform with the pulse frequency

        Returns:
        envelope_i (list) of (float), envelope_q (list) of (float): a tuple with I & Q waveform samples
        """
        envelope_i = pulse.envelope_i
        envelope_q = pulse.envelope_q # np.zeros(int(pulse.duration))
        assert len(envelope_i) == len(envelope_q)
        if modulate:
            time = np.arange(pulse.duration) * 1e-9
            # FIXME: There should be a simpler way to construct this array
            cosalpha = np.cos(2 * np.pi * pulse.frequency * time + pulse.phase)
            sinalpha = np.sin(2 * np.pi * pulse.frequency * time + pulse.phase)
            mod_matrix = np.array([[cosalpha,sinalpha], [-sinalpha,cosalpha]])
            result = []
            for it, t, ii, qq in zip(np.arange(pulse.duration), time, envelope_i, envelope_q):
                result.append(mod_matrix[:, :, it] @ np.array([ii, qq]))
            mod_signals = np.array(result)
            return mod_signals[:, 0] + pulse.offset_i, mod_signals[:, 1] + pulse.offset_q
        else:
            return envelope_i, envelope_q
            # TODO: Check if offsets are to be added to the envelopes

    def upload(self):
        """Uploads waveforms and programs all sequencers and arms them in preparation for execution."""
        if not self.current_pulsesequence_hash == self.last_pulsequence_hash:
            self.last_pulsequence_hash = self.current_pulsesequence_hash
            import os
            # TODO: dont upload if the same 
            # Upload waveforms and program
            qblox_dict = {}
            data_folder = self.data_folder
            for sequencer in self.sequencers:
                # Reformat waveforms to lists
                for name, waveform in self.waveforms[sequencer].items():
                    if isinstance(waveform["data"], np.ndarray):
                        self.waveforms[sequencer][name]["data"] = self.waveforms[sequencer][name]["data"].tolist()  # JSON only supports lists

                # Add sequence program and waveforms to single dictionary and write to JSON file
                filename = f"{data_folder}/{self.name}_sequencer{sequencer}_sequence.json"
                qblox_dict[sequencer] = {
                    "waveforms": self.waveforms[sequencer],
                    "weights": {}, #self.weights,
                    "acquisitions": self.acquisitions[sequencer],
                    "program": self.program[sequencer]
                    }
                if not os.path.exists(data_folder):
                    os.makedirs(data_folder)
                with open(filename, "w", encoding="utf-8") as file:
                    json.dump(qblox_dict[sequencer], file, indent=4)

                # Route sequencers to specific outputs.

                port = int(self.channel_port_map[self.sequencer_channel_map[sequencer]][1:])-1
                if hasattr(self.device, f"sequencer{sequencer}_channel_map_path0_out{2*port}_en"):
                    self.device.set(f"sequencer{sequencer}_channel_map_path0_out{2*port}_en", True)
                    self.device.set(f"sequencer{sequencer}_channel_map_path1_out{2*port+1}_en", True)
                else:
                    raise_error(Exception, f"The device does not have port {port}")
                
                self.device.set(f"sequencer{sequencer}_sync_en", self.sync_en)


                # Upload json file to the device sequencers
                self.device.set(f"sequencer{sequencer}_waveforms_and_program", os.path.join(os.getcwd(), filename))
        # Arm all sequencers
        self.device.arm_sequencer()
        # DEBUG:
        # print(self.name)
        # self.device.print_readable_snapshot(update=True)

    def play_sequence(self):
        """Executes the sequence of instructions."""
        # Start playing sequences in all sequencers
        self.device.start_sequencer()

    def play_sequence_and_acquire(self):
        """Executes the sequence of instructions and retrieves the readout results.
        """
        # Start playing sequences in all sequencers
        self.device.start_sequencer()
        
        acquisition_results = {}
        # Retrieve data
        for sequencer in self.sequencers:
            # Wait for the sequencer to stop with a timeout period of one minute.
            self.device.get_sequencer_state(sequencer, timeout = 1)
            #Wait for the acquisition to finish with a timeout period of one second.
            self.device.get_acquisition_state(sequencer, timeout = 1)
            acquisition_results[sequencer] = {}
            for acquisition in self.acquisitions[sequencer]:
                #Move acquisition data from temporary memory to acquisition list.
                self.device.store_scope_acquisition(sequencer, acquisition)
                #Get acquisitions from instrument.
                raw_results = self.device.get_acquisitions(sequencer)
                i, q = self._demodulate_and_integrate(raw_results, acquisition)
                acquisition_results[sequencer][acquisition] = np.sqrt(i**2 + q**2), np.arctan2(q, i), i, q
        
        return acquisition_results

    def _demodulate_and_integrate(self, raw_results, acquisition):
        acquisition_name = acquisition
        acquisition_frequency = 20_000_000
        #TODO: obtain from acquisition info
        #DOWN Conversion
        norm_factor = 1. / (self.integration_length)
        n0 = 0 # self.start_sample
        n1 = self.integration_length # self.start_sample + self.integration_length
        input_vec_I = np.array(raw_results[acquisition_name]["acquisition"]["scope"]["path0"]["data"][n0: n1])
        input_vec_Q = np.array(raw_results[acquisition_name]["acquisition"]["scope"]["path1"]["data"][n0: n1])
        input_vec_I -= np.mean(input_vec_I)
        input_vec_Q -= np.mean(input_vec_Q)

        if self.mode == 'ssb':
            modulated_i = input_vec_I
            modulated_q = input_vec_Q
            time = np.arange(modulated_i.shape[0])*1e-9
            cosalpha = np.cos(2 * np.pi * acquisition_frequency * time)
            sinalpha = np.sin(2 * np.pi * acquisition_frequency * time)
            demod_matrix = 2 * np.array([[cosalpha, -sinalpha], [sinalpha, cosalpha]])
            result = []
            for it, t, ii, qq in zip(np.arange(modulated_i.shape[0]), time,modulated_i, modulated_q):
                result.append(demod_matrix[:,:,it] @ np.array([ii, qq]))
            demodulated_signal = np.array(result)
            integrated_signal = norm_factor*np.sum(demodulated_signal,axis=0)

        elif self.mode == 'optimal':
            raise_error(NotImplementedError, "Optimal Demodulation Mode not coded yet.")
        else:
            raise_error(NotImplementedError, "Demodulation mode not understood.")
        return integrated_signal


    def start(self):
        pass

    def stop(self):
        """Stops all sequencers"""
        self.device.stop_sequencer()

    def disconnect(self):
        """Disconnects from the instrument."""
        if self.is_connected:
            self.stop()
            self.device.close()
            self.is_connected = False
    
    def __del__(self):
        self.disconnect()




class ClusterQCM(AbstractInstrument):

    def __init__(self, name, ip):
        super().__init__(name, ip)

        self.sequencer_channel_map = {} 
        self.last_pulsequence_hash = ""
        self.current_pulsesequence_hash = ""

    def connect(self):
        """
        Connects to the instrument using the IP address set in the runcard.
        """
        if not self.is_connected:
            try:
                from cluster.cluster import cluster_qcm
                self.device = cluster_qcm(self.name, self.ip)
            except Exception as exc:
                raise InstrumentException(self, str(exc))
            self.is_connected = True
        else:
            raise_error(Exception,'There is an open connection to the instrument already')

    # Device Property Wrappers
    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, gain):
        #TODO: Add support for independent control gains for each sequencer / path
        self._gain = gain
        for n in range(self.device._num_sequencers):
            self.device.set(f"sequencer{n}_gain_awg_path0", gain)
            self.device.set(f"sequencer{n}_gain_awg_path1", gain)

    def setup(self, **kwargs):
        """
        Sets up the instrument using the parameters of the runcard. 
        A connection needs to be established before calling this method.
        
        Args:
            ref_clock (str): {'internal', 'external'} the source of the instrument clock
            sync_en (bool): {True, False} syncronise with other instruments
            gain (float): {0 .. 1} the gain applied to all sequencers on their output paths
            channel_port_map (dict): a dictionary of {channel (int): port (str) {'o1', 'o2', ...}}
        """
        # Load settings as class attributes
        gain = kwargs.pop('gain')
        self.__dict__.update(kwargs)

        # Hardcoded values used to generate sequence program
        self.wait_loop_step = 1000
        self.waveform_max_length = 16384//2 # maximum length of the combination of waveforms, per sequencer, in number of samples (defined by the sequencer memory).

        if self.is_connected:
            # Reset
            self.device.reset()
            self.gain = gain
            # Configure clock source
            self.device.reference_source(self.ref_clock)
            for n in range(self.device._num_sequencers):
                # Configure the sequencers synchronization.
                self.device.set(f"sequencer{n}_sync_en", False)
                # Disable all sequencer - port connections
                for out in range(0, 4):
                    if hasattr(self.device, f"sequencer{n}_channel_map_path{out%2}_out{out}_en"):
                        self.device.set(f"sequencer{n}_channel_map_path{out%2}_out{out}_en", False)
                # The mapping of sequencers to ports is done in upload() as the number of sequencers needed 
                # can only be determined after examining the pulse sequence
        else:
            raise_error(Exception,'There is no connection to the instrument')

    def _get_start(self, pulse):
        # This is a helper method used to sort a list of pulses by their start time
        return pulse.start

    def process_pulse_sequence(self, channel_pulses, nshots):
        """
        Processes a list of pulses, generating the waveforms and sequence program required by the instrument to synthesise them.
        
        Args:
        channel_pulses (dict): a dictionary of {channel (int): pulses (list)}
        nshots (int): the number of times the sequence of pulses will be repeated
        """
        channels = self.channel_port_map.keys()

        # Check if the sequence to be processed is the same as the last one. If so, there is no need to generate waveforms and program
        self.current_pulsesequence_hash = ""
        for channel in channels:
            for pulse in channel_pulses[channel]:
                self.current_pulsesequence_hash += pulse.serial
        if not self.current_pulsesequence_hash == self.last_pulsequence_hash:

            # Sort pulses by their start time 
            for channel in channels:
                channel_pulses[channel].sort(key=self._get_start) 
            
            # Check if pulses on the same channel overlap
            for channel in channels:
                for m in range(1, len(channel_pulses[channel])):
                    channel_pulses[channel][m].overlaps = []
                    for n in range(m):
                        if channel_pulses[channel][m].start - channel_pulses[channel][n].start < channel_pulses[channel][n].duration:
                            channel_pulses[channel][m].overlaps.append(n)
                for m in range(1, len(channel_pulses[channel])):
                        raise_error(NotImplementedError, "Overlaping pulses on the same channel are not yet supported.")
            

            # Allocate channel pulses to sequencers. 
            #   At least one sequencer is needed for each channel
            #   If the memory required for the sequence of pulses exceeds 16384/2, 
            #   additional sequencers are used to synthesise it
            #   TODO: Improve the implementation to support:
            #           - pulses that span more than two sequencers
            #           - avoid pulse splitting if possible

            self.sequencers = []        # a list of sequencers (int) used
            sequencer_pulses = {}       # a dictionary of {sequencer (int): pulses (list)}     
            sequencer = -1              # initialised to -1 so that in the first iteration it becomes 0, the first sequencer number
            waveforms_length = 0        # accumulates the length of unique pulses per sequencer
            
            for channel in channels:
                # Select a sequencer and add it to the sequencer_channel_map
                sequencer += 1
                self.sequencers.append(sequencer)
                self.sequencer_channel_map[sequencer]=channel
                sequencer_pulses[sequencer]=[]
                wc = 0                  # unique waveform counter
                unique_pulses = {}      # a dictionary of {unique pulse IDs (str): and their I & Q indices (int)}
                
                # Iterate over the list of pulses to check if they are unique and if the overal length of the waveform exceeds the memory available
                for pulse in channel_pulses[channel]:
                    pulse_serial = pulse.serial[pulse.serial.find(',',pulse.serial.find(',')+1)+2:-1] # removes the channel and start information from Pulse.serial to compare between pulses
                    if pulse_serial not in unique_pulses.keys():
                        # If the pulse is unique (it hasn't been saved before):
                        # Mark it as unique
                        pulse.is_unique = True
                        # Check if the overall length of the waveform exceeds memory size (waveform_max_length)
                        waveforms_length += pulse.duration
                        if waveforms_length <= self.waveform_max_length:
                            # If not, add the pulse to the list of pulses for the current sequencer and save the waveform indexes
                            pulse.is_split = False
                            sequencer_pulses[sequencer].append(pulse)
                            pulse.waveform_indexes = [0 + wc, 1 + wc]
                            unique_pulses[pulse_serial] =  [0 + wc, 1 + wc]
                            wc += 2
                        else:
                            # The pulse needs to be split
                            # This implementation only supports the pulse to be split between two sequencers
                            #   first part of the pulse (up to split_point) is played in the current sequencer
                            #   the second part is played in a new sequencer. The length of the second part 
                            #   should not exceed waveform_max_length.

                            # Mark the pulse to be split, save the splitting point and add the pulse to the current sequencer (that will play only the first part)
                            pulse.is_split = True
                            pulse.split_point = pulse.duration - (waveforms_length - self.waveform_max_length)
                            sequencer_pulses[sequencer].append(pulse)
                            pulse.waveform_indexes = [0 + wc, 1 + wc]
                            wc += 2

                            # Select a new sequencer
                            sequencer += 1
                            if sequencer > self.device._num_sequencers:
                                    raise_error(Exception, f"The number of sequencers requried to play the sequence exceeds the number available {self.device._num_sequencers}.")
                            # Initialise the corresponding variables 
                            self.sequencers.append(sequencer)
                            self.sequencer_channel_map[sequencer]=channel
                            sequencer_pulses[sequencer]=[]

                            unique_pulses = {}

                            # Add the pulse to the new sequencer so that it plays its second part
                            sequencer_pulses[sequencer].append(pulse)
                            waveforms_length = waveforms_length - self.waveform_max_length
                            if waveforms_length > self.waveform_max_length:
                                raise_error(NotImplementedError, f"The pulse being played spans more than 2 sequencers. Not implemented yet.")
                    
                    else:
                        pulse.is_unique = False
                        pulse.is_split = False
                        sequencer_pulses[sequencer].append(pulse)
                        pulse.waveform_indexes = unique_pulses[pulse_serial]

            # Generate waveforms and programs for each sequencer
            pulses = sequencer_pulses
            self.waveforms = {}
            self.program = {}

            for sequencer in self.sequencers:
                # Waveforms
                self.waveforms[sequencer] = {}
                for pulse in pulses[sequencer]:
                    if pulse.is_unique:
                        # If the pulse is unique, generate the waveforms and save them to the dictionary
                        I, Q = self.generate_waveforms_from_pulse(pulse)
                        if not pulse.is_split:
                            self.waveforms[sequencer][f"{self.name}_{pulse}_pulse_I"] = {"data": I, "index": pulse.waveform_indexes[0]}
                            self.waveforms[sequencer][f"{self.name}_{pulse}_pulse_Q"] = {"data": Q, "index": pulse.waveform_indexes[1]}
                        else:
                            if pulse == pulses[sequencer][0]: # First pulse in the sequence
                                self.waveforms[sequencer][f"{self.name}_{pulse}_2nd_pulse_I"] = {"data": I[pulse.split_point:], "index": pulse.waveform_indexes[0]}
                                self.waveforms[sequencer][f"{self.name}_{pulse}_2nd_pulse_Q"] = {"data": Q[pulse.split_point:], "index": pulse.waveform_indexes[1]}
                            elif pulse == pulses[sequencer][-1]: # Last pulse in the sequence
                                self.waveforms[sequencer][f"{self.name}_{pulse}_1st_pulse_I"] = {"data": I[:pulse.split_point], "index": pulse.waveform_indexes[0]}
                                self.waveforms[sequencer][f"{self.name}_{pulse}_1st_pulse_Q"] = {"data": Q[:pulse.split_point], "index": pulse.waveform_indexes[1]}
                        
                        # If it isn't unique, there is nothing to be done, it was already saved.
                
                # Program
                # TODO: the sequence_total_duration should be corrected if the last pulse of the sequence is_split, 
                # but it has no impact as a split pulse is expected to be the last in the sequencer sequence and the time_between_repetitions compensates the difference.
                sequence_total_duration = pulses[sequencer][-1].start + pulses[sequencer][-1].duration + self.minimum_delay_between_instructions # the minimum delay between instructions is 4ns
                time_between_repetitions = self.repetition_duration - sequence_total_duration
                assert time_between_repetitions > 0
                extra_wait = time_between_repetitions % self.wait_loop_step
                num_wait_loops = (time_between_repetitions - extra_wait) // self. wait_loop_step

                header = f"""
                move {nshots},R0 # nshots
                nop
                wait_sync {self.minimum_delay_between_instructions}
                loop:"""
                body = ""

                footer = f"""
                    # wait {time_between_repetitions} ns"""
                if num_wait_loops > 0:
                    footer += f"""
                    move {num_wait_loops},R1
                    nop
                    waitloop:
                        wait {self.wait_loop_step}
                        loop R1,@waitloop"""
                footer += f"""
                    wait {extra_wait}
                loop R0,@loop
                stop 
                """

                # Add an initial wait instruction for the first pulse of the sequence
                if pulses[sequencer][0].start != 0:
                    if not pulses[sequencer][0].is_split:
                        initial_wait_instruction = f"\t\t\twait {pulses[sequencer][0].start}"
                    else:
                        initial_wait_instruction = f"\t\t\twait {pulses[sequencer][0].start + pulses[sequencer][0].split_point}"
                else:
                    initial_wait_instruction = "\t\t\t# wait 0"
                body += "\n" + initial_wait_instruction

                for n in range(len(pulses[sequencer])):
                    # Calculate the delay_after_play that is to be used as an argument to the play instruction
                    if not pulses[sequencer][n].is_split or n > 0:
                        if len(pulses[sequencer]) > n + 1:
                            # If there are more pulses to be played, the delay is the time between the pulse end and the next pulse start
                            delay_after_play = pulses[sequencer][n + 1].start - pulses[sequencer][n].start
                        else:
                            delay_after_play = sequence_total_duration - pulses[sequencer][n].start
                    else:
                        # If it is split and the first pulse in the sequence:
                        if len(pulses[sequencer]) > n + 1:
                            # If there are more pulses to be played, the delay is the time between the pulse end and the next pulse start
                            delay_after_play = pulses[sequencer][n + 1].start - pulses[sequencer][n].start - pulses[sequencer][n].split_point
                        else:
                            delay_after_play = sequence_total_duration - pulses[sequencer][n].start - pulses[sequencer][n].split_point

                        
                        
                    if delay_after_play < self.minimum_delay_between_instructions:
                            raise_error(Exception, f"The minimum delay between pulses is {self.minimum_delay_between_instructions}ns.")
                    # Prepare play instruction: play arg0, arg1, arg2. 
                    #   arg0 is the index of the I waveform 
                    #   arg1 is the index of the Q waveform
                    #   arg2 is the delay between starting the instruction and the next instruction
                    play_instruction = f"\t\t\tplay {pulses[sequencer][n].waveform_indexes[0]},{pulses[sequencer][n].waveform_indexes[1]},{delay_after_play}"
                    # Add the serial of the pulse as a comment
                    play_instruction += " "*(26-len(play_instruction)) + f"# play waveforms {pulses[sequencer][n]}" 
                    body += "\n" + play_instruction

                self.program[sequencer] = header + body + footer

                # DEBUG:
                # print(f"{self.name} sequencer {sequencer} program:\n" + self.program[sequencer]) 


    def generate_waveforms_from_pulse(self, pulse, modulate = True):
        """
        Generates I & Q waveforms for the pulse passed as a parameter.

        Args:
        pulse (Pulse): a Pulse object
        modulate (bool): {True, False} module the waveform with the pulse frequency

        Returns:
        envelope_i (list) of (float), envelope_q (list) of (float): a tuple with I & Q waveform samples
        """
        envelope_i = pulse.envelope_i
        envelope_q = pulse.envelope_q # np.zeros(int(pulse.duration))
        assert len(envelope_i) == len(envelope_q)
        if modulate:
            time = np.arange(pulse.duration) * 1e-9
            # FIXME: There should be a simpler way to construct this array
            cosalpha = np.cos(2 * np.pi * pulse.frequency * time + pulse.phase)
            sinalpha = np.sin(2 * np.pi * pulse.frequency * time + pulse.phase)
            mod_matrix = np.array([[cosalpha,sinalpha], [-sinalpha,cosalpha]])
            result = []
            for it, t, ii, qq in zip(np.arange(pulse.duration), time, envelope_i, envelope_q):
                result.append(mod_matrix[:, :, it] @ np.array([ii, qq]))
            mod_signals = np.array(result)
            return mod_signals[:, 0] + pulse.offset_i, mod_signals[:, 1] + pulse.offset_q
        else:
            return envelope_i, envelope_q
            # TODO: Check if offsets are to be added to the envelopes

    def upload(self):
        """Uploads waveforms and programs all sequencers and arms them in preparation for execution."""
        if not self.current_pulsesequence_hash == self.last_pulsequence_hash:
            self.last_pulsequence_hash = self.current_pulsesequence_hash
            import os
            # TODO: dont upload if the same 
            # Upload waveforms and program
            qblox_dict = {}
            data_folder = self.data_folder
            for sequencer in self.sequencers:
                # Reformat waveforms to lists
                for name, waveform in self.waveforms[sequencer].items():
                    if isinstance(waveform["data"], np.ndarray):
                        self.waveforms[sequencer][name]["data"] = self.waveforms[sequencer][name]["data"].tolist()  # JSON only supports lists

                # Add sequence program and waveforms to single dictionary and write to JSON file
                filename = f"{data_folder}/{self.name}_sequencer{sequencer}_sequence.json"
                qblox_dict[sequencer] = {
                    "waveforms": self.waveforms[sequencer],
                    "weights": {}, #self.weights,
                    "acquisitions": {}, #self.acquisitions,
                    "program": self.program[sequencer]
                    }
                if not os.path.exists(data_folder):
                    os.makedirs(data_folder)
                with open(filename, "w", encoding="utf-8") as file:
                    json.dump(qblox_dict[sequencer], file, indent=4)

                # Route sequencers to specific outputs.

                port = int(self.channel_port_map[self.sequencer_channel_map[sequencer]][1:])-1
                if hasattr(self.device, f"sequencer{sequencer}_channel_map_path0_out{2*port}_en"):
                    self.device.set(f"sequencer{sequencer}_channel_map_path0_out{2*port}_en", True)
                    self.device.set(f"sequencer{sequencer}_channel_map_path1_out{2*port+1}_en", True)
                else:
                    raise_error(Exception, f"The device does not have port {port}")
                
                # Configure syncronisation
                self.device.set(f"sequencer{sequencer}_sync_en", self.sync_en)

                # Upload json file to the device sequencers
                self.device.set(f"sequencer{sequencer}_waveforms_and_program", os.path.join(os.getcwd(), filename))
        # Arm all sequencers
        self.device.arm_sequencer()
        # DEBUG:
        # print(self.name)
        # self.device.print_readable_snapshot(update=True)

    def play_sequence(self):
        """Executes the sequence of instructions."""
        # Start playing sequences in all sequencers
        self.device.start_sequencer()

    def start(self):
        pass

    def stop(self):
        """Stops all sequencers"""
        self.device.stop_sequencer()

    def disconnect(self):
        """Disconnects from the instrument."""
        if self.is_connected:
            self.stop()
            self.device.close()
            self.is_connected = False
    
    def __del__(self):
        self.disconnect()




class ClusterQRM_blank(AbstractInstrument):

    def connect(self):
        pass
    def setup(self, **kwargs):
        self.__dict__.update(kwargs)
    def start(self):
        pass
    def stop(self):
        pass
    def disconnect(self):
        pass
    def process_pulse_sequence(self, channel_pulses, nshots):
        pass
    def upload(self):
        pass
    def play_sequence_and_acquire(self):
        pass
    def __del__(self):
        self.disconnect()





class ClusterQCMRF(AbstractInstrument):

    def connect(self):
        pass
    def setup(self, **kwargs):
        self.__dict__.update(kwargs)
    def start(self):
        pass
    def stop(self):
        pass
    def disconnect(self):
        pass
    
    def __del__(self):
        self.disconnect()

class ClusterQRMRF(AbstractInstrument):

    def connect(self):
        pass
    def setup(self, **kwargs):
        self.__dict__.update(kwargs)
    def start(self):
        pass
    def stop(self):
        pass
    def disconnect(self):
        pass    

    def __del__(self):
        self.disconnect()
