import json
import numpy as np
import qblox_instruments
from qibolab.instruments.abstract import AbstractInstrument, InstrumentException


class Cluster(AbstractInstrument):
    def __init__(self, name, address):
        super().__init__(name, address)

    def connect(self):
        global cluster
        if not self.is_connected:
            for attempt in range(3):
                try:
                    qblox_instruments.Cluster.close_all()
                    self.device = qblox_instruments.Cluster(self.name, self.address)
                    self.device.reset()
                    cluster = self.device
                    self.is_connected = True
                    # DEBUG: Print Cluster Status                
                    # print(self.device.get_system_status())
                    break
                except Exception as exc:
                    print(f"Unable to connect:\n{str(exc)}\nRetrying...")
            if not self.is_connected:
                raise InstrumentException(self, f'Unable to connect to {self.name}')
            
    def setup(self, **kwargs):
        self.reference_clock_source = kwargs['reference_clock_source']
        self.device.reference_source(self.reference_clock_source)

    def start(self):
        pass

    def stop(self):
        pass

    def disconnect(self):
        qblox_instruments.Cluster.close_all()
        global cluster
        cluster = None


cluster : Cluster = None


class ClusterQRM_RF(AbstractInstrument):
    """
    Qblox Cluster Qubit Readout Module RF driver.
    
    Args:
        name (str): unique name given to the instrument
        address (str): IP_address:module_number

    """
    DEFAULT_SEQUENCER = 0

    property_wrapper = lambda parent, *parameter: property(lambda self: parent.device.get(parameter[0]), lambda self,x: parent.set_device_parameter(parent.device, *parameter, value = x))
    sequencer_property_wrapper = lambda parent, sequencer, *parameter: property(lambda self: parent.device.sequencers[sequencer].get(parameter[0]), lambda self,x: parent.set_device_parameter(parent.device.sequencers[sequencer], *parameter, value = x))
    

    def __init__(self, name, address):
        super().__init__(name, address)
        self.sequencers = []
        self.sequencer_channel_map = {} 
        self.last_pulsequence_hash = "uninitialised"
        self.current_pulsesequence_hash = ""
        self.device_parameters = {}
        self.ports = {}

    def connect(self):
        """
        Connects to the instrument using the IP address set in the runcard.
        """
        global cluster
        if not self.is_connected:
            if cluster:
                self.device = cluster.modules[int(self.address.split(':')[1])-1]

                self.ports['o1'] = type(f'port_o1', (), 
                    {'attenuation': self.property_wrapper('out0_att'), 
                     'lo_enabled': self.property_wrapper('out0_in0_lo_en'), 
                     'lo_frequency': self.property_wrapper('out0_in0_lo_freq'), 
                     'gain': self.sequencer_property_wrapper(self.DEFAULT_SEQUENCER, 'gain_awg_path0', 'gain_awg_path1'), 
                     'hardware_mod_en': self.sequencer_property_wrapper(self.DEFAULT_SEQUENCER, 'mod_en_awg'),
                     'nco_freq': self.sequencer_property_wrapper(self.DEFAULT_SEQUENCER, 'nco_freq'),
                     'nco_phase_offs': self.sequencer_property_wrapper(self.DEFAULT_SEQUENCER, 'nco_phase_offs')
                    })()

                self.ports['i1'] = type(f'port_i1', (), 
                    {'hardware_demod_en': self.sequencer_property_wrapper(self.DEFAULT_SEQUENCER, 'demod_en_acq')
                    })()

                setattr(self.__class__, 'acquisition_duration', self.sequencer_property_wrapper(self.DEFAULT_SEQUENCER, 'integration_length_acq'))
                setattr(self.__class__, 'discretization_threshold_acq', self.sequencer_property_wrapper(self.DEFAULT_SEQUENCER, 'discretization_threshold_acq'))
                setattr(self.__class__, 'phase_rotation_acq', self.sequencer_property_wrapper(self.DEFAULT_SEQUENCER, 'phase_rotation_acq'))

                self.cluster = cluster
                self.is_connected = True
                
                self.set_device_parameter(self.device, 'in0_att', value = 0) 
                self.set_device_parameter(self.device, 'out0_offset_path0', 'out0_offset_path1', value = 0) # Default after reboot = 7.625
                self.set_device_parameter(self.device, 'scope_acq_avg_mode_en_path0', 'scope_acq_avg_mode_en_path1', value = True)
                self.set_device_parameter(self.device, 'scope_acq_sequencer_select', value = self.DEFAULT_SEQUENCER) 
                self.set_device_parameter(self.device, 'scope_acq_trigger_level_path0', 'scope_acq_trigger_level_path1', value = 0) 
                self.set_device_parameter(self.device, 'scope_acq_trigger_mode_path0', 'scope_acq_trigger_mode_path1', value = 'sequencer') 
                    
                target  = self.device.sequencers[self.DEFAULT_SEQUENCER]

                self.set_device_parameter(target, 'channel_map_path0_out0_en', value = True)
                self.set_device_parameter(target, 'channel_map_path1_out1_en', value = True)
                self.set_device_parameter(target, 'cont_mode_en_awg_path0', 'cont_mode_en_awg_path1', value = False)
                self.set_device_parameter(target, 'cont_mode_waveform_idx_awg_path0', 'cont_mode_waveform_idx_awg_path1', value = 0)
                self.set_device_parameter(target, 'marker_ovr_en', value = True) # Default after reboot = False
                self.set_device_parameter(target, 'marker_ovr_value', value = 15) # Default after reboot = 0
                self.set_device_parameter(target, 'mixer_corr_gain_ratio', value = 1)
                self.set_device_parameter(target, 'mixer_corr_phase_offset_degree', value = 0)
                self.set_device_parameter(target, 'offset_awg_path0', value = 0)
                self.set_device_parameter(target, 'offset_awg_path1', value = 0)
                self.set_device_parameter(target, 'sync_en', value = True) # Default after reboot = False
                self.set_device_parameter(target, 'upsample_rate_awg_path0', 'upsample_rate_awg_path1', value = 0)

                self.set_device_parameter(target, 'channel_map_path0_out0_en', 'channel_map_path1_out1_en', value = True)
                
                self.device_num_sequencers = len(self.device.sequencers)
                for sequencer in range(1,  self.device_num_sequencers):
                    self.set_device_parameter(self.device.sequencers[sequencer], 'channel_map_path0_out0_en', 'channel_map_path1_out1_en', value = False) # Default after reboot = True

    def set_device_parameter(self, target, *parameters, value):
        if self.is_connected:
            key = target.name + '.' + parameters[0]
            if not key in self.device_parameters:
                for parameter in parameters:
                    if not hasattr(target, parameter):
                         raise Exception(f'The instrument {self.name} does not have parameters {parameter}')
                    target.set(parameter, value)
                self.device_parameters[key] = value
            elif self.device_parameters[key] != value:
                for parameter in parameters:
                    target.set(parameter, value)
                self.device_parameters[key] = value
        else:
            raise Exception('There is no connection to the instrument  {self.name}')
    
    def erase_device_parameters_cache(self):
        self.device_parameters = {}

    def setup(self, **kwargs):
        """
        Sets up the instrument using the parameters of the runcard. 
        A connection needs to be established before calling this method.
        
        Args:

        """

        # Hardcoded values used to generate sequence program
        self.wait_loop_step = 1000
        self.waveform_max_length = 16384//2 # maximum length of the combination of waveforms, per sequencer, in number of samples (defined by the sequencer memory).
        self.device_num_ports = 1
        if self.is_connected:
            # Reset
            # if self.current_pulsesequence_hash != self.last_pulsequence_hash:
            #     self.device_parameters = {}
            # TODO: Check when Reset was necessary

            # Load settings
            self.hardware_avg = kwargs['hardware_avg']
            self.sampling_rate = kwargs['sampling_rate']
            self.repetition_duration = kwargs['repetition_duration']
            self.minimum_delay_between_instructions = kwargs['minimum_delay_between_instructions']
            # TODO: Remove minimum_delay_between_instructions

            self.channel_port_map = kwargs['channel_port_map']
            
            self.ports['o1'].attenuation = kwargs['ports']['o1']['attenuation']                         # Default after reboot = 7
            self.ports['o1'].lo_enabled = kwargs['ports']['o1']['lo_enabled']                           # Default after reboot = True
            self.ports['o1'].lo_frequency = kwargs['ports']['o1']['lo_frequency']                       # Default after reboot = 6_000_000_000
            self.ports['o1'].gain = kwargs['ports']['o1']['gain']                                       # Default after reboot = 1
            self.ports['o1'].hardware_mod_en = kwargs['ports']['o1']['hardware_mod_en']                 # Default after reboot = False
            
            self.ports['o1'].nco_freq = 0                                                               # Default after reboot = 1
            self.ports['o1'].nco_phase_offs = 0                                                         # Default after reboot = 1
            
            self.ports['i1'].hardware_demod_en = kwargs['ports']['i1']['hardware_demod_en']             # Default after reboot = False
            
            self.acquisition_hold_off = kwargs['acquisition_hold_off']
            self.acquisition_duration = kwargs['acquisition_duration']
            self.discretization_threshold_acq = 0                                                       # Default after reboot = 1
            self.phase_rotation_acq = 0                                                                 # Default after reboot = 1

        else:
            raise Exception('There is no connection to the instrument')

    def process_pulse_sequence(self, channel_pulses, nshots):
        """
        Processes a list of pulses, generating the waveforms and sequence program required by the instrument to synthesise them.
        
        Args:
        channel_pulses (dict): a dictionary of {channel (int): pulses (list)}
        nshots (int): the number of times the sequence of pulses will be repeated
        """
        # Load the channels to which the instrument is connected
        channels = self.channel_port_map.keys()

        # Check if the sequence to be processed is the same as the last one. If so, there is no need to generate waveforms and program
        self.current_pulsesequence_hash = ""
        for channel in channels:
            for pulse in channel_pulses[channel]:
                self.current_pulsesequence_hash += pulse.serial
        if True: # self.current_pulsesequence_hash != self.last_pulsequence_hash:
            # Sort pulses by their start time 
            for channel in channels:
                channel_pulses[channel].sort(key=lambda pulse: pulse.start) 
            
            # Check if pulses on the same channel overlap
            for channel in channels:
                for m in range(1, len(channel_pulses[channel])):
                    channel_pulses[channel][m].overlaps = []
                    for n in range(m):
                        if channel_pulses[channel][m].start - channel_pulses[channel][n].start < channel_pulses[channel][n].duration:
                            channel_pulses[channel][m].overlaps.append(n)
                for m in range(1, len(channel_pulses[channel])):
                    if len(channel_pulses[channel][m].overlaps) > 0:
                        # TODO: Urgently needed in order to implement multiplexed readout
                        raise(NotImplementedError, "Overlaping pulses on the same channel are not yet supported.")

            # Allocate channel pulses to sequencers. 
            #   At least one sequencer is needed for each channel
            #   If the memory required for the sequence of pulses exceeds 16384/2, 
            #   additional sequencers are used to synthesise it
            self.sequencers = []        # a list of sequencers (int) used
            sequencer_pulses = {}       # a dictionary of {sequencer (int): pulses (list)}     
            sequencer = -1              # initialised to -1 so that in the first iteration it becomes 0, the first sequencer number
            self.waveforms = {}
            for channel in channels:
                if len(channel_pulses[channel]) > 0:
                    # Select a sequencer and add it to the sequencer_channel_map
                    sequencer += 1
                    if sequencer > self.device_num_sequencers:
                        raise Exception(f"The number of sequencers requried to play the sequence exceeds the number available {self.device_num_sequencers}.")
                    # Initialise the corresponding variables 
                    self.sequencers.append(sequencer)
                    self.sequencer_channel_map[sequencer]=channel
                    sequencer_pulses[sequencer]=[]
                    self.waveforms[sequencer]={}
                    unique_pulses = {}      # a dictionary of {unique pulse IDs (str): and their I & Q indices (int)}
                    wc = 0                  # unique waveform counter
                    waveforms_length = 0        # accumulates the length of unique pulses per sequencer

                    # Iterate over the list of pulses to check if they are unique and if the overal length of the waveform exceeds the memory available
                    n = 0
                    while n < len(channel_pulses[channel]):
                        pulse = channel_pulses[channel][n]
                        pulse_serial = pulse.serial[pulse.serial.find(',',pulse.serial.find(',')+1)+2:-1] # removes the channel and start information from Pulse.serial to compare between pulses
                        if pulse_serial not in unique_pulses.keys():
                            # If the pulse is unique (it hasn't been saved before):
                            I, Q = self.generate_waveforms_from_pulse(pulse, modulate = True)
                            # Check if the overall length of the waveform exceeds memory size (waveform_max_length) and split it if necessary
                            is_split = False
                            part = 0
                            while waveforms_length + pulse.duration > self.waveform_max_length:
                                if pulse.type == 'ro':
                                    raise(NotImplementedError, f"Readout pulses longer than the memory available for a sequencer ({self.waveform_max_length}) are not supported.")
                                import copy
                                first_part = copy.deepcopy(pulse)
                                first_part.duration = self.waveform_max_length - waveforms_length 
                                channel_pulses[channel].insert(n, first_part)
                                n += 1
                                sequencer_pulses[sequencer].append(first_part)
                                first_part.waveform_indexes = [0 + wc, 1 + wc]
                                self.waveforms[sequencer][f"{self.name}_{pulse}_{part}_pulse_I"] = {"data": I[:first_part.duration], "index": 0 + wc}
                                self.waveforms[sequencer][f"{self.name}_{pulse}_{part}_pulse_Q"] = {"data": Q[:first_part.duration], "index": 1 + wc}
                                
                                pulse = copy.deepcopy(pulse)
                                I, Q = I[first_part.duration:], Q[first_part.duration:]
                                pulse.start = pulse.start + self.waveform_max_length - waveforms_length
                                pulse.duration = pulse.duration - (self.waveform_max_length - waveforms_length)
                                is_split = True
                                part += 1

                                # Select a new sequencer
                                sequencer += 1
                                if sequencer > self.device_num_sequencers:
                                        raise Exception(f"The number of sequencers requried to play the sequence exceeds the number available {self.device_num_sequencers}.")
                                # Initialise the corresponding variables 
                                self.sequencers.append(sequencer)
                                self.sequencer_channel_map[sequencer]=channel
                                sequencer_pulses[sequencer]=[]
                                self.waveforms[sequencer]={}
                                unique_pulses = {} 
                                wc = 0                 
                                waveforms_length = 0
    
                            # Add the pulse to the list of pulses for the current sequencer and save the waveform indexes
                            sequencer_pulses[sequencer].append(pulse)
                            pulse.waveform_indexes = [0 + wc, 1 + wc]
                            if not is_split:
                                unique_pulses[pulse_serial] =  [0 + wc, 1 + wc]              
                            waveforms_length += pulse.duration
                            self.waveforms[sequencer][f"{self.name}_{pulse}_{part}_pulse_I"] = {"data": I, "index": 0 + wc}
                            self.waveforms[sequencer][f"{self.name}_{pulse}_{part}_pulse_Q"] = {"data": Q, "index": 1 + wc}
                            wc += 2

                        else:
                            sequencer_pulses[sequencer].append(pulse)
                            pulse.waveform_indexes = unique_pulses[pulse_serial]
                        n += 1

            # Generate programs for each sequencer
            pulses = sequencer_pulses
            self.acquisitions = {}
            self.program = {}
            self.weights = {}

            for sequencer in self.sequencers:
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

                wait_time = time_between_repetitions
                extra_wait = wait_time % self.wait_loop_step
                while wait_time > 0 and extra_wait < 4 :
                    self.wait_loop_step += 1
                    extra_wait = wait_time % self.wait_loop_step
                num_wait_loops = (wait_time - extra_wait) // self. wait_loop_step

                header = f"""
                move {nshots},R0 # nshots
                nop
                wait_sync {self.minimum_delay_between_instructions}
                loop:"""
                body = ""

                footer = f"""
                    # wait {wait_time} ns"""
                if num_wait_loops > 0:
                    footer += f"""
                    move {num_wait_loops},R2
                    nop
                    waitloop2:
                        wait {self.wait_loop_step}
                        loop R2,@waitloop2"""
                if extra_wait > 0: 
                    footer += f"""
                        wait {extra_wait}"""
                else:
                    footer += f"""
                        # wait 0"""

                footer += f"""
                loop R0,@loop
                stop 
                """

                # Add an initial wait instruction for the first pulse of the sequence
                wait_time = pulses[sequencer][0].start
                extra_wait = wait_time % self.wait_loop_step
                while wait_time > 0 and extra_wait < 4 :
                    self.wait_loop_step += 1
                    extra_wait = wait_time % self.wait_loop_step
                num_wait_loops = (wait_time - extra_wait) // self. wait_loop_step

                if wait_time > 0:
                    initial_wait_instruction = f"""
                    # wait {wait_time} ns"""
                    if num_wait_loops > 0:
                        initial_wait_instruction += f"""
                    move {num_wait_loops},R1
                    nop
                    waitloop1:
                        wait {self.wait_loop_step}
                        loop R1,@waitloop1"""
                    if extra_wait > 0: 
                        initial_wait_instruction += f"""
                        wait {extra_wait}"""
                    else:
                        initial_wait_instruction += f"""
                        # wait 0"""
                else:
                    initial_wait_instruction = """
                    # wait 0"""

                body += initial_wait_instruction

                for n in range(len(pulses[sequencer])):
                    if pulses[sequencer][n].type == 'ro':
                        delay_after_play = self.minimum_delay_between_instructions # self.acquisition_start #FIXME: This would not work for split pulses
                        
                        if len(pulses[sequencer]) > n + 1:
                            # If there are more pulses to be played, the delay is the time between the pulse end and the next pulse start
                            delay_after_acquire = pulses[sequencer][n + 1].start - pulses[sequencer][n].start - self.minimum_delay_between_instructions # self.acquisition_start
                        else:
                            delay_after_acquire = sequence_total_duration - pulses[sequencer][n].start - self.minimum_delay_between_instructions # self.acquisition_start
                            
                        if delay_after_acquire < self.minimum_delay_between_instructions:
                                raise Exception(f"The minimum delay before starting acquisition is {self.minimum_delay_between_instructions}ns.")
                        
                        # Prepare play instruction: play arg0, arg1, arg2. 
                        #   arg0 is the index of the I waveform 
                        #   arg1 is the index of the Q waveform
                        #   arg2 is the delay between starting the instruction and the next instruction
                        play_instruction = f"                    play {pulses[sequencer][n].waveform_indexes[0]},{pulses[sequencer][n].waveform_indexes[1]},{delay_after_play}"
                        # Add the serial of the pulse as a comment
                        play_instruction += " "*(34-len(play_instruction)) + f"# play waveforms {pulses[sequencer][n]}" 
                        body += "\n" + play_instruction

                        # Prepare acquire instruction: acquire arg0, arg1, arg2. 
                        #   arg0 is the index of the acquisition 
                        #   arg1 is the index of the data bin
                        #   arg2 is the delay between starting the instruction and the next instruction
                        acquire_instruction = f"                    acquire {pulses[sequencer][n].acquisition_index},0,{delay_after_acquire}"
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
                                raise Exception(f"The minimum delay between pulses is {self.minimum_delay_between_instructions}ns.")
                        
                        # Prepare play instruction: play arg0, arg1, arg2. 
                        #   arg0 is the index of the I waveform 
                        #   arg1 is the index of the Q waveform
                        #   arg2 is the delay between starting the instruction and the next instruction
                        play_instruction = f"                    play {pulses[sequencer][n].waveform_indexes[0]},{pulses[sequencer][n].waveform_indexes[1]},{delay_after_play}"
                        # Add the serial of the pulse as a comment
                        play_instruction += " "*(34-len(play_instruction)) + f"# play waveforms {pulses[sequencer][n]}" 
                        body += "\n" + play_instruction

                self.program[sequencer] = header + body + footer

                # DEBUG: QRM print sequencer program
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
        envelope_q = pulse.envelope_q 
        assert len(envelope_i) == len(envelope_q)
        envelopes = np.array([envelope_i, envelope_q])
        if modulate:
            time = np.arange(pulse.duration) / self.sampling_rate
            cosalpha = np.cos(2 * np.pi * pulse.frequency * time + pulse.phase)
            sinalpha = np.sin(2 * np.pi * pulse.frequency * time + pulse.phase)
            mod_matrix = np.array([[ cosalpha, -sinalpha], 
                                   [sinalpha, cosalpha]])
            # mod_signals = np.einsum("abt,bt->ta", mod_matrix, envelopes)
            result = []
            for it, t, ii, qq in zip(np.arange(pulse.duration), time, envelope_i, envelope_q):
                result.append(mod_matrix[:, :, it] @ np.array([ii, qq]))
            mod_signals = np.array(result)

            # DEBUG: QRM plot envelopes
            # import matplotlib.pyplot as plt
            # plt.plot(mod_signals[:, 0] + pulse.offset_i)
            # plt.plot(mod_signals[:, 1] + pulse.offset_q)
            # plt.show()

            return mod_signals[:, 0] + pulse.offset_i, mod_signals[:, 1] + pulse.offset_q
        else:
            return envelope_i, envelope_q

    def upload(self):
        """Uploads waveforms and programs all sequencers and arms them in preparation for execution."""
    
        # Setup
        for sequencer in range(self.device_num_sequencers):
            target  = self.device.sequencers[sequencer]
            if sequencer in self.sequencers:
                # Enable sequencer syncronisation
                self.set_device_parameter(target, 'sync_en', value = True)
            else:
                self.set_device_parameter(target, 'sync_en', value = False)

        # Upload
        if True: # self.current_pulsesequence_hash != self.last_pulsequence_hash:
            self.last_pulsequence_hash = self.current_pulsesequence_hash
            # Upload waveforms and program
            qblox_dict = {}
            for sequencer in self.sequencers:
                # Reformat waveforms to lists
                for name, waveform in self.waveforms[sequencer].items():
                    if isinstance(waveform["data"], np.ndarray):
                        self.waveforms[sequencer][name]["data"] = self.waveforms[sequencer][name]["data"].tolist()  # JSON only supports lists

                # Add sequence program and waveforms to single dictionary and write to JSON file
                filename = f"{self.name}_sequencer{sequencer}_sequence.json"
                qblox_dict[sequencer] = {
                    "waveforms": self.waveforms[sequencer],
                    "weights": {}, #self.weights,
                    "acquisitions": self.acquisitions[sequencer],
                    "program": self.program[sequencer]
                    }
                with open(self.data_folder / filename, "w", encoding="utf-8") as file:
                    json.dump(qblox_dict[sequencer], file, indent=4)
                    
                # Upload json file to the device sequencers
                self.device.sequencers[sequencer].sequence(str(self.data_folder / filename))
        
        # Arm
        for sequencer in self.sequencers:
            # Arm sequencer
            self.device.arm_sequencer(sequencer)

        # DEBUG: QRM Print Readable Snapshot
        # print(self.name)
        # self.device.print_readable_snapshot(update=True)

    def play_sequence(self):
        """Executes the sequence of instructions."""
        # Start playing sequences in all sequencers
        for sequencer in self.sequencers:
            self.device.start_sequencer(sequencer)

    def play_sequence_and_acquire(self):
        """Executes the sequence of instructions and retrieves the readout results.
        """
        # Start playing sequences in all sequencers
        for sequencer in self.sequencers:
            self.device.start_sequencer(sequencer)
        
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
                i, q = self._demodulate_and_integrate(raw_results, acquisition, demodulate = True)
                acquisition_results[sequencer][acquisition] = np.sqrt(i**2 + q**2), np.arctan2(q, i), i, q
                # DEBUG: QRM Plot Incomming Pulses
                # import qibolab.instruments.debug.incomming_pulse_plotting as pp
                # pp.plot(raw_results)
        return acquisition_results

    def _demodulate_and_integrate(self, raw_results, acquisition, demodulate = True):
        if demodulate:
            acquisition_name = acquisition
            acquisition_frequency = 20_000_000
            #TODO: obtain from acquisition info
            #DOWN Conversion
            n0 = self.acquisition_hold_off # 0
            n1 = self.acquisition_hold_off + self.acquisition_duration # self.acquisition_duration # 
            input_vec_I = np.array(raw_results[acquisition_name]["acquisition"]["scope"]["path0"]["data"][n0: n1])
            input_vec_Q = np.array(raw_results[acquisition_name]["acquisition"]["scope"]["path1"]["data"][n0: n1])
            input_vec_I -= np.mean(input_vec_I)
            input_vec_Q -= np.mean(input_vec_Q)

            modulated_i = input_vec_I
            modulated_q = input_vec_Q
            time = np.arange(modulated_i.shape[0])*1e-9
            cosalpha = np.cos(2 * np.pi * acquisition_frequency * time)
            sinalpha = np.sin(2 * np.pi * acquisition_frequency * time)
            demod_matrix = 2 * np.array([[cosalpha, sinalpha], [-sinalpha, cosalpha]])
            result = []
            for it, t, ii, qq in zip(np.arange(modulated_i.shape[0]), time,modulated_i, modulated_q):
                result.append(demod_matrix[:,:,it] @ np.array([ii, qq]))
            demodulated_signal = np.array(result)
            integrated_signal = np.mean(demodulated_signal,axis=0)

            # import matplotlib.pyplot as plt
            # plt.plot(input_vec_I[:400])
            # plt.plot(list(map(list, zip(*demodulated_signal)))[0][:400])
            # plt.show()
        else:
            int_len = self.acquisition_duration
            i = [(val/int_len) for val in raw_results[acquisition]["acquisition"]["bins"]["integration"]["path0"]][0]
            q = [(val/int_len) for val in raw_results[acquisition]["acquisition"]["bins"]["integration"]["path1"]][0]
            integrated_signal = i, q
        return integrated_signal

    def start(self):
        pass

    def stop(self):
        """Stops all sequencers"""
        self.device.stop_sequencer()

    def disconnect(self):
        pass
        

class ClusterQCM_RF(AbstractInstrument):
    """
    Qblox Cluster Qubit Control Module RF driver.
    
    Args:
        name (str): unique name given to the instrument
        address (str): IP_address:module_number

    """
    O1_DEFAULT_SEQUENCER = 0
    O2_DEFAULT_SEQUENCER = 1

    property_wrapper = lambda parent, *parameter: property(lambda self: parent.device.get(parameter[0]), lambda self,x: parent.set_device_parameter(parent.device, *parameter, value = x))
    sequencer_property_wrapper = lambda parent, sequencer, *parameter: property(lambda self: parent.device.sequencers[sequencer].get(parameter[0]), lambda self,x: parent.set_device_parameter(parent.device.sequencers[sequencer], *parameter, value = x))
    

    def __init__(self, name, address):
        super().__init__(name, address)
        self.sequencers = []
        self.sequencer_channel_map = {} 
        self.last_pulsequence_hash = "uninitialised"
        self.current_pulsesequence_hash = ""
        self.device_parameters = {}
        self.ports = {}

    def connect(self):
        """
        Connects to the instrument using the IP address set in the runcard.
        """
        global cluster
        if not self.is_connected:
            if cluster:
                self.device = cluster.modules[int(self.address.split(':')[1])-1]

                self.ports['o1'] = type(f'port_o1', (), 
                    {'attenuation': self.property_wrapper('out0_att'), 
                    'lo_enabled': self.property_wrapper('out0_lo_en'), 
                    'lo_frequency': self.property_wrapper('out0_lo_freq'), 
                    'gain': self.sequencer_property_wrapper(self.O1_DEFAULT_SEQUENCER, 'gain_awg_path0', 'gain_awg_path1'), 
                    'hardware_mod_en': self.sequencer_property_wrapper(self.O1_DEFAULT_SEQUENCER, 'mod_en_awg'),
                    'nco_freq': self.sequencer_property_wrapper(self.O1_DEFAULT_SEQUENCER, 'nco_freq'),
                    'nco_phase_offs': self.sequencer_property_wrapper(self.O1_DEFAULT_SEQUENCER, 'nco_phase_offs')
                    })()

                self.ports['o2'] = type(f'port_o1', (), 
                    {'attenuation': self.property_wrapper('out1_att'), 
                    'lo_enabled': self.property_wrapper('out1_lo_en'), 
                    'lo_frequency': self.property_wrapper('out1_lo_freq'), 
                    'gain': self.sequencer_property_wrapper(self.O2_DEFAULT_SEQUENCER, 'gain_awg_path0', 'gain_awg_path1'), 
                    'hardware_mod_en': self.sequencer_property_wrapper(self.O2_DEFAULT_SEQUENCER, 'mod_en_awg'),
                    'nco_freq': self.sequencer_property_wrapper(self.O2_DEFAULT_SEQUENCER, 'nco_freq'),
                    'nco_phase_offs': self.sequencer_property_wrapper(self.O2_DEFAULT_SEQUENCER, 'nco_phase_offs')
                    })()


                self.cluster = cluster
                self.is_connected = True
                self.set_device_parameter(self.device, 'out0_offset_path0', 'out0_offset_path1', value = 0) # Default after reboot = 7.625
                self.set_device_parameter(self.device, 'out1_offset_path0', 'out1_offset_path1', value = 0) # Default after reboot = 7.625
                                    
                for target in [self.device.sequencers[self.O1_DEFAULT_SEQUENCER], self.device.sequencers[self.O2_DEFAULT_SEQUENCER]]:

                    self.set_device_parameter(target, 'cont_mode_en_awg_path0', 'cont_mode_en_awg_path1', value = False)
                    self.set_device_parameter(target, 'cont_mode_waveform_idx_awg_path0', 'cont_mode_waveform_idx_awg_path1', value = 0)
                    self.set_device_parameter(target, 'marker_ovr_en', value = True) # Default after reboot = False
                    self.set_device_parameter(target, 'marker_ovr_value', value = 15) # Default after reboot = 0
                    self.set_device_parameter(target, 'mixer_corr_gain_ratio', value = 1)
                    self.set_device_parameter(target, 'mixer_corr_phase_offset_degree', value = 0)
                    self.set_device_parameter(target, 'offset_awg_path0', value = 0)
                    self.set_device_parameter(target, 'offset_awg_path1', value = 0)
                    self.set_device_parameter(target, 'sync_en', value = True) # Default after reboot = False
                    self.set_device_parameter(target, 'upsample_rate_awg_path0', 'upsample_rate_awg_path1', value = 0)

                self.set_device_parameter(self.device.sequencers[self.O1_DEFAULT_SEQUENCER], 'channel_map_path0_out0_en', 'channel_map_path1_out1_en', value = True)
                self.set_device_parameter(self.device.sequencers[self.O1_DEFAULT_SEQUENCER], 'channel_map_path0_out2_en', 'channel_map_path1_out3_en', value = False)
                self.set_device_parameter(self.device.sequencers[self.O2_DEFAULT_SEQUENCER], 'channel_map_path0_out0_en', 'channel_map_path1_out1_en', value = False)
                self.set_device_parameter(self.device.sequencers[self.O2_DEFAULT_SEQUENCER], 'channel_map_path0_out2_en', 'channel_map_path1_out3_en', value = True)
                self.device_num_sequencers = len(self.device.sequencers)
                for sequencer in range(2,  self.device_num_sequencers):
                    self.set_device_parameter(self.device.sequencers[sequencer], 'channel_map_path0_out0_en', 'channel_map_path1_out1_en', value = False) # Default after reboot = True
                    self.set_device_parameter(self.device.sequencers[sequencer], 'channel_map_path0_out2_en', 'channel_map_path1_out3_en', value = False) # Default after reboot = True

    def set_device_parameter(self, target, *parameters, value):
        if self.is_connected:
            key = target.name + '.' + parameters[0]
            if not key in self.device_parameters:
                for parameter in parameters:
                    if not hasattr(target, parameter):
                         raise Exception(f'The instrument {self.name} does not have parameters {parameter}')
                    target.set(parameter, value)
                self.device_parameters[key] = value
            elif self.device_parameters[key] != value:
                for parameter in parameters:
                    target.set(parameter, value)
                self.device_parameters[key] = value
        else:
            raise Exception('There is no connection to the instrument  {self.name}')
    
    def erase_device_parameters_cache(self):
        self.device_parameters = {}


    def setup(self, **kwargs):
        """
        Sets up the instrument using the parameters of the runcard. 
        A connection needs to be established before calling this method.
        
        Args:

        """

        # Hardcoded values used to generate sequence program
        self.wait_loop_step = 1000
        self.waveform_max_length = 16384//2 # maximum length of the combination of waveforms, per sequencer, in number of samples (defined by the sequencer memory).
        self.device_num_ports = 2
        if self.is_connected:
            # Reset
            # if self.current_pulsesequence_hash != self.last_pulsequence_hash:
            #     self.device_parameters = {}
            # TODO: Check when Reset was necessary

            # Load settings
            self.hardware_avg = kwargs['hardware_avg']
            self.sampling_rate = kwargs['sampling_rate']
            self.repetition_duration = kwargs['repetition_duration']
            self.minimum_delay_between_instructions = kwargs['minimum_delay_between_instructions']
            # TODO: Remove minimum_delay_between_instructions

            self.channel_port_map = kwargs['channel_port_map']

            self.ports['o1'].attenuation = kwargs['ports']['o1']['attenuation']                        # Default after reboot = 7
            self.ports['o1'].lo_enabled = kwargs['ports']['o1']['lo_enabled']                          # Default after reboot = True
            self.ports['o1'].lo_frequency = kwargs['ports']['o1']['lo_frequency']                      # Default after reboot = 6_000_000_000
            self.ports['o1'].gain = kwargs['ports']['o1']['gain']                                      # Default after reboot = 1
            self.ports['o1'].hardware_mod_en = kwargs['ports']['o1']['hardware_mod_en']                # Default after reboot = False
            self.ports['o1'].nco_freq = 0                                                              # Default after reboot = 1
            self.ports['o1'].nco_phase_offs = 0                                                        # Default after reboot = 1

            self.ports['o2'].attenuation = kwargs['ports']['o2']['attenuation']                        # Default after reboot = 7
            self.ports['o2'].lo_enabled = kwargs['ports']['o2']['lo_enabled']                          # Default after reboot = True
            self.ports['o2'].lo_frequency = kwargs['ports']['o2']['lo_frequency']                      # Default after reboot = 6_000_000_000
            self.ports['o2'].gain = kwargs['ports']['o2']['gain']                                      # Default after reboot = 1
            self.ports['o2'].hardware_mod_en = kwargs['ports']['o2']['hardware_mod_en']                # Default after reboot = False
            self.ports['o2'].nco_freq = 0                                                              # Default after reboot = 1
            self.ports['o2'].nco_phase_offs = 0                                                        # Default after reboot = 1


        else:
            raise Exception('There is no connection to the instrument')

    def process_pulse_sequence(self, channel_pulses, nshots):
        """
        Processes a list of pulses, generating the waveforms and sequence program required by the instrument to synthesise them.
        
        Args:
        channel_pulses (dict): a dictionary of {channel (int): pulses (list)}
        nshots (int): the number of times the sequence of pulses will be repeated
        """
        # Load the channels to which the instrument is connected
        channels = self.channel_port_map.keys()

        # Check if the sequence to be processed is the same as the last one. If so, there is no need to generate waveforms and program
        self.current_pulsesequence_hash = ""
        for channel in channels:
            for pulse in channel_pulses[channel]:
                self.current_pulsesequence_hash += pulse.serial
        if self.current_pulsesequence_hash != self.last_pulsequence_hash:
            # Sort pulses by their start time 
            for channel in channels:
                channel_pulses[channel].sort(key=lambda pulse: pulse.start) 
            
            # Check if pulses on the same channel overlap
            for channel in channels:
                for m in range(1, len(channel_pulses[channel])):
                    channel_pulses[channel][m].overlaps = []
                    for n in range(m):
                        if channel_pulses[channel][m].start - channel_pulses[channel][n].start < channel_pulses[channel][n].duration:
                            channel_pulses[channel][m].overlaps.append(n)
                for m in range(1, len(channel_pulses[channel])):
                    if len(channel_pulses[channel][m].overlaps) > 0:
                        raise(NotImplementedError, "Overlaping pulses on the same channel are not yet supported.")
            
            # Allocate channel pulses to sequencers. 
            #   At least one sequencer is needed for each channel
            #   If the memory required for the sequence of pulses exceeds 16384/2, 
            #   additional sequencers are used to synthesise it
            self.sequencers = []        # a list of sequencers (int) used
            sequencer_pulses = {}       # a dictionary of {sequencer (int): pulses (list)}     
            sequencer = -1              # initialised to -1 so that in the first iteration it becomes 0, the first sequencer number
            self.waveforms = {}
            for channel in channels:
                sequencer += 1
                if len(channel_pulses[channel]) > 0:
                    # Select a sequencer and add it to the sequencer_channel_map
                    if sequencer > self.device_num_sequencers:
                        raise Exception(f"The number of sequencers requried to play the sequence exceeds the number available {self.device_num_sequencers}.")
                    # Initialise the corresponding variables 
                    self.sequencers.append(sequencer)
                    self.sequencer_channel_map[sequencer]=channel
                    sequencer_pulses[sequencer]=[]
                    self.waveforms[sequencer]={}
                    unique_pulses = {}      # a dictionary of {unique pulse IDs (str): and their I & Q indices (int)}
                    wc = 0                  # unique waveform counter
                    waveforms_length = 0        # accumulates the length of unique pulses per sequencer

                    # Iterate over the list of pulses to check if they are unique and if the overal length of the waveform exceeds the memory available
                    n = 0
                    while n < len(channel_pulses[channel]):
                        pulse = channel_pulses[channel][n]
                        pulse_serial = pulse.serial[pulse.serial.find(',',pulse.serial.find(',')+1)+2:-1] # removes the channel and start information from Pulse.serial to compare between pulses
                        if pulse_serial not in unique_pulses.keys():
                            # If the pulse is unique (it hasn't been saved before):
                            I, Q = self.generate_waveforms_from_pulse(pulse, modulate = True)
                            # Check if the overall length of the waveform exceeds memory size (waveform_max_length) and split it if necessary
                            is_split = False
                            part = 0
                            while waveforms_length + pulse.duration > self.waveform_max_length:
                                import copy
                                first_part = copy.deepcopy(pulse)
                                first_part.duration = self.waveform_max_length - waveforms_length 
                                channel_pulses[channel].insert(n, first_part)
                                n += 1
                                sequencer_pulses[sequencer].append(first_part)
                                first_part.waveform_indexes = [0 + wc, 1 + wc]
                                self.waveforms[sequencer][f"{self.name}_{pulse}_{part}_pulse_I"] = {"data": I[:first_part.duration], "index": 0 + wc}
                                self.waveforms[sequencer][f"{self.name}_{pulse}_{part}_pulse_Q"] = {"data": Q[:first_part.duration], "index": 1 + wc}
                                
                                pulse = copy.deepcopy(pulse)
                                I, Q = I[first_part.duration:], Q[first_part.duration:]
                                pulse.start = pulse.start + self.waveform_max_length - waveforms_length
                                pulse.duration = pulse.duration - (self.waveform_max_length - waveforms_length)
                                is_split = True
                                part += 1

                                # Select a new sequencer
                                sequencer += 1
                                if sequencer > self.device_num_sequencers:
                                        raise Exception(f"The number of sequencers requried to play the sequence exceeds the number available {self.device_num_sequencers}.")
                                # Initialise the corresponding variables 
                                self.sequencers.append(sequencer)
                                self.sequencer_channel_map[sequencer]=channel
                                sequencer_pulses[sequencer]=[]
                                self.waveforms[sequencer]={}
                                unique_pulses = {} 
                                wc = 0                 
                                waveforms_length = 0
    
                            # Add the pulse to the list of pulses for the current sequencer and save the waveform indexes
                            sequencer_pulses[sequencer].append(pulse)
                            pulse.waveform_indexes = [0 + wc, 1 + wc]
                            if not is_split:
                                unique_pulses[pulse_serial] =  [0 + wc, 1 + wc]                  
                            waveforms_length += pulse.duration
                            self.waveforms[sequencer][f"{self.name}_{pulse}_{part}_pulse_I"] = {"data": I, "index": 0 + wc}
                            self.waveforms[sequencer][f"{self.name}_{pulse}_{part}_pulse_Q"] = {"data": Q, "index": 1 + wc}
                            wc += 2    

                        else:
                            sequencer_pulses[sequencer].append(pulse)
                            pulse.waveform_indexes = unique_pulses[pulse_serial]
                        n += 1

            # Generate programs for each sequencer
            pulses = sequencer_pulses
            self.program = {}

            for sequencer in self.sequencers:
                # Program
                sequence_total_duration = pulses[sequencer][-1].start + pulses[sequencer][-1].duration + self.minimum_delay_between_instructions # the minimum delay between instructions is 4ns
                time_between_repetitions = self.repetition_duration - sequence_total_duration
                assert time_between_repetitions > 0

                wait_time = time_between_repetitions
                extra_wait = wait_time % self.wait_loop_step
                while wait_time > 0 and extra_wait < 4 :
                    self.wait_loop_step += 1
                    extra_wait = wait_time % self.wait_loop_step
                num_wait_loops = (wait_time - extra_wait) // self.wait_loop_step

                header = f"""
                move {nshots},R0 # nshots
                nop
                wait_sync {self.minimum_delay_between_instructions}
                loop:"""
                body = ""

                footer = f"""
                    # wait {wait_time} ns"""
                if num_wait_loops > 0:
                    footer += f"""
                    move {num_wait_loops},R2
                    nop
                    waitloop2:
                        wait {self.wait_loop_step}
                        loop R2,@waitloop2"""
                if extra_wait > 0: 
                    footer += f"""
                        wait {extra_wait}"""
                else:
                    footer += f"""
                        # wait 0"""

                footer += f"""
                loop R0,@loop
                stop 
                """

                # Add an initial wait instruction for the first pulse of the sequence
                wait_time = pulses[sequencer][0].start
                extra_wait = wait_time % self.wait_loop_step
                while wait_time > 0 and extra_wait < 4 :
                    self.wait_loop_step += 1
                    extra_wait = wait_time % self.wait_loop_step
                num_wait_loops = (wait_time - extra_wait) // self. wait_loop_step

                if wait_time > 0:
                    initial_wait_instruction = f"""
                    # wait {wait_time} ns"""
                    if num_wait_loops > 0:
                        initial_wait_instruction += f"""
                    move {num_wait_loops},R1
                    nop
                    waitloop1:
                        wait {self.wait_loop_step}
                        loop R1,@waitloop1"""
                    if extra_wait > 0: 
                        initial_wait_instruction += f"""
                    wait {extra_wait}"""
                    else:
                        initial_wait_instruction += f"""
                    # wait 0"""
                else:
                    initial_wait_instruction = """
                    # wait 0"""

                body += initial_wait_instruction

                for n in range(len(pulses[sequencer])):
                    # Calculate the delay_after_play that is to be used as an argument to the play instruction
                    if len(pulses[sequencer]) > n + 1:
                        # If there are more pulses to be played, the delay is the time between the pulse end and the next pulse start
                        delay_after_play = pulses[sequencer][n + 1].start - pulses[sequencer][n].start
                    else:
                        delay_after_play = sequence_total_duration - pulses[sequencer][n].start

                    if delay_after_play < self.minimum_delay_between_instructions:
                            raise Exception(f"The minimum delay between pulses is {self.minimum_delay_between_instructions}ns.")
                    
                    # Prepare play instruction: play arg0, arg1, arg2. 
                    #   arg0 is the index of the I waveform 
                    #   arg1 is the index of the Q waveform
                    #   arg2 is the delay between starting the instruction and the next instruction
                    play_instruction = f"                    play {pulses[sequencer][n].waveform_indexes[0]},{pulses[sequencer][n].waveform_indexes[1]},{delay_after_play}"
                    # Add the serial of the pulse as a comment
                    play_instruction += " "*(34-len(play_instruction)) + f"# play waveforms {pulses[sequencer][n]}" #TODO: change for split pulses
                    body += "\n" + play_instruction

                self.program[sequencer] = header + body + footer

                # DEBUG: QCM print sequencer program
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
        envelope_q = pulse.envelope_q 
        assert len(envelope_i) == len(envelope_q)
        envelopes = np.array([envelope_i, envelope_q])
        if modulate:
            time = np.arange(pulse.duration) / self.sampling_rate
            cosalpha = np.cos(2 * np.pi * pulse.frequency * time + pulse.phase)
            sinalpha = np.sin(2 * np.pi * pulse.frequency * time + pulse.phase)
            mod_matrix = np.array([[ cosalpha, -sinalpha], 
                                   [sinalpha, cosalpha]])
            # mod_signals = np.einsum("abt,bt->ta", mod_matrix, envelopes)
            result = []
            for it, t, ii, qq in zip(np.arange(pulse.duration), time, envelope_i, envelope_q):
                result.append(mod_matrix[:, :, it] @ np.array([ii, qq]))
            mod_signals = np.array(result)

            # DEBUG: QCM plot envelopes
            # import matplotlib.pyplot as plt
            # plt.plot(mod_signals[:, 0] + pulse.offset_i)
            # plt.plot(mod_signals[:, 1] + pulse.offset_q)
            # plt.show()

            return mod_signals[:, 0] + pulse.offset_i, mod_signals[:, 1] + pulse.offset_q
        else:
            return envelope_i, envelope_q

    def upload(self):
        """Uploads waveforms and programs all sequencers and arms them in preparation for execution."""
        # Setup
        for sequencer in range(self.device_num_sequencers):
            target  = self.device.sequencers[sequencer]
            if sequencer in self.sequencers:
                # Enable sequencer syncronisation
                self.set_device_parameter(target, 'sync_en', value = True)
            else:
                self.set_device_parameter(target, 'sync_en', value = False)
                
        # Upload
        if self.current_pulsesequence_hash != self.last_pulsequence_hash:
            self.last_pulsequence_hash = self.current_pulsesequence_hash
            # TODO: dont upload if the same 
            # Upload waveforms and program
            qblox_dict = {}
            for sequencer in self.sequencers:
                # Reformat waveforms to lists
                for name, waveform in self.waveforms[sequencer].items():
                    if isinstance(waveform["data"], np.ndarray):
                        self.waveforms[sequencer][name]["data"] = self.waveforms[sequencer][name]["data"].tolist()  # JSON only supports lists

                # Add sequence program and waveforms to single dictionary and write to JSON file
                filename = f"{self.name}_sequencer{sequencer}_sequence.json"
                qblox_dict[sequencer] = {
                    "waveforms": self.waveforms[sequencer],
                    "weights": {}, #self.weights,
                    "acquisitions": {}, #self.acquisitions,
                    "program": self.program[sequencer]
                    }
                with open(self.data_folder / filename, "w", encoding="utf-8") as file:
                    json.dump(qblox_dict[sequencer], file, indent=4)
                # Upload json file to the device sequencers
                self.device.sequencers[sequencer].sequence(str(self.data_folder / filename))

        # Arm
        for sequencer in self.sequencers:
            # Arm sequencer
            self.device.arm_sequencer(sequencer)

        # DEBUG: QCM Print Readable Snapshot
        # print(self.name)
        # self.device.print_readable_snapshot(update=True)


    def play_sequence(self):
        """Executes the sequence of instructions."""
        for sequencer in self.sequencers:
            # Start sequencer
            self.device.start_sequencer(sequencer)

    def start(self):
        pass

    def stop(self):
        """Stops all sequencers"""
        self.device.stop_sequencer()

    def disconnect(self):
        pass
    

