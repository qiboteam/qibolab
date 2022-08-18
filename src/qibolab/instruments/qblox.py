import json
import numpy as np
import qblox_instruments
from qibolab.instruments.abstract import AbstractInstrument, InstrumentException
from qibolab.pulses import PulseSequence, Pulse, PulseShape, PulseType, Waveform


class WaveformsBuffer():

    SIZE: int = 16383

    class NotEnoughMemory(Exception):
        pass

    def __init__(self):
        self.unique_waveforms: list = [] # Waveform
        self.available_memory: int = WaveformsBuffer.SIZE

    def add_waveforms(self, waveform_i: Waveform, waveform_q: Waveform):

        if waveform_i not in self.unique_waveforms or waveform_q not in self.unique_waveforms:
            memory_needed = 0
            if not waveform_i in self.unique_waveforms:
                memory_needed +=  len(waveform_i) 
            if not waveform_q in self.unique_waveforms:
                memory_needed +=  len(waveform_q) 

            if self.available_memory >= memory_needed:
                if not waveform_i in self.unique_waveforms:
                    self.unique_waveforms.append(waveform_i)
                if not waveform_q in self.unique_waveforms:
                    self.unique_waveforms.append(waveform_q)
            else:
                raise WaveformsBuffer.NotEnoughMemory


class Sequencer():
    def __init__(self, number):
        self.number: int = number
        self.waveforms_buffer: WaveformsBuffer = WaveformsBuffer()
        self.pulses: PulseSequence = PulseSequence()
        self.waveforms: dict = {}
        self.acquisitions: dict = {}
        self.weights: dict = {}
        self.program: str = ""


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
    DEFAULT_SEQUENCERS = {'o1': 0}

    property_wrapper = lambda parent, *parameter: property(
        lambda self: parent.device.get(parameter[0]), 
        lambda self,x: parent.set_device_parameter(parent.device, *parameter, value = x)
        )
    sequencer_property_wrapper = lambda parent, sequencer, *parameter: property(
        lambda self: parent.device.sequencers[sequencer].get(parameter[0]), 
        lambda self,x: parent.set_device_parameter(parent.device.sequencers[sequencer], *parameter, value = x)
        )

    def __init__(self, name, address):
        super().__init__(name, address)
        self.ports = {}
        self.input_ports_keys = ['i1']
        self.output_ports_keys = ['o1']
        self.sequencers:dict[WaveformsBuffer] = {'o1': []}
        self.last_pulsequence_hash:int = 0
        self.current_pulsesequence_hash:int
        self.device_parameters = {}


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
                     'gain': self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS['o1'], 'gain_awg_path0', 'gain_awg_path1'), 
                     'hardware_mod_en': self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS['o1'], 'mod_en_awg'),
                     'nco_freq': self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS['o1'], 'nco_freq'),
                     'nco_phase_offs': self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS['o1'], 'nco_phase_offs')
                    })()

                self.ports['i1'] = type(f'port_i1', (), 
                    {'hardware_demod_en': self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS['o1'], 'demod_en_acq')
                    })()

                setattr(self.__class__, 'acquisition_duration', self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS['o1'], 'integration_length_acq'))
                setattr(self.__class__, 'discretization_threshold_acq', self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS['o1'], 'discretization_threshold_acq'))
                setattr(self.__class__, 'phase_rotation_acq', self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS['o1'], 'phase_rotation_acq'))

                self.cluster = cluster
                self.is_connected = True
                
                self.set_device_parameter(self.device, 'in0_att', value = 0) 
                self.set_device_parameter(self.device, 'out0_offset_path0', 'out0_offset_path1', value = 0) # Default after reboot = 7.625
                self.set_device_parameter(self.device, 'scope_acq_avg_mode_en_path0', 'scope_acq_avg_mode_en_path1', value = True)
                self.set_device_parameter(self.device, 'scope_acq_sequencer_select', value = self.DEFAULT_SEQUENCERS['o1']) 
                self.set_device_parameter(self.device, 'scope_acq_trigger_level_path0', 'scope_acq_trigger_level_path1', value = 0) 
                self.set_device_parameter(self.device, 'scope_acq_trigger_mode_path0', 'scope_acq_trigger_mode_path1', value = 'sequencer') 
                    
                target  = self.device.sequencers[self.DEFAULT_SEQUENCERS['o1']]

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
        self.device_num_ports = 1
        if self.is_connected:
            # Reset
            # if self.current_pulsesequence_hash != self.last_pulsequence_hash:
            #     self.device_parameters = {}
            # TODO: Check when Reset was necessary

            # Load settings
            self.sampling_rate = kwargs['sampling_rate']

            self.channel_port_map = kwargs['channel_port_map']
            self.port_channel_map = {v: k for k, v in self.channel_port_map.items()}
            self.channels = list(self.channel_port_map.keys())
            
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

    def process_pulse_sequence(self, instrument_pulses: PulseSequence, nshots, repetition_duration):
        """
        Processes a list of pulses, generating the waveforms and sequence program required by the instrument to synthesise them.
        
        Args:
        channel_pulses (dict): a dictionary of {channel (int): pulses (list)}
        nshots (int): the number of times the sequence of pulses will be repeated
        """

        # Save the hash of the current sequence of pulses.
        self.current_pulsesequence_hash = hash((instrument_pulses, nshots, repetition_duration, self.ports['o1'].hardware_mod_en, self.ports['i1'].hardware_demod_en))
        
        # Check if the sequence to be processed is the same as the last one. 
        # If so, there is no need to generate new waveforms and program
        # Except if hardware demodulation is activated (to force a memory reset until qblox fix the issue)
        if self.ports['i1'].hardware_demod_en or self.current_pulsesequence_hash != self.last_pulsequence_hash:

            # split the collection of instruments pulses by ports
            self.free_sequencers = [0,1,2,3,4,5]
            port = 'o1'
            port_pulses: PulseSequence = instrument_pulses.get_channel_pulses(self.port_channel_map[port])
            if not port_pulses.is_empty:
                self.sequencers[port] = []
                # if port_pulses.pulses_overlap:
                #     raise NotImplementedError("Overlaping pulses on the same channel are not yet supported.")
                for non_overlapping_pulses in port_pulses.separate_overlapping_pulses():
                    if len(self.free_sequencers) == 0:
                        raise Exception(f"The number of sequencers requried to play the sequence exceeds the number available {self.device_num_sequencers}.")
                    next_sequencer_number = self.free_sequencers.pop(0)
                    if next_sequencer_number != self.DEFAULT_SEQUENCERS[port]:
                        for parameter in self.device.sequencers[self.DEFAULT_SEQUENCERS[port]].parameters:
                            value = self.device.sequencers[self.DEFAULT_SEQUENCERS[port]].get(param_name = parameter)
                            if not value is None:
                                target  = self.device.sequencers[next_sequencer_number]
                                self.set_device_parameter(target, parameter, value = value)
                    self.set_device_parameter(self.device.sequencers[next_sequencer_number], 'nco_freq', value = non_overlapping_pulses[0].frequency)
                    sequencer = Sequencer(next_sequencer_number)
                    self.sequencers[port].append(sequencer)

                    pulses_to_be_processed = non_overlapping_pulses.shallow_copy()
                    while not pulses_to_be_processed.is_empty:
                        try:
                            pulse: Pulse = pulses_to_be_processed[0]
                            if self.ports[port].hardware_mod_en:
                                pulse.waveform_i, pulse.waveform_q = pulse.envelope_waveforms
                            else:
                                pulse.waveform_i, pulse.waveform_q = pulse.modulated_waveforms
                            sequencer.waveforms_buffer.add_waveforms(pulse.waveform_i, pulse.waveform_q)
                            sequencer.pulses.add(pulse)
                            pulses_to_be_processed.remove(pulse)
                        except WaveformsBuffer.NotEnoughMemory:
                            if len(pulse.waveform_i) + len(pulse.waveform_q) > WaveformsBuffer.SIZE:
                                raise NotImplementedError(f"Pulses with waveforms longer than the memory of a sequencer ({WaveformsBuffer.SIZE // 2}) are not supported.")
                            if len(self.free_sequencers) == 0:
                                raise Exception(f"The number of sequencers requried to play the sequence exceeds the number available {self.device_num_sequencers}.")
                            next_sequencer_number = self.free_sequencers.pop(0)
                            sequencer = Sequencer(next_sequencer_number)
                            for parameter in self.device.sequencers[self.DEFAULT_SEQUENCERS[port]].parameters:
                                value = self.device.sequencers[self.DEFAULT_SEQUENCERS[port]].get(param_name = parameter)
                                if not value is None:
                                    target  = self.device.sequencers[next_sequencer_number]
                                    self.set_device_parameter(target, parameter, value = value)
                            self.sequencers[port].append(sequencer)
                self.used_sequencers = []
                for port in self.output_ports_keys:                
                    for sequencer in self.sequencers[port]:
                        self.used_sequencers.append(sequencer.number)

                self.unused_sequencers = []
                for n in range(self.device_num_sequencers):
                    if not n in self.used_sequencers:
                        self.unused_sequencers.append(n)


                for port in self.output_ports_keys:                
                    for sequencer in self.sequencers[port]:

                        # Waveforms
                        for index, waveform in enumerate(sequencer.waveforms_buffer.unique_waveforms):
                            sequencer.waveforms[waveform.serial] = {"data": waveform.data.tolist(), "index": index}

                        # Acquisitions
                        for acquisition_index, pulse in enumerate(sequencer.pulses.ro_pulses):
                            sequencer.acquisitions[pulse.serial] = {"num_bins": 1, "index":acquisition_index}

                        # Program
                        minimum_delay_between_instructions = 4
                        wait_loop_step:int = 1000

                        pulses = sequencer.pulses
                        sequence_total_duration = pulses.start + pulses.duration + minimum_delay_between_instructions # the minimum delay between instructions is 4ns
                        time_between_repetitions = repetition_duration - sequence_total_duration
                        assert time_between_repetitions > 0

                        wait_time = time_between_repetitions
                        extra_wait = wait_time % wait_loop_step
                        while wait_time > 0 and extra_wait < 4 :
                            wait_loop_step += 1
                            extra_wait = wait_time % wait_loop_step
                        num_wait_loops = (wait_time - extra_wait) // wait_loop_step

                        header = f"""
                        move {nshots},R0 # nshots
                        nop
                        wait_sync {minimum_delay_between_instructions}
                        loop:
                        reset_ph"""
                        body = ""

                        footer = f"""
                            # wait {wait_time} ns"""
                        if num_wait_loops > 0:
                            footer += f"""
                            move {num_wait_loops},R2
                            nop
                            waitloop2:
                                wait {wait_loop_step}
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
                        wait_time = pulses[0].start
                        extra_wait = wait_time % wait_loop_step
                        while wait_time > 0 and extra_wait < 4 :
                            wait_loop_step += 1
                            extra_wait = wait_time % wait_loop_step
                        num_wait_loops = (wait_time - extra_wait) // wait_loop_step

                        if wait_time > 0:
                            initial_wait_instruction = f"""
                            # wait {wait_time} ns"""
                            if num_wait_loops > 0:
                                initial_wait_instruction += f"""
                            move {num_wait_loops},R1
                            nop
                            waitloop1:
                                wait {wait_loop_step}
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

                        for n in range(pulses.count):
                            if pulses[n].type == PulseType.READOUT:
                                delay_after_play = self.acquisition_hold_off #FIXME: This would not work for split pulses
                                
                                if len(pulses) > n + 1:
                                    # If there are more pulses to be played, the delay is the time between the pulse end and the next pulse start
                                    delay_after_acquire = pulses[n + 1].start - pulses[n].start - self.acquisition_hold_off
                                else:
                                    delay_after_acquire = sequence_total_duration - pulses[n].start - self.acquisition_hold_off
                                    
                                if delay_after_acquire < minimum_delay_between_instructions:
                                        raise Exception(f"The minimum delay after starting acquisition is {minimum_delay_between_instructions}ns.")
                                
                                # Prepare play instruction: play arg0, arg1, arg2. 
                                #   arg0 is the index of the I waveform 
                                #   arg1 is the index of the Q waveform
                                #   arg2 is the delay between starting the instruction and the next instruction
                                play_instruction = f"                    play {sequencer.waveforms_buffer.unique_waveforms.index(pulses[n].waveform_i)},{sequencer.waveforms_buffer.unique_waveforms.index(pulses[n].waveform_q)},{delay_after_play}"
                                # Add the serial of the pulse as a comment
                                play_instruction += " "*(34-len(play_instruction)) + f"# play waveforms {pulses[n]}" 
                                body += "\n" + play_instruction

                                # Prepare acquire instruction: acquire arg0, arg1, arg2. 
                                #   arg0 is the index of the acquisition 
                                #   arg1 is the index of the data bin
                                #   arg2 is the delay between starting the instruction and the next instruction
                                acquire_instruction = f"                    acquire {pulses.ro_pulses.index(pulses[n])},0,{delay_after_acquire}"
                                # Add the serial of the pulse as a comment
                                body += "\n" + acquire_instruction

                            else:
                                # Calculate the delay_after_play that is to be used as an argument to the play instruction
                                if len(pulses) > n + 1:
                                    # If there are more pulses to be played, the delay is the time between the pulse end and the next pulse start
                                    delay_after_play = pulses[n + 1].start - pulses[n].start
                                else:
                                    delay_after_play = sequence_total_duration - pulses[n].start
                                    
                                if delay_after_play < minimum_delay_between_instructions:
                                        raise Exception(f"The minimum delay between pulses is {minimum_delay_between_instructions}ns.")
                                
                                # Prepare play instruction: play arg0, arg1, arg2. 
                                #   arg0 is the index of the I waveform 
                                #   arg1 is the index of the Q waveform
                                #   arg2 is the delay between starting the instruction and the next instruction
                                play_instruction = f"                    play {sequencer.waveforms_buffer.unique_waveforms.index(pulses[n].waveform_i)},{sequencer.waveforms_buffer.unique_waveforms.index(pulses[n].waveform_q)},{delay_after_play}"
                                # Add the serial of the pulse as a comment
                                play_instruction += " "*(34-len(play_instruction)) + f"# play waveforms {pulses[n]}" 
                                body += "\n" + play_instruction

                        sequencer.program = header + body + footer

    def upload(self):
        """Uploads waveforms and programs all sequencers and arms them in preparation for execution."""
        if self.ports['i1'].hardware_demod_en or self.current_pulsesequence_hash != self.last_pulsequence_hash:
            self.last_pulsequence_hash = self.current_pulsesequence_hash
    
            # Setup

            for sequencer_number in self.used_sequencers:
                target  = self.device.sequencers[sequencer_number]
                self.set_device_parameter(target, 'sync_en', value = True)
                self.set_device_parameter(target, 'marker_ovr_en', value = True) # Default after reboot = False
                self.set_device_parameter(target, 'marker_ovr_value', value = 15) # Default after reboot = 0
            
            for sequencer_number in self.unused_sequencers:
                target  = self.device.sequencers[sequencer_number]
                self.set_device_parameter(target, 'sync_en', value = False)
                self.set_device_parameter(target, 'marker_ovr_en', value = True) # Default after reboot = False
                self.set_device_parameter(target, 'marker_ovr_value', value = 0) # Default after reboot = 0

            # Upload waveforms and program
            qblox_dict = {}
            sequencer:Sequencer
            for port in self.output_ports_keys:                
                for sequencer in self.sequencers[port]:
                    # Add sequence program and waveforms to single dictionary and write to JSON file
                    filename = f"{self.name}_sequencer{sequencer.number}_sequence.json"
                    qblox_dict[sequencer] = {
                        "waveforms": sequencer.waveforms,
                        "weights": sequencer.weights,
                        "acquisitions": sequencer.acquisitions,
                        "program": sequencer.program
                        }
                    with open(self.data_folder / filename, "w", encoding="utf-8") as file:
                        json.dump(qblox_dict[sequencer], file, indent=4)
                        
                    # Upload json file to the device sequencers
                    self.device.sequencers[sequencer.number].sequence(str(self.data_folder / filename))
        
        # Arm sequencers
        for sequencer in self.used_sequencers:
            self.device.arm_sequencer(sequencer)

        # DEBUG: QRM Print Readable Snapshot
        # print(self.name)
        # self.device.print_readable_snapshot(update=True)

    def play_sequence(self):
        """Executes the sequence of instructions."""
        # Start playing sequences in all sequencers
        for sequencer in self.used_sequencers:
            self.device.start_sequencer(sequencer)

    def play_sequence_and_acquire(self):
        """Executes the sequence of instructions and retrieves the readout results.
        """
        # Start playing sequences in all sequencers
        for sequencer_number in self.used_sequencers:
            self.device.start_sequencer(sequencer_number)
        

        # Retrieve data
        acquisition_results = {}
        for port in self.output_ports_keys:                
            for sequencer in self.sequencers[port]:
                sequencer_number = sequencer.number
                # Wait for the sequencer to stop with a timeout period of one minute.
                self.device.get_sequencer_state(sequencer_number, timeout = 1)
                #Wait for the acquisition to finish with a timeout period of one second.
                self.device.get_acquisition_state(sequencer_number, timeout = 1)

                for pulse in sequencer.pulses.ro_pulses:
                    acquisition_name = pulse.serial
                    #Move acquisition data from temporary memory to acquisition list.
                    self.device.store_scope_acquisition(sequencer_number, acquisition_name)
                    #Get acquisitions from instrument.
                    raw_results = self.device.get_acquisitions(sequencer_number)
                    i, q = self.process_acquisition_results(raw_results[acquisition_name], pulse, demodulate = (not self.ports['i1'].hardware_demod_en))
                    acquisition_results[pulse] = np.sqrt(i**2 + q**2), np.arctan2(q, i), i, q
                    # DEBUG: QRM Plot Incomming Pulses
                    # import qibolab.instruments.debug.incomming_pulse_plotting as pp
                    # pp.plot(raw_results)
        return acquisition_results

    def process_acquisition_results(self, acquisition_results, readout_pulse: Pulse,  demodulate = True):
        if demodulate:
            acquisition_frequency = readout_pulse.frequency

            #DOWN Conversion
            n0 = 0 
            n1 = self.acquisition_duration
            input_vec_I = np.array(acquisition_results["acquisition"]["scope"]["path0"]["data"][n0: n1])
            input_vec_Q = np.array(acquisition_results["acquisition"]["scope"]["path1"]["data"][n0: n1])
            input_vec_I -= np.mean(input_vec_I)
            input_vec_Q -= np.mean(input_vec_Q)

            modulated_i = input_vec_I
            modulated_q = input_vec_Q
        
            num_samples = modulated_i.shape[0]
            time = np.arange(num_samples) / PulseShape.SAMPLING_RATE

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
            i = np.sqrt(2) * [(val/int_len) for val in acquisition_results["acquisition"]["bins"]["integration"]["path0"]][0]
            q = np.sqrt(2) * [(val/int_len) for val in acquisition_results["acquisition"]["bins"]["integration"]["path1"]][0]
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

    DEFAULT_SEQUENCERS = {'o1': 0, 'o2': 1}

    property_wrapper = lambda parent, *parameter: property(
        lambda self: parent.device.get(parameter[0]), 
        lambda self,x: parent.set_device_parameter(parent.device, *parameter, value = x)
        )
    sequencer_property_wrapper = lambda parent, sequencer, *parameter: property(
        lambda self: parent.device.sequencers[sequencer].get(parameter[0]), 
        lambda self,x: parent.set_device_parameter(parent.device.sequencers[sequencer], *parameter, value = x)
        )

    def __init__(self, name, address):
        super().__init__(name, address)
        self.ports = {}
        self.output_ports_keys = ['o1', 'o2']
        self.sequencers:dict[WaveformsBuffer] = {'o1': [], 'o2': []}
        self.last_pulsequence_hash:int = 0
        self.current_pulsesequence_hash:int
        self.device_parameters = {}
        self.free_sequencers = [2,3,4,5]

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
                    'gain': self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS['o1'], 'gain_awg_path0', 'gain_awg_path1'), 
                    'hardware_mod_en': self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS['o1'], 'mod_en_awg'),
                    'nco_freq': self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS['o1'], 'nco_freq'),
                    'nco_phase_offs': self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS['o1'], 'nco_phase_offs')
                    })()

                self.ports['o2'] = type(f'port_o1', (), 
                    {'attenuation': self.property_wrapper('out1_att'), 
                    'lo_enabled': self.property_wrapper('out1_lo_en'), 
                    'lo_frequency': self.property_wrapper('out1_lo_freq'), 
                    'gain': self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS['o2'], 'gain_awg_path0', 'gain_awg_path1'), 
                    'hardware_mod_en': self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS['o2'], 'mod_en_awg'),
                    'nco_freq': self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS['o2'], 'nco_freq'),
                    'nco_phase_offs': self.sequencer_property_wrapper(self.DEFAULT_SEQUENCERS['o2'], 'nco_phase_offs')
                    })()


                self.cluster = cluster
                self.is_connected = True
                self.set_device_parameter(self.device, 'out0_offset_path0', 'out0_offset_path1', value = 0) # Default after reboot = 7.625
                self.set_device_parameter(self.device, 'out1_offset_path0', 'out1_offset_path1', value = 0) # Default after reboot = 7.625
                                    
                for target in [self.device.sequencers[self.DEFAULT_SEQUENCERS['o1']], self.device.sequencers[self.DEFAULT_SEQUENCERS['o2']]]:

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

                self.set_device_parameter(self.device.sequencers[self.DEFAULT_SEQUENCERS['o1']], 'channel_map_path0_out0_en', 'channel_map_path1_out1_en', value = True)
                self.set_device_parameter(self.device.sequencers[self.DEFAULT_SEQUENCERS['o1']], 'channel_map_path0_out2_en', 'channel_map_path1_out3_en', value = False)
                self.set_device_parameter(self.device.sequencers[self.DEFAULT_SEQUENCERS['o2']], 'channel_map_path0_out0_en', 'channel_map_path1_out1_en', value = False)
                self.set_device_parameter(self.device.sequencers[self.DEFAULT_SEQUENCERS['o2']], 'channel_map_path0_out2_en', 'channel_map_path1_out3_en', value = True)
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
        self.device_num_ports = 2
        if self.is_connected:
            # Reset
            # if self.current_pulsesequence_hash != self.last_pulsequence_hash:
            #     self.device_parameters = {}
            # TODO: Check when Reset was necessary

            # Load settings
            self.sampling_rate = kwargs['sampling_rate']

            self.channel_port_map = kwargs['channel_port_map']
            self.port_channel_map = {v: k for k, v in self.channel_port_map.items()}
            self.channels = list(self.channel_port_map.keys())

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

    def process_pulse_sequence(self, instrument_pulses: PulseSequence, nshots, repetition_duration):
        """
        Processes a list of pulses, generating the waveforms and sequence program required by the instrument to synthesise them.
        
        Args:
        channel_pulses (dict): a dictionary of {channel (int): pulses (list)}
        nshots (int): the number of times the sequence of pulses will be repeated
        """

        # Save the hash of the current sequence of pulses.
        self.current_pulsesequence_hash = hash((instrument_pulses, nshots, repetition_duration, self.ports['o1'].hardware_mod_en, self.ports['o2'].hardware_mod_en))
        
        # Check if the sequence to be processed is the same as the last one. 
        # If so, there is no need to generate new waveforms and program
        if self.current_pulsesequence_hash != self.last_pulsequence_hash:

            # split the collection of instruments pulses by ports
            for port in self.ports:
                port_pulses: PulseSequence = instrument_pulses.get_channel_pulses(self.port_channel_map[port])
                if not port_pulses.is_empty:
                    if port_pulses.pulses_overlap:
                        # TODO: Urgently needed in order to implement multiplexed readout
                        raise NotImplementedError("Overlaping pulses on the same channel are not yet supported.")

                    sequencer = Sequencer(self.DEFAULT_SEQUENCERS[port])
                    self.sequencers[port] = [sequencer]

                    port_pulses_to_be_processed = port_pulses.shallow_copy()
                    while not port_pulses_to_be_processed.is_empty:
                        try:
                            pulse: Pulse = port_pulses_to_be_processed[0]
                            if self.ports[port].hardware_mod_en:
                                pulse.waveform_i, pulse.waveform_q = pulse.envelope_waveforms
                            else:
                                pulse.waveform_i, pulse.waveform_q = pulse.modulated_waveforms
                            sequencer.waveforms_buffer.add_waveforms(pulse.waveform_i, pulse.waveform_q)
                            sequencer.pulses.add(pulse)
                            port_pulses_to_be_processed.remove(pulse)
                        except WaveformsBuffer.NotEnoughMemory:
                            if len(pulse.waveform_i) + len(pulse.waveform_q) > WaveformsBuffer.SIZE:
                                raise NotImplementedError(f"Pulses with waveforms longer than the memory of a sequencer ({WaveformsBuffer.SIZE // 2}) are not supported.")
                            if len(self.free_sequencers) > 0:
                                next_sequencer_number = self.free_sequencers.pop(0)
                                for parameter in self.device.sequencers[self.DEFAULT_SEQUENCERS[port]].parameters:
                                    value = self.device.sequencers[self.DEFAULT_SEQUENCERS[port]].get(param_name = parameter)
                                    if not value is None:
                                        target  = self.device.sequencers[next_sequencer_number]
                                        self.set_device_parameter(target, parameter, value = value)
                                sequencer = Sequencer(next_sequencer_number)
                                self.sequencers[port].append(sequencer)
                            else:
                                raise Exception(f"The number of sequencers requried to play the sequence exceeds the number available {self.device_num_sequencers}.")

                    for index, waveform in enumerate(sequencer.waveforms_buffer.unique_waveforms):
                        sequencer.waveforms[waveform.serial] = {"data": waveform.data.tolist(), "index": index}

                    for acquisition_index, pulse in enumerate(sequencer.pulses.ro_pulses):
                        sequencer.acquisitions[pulse.serial] = {"num_bins": 1, "index":acquisition_index}


                    # Program
                    minimum_delay_between_instructions = 4
                    wait_loop_step:int = 1000

                    pulses = port_pulses
                    sequence_total_duration = pulses.start + pulses.duration + minimum_delay_between_instructions # the minimum delay between instructions is 4ns
                    time_between_repetitions = repetition_duration - sequence_total_duration
                    assert time_between_repetitions > 0

                    wait_time = time_between_repetitions
                    extra_wait = wait_time % wait_loop_step
                    while wait_time > 0 and extra_wait < 4 :
                        wait_loop_step += 1
                        extra_wait = wait_time % wait_loop_step
                    num_wait_loops = (wait_time - extra_wait) // wait_loop_step

                    header = f"""
                    move {nshots},R0 # nshots
                    nop
                    wait_sync {minimum_delay_between_instructions}
                    loop:
                    reset_ph"""
                    body = ""

                    footer = f"""
                        # wait {wait_time} ns"""
                    if num_wait_loops > 0:
                        footer += f"""
                        move {num_wait_loops},R2
                        nop
                        waitloop2:
                            wait {wait_loop_step}
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
                    wait_time = pulses[0].start
                    extra_wait = wait_time % wait_loop_step
                    while wait_time > 0 and extra_wait < 4 :
                        wait_loop_step += 1
                        extra_wait = wait_time % wait_loop_step
                    num_wait_loops = (wait_time - extra_wait) // wait_loop_step

                    if wait_time > 0:
                        initial_wait_instruction = f"""
                        # wait {wait_time} ns"""
                        if num_wait_loops > 0:
                            initial_wait_instruction += f"""
                        move {num_wait_loops},R1
                        nop
                        waitloop1:
                            wait {wait_loop_step}
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

                    for n in range(pulses.count):
                        # Calculate the delay_after_play that is to be used as an argument to the play instruction
                        if len(pulses) > n + 1:
                            # If there are more pulses to be played, the delay is the time between the pulse end and the next pulse start
                            delay_after_play = pulses[n + 1].start - pulses[n].start
                        else:
                            delay_after_play = sequence_total_duration - pulses[n].start
                            
                        if delay_after_play < minimum_delay_between_instructions:
                                raise Exception(f"The minimum delay between pulses is {minimum_delay_between_instructions}ns.")
                        
                        # Prepare play instruction: play arg0, arg1, arg2. 
                        #   arg0 is the index of the I waveform 
                        #   arg1 is the index of the Q waveform
                        #   arg2 is the delay between starting the instruction and the next instruction
                        play_instruction = f"                    play {sequencer.waveforms_buffer.unique_waveforms.index(pulses[n].waveform_i)},{sequencer.waveforms_buffer.unique_waveforms.index(pulses[n].waveform_q)},{delay_after_play}"
                        # Add the serial of the pulse as a comment
                        play_instruction += " "*(34-len(play_instruction)) + f"# play waveforms {pulses[n]}" 
                        body += "\n" + play_instruction

                    sequencer.program = header + body + footer

    def upload(self):
        """Uploads waveforms and programs all sequencers and arms them in preparation for execution."""
        if self.current_pulsesequence_hash != self.last_pulsequence_hash:
            self.last_pulsequence_hash = self.current_pulsesequence_hash
    
            # Setup
            self.used_sequencers = []
            for port in self.output_ports_keys:                
                for sequencer in self.sequencers[port]:
                    self.used_sequencers.append(sequencer.number)
            for sequencer_number in self.used_sequencers:
                target  = self.device.sequencers[sequencer_number]
                self.set_device_parameter(target, 'sync_en', value = True)
                self.set_device_parameter(target, 'marker_ovr_en', value = True) # Default after reboot = False
                self.set_device_parameter(target, 'marker_ovr_value', value = 15) # Default after reboot = 0
            
            self.unused_sequencers = []
            for n in range(self.device_num_sequencers):
                if not n in self.used_sequencers:
                    self.unused_sequencers.append(n)
            for sequencer_number in self.unused_sequencers:
                target  = self.device.sequencers[sequencer_number]
                self.set_device_parameter(target, 'sync_en', value = False)
                self.set_device_parameter(target, 'marker_ovr_en', value = True) # Default after reboot = False
                self.set_device_parameter(target, 'marker_ovr_value', value = 0) # Default after reboot = 0

            # Upload waveforms and program
            qblox_dict = {}
            sequencer:Sequencer
            for port in self.output_ports_keys:                
                for sequencer in self.sequencers[port]:
                    # Add sequence program and waveforms to single dictionary and write to JSON file
                    filename = f"{self.name}_sequencer{sequencer.number}_sequence.json"
                    qblox_dict[sequencer] = {
                        "waveforms": sequencer.waveforms,
                        "weights": sequencer.weights,
                        "acquisitions": sequencer.acquisitions,
                        "program": sequencer.program
                        }
                    with open(self.data_folder / filename, "w", encoding="utf-8") as file:
                        json.dump(qblox_dict[sequencer], file, indent=4)
                        
                    # Upload json file to the device sequencers
                    self.device.sequencers[sequencer.number].sequence(str(self.data_folder / filename))
        
        # Arm sequencers
        for sequencer in self.used_sequencers:
            self.device.arm_sequencer(sequencer)

        # DEBUG: QRM Print Readable Snapshot
        # print(self.name)
        # self.device.print_readable_snapshot(update=True)


    def play_sequence(self):
        """Executes the sequence of instructions."""
        for sequencer in self.used_sequencers:
            # Start sequencer
            self.device.start_sequencer(sequencer)

    def start(self):
        pass

    def stop(self):
        """Stops all sequencers"""
        self.device.stop_sequencer()

    def disconnect(self):
        pass
    

