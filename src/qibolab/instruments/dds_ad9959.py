# -*- coding: utf-8 -*-
import sys
import time

import numpy as np
import usb

from qibolab.instruments.abstract import AbstractInstrument, InstrumentException
from qibolab.instruments.customhandler import DeviceHandle
from qibolab.paths import qibolab_folder


# this code is developed on the exisitng minimalistic driver located on: https://github.com/Schlabonski/evalcontrol by the user @schlabonski
# the additions include programming for all the functions for the dds ad9959 which was not present intially
# modulation using the pc and the implementation of the abstract class is left
class AD9959:
    def __init__(
        self,
        vid=0x0456,
        pid=0xEE25,
        port_numbers=None,
        bus_number=None,
        auto_update=True,
        rfclk=20e6,
        channel=0,
        clkfactor=1,
    ):

        self.channel = channel

        # find all usb devices with matching vid/pid
        devs = list(usb.core.find(idVendor=vid, idProduct=pid, find_all=True))
        dev = None
        dev_mess = "No devices with matching vID/pID {}/{} found!".format(
            hex(vid), hex(pid)
        )
        assert len(devs) > 0, dev_mess
        # if more than one AD9959 is present, decide by usb port address
        if len(devs) > 1:
            assert (
                port_numbers is not None and bus_number is not None
            ), "More than one AD9959 present. Specify USB bus and port numbers!"
            for d in devs:
                if d.port_numbers == port_numbers and d.bus == bus_number:
                    dev = d
                    break
            assert (
                dev is not None
            ), "No matching device was found. Check bus and port numbers!"

        else:
            dev = devs[0]

        dev.set_configuration()
        cnf = dev.configurations()[0]
        intf = cnf[(0, 0)]
        self.dev = dev

        # retrieve important endpoints of usb controller
        self._ep1 = intf[0]
        self._ep81 = intf[1]
        self._ep4 = intf[3]
        self._ep88 = intf[5]

        # set default values for physical variables

        self.ref_clock_frequency = rfclk
        self.system_clock_frequency = rfclk

        # set default value for auto IO update
        self.auto_update = auto_update

        # try to access device, it might still be in use by another handler,
        # in this case, reset it
        try:
            print("this is the initialiser of the board with default values")
            self.set_clock_multiplier(clkfactor)
        except usb.USBError:
            self._reset_usb_handler()
        print("this is the initialiser of the board with default values")
        self.set_clock_multiplier(clkfactor)
        print("resetting the values to the default values")
        self.__reset_default_values__()

    def __del__(self):
        """disconnects the device assosciated with usb"""
        usb.util.dispose_resources(self.dev)

    def _reset_usb_handler(self):
        """Resets the usb handler via which communication takes place.

        This method can be used to prevent USBErrors that occur when communication with the device times out
        because it is still in use by another process.
        """
        self.dev.reset()

    def _write_to_dds_register(self, register, word):
        """Writes a word to the given register of the dds chip.

        Any words that are supposed to be written directly to the DDS
        chip are sent to endpoint 0x04 of the microcrontroler.

        :register: hex
            ID of the target register on the DDS chip.
        :word: bytearray
            word that is written to the respective register. Each byte in
            the bytearray should be either 0x00 or 0x01.
            the default value that is followed is msb first format

        """
        # express the register name as a binary. Maintain the format that is
        # understood by endpoint 0x04. The first bit signifies read/write,
        # the next 7 bits specify the register
        register_bin = bin(register).lstrip("0b")
        if len(register_bin) < 7:
            register_bin = (7 - len(register_bin)) * "0" + register_bin

        register_bin = "".join(" 0" + b for b in register_bin)

        # construct the full message that is sent to the endpoint
        message = "00" + register_bin + " " + word
        message = bytearray.fromhex(message)

        # endpoint 0x04 will forward the word to the specified register
        with DeviceHandle(self.dev) as dh:
            dh.bulkWrite(self._ep4, message)

    def _read_from_register(self, register, size):
        """Reads a word of length `size` from `register` of the DDS chip.

        :register: hex
            register of the DDS chip from which to read
        :size: int
            length of the word (in bytes) that is read from the register
        :returns: bytearray
            the readout of the register

        """
        # convert the size to hex
        size_hex = hex(size).lstrip("0x")
        if len(size_hex) < 2:
            size_hex = "0" + size_hex

        # set the controler to readback mode
        begin_readback = bytearray.fromhex("07 00 " + size_hex)
        with DeviceHandle(self.dev) as dh:
            dh.bulkWrite(self._ep1, begin_readback)

        # construct the command to read out the register
        register_bin = bin(register).lstrip("0b")
        if len(register_bin) < 7:
            register_bin = (7 - len(register_bin)) * "0" + register_bin
        register_bin = "".join(" 0" + b for b in register_bin)
        readout_command = bytearray.fromhex("01" + register_bin)
        with DeviceHandle(self.dev) as dh:
            dh.bulkWrite(self._ep4, readout_command)

        time.sleep(0.1)
        # read the message from endpoint 88
        with DeviceHandle(self.dev) as dh:
            readout = dh.bulkRead(self._ep88, size=size)

        # turn off readback mode
        end_readback = bytearray.fromhex("04 00")
        with DeviceHandle(self.dev) as dh:
            dh.bulkWrite(self._ep1, end_readback)
        return readout

    def _load_IO(self):
        """Loads the I/O update line (same as GUI load function)."""
        load_message = bytearray.fromhex("0C 00")
        with DeviceHandle(self.dev) as dh:
            dh.bulkWrite(self._ep1, load_message)
            readout = dh.bulkRead(self._ep81, 1)
        return readout

    def _update_IO(self):
        """Updates the IO to the DDS chip (same as GUI function)."""
        update_message = bytearray.fromhex("0C 10")
        with DeviceHandle(self.dev) as dh:
            dh.bulkWrite(self._ep1, update_message)
            readout = dh.bulkRead(self._ep81, 1)
        return readout

    def __reset_default_values__(self):
        """
        sets the default values of all the registers as mentioned in the datasheet
        check if the multiple I/O function is required for writing the fucntion into the registers
        """
        csr = "11110000"
        csr_default = "".join(" 0" + b for b in csr)
        csr_default = csr_default[1:]
        fr1 = "000000000000000000000000"
        fr1_default = "".join(" 0" + b for b in fr1)
        fr1_default = fr1_default[1:]
        fr2 = "0000000000000000"
        fr2_default = "".join(" 0" + b for b in fr2)
        fr2_default = fr2_default[1:]
        cfr = "000000100000001100000000"
        cfr_default = "".join(" 0" + b for b in cfr)
        cfr_default = cfr_default[1:]
        cftw0 = "00000000000000000000000000000000"
        cftw0_default = "".join(" 0" + b for b in cftw0)
        cftw0_default = cftw0_default[1:]
        cpow0 = "0000000000000000"
        cpow0_default = "".join(" 0" + b for b in cpow0)
        cpow0_default = cpow0_default[1:]
        acr = "000000000000000000000000"
        acr_default = "".join(" 0" + b for b in acr)
        acr_default = acr_default[1:]

        self._write_to_dds_register(0x00, csr_default)
        self._load_IO()
        if self.auto_update:
            self._update_IO()
        self._write_to_dds_register(0x01, fr1_default)
        self._load_IO()
        if self.auto_update:
            self._update_IO()
        self._write_to_dds_register(0x02, fr2_default)
        self._load_IO()
        if self.auto_update:
            self._update_IO()
        self._write_to_dds_register(0x03, cfr_default)
        self._load_IO()
        if self.auto_update:
            self._update_IO()
        self._write_to_dds_register(0x04, cftw0_default)
        self._load_IO()
        if self.auto_update:
            self._update_IO()
        self._write_to_dds_register(0x05, cpow0_default)
        self._load_IO()
        if self.auto_update:
            self._update_IO()
        self._write_to_dds_register(0x06, acr_default)
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def set_clock_multiplier(self, factor):
        """Sets the multiplier for the reference clock.
        The system clock frequency is given by
           f_sys = multiplier * f_ref
        where f_ref is the frequency of the reference clock.
        :factor: int
           Multiplying factor between 4 and 20. A `factor` of 1 disables
           multiplication.

        """
        # in case of a factor of one, we want to disable multiplication
        if (factor >= 4) & (factor <= 20):
            self.system_clock_frequency = self.ref_clock_frequency * factor
            print("multipier-active")

        else:
            factor = 0
            self.system_clock_frequency = self.ref_clock_frequency
            print("multiplier-not-active")
        # construct the multiplier word
        multi_bin = bin(factor).lstrip("0b")
        if len(multi_bin) < 5:
            multi_bin = (5 - len(multi_bin)) * "0" + multi_bin
        # get the current state of function register 1
        fr1_old = self._read_from_register(0x01, 24)
        fr1_old_bitstring = "".join(str(b) for b in fr1_old)

        # update the multiplier section of the bitstring
        fr1_new_bitstring = fr1_old_bitstring[0] + multi_bin + fr1_old_bitstring[6:]
        if self.system_clock_frequency > 255e6:
            l = list(fr1_new_bitstring)
            l[0] = "1"
            fr1_new_bitstring = "".join(l)
            print("System clock exceeds 255MHz, VCO gain bit was set to True!")
        else:
            l = list(fr1_new_bitstring)
            l[0] = "0"
            fr1_new_bitstring = "".join(l)
            print("System clock does not exceed 255 MHz. VCO gain bit is set to false ")
        fr1_word = "".join(" 0" + b for b in fr1_new_bitstring)[1:]

        # write the new multiplier to the register on dds chip
        self._write_to_dds_register(0x01, fr1_word)
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def _channel_select(self, channel):
        """Selects the chosen channels in the channel select register. this should be from 0 1 2 and 3. These correspond to channel 0, channel 1, channel 2 and channel 3.

        :channels: int or list
            ID or list of the channel IDs to select e.g. [0,2,3]
            select 0 to enable channel 0
            select 1 to enable channel 1
            select 2 to enable channel 2
            select 3 to enable channel 3

        """
        if np.issubdtype(type(channel), np.integer):
            channels = [channel]
        else:
            channels = [c for c in channel]

        # set the channels in the channel select register
        channel_select_bin = list("0000")
        print(channel_select_bin)
        for ch in channels:
            channel_select_bin[ch] = "1"
        channel_select_bin = channel_select_bin[::-1]  # we have inverse order
        print(channel_select_bin)
        channel_select_bin = "".join(channel_select_bin)
        # in the register
        csr_old = self._read_from_register(0x00, 8)
        csr_old_bin = "".join(str(b) for b in csr_old)
        csr_new_bin = channel_select_bin[:4] + csr_old_bin[4:]
        print(csr_new_bin)
        if csr_new_bin[4] == "1":
            sys.exit("this bit must be 0 and is predetermined in the datasheet")
        csr_word = "".join(" 0" + b for b in csr_new_bin)
        csr_word = csr_word[1:]
        print(csr_word)

        self._write_to_dds_register(0x00, csr_word)
        print("channels enabled", channels)

    def precompute_frequency_word(self, channel, frequency):
        """Precomputes a frequency tuning word for given channel(s) and frequency.
        can be used to set a single frequency for multiple channels as well as set different frequencies for different channels
        The current implementation of this method is repetitive and clumsy!

        :channel:  int or list
            Channel number or list of channel numbers.
        :frequency: float
            frequency to be set on channel(s)
        :returns: bytearray, bytearray
           The message to select the channels and set the frequency word, which can both
           be sent to self._ep4

        """
        if np.issubdtype(type(channel), np.integer):
            channels = [channel]
        else:
            channels = [c for c in channel]

        # set the channels in the channel select register
        channel_select_bin = list("0000")
        for ch in channels:
            channel_select_bin[ch] = "1"
        channel_select_bin = channel_select_bin[::-1]  # we have inverse order
        channel_select_bin = "".join(channel_select_bin)
        # in the register
        # preserve the information that is stored in the CSR
        csr_old = self._read_from_register(0x00, 8)
        csr_old_bin = "".join(str(b) for b in csr_old)
        csr_new_bin = channel_select_bin[:4] + csr_old_bin[4:]
        if csr_new_bin[4] != 0:
            sys.exit("this bit must be 0 and is predetermined in the datasheet")
        csr_word = "".join(" 0" + b for b in csr_new_bin)
        csr_word = csr_word[1:]
        word = csr_word

        # express the register name as a binary. Maintain the format that is
        # understood by endpoint 0x04. The first bit signifies read/write,
        # the next 7 bits specify the register
        register_bin = bin(0x00).lstrip("0b")
        if len(register_bin) < 7:
            register_bin = (7 - len(register_bin)) * "0" + register_bin

        register_bin = "".join(" 0" + b for b in register_bin)

        # construct the full message that is sent to the endpoint
        message = "00" + register_bin + " " + word
        message_channel_select = bytearray.fromhex(message)

        ## Now compute the same message to select the frequency
        assert frequency <= self.system_clock_frequency, (
            "Frequency should not"
            + " exceed system clock frequency! System clock frequency is {}Hz".format(
                self.system_clock_frequency
            )
        )

        # calculate the fraction of the full frequency
        fraction = frequency / self.system_clock_frequency
        fraction_bin = bin(round(fraction * (2**32 - 1))).lstrip(
            "0b"
        )  # full range are 32 bit
        if len(fraction_bin) < 32:
            fraction_bin = (32 - len(fraction_bin)) * "0" + fraction_bin
        closest_possible_value = (
            int(fraction_bin, base=2) / (2**32 - 1) * self.system_clock_frequency
        )
        print(
            "Frequency of channel {1} encoded as closest possible value {0}MHz".format(
                closest_possible_value / 1e6, channel
            )
        )

        # set the frequency word in the frequency register
        frequency_word = "".join(" 0" + b for b in fraction_bin)
        frequency_word = frequency_word[1:]
        word = frequency_word

        # express the register name as a binary. Maintain the format that is
        # understood by endpoint 0x04. The first bit signifies read/write,
        # the next 7 bits specify the register
        register_bin = bin(0x04).lstrip("0b")  # 0x04 = frequency register
        if len(register_bin) < 7:
            register_bin = (7 - len(register_bin)) * "0" + register_bin

        register_bin = "".join(" 0" + b for b in register_bin)

        # construct the full message that is sent to the endpoint
        message = "00" + register_bin + " " + word
        message_frequency_word = bytearray.fromhex(message)

        return message_channel_select, message_frequency_word

    def set_precomputed_frequency(self, message_channel_select, message_frequency_word):
        """This method sets the frequency from precomputed byte-encoded words.

        The input for this method should be the ouput of self.precompute_frequency_word.
        This method only send the precomputed byte arrays to enpoint 4 of the USB handler and
        thus be faster than the self.set_frequency method.

        :message_channel_select: bytearray
            The bytearray that encodes a given channel selection.

        :message_frequency_word: bytearray
            Encodes the setting of the frequency word.

        """
        with DeviceHandle(self.dev) as dh:
            dh.bulkWrite(self._ep4, message_channel_select)
            dh.bulkWrite(self._ep4, message_frequency_word)

    def amplitude_multiplier_enable(self, channel):

        """
        :channel: int
        Select and activate for 1 channel
        """

        if channel is None:
            channel = self.channel

        # select the chosen channels
        self._channel_select(channel)

        # load the current amplitude control register word
        acr = self._read_from_register(0x06, 24)
        acr[11] = 1
        # acr[12]=0
        acr_new_bin = "".join(" 0" + str(b) for b in acr)
        acr_new_bin = acr_new_bin[1:]
        self._write_to_dds_register(0x06, acr_new_bin)

        # update I/O
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def auto_ru_rd_mode_enable(self, channel):

        """
        :channel: int or list
            Channels for which the amplitude scaling should be set to the manual mode.
        """

        if channel is None:
            channel = self.channel

        # select the chosen channels
        self._channel_select(channel)

        # load the current amplitude control register word
        acr = self._read_from_register(0x06, 24)
        if acr[11] == 1:
            acr[12] = 1
        else:
            assert acr[11] is None, "Enable the amplitude multiplier bit"

        acr_new_bin = "".join(" 0" + str(b) for b in acr)
        acr_new_bin = acr_new_bin[1:]
        self._write_to_dds_register(0x06, acr_new_bin)

        # update I/O
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def auto_ru_rd_mode_disable(self, channel):

        """
        :channel: int or list
            Channels for which the amplitude scaling should be set to the manual mode.
        """

        if channel is None:
            channel = self.channel

        # select the chosen channels
        self._channel_select(channel)

        # load the current amplitude control register word
        acr = self._read_from_register(0x06, 24)
        if acr[11] == 1:
            acr[12] = 0
        else:
            assert acr[11] is None, "Enable the amplitude multiplier bit"

        acr_new_bin = "".join(" 0" + str(b) for b in acr)
        acr_new_bin = acr_new_bin[1:]
        self._write_to_dds_register(0x06, acr_new_bin)

        # update I/O
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    # generally is present during all kinds of wave generation,

    def amplitude_multiplier_disable(self, channel):

        """
        :channel: int or list
            Channels for which the amplitude scaling should be set to the manual mode.
        """

        if channel is None:
            channel = self.channel

        # select the chosen channels
        self._channel_select(channel)
        acr = self._read_from_register(0x06, 24)
        acr[11] = 0
        acr_new_bin = "".join(" 0" + str(b) for b in acr)
        acr_new_bin = acr_new_bin[1:]
        self._write_to_dds_register(0x06, acr_new_bin)

        # update I/O
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def serial_I_O_Mode(self, mode):
        """
        function to set the serial mode of to 2 wire mode, 3 wire mode
        input 0 for single bit serial(2 wire mode)
        input 1 for single bit serial(3 wire mode)
        input 2 for 2 bit serial mode
        input 3 for 4 bit serial mode
        """
        # this needs input on the SDIO_pin add exception here
        # 1. read the old CFR content
        csr_old = self._read_from_register(0x00, 8)
        # csr_new = csr_old
        csr_old_bin = "".join(str(b) for b in csr_old)
        # 2. replace the CFR by one with updated LS enable bit
        if mode == 0:
            enable_bin = "00"
        elif mode == 1:
            enable_bin = "01"
        elif mode == 2:
            enable_bin = "10"
        elif mode == 3:
            enable_bin = "11"
        else:
            assert mode is None, "invalid input, moving to default value"

        csr_new_bin = csr_old_bin[:4] + enable_bin + csr_old_bin[7]
        if csr_new_bin[4] != 0:
            sys.exit("this bit must be 0 and is predetermined in the datasheet")
        csr_word_new = "".join(" 0" + b for b in csr_new_bin)  # translate to bytes
        csr_word_new = csr_word_new[1:]  # crop the first white space
        self._write_to_dds_register(0x00, csr_word_new)
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def charge_pump_control(self, mode):
        """
        function to set the current in the charge pump control


        function to set the serial mode of to 2 wire mode, 3 wire mode
        input 0 for the charge pump current is 75 μA.
        input 1 for pump current is 100 μA.
        input 2 for charge pump current is 125 μA.
        input 3 for charge pump current is 150 μA.
        """
        # 1. read the old CFR content
        fr1_old = self._read_from_register(0x01, 24)
        fr1_old_bin = "".join(str(b) for b in fr1_old)
        # 2. replace the CFR by one with updated LS enable bit
        if mode == 0:
            enable_bin = "00"
        elif mode == 1:
            enable_bin = "01"
        elif mode == 2:
            enable_bin = "10"
        elif mode == 3:
            enable_bin = "11"
        else:
            assert mode is None, "invalid input. Value remains as it is="
        fr1_new_bin = fr1_old_bin[:6] + enable_bin + fr1_old_bin[8:]
        fr1_word_new = "".join(" 0" + b for b in fr1_new_bin)  # translate to bytes
        fr1_word_new = fr1_word_new[1:]  # crop the first white space
        self._write_to_dds_register(0x01, fr1_word_new)
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def ru_rd_amplitude(self, mode):
        """
        function to change the amplitude modulation using the ru/rd mode
        0  for disabling RU/RD
        01 P2 and P3 available for RU/RD operation
        10 only P3 available for RU/RD operation
        11 only SDIO_1, SDIO_2 and SDIO_3 are available for operation. The serial I/O is now used only in 1 bit mode

        """
        fr1_old = self._read_from_register(0x01, 24)
        fr1_old_bin = "".join(str(b) for b in fr1_old)
        # 2. replace the CFR by one with updated LS enable bit
        if mode == 0:
            enable_bin = "00"
        elif mode == 1:
            enable_bin = "01"
        elif mode == 2:
            enable_bin = "10"
        elif mode == 3:
            enable_bin = "11"
        else:
            assert mode is None, "invalid input. Value remains as it is="
        fr1_new_bin = fr1_old_bin[:12] + enable_bin + fr1_old_bin[14:]
        fr1_word_new = "".join(" 0" + b for b in fr1_new_bin)  # translate to bytes
        fr1_word_new = fr1_word_new[1:]  # crop the first white space
        self._write_to_dds_register(0x01, fr1_word_new)
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def modulation_level(self, mode):
        """
        function to select the modulation.
        select 0 for two-level modulation
        select 1 for four-level modulation
        select 2 for eight-level modulation
        select 3 for 16 level modulation
        """
        fr1_old = self._read_from_register(0x01, 24)
        fr1_old_bin = "".join(str(b) for b in fr1_old)
        # 2. replace the CFR by one with updated LS enable bit
        if mode == 0:
            enable_bin = "00"
        elif mode == 1:
            enable_bin = "01"
        elif mode == 2:
            enable_bin = "10"
        elif mode == 3:
            enable_bin = "11"
        else:
            assert mode is None, "invalid input. Value remains as it is="
        fr1_new_bin = fr1_old_bin[:14] + enable_bin + fr1_old_bin[16:]
        fr1_word_new = "".join(" 0" + b for b in fr1_new_bin)  # translate to bytes
        fr1_word_new = fr1_word_new[1:]  # crop the first white space
        self._write_to_dds_register(0x01, fr1_word_new)
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def two_level_modulation_no_ru_rd(self):
        """
        function to create 2 level modulation without RU/RD. The profile pic configuration bits are dont care.
        P0 controls the modulation of CH0, P1= CH1, P2 = CH2 and P3 = CH3
        """
        self.ru_rd_amplitude(0)
        # fr1_old = self._read_from_register(0x01,24)
        # fr1_old_bin = ''.join(str(b) for b in fr1_old)
        print("changing the modulatiion mode to 2 level")
        self.modulation_level(0)
        print(
            "this is dont care for the profile pin configuration, hence the bits for the profile pin are being set to 0"
        )

    def four_level_modulation_no_ru_rd(self, mode):
        """
        function to set the configuration of the profile pin and the SDIO_x pins for different modulation modes
        there are 4 types of modulation that are available
        0 for 2 level modulation all channels, no RU/RD, P0 = CH0, P1 = CH1, P2 = CH2, P3 = CH3, the bits for controlling the profile pin configuration is dont care
        1 for 4 level modulation on CH0 and CH1, no RU/RD, P0 = CH0, P1 = CH0, P2 = CH1, P3 = CH1,
        2 for 4 level modulation on CH0 and CH2 no RU/RD, P0 = CH0, P1 = CH0, P2 = CH2, P3 = CH2
        3 for 4 level modulation on CH0 and CH3, no RU/RD, P0 = CH0, P1 = CH0, P2 = CH3, P3 = CH3
        4 for 4 level modulation on CH1 and CH2 no RU/RD, P0 = CH1, P1 = CH1, P2 = CH2, P3 = CH2
        5 for 4 level modulation on CH1 and CH3, no RU/RD, P0 = CH1, P1 = CH1, P2 = CH3, P3 = CH3
        6 for 4 level modulation on CH2 and CH3, no RU/RD, P0 = CH2, P1 = CH2, P2 = CH3, P3 = CH3
        """
        self.ru_rd_amplitude(0)
        self.modulation_level(1)
        fr1_old = self._read_from_register(0x01, 24)
        fr1_old_bin = "".join(str(b) for b in fr1_old)
        if mode == 1:
            enable_bin = "000"
        elif mode == 2:
            enable_bin = "001"
        elif mode == 3:
            enable_bin = "010"
        elif mode == 4:
            enable_bin = "011"
        elif mode == 5:
            enable_bin = "100"
        elif mode == 6:
            enable_bin = "101"
        else:
            assert mode is None, "invalid input. Value remains as it is"
        fr1_new_bin = fr1_old_bin[:9] + enable_bin + fr1_old_bin[12:]
        fr1_word_new = "".join(" 0" + b for b in fr1_new_bin)  # translate to bytes
        fr1_word_new = fr1_word_new[1:]  # crop the first white space
        self._write_to_dds_register(0x01, fr1_word_new)
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def eight_level_modulation_no_ru_rd(self, mode):
        """
        function to set the configuration of the profile pin and the SDIO_x pins for different modulation modes
        there are 4 types of modulation that are available
        mode 0 for eight level modulation on ch0, no ru/rd
        mode 1 for eight level modulation on ch1, no ru/rd
        mode 2 for eight level modulation on ch2, no ru/rd
        mode 3 for eight level modulation on ch3, no ru/rd
        """
        # for dont care bits, generally the programmers make it 0

        self.ru_rd_amplitude(0)
        self.modulation_level(2)
        fr1_old = self._read_from_register(0x01, 24)
        fr1_old_bin = "".join(str(b) for b in fr1_old)
        if mode == 0:
            enable_bin = "000"
        elif mode == 1:
            enable_bin = "001"
        elif mode == 2:
            enable_bin = "010"
        elif mode == 3:
            enable_bin = "011"
        else:
            assert mode is None, "invalid input. Value remains as it is"
        fr1_new_bin = fr1_old_bin[:9] + enable_bin + fr1_old_bin[12:]
        fr1_word_new = "".join(" 0" + b for b in fr1_new_bin)  # translate to bytes
        fr1_word_new = fr1_word_new[1:]  # crop the first white space
        self._write_to_dds_register(0x01, fr1_word_new)
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def sixteen_level_modulation_no_ru_rd(self, mode):
        """
        function to set the configuration of the profile pin and the SDIO_x pins for different modulation modes
        there are 4 types of modulation that are available
        mode 0 for 16 level modulation on ch0, no ru/rd
        mode 1 for 16 level modulation on ch1, no ru/rd
        mode 2 for 16 level modulation on ch2, no ru/rd
        mode 3 for 16 level modulation on ch3, no ru/rd
        """
        # for dont care bits, generally the programmers make it 0

        self.ru_rd_amplitude(0)
        self.modulation_level(3)
        fr1_old = self._read_from_register(0x01, 24)
        fr1_old_bin = "".join(str(b) for b in fr1_old)
        if mode == 0:
            enable_bin = "000"
        elif mode == 1:
            enable_bin = "001"
        elif mode == 2:
            enable_bin = "010"
        elif mode == 3:
            enable_bin = "011"
        else:
            assert mode is None, "invalid input. Value remains as it is"
        fr1_new_bin = fr1_old_bin[:9] + enable_bin + fr1_old_bin[12:]
        fr1_word_new = "".join(" 0" + b for b in fr1_new_bin)  # translate to bytes
        fr1_word_new = fr1_word_new[1:]  # crop the first white space
        self._write_to_dds_register(0x01, fr1_word_new)
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def two_level_modulation_ru_rd(self, mode):
        """
        function to create 2 level modulation with RU/RD.
        mode 0 corresponds to 2 level modulation on CH0 and CH1 with RU/RD
        mode 1 corresponds to 2 level modulation on CH0 and CH2 with RU/RD
        mode 2 corresponds to 2 level modulation on CH0 and CH3 with RU/RD
        mode 3 corresponds to 2 level modulation on CH1 and CH2 with RU/RD
        mode 4 corresponds to 2 level modulation on CH1 and CH3 with RU/RD
        mode 5 corresponds to 2 level modulation on CH2 and CH3 with RU/RD
        """
        self.ru_rd_amplitude(1)
        # fr1_old = self._read_from_register(0x01,24)
        # fr1_old_bin = ''.join(str(b) for b in fr1_old)
        print("changing the modulatiion mode to 2 level")
        self.modulation_level(0)
        fr1_old = self._read_from_register(0x01, 24)
        fr1_old_bin = "".join(str(b) for b in fr1_old)
        if mode == 0:
            enable_bin = "000"
        elif mode == 1:
            enable_bin = "001"
        elif mode == 2:
            enable_bin = "010"
        elif mode == 3:
            enable_bin = "011"
        elif mode == 4:
            enable_bin = "100"
        elif mode == 5:
            enable_bin = "101"
        else:
            assert mode is None, "invalid input. Value remains as it is"
        fr1_new_bin = fr1_old_bin[:9] + enable_bin + fr1_old_bin[12:]
        fr1_word_new = "".join(" 0" + b for b in fr1_new_bin)  # translate to bytes
        fr1_word_new = fr1_word_new[1:]  # crop the first white space
        self._write_to_dds_register(0x01, fr1_word_new)
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def eight_level_modulation_ru_rd(self, mode):
        """
        function to set the configuration of the profile pin and the SDIO_x pins for different modulation modes
        there are 4 types of modulation that are available
        mode 0 for eight level modulation on ch0, with ru/rd
        mode 1 for eight level modulation on ch1, with ru/rd
        mode 2 for eight level modulation on ch2, with ru/rd
        mode 3 for eight level modulation on ch3, with ru/rd
        """
        # for dont care bits, generally the programmers make it 0

        self.ru_rd_amplitude(2)
        self.modulation_level(2)
        fr1_old = self._read_from_register(0x01, 24)
        fr1_old_bin = "".join(str(b) for b in fr1_old)
        if mode == 0:
            enable_bin = "000"
        elif mode == 1:
            enable_bin = "001"
        elif mode == 2:
            enable_bin = "010"
        elif mode == 3:
            enable_bin = "011"
        else:
            assert mode is None, "invalid input. Value remains as it is"
        fr1_new_bin = fr1_old_bin[:9] + enable_bin + fr1_old_bin[12:]
        fr1_word_new = "".join(" 0" + b for b in fr1_new_bin)  # translate to bytes
        fr1_word_new = fr1_word_new[1:]  # crop the first white space
        self._write_to_dds_register(0x01, fr1_word_new)
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def two_level_modulation_sdio_ru_rd(self):
        """
        two level modulation using SDIO pins for ru/rd. The profile pin configuration is dont care.
        P0 corresponds to CH0
        P1 corresponds to CH1
        P2 corresponds to CH2
        P3 corresponds to CH3
        """
        self.ru_rd_amplitude(3)
        self.modulation_level(0)
        fr1_old = self._read_from_register(0x01, 24)
        fr1_old_bin = "".join(str(b) for b in fr1_old)
        enable_bin = "000"
        fr1_new_bin = fr1_old_bin[:9] + enable_bin + fr1_old_bin[12:]
        fr1_word_new = "".join(" 0" + b for b in fr1_new_bin)  # translate to bytes
        fr1_word_new = fr1_word_new[1:]  # crop the first white space
        self._write_to_dds_register(0x01, fr1_word_new)
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def four_level_modulation_sdio_ru_rd(self, mode):
        """ "
        4 level modulation using SDIO_1, SDIO_2 and SDIO_3 pin assignments.SDIO_3 is N/A for all the modes
        mode 0 : P0, P1 controls CH0, P2, P3 controls CH1, SDIO_1 = CH0 RU/RD, SDIO_2= CH1 RU/RD
        mode 1 : P0, P1 controls CH0, P2, P3 controls CH2, SDIO_1 = CH0 RU/RD, SDIO_2= CH2 RU/RD
        mode 2 : P0, P1 controls CH0, P2, P3 controls CH3, SDIO_1 = CH0 RU/RD, SDIO_2= CH3 RU/RD
        mode 3 : P0, P1 controls CH1, P2, P3 controls CH2, SDIO_1 = CH1 RU/RD, SDIO_2= CH2 RU/RD
        mode 4 : P0, P1 controls CH1, P2, P3 controls CH3, SDIO_1 = CH1 RU/RD, SDIO_2= CH3 RU/RD
        mode 5 : P0, P1 controls CH2, P2, P3 controls CH3, SDIO_1 = CH2 RU/RD, SDIO_2= CH3 RU/RD
        """
        self.ru_rd_amplitude(3)
        self.modulation_level(1)
        fr1_old = self._read_from_register(0x01, 24)
        fr1_old_bin = "".join(str(b) for b in fr1_old)
        if mode == 1:
            enable_bin = "000"
        elif mode == 2:
            enable_bin = "001"
        elif mode == 3:
            enable_bin = "010"
        elif mode == 4:
            enable_bin = "011"
        elif mode == 5:
            enable_bin = "100"
        elif mode == 6:
            enable_bin = "101"
        else:
            assert mode is None, "invalid input. Value remains as it is"
        fr1_new_bin = fr1_old_bin[:9] + enable_bin + fr1_old_bin[12:]
        fr1_word_new = "".join(" 0" + b for b in fr1_new_bin)  # translate to bytes
        fr1_word_new = fr1_word_new[1:]  # crop the first white space
        self._write_to_dds_register(0x01, fr1_word_new)
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def sixteen_level_modulation_sdio_ru_rd(self, mode):
        """ "
        16 level modulation using SDIO_1, SDIO_2,SDIO_3 is N/A for all the modes
        mode 0 : P0, P1, P2 and P3 controls CH0, SDIO_1 = CH0 RU/RD
        mode 1 : P0, P1, P2 and P3 controls CH1, SDIO_1 = CH1 RU/RD
        mode 2 : P0, P1, P2 and P3 controls CH2, SDIO_1 = CH2 RU/RD
        mode 3 : P0, P1, P2 and P3 controls CH3, SDIO_1 = CH3 RU/RD
        """
        self.ru_rd_amplitude(3)
        self.modulation_level(1)
        fr1_old = self._read_from_register(0x01, 24)
        fr1_old_bin = "".join(str(b) for b in fr1_old)
        if mode == 0:
            enable_bin = "000"
        elif mode == 1:
            enable_bin = "001"
        elif mode == 2:
            enable_bin = "010"
        elif mode == 3:
            enable_bin = "011"
        else:
            assert mode is None, "invalid input. Value remains as it is"
        fr1_new_bin = fr1_old_bin[:9] + enable_bin + fr1_old_bin[12:]
        fr1_word_new = "".join(" 0" + b for b in fr1_new_bin)  # translate to bytes
        fr1_word_new = fr1_word_new[1:]  # crop the first white space
        self._write_to_dds_register(0x01, fr1_word_new)
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def reference_clock_input_enable(self):
        """
        function to enable the clock input circuitry
        """
        fr1 = self._read_from_register(0x01, 24)
        print(fr1)
        # bit to perform the required function
        fr1[16] = 1
        print(fr1)
        # construct the command for the cypress chip
        fr1_new_bin = "".join(" 0" + str(b) for b in fr1)
        fr1_new_bin = fr1_new_bin[1:]
        print(fr1_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x01, fr1_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def reference_clock_input_disable(self):
        """
        function to disable the clock input circuitry
        """

        fr1 = self._read_from_register(0x01, 24)
        print(fr1)
        # bit to perform the required function
        fr1[16] = 0
        print(fr1)
        # construct the command for the cypress chip
        fr1_new_bin = "".join(" 0" + str(b) for b in fr1)
        fr1_new_bin = fr1_new_bin[1:]
        print(fr1_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x01, fr1_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():

            self._update_IO

    def external_power_mode_enable(self):
        """
        function to enable the external power down mode
        """

        fr1 = self._read_from_register(0x01, 24)
        print(fr1)
        # bit to perform the required function
        fr1[17] = 1
        print(fr1)
        # construct the command for the cypress chip
        fr1_new_bin = "".join(" 0" + str(b) for b in fr1)
        fr1_new_bin = fr1_new_bin[1:]
        print(fr1_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x01, fr1_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def external_power_mode_disable(self):
        """
        function to disable the external power down mode
        """

        fr1 = self._read_from_register(0x01, 24)
        print(fr1)
        # bit to perform the required function
        fr1[17] = 0
        print(fr1)
        # construct the command for the cypress chip
        fr1_new_bin = "".join(" 0" + str(b) for b in fr1)
        fr1_new_bin = fr1_new_bin[1:]
        print(fr1_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x01, fr1_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def sync_clck_enable(self):
        """
        enable synchronisation of the clocks
        """
        fr1 = self._read_from_register(0x01, 24)
        print(fr1)
        # bit to perform the required function
        fr1[18] = 1
        print(fr1)
        # construct the command for the cypress chip
        fr1_new_bin = "".join(" 0" + str(b) for b in fr1)
        fr1_new_bin = fr1_new_bin[1:]
        print(fr1_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x01, fr1_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def sync_clck_disable(self):
        """
        disable synchronisation of the clocks
        """
        fr1 = self._read_from_register(0x01, 24)
        print(fr1)
        # bit to perform the required function
        fr1[18] = 0
        print(fr1)
        # construct the command for the cypress chip
        fr1_new_bin = "".join(" 0" + str(b) for b in fr1)
        fr1_new_bin = fr1_new_bin[1:]
        print(fr1_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x01, fr1_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def dac_reference_power_enable(self):
        """
        dac reference power enable
        """
        """
        function to enable the clock input circuitry
        """
        fr1 = self._read_from_register(0x01, 24)
        print(fr1)
        # bit to perform the required function
        fr1[19] = 1
        print(fr1)
        # construct the command for the cypress chip
        fr1_new_bin = "".join(" 0" + str(b) for b in fr1)
        fr1_new_bin = fr1_new_bin[1:]
        print(fr1_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x01, fr1_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def dac_reference_power_disable(self):
        """dac reference power disable"""

        """
        function to enable the clock input circuitry
        """
        fr1 = self._read_from_register(0x01, 24)
        print(fr1)
        # bit to perform the required function
        fr1[19] = 0
        print(fr1)
        # construct the command for the cypress chip
        fr1_new_bin = "".join(" 0" + str(b) for b in fr1)
        fr1_new_bin = fr1_new_bin[1:]
        print(fr1_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x01, fr1_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def manual_hardware_sync_enable(self):
        """
        enable the manual hardware synchronisation of the devices."""
        fr1 = self._read_from_register(0x01, 24)
        print(fr1)
        # bit to perform the required function
        fr1[22] = 1
        print(fr1)
        # construct the command for the cypress chip
        fr1_new_bin = "".join(" 0" + str(b) for b in fr1)
        fr1_new_bin = fr1_new_bin[1:]
        print(fr1_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x01, fr1_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def manual_hardware_sync_disable(self):
        """
        disable the manual hardware synchronisation of the devices
        """
        fr1 = self._read_from_register(0x01, 24)
        print(fr1)
        # bit to perform the required function
        fr1[22] = 0
        print(fr1)
        # construct the command for the cypress chip
        fr1_new_bin = "".join(" 0" + str(b) for b in fr1)
        fr1_new_bin = fr1_new_bin[1:]
        print(fr1_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x01, fr1_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def manual_software_sync_enable(self):
        """
        enable the manual software syncronisation of the devices.
        """
        fr1 = self._read_from_register(0x01, 24)
        print(fr1)
        # bit to perform the required function
        fr1[23] = 1
        print(fr1)
        # construct the command for the cypress chip
        fr1_new_bin = "".join(" 0" + str(b) for b in fr1)
        fr1_new_bin = fr1_new_bin[1:]
        print(fr1_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x01, fr1_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def manual_software_sync_disable(self):
        """
        disable the manual software syncronisation of the devices.
        """
        fr1 = self._read_from_register(0x01, 24)
        print(fr1)
        # bit to perform the required function
        fr1[23] = 0
        print(fr1)
        # construct the command for the cypress chip
        fr1_new_bin = "".join(" 0" + str(b) for b in fr1)
        fr1_new_bin = fr1_new_bin[1:]
        print(fr1_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x01, fr1_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def all_channel_autoclear_sweep_accumulator_enable(self):
        """
        channel autoclear sweep accumulator enable
        """
        fr2 = self._read_from_register(0x02, 16)
        print(fr2)
        # bit to perform the required function
        fr2[0] = 1
        print(fr2)
        # construct the command for the cypress chip
        fr2_new_bin = "".join(" 0" + str(b) for b in fr2)
        fr2_new_bin = fr2_new_bin[1:]
        print(fr2_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x02, fr2_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def all_channel_autoclear_sweep_accumulator_disable(self):
        """
        channel autoclear sweep accumulator disbale
        """
        fr2 = self._read_from_register(0x02, 16)
        print(fr2)
        # bit to perform the required function
        fr2[0] = 0
        print(fr2)
        # construct the command for the cypress chip
        fr2_new_bin = "".join(" 0" + str(b) for b in fr2)
        fr2_new_bin = fr2_new_bin[1:]
        print(fr2_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x02, fr2_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def all_channel_clear_sweep_accumulator_enable(self):
        """
        channel clear sweep accumulator enable
        """
        fr2 = self._read_from_register(0x02, 16)
        print(fr2)
        # bit to perform the required function
        fr2[1] = 1
        print(fr2)
        # construct the command for the cypress chip
        fr2_new_bin = "".join(" 0" + str(b) for b in fr2)
        fr2_new_bin = fr2_new_bin[1:]
        print(fr2_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x02, fr2_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def all_channel_clear_sweep_accumulator_disable(self):
        """
        channel clear sweep accumulator disbale
        """
        fr2 = self._read_from_register(0x02, 16)
        print(fr2)
        # bit to perform the required function
        fr2[1] = 0
        print(fr2)
        # construct the command for the cypress chip
        fr2_new_bin = "".join(" 0" + str(b) for b in fr2)
        fr2_new_bin = fr2_new_bin[1:]
        print(fr2_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x02, fr2_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def all_channel_autoclear_phase_accumulator_enable(self):
        """
        channel autoclear sweep accumulator enable
        """
        fr2 = self._read_from_register(0x02, 16)
        print(fr2)
        # bit to perform the required function
        fr2[2] = 1
        print(fr2)
        # construct the command for the cypress chip
        fr2_new_bin = "".join(" 0" + str(b) for b in fr2)
        fr2_new_bin = fr2_new_bin[1:]
        print(fr2_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x02, fr2_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def all_channel_autoclear_phase_accumulator_disable(self):
        """
        channel autoclear sweep accumulator disbale
        """
        fr2 = self._read_from_register(0x02, 16)
        print(fr2)
        # bit to perform the required function
        fr2[2] = 0
        print(fr2)
        # construct the command for the cypress chip
        fr2_new_bin = "".join(" 0" + str(b) for b in fr2)
        fr2_new_bin = fr2_new_bin[1:]
        print(fr2_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x02, fr2_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def all_channel_clear_phase_accumulator_enable(self):
        """
        channel clear sweep accumulator enable
        """
        fr2 = self._read_from_register(0x02, 16)
        print(fr2)
        # bit to perform the required function
        fr2[3] = 1
        print(fr2)
        # construct the command for the cypress chip
        fr2_new_bin = "".join(" 0" + str(b) for b in fr2)
        fr2_new_bin = fr2_new_bin[1:]
        print(fr2_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x02, fr2_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def all_channel_clear_phase_accumulator_disable(self):
        """
        channel clear sweep accumulator disbale
        """
        fr2 = self._read_from_register(0x02, 16)
        print(fr2)
        # bit to perform the required function
        fr2[3] = 0
        print(fr2)
        # construct the command for the cypress chip
        fr2_new_bin = "".join(" 0" + str(b) for b in fr2)
        fr2_new_bin = fr2_new_bin[1:]
        print(fr2_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x02, fr2_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def auto_sync_enable(self):
        """
        enables automatic syncronisation of multiple devices
        """
        fr2 = self._read_from_register(0x02, 16)
        print(fr2)
        # bit to perform the required function
        fr2[8] = 1
        print(fr2)
        # construct the command for the cypress chip
        fr2_new_bin = "".join(" 0" + str(b) for b in fr2)
        fr2_new_bin = fr2_new_bin[1:]
        print(fr2_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x02, fr2_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def auto_sync_disable(self):
        """
        enables automatic syncronisation of multiple devices
        """
        fr2 = self._read_from_register(0x02, 16)
        print(fr2)
        # bit to perform the required function
        fr2[8] = 0
        print(fr2)
        # construct the command for the cypress chip
        fr2_new_bin = "".join(" 0" + str(b) for b in fr2)
        fr2_new_bin = fr2_new_bin[1:]
        print(fr2_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x02, fr2_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def multi_device_sync_master_enable(self):
        """
        synchronisation of multiple devices
        """
        fr2 = self._read_from_register(0x02, 16)
        print(fr2)
        # bit to perform the required function
        fr2[9] = 1
        print(fr2)
        # construct the command for the cypress chip
        fr2_new_bin = "".join(" 0" + str(b) for b in fr2)
        fr2_new_bin = fr2_new_bin[1:]
        print(fr2_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x02, fr2_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def multi_device_sync_slave_enable(self):
        """
        synchronisation of multiple devices
        """
        fr2 = self._read_from_register(0x02, 16)
        print(fr2)
        # bit to perform the required function
        fr2[9] = 0
        print(fr2)
        # construct the command for the cypress chip
        fr2_new_bin = "".join(" 0" + str(b) for b in fr2)
        fr2_new_bin = fr2_new_bin[1:]
        print(fr2_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x02, fr2_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def multi_device_sync_status(self):
        """
        check the status of the multiple devices that are being synchronised
        returns 1, if the devices are out of sync
        """
        fr2 = self._read_from_register(0x02, 16)
        check = fr2[10]
        return check

    def multi_device_sync_mask_enable(self):
        """
        mask the device for multidevice that are being synchronised
        """
        fr2 = self._read_from_register(0x02, 16)
        print(fr2)
        # bit to perform the required function
        fr2[11] = 1
        print(fr2)
        # construct the command for the cypress chip
        fr2_new_bin = "".join(" 0" + str(b) for b in fr2)
        fr2_new_bin = fr2_new_bin[1:]
        print(fr2_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x02, fr2_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def multi_device_sync_mask_disable(self):
        """
        mask the device for multidevice that are being synchronised
        """
        fr2 = self._read_from_register(0x02, 16)
        print(fr2)
        # bit to perform the required function
        fr2[11] = 0
        print(fr2)
        # construct the command for the cypress chip
        fr2_new_bin = "".join(" 0" + str(b) for b in fr2)
        fr2_new_bin = fr2_new_bin[1:]
        print(fr2_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x02, fr2_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def system_clock_offset(self, mode):
        """
        add an offset to the synchronisation of the multiple clocks.
        mode 00 = 0 <= delay <=1
        mode 01 = 1 <= delay <=2
        mode 10 = 2 <= delay <=3
        mode 11 = 3 <= delay <=4
        """
        fr2_old = self._read_from_register(0x02, 16)
        fr2_old_bin = "".join(str(b) for b in fr2_old)
        # 2. replace the CFR by one with updated LS enable bit
        if mode == 0:
            enable_bin = "00"
        elif mode == 1:
            enable_bin = "01"
        elif mode == 2:
            enable_bin = "10"
        elif mode == 3:
            enable_bin = "11"
        else:
            assert mode is None, "invalid input. Value remains as it is="
        fr1_new_bin = enable_bin + fr2_old_bin[2:]
        fr1_word_new = "".join(" 0" + b for b in fr1_new_bin)  # translate to bytes
        fr1_word_new = fr1_word_new[1:]  # crop the first white space
        self._write_to_dds_register(0x01, fr1_word_new)
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def linear_sweep_no_dwell_disable(self, channel=None):
        """
        disable the channel linear sweep no-dwell function

        channel is entered either as integer or as an array
        """
        if channel is None:
            channel = self.channel

        # select the chosen channel
        self._channel_select(channel)
        cfr = self._read_from_register(0x03, 24)

        cfr[8] = 0
        if cfr[13] == 1:
            sys.exit("this bit must be 0 and is predetermined in the datasheet")
        cfr_new_bin = "".join(" 0" + str(b) for b in cfr)

        cfr_new_bin = cfr_new_bin[1:]
        self._write_to_dds_register(0x03, cfr_new_bin)

        # update I/O
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def linear_sweep_no_dwell_enable(self, channel=None):
        """
        enable the linear sweep no dwell function
        """
        if channel is None:
            channel = self.channel

        # select the chosen channel

        self._channel_select(channel)
        cfr = self._read_from_register(0x03, 24)
        if cfr[9] == 0:
            assert (
                cfr[9] is None
            ), " This mode cannot be activated as linear sweep enable mode is inactive. "
        if cfr[13] == 1:
            sys.exit("this bit must be 0 and is predetermined in the datasheet")
        cfr[8] = 1
        cfr_new_bin = "".join(" 0" + str(b) for b in cfr)
        cfr_new_bin = cfr_new_bin[1:]
        self._write_to_dds_register(0x03, cfr_new_bin)

        # update I/O
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def load_srr_timeout_enable(self, channel):
        """
        the linear sweep ramp rate timer is loaded only upon timeout (timer = 1) and is not loaded because of an I/O_UPDATE input signal (default).

        """
        if channel is None:
            channel = self.channel

        # select the chosen channel
        self._channel_select(channel)
        cfr = self._read_from_register(0x03, 24)
        cfr[10] = 1
        if cfr[13] == 1:
            sys.exit("this bit must be 0 and is predetermined in the datasheet")
        cfr_new_bin = "".join(" 0" + str(b) for b in cfr)
        cfr_new_bin = cfr_new_bin[1:]
        self._write_to_dds_register(0x03, cfr_new_bin)

        # update I/O
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def load_srr_timeout_disable(self, channel):
        """
        the linear sweep ramp rate timer is loaded upon timeout (timer = 1) or at the time of an I/O_UPDATE input signal.
        """
        if channel is None:
            channel = self.channel

        # select the chosen channels
        self._channel_select(channel)
        cfr = self._read_from_register(0x03, 24)
        cfr[10] = 0
        if cfr[13] == 1:
            sys.exit("this bit must be 0 and is predetermined in the datasheet")
        cfr_new_bin = "".join(" 0" + str(b) for b in cfr)
        cfr_new_bin = cfr_new_bin[1:]
        self._write_to_dds_register(0x03, cfr_new_bin)

        # update I/O
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def automatic_amplitude_ru_rd_mode_acitvate(self, mode, channel=None):
        """
        activates the automatic RU/RD mode for the amplitude control
        select the channel from 0 to 3.
        select mode 0 for step size to be 1
        select mode 1 for step size to be 2
        select mode 2 for step size to be 4
        select mode 3 for step size to be 8
        ramp_rate: int
        maximum size is of 8 bit

        """
        if channel is None:
            channel = self.channel

        self._channel_select(channel)
        acr = self._read_from_register(0x06, 24)
        acr[3] = 1
        acr[4] = 1
        acr_old = "".join(str(b) for b in acr)
        # 2. replace the CFR by one with updated LS enable bit
        if mode == 0:
            enable_bin = "00"
        elif mode == 1:
            enable_bin = "01"
        elif mode == 2:
            enable_bin = "10"
        elif mode == 3:
            enable_bin = "11"
        else:
            assert mode is None, "invalid input. Value remains as it is="
        acr_new = enable_bin + acr_old[2:]
        acr_new = "".join(" 0" + b for b in acr_new)  # translate to bytes
        acr_new = acr_new[1:]  # crop the first white space

        # write to the register
        self._write_to_dds_register(0x06, acr_new)

        # update I/O
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def amplitude_ramp_rate(
        self,
        value,
        channel=None,
    ):
        """
        Increment/decrement the step size of the amplitude
        channel selec ts the channel that is involved,
        value selects the amplitude ramp rate for the channel is of the typ[e integer]
        """
        if channel is None:
            channel = self.channel
        self._channel_select(channel)
        acr = self._read_from_register(0x06, 24)
        assert (
            value <= 255
        ), "amplitude shoukd not be greater than system clock frequency"
        value_bin = bin(value)[2:]
        if len(value_bin) < 8:
            value_new = (8 - len(value_bin)) * "0" + value_bin
        acr_new = value_new + acr[8:]

        self._write_to_dds_register(0x06, acr_new)
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def load_arr_io_enable(self, channel=None):
        """
        load the amplitude ramp rate upon time_out or at I/O update signal
        """
        if channel is None:
            channel = self.channel

        # select the chosen channels
        self._channel_select(channel)
        acr = self._read_from_register(0x06, 24)
        acr[13] = 1
        acr_new_bin = "".join(" 0" + str(b) for b in acr)
        acr_new_bin = acr_new_bin[1:]
        self._write_to_dds_register(0x06, acr_new_bin)

        # update I/O
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def load_arr_io_disbale(self, channel):
        """
        load the amplitude ramp rate upon time_out and is not loaded at I/O update signal
        """
        if channel is None:
            channel = self.channel

        # select the chosen channels
        self._channel_select(channel)
        acr = self._read_from_register(0x06, 24)
        acr[13] = 0
        acr_new_bin = "".join(" 0" + str(b) for b in acr)
        acr_new_bin = acr_new_bin[1:]
        self._write_to_dds_register(0x06, acr_new_bin)

        # update I/O
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    # documented from here in the class diagram.
    def set_amplitude(self, asf, channel=None):
        if channel is None:
            channel = self.channel
        assert 0 <= asf <= 1, "Factor should be between 0 and 1"

        self._channel_select(channel)
        asf_bin = bin(round(asf * (2**10))).lstrip("0b")
        if len(asf_bin) < 10:
            asf_bin = (10 - len(asf_bin)) * "0" + asf_bin
        print(asf_bin)
        acr_old = self._read_from_register(0x06, 24)
        acr_old[11] = 1
        acr_old[12] = 0
        acr = "".join(str(b) for b in acr_old)
        print(acr)
        acr_bin = acr[:14] + asf_bin
        print(acr_bin)
        asf_bin_word = "".join(" 0" + b for b in acr_bin)
        asf_bin_word = asf_bin_word[1:]
        print(asf_bin_word)
        self._write_to_dds_register(0x06, asf_bin_word)
        # update I/O
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    # def set_amplitude(self, asf, channel=None, channel_word=0):
    #     """ Sets the amplitude scaling factor for a given channel.

    #     :asf: float
    #         Between 0 and 1. DACs have 10-bit resolution.
    #     :channel: int
    #         Channel for which the amplitude scaling factor is set.
    #     :channel_word: int
    #         Channel word for the given channel. Every channel has 16 channel word registers that
    #         can be used for modulation. Data is written on the channel word using registers.
    #     """
    #     if channel is None:
    #         channel = self.channel

    #     assert channel_word < 16, ("Channel word cannot exceed 15, input was {0}".format(channel_word))

    #     # select the chosen channels
    #     self._channel_select(channel)

    #     # construct the asf word
    #     fraction_bin = bin(int(round(asf * (2**10 - 1)))).lstrip('0b') # full range are 10 bit

    #     # load the current amplitude control register word
    #     acr_old = self._read_from_register(0x06, 24)
    #     acr_old[11] = 0
    #     acr_old[12] = 1
    #     acr = ''.join(str(b) for b in acr_old)
    #     # set the asf word in the amplitude control register
    #     # asf_word = ''.join(' 0' + b for b in fraction_bin)
    #     # asf_word = asf_word[1:]

    #     acr_new_bin = acr[:15] + fraction_bin
    #     acr_new = ''.join(' 0' + b for b in acr_new_bin)
    #     acr_new = acr_new[1:]

    #     if channel_word == 0:
    #         self._write_to_dds_register(0x06,acr_new)
    #     else:
    #         register = channel_word - 1 + 0x0A
    #         self._write_to_dds_register(register, fraction_bin)

    def set_frequency(self, frequency, channel=None, channel_word=0):
        """Sets a new frequency for a given channel.

        :frequency: float
            The new frequency in Hz. Should not exceed `system_clock_frequency`.
        :channel: int or seq
            Channel(s) for which the frequency should be set.
        :channel_word: int
            Determines the channel_word to which the frequency is written. Each channel has 16
            channel_words that can be used.

        """

        if channel is None:
            channel = self.channel
        assert frequency <= self.system_clock_frequency, (
            "Frequency should not"
            + " exceed system clock frequency! System clock frequency is {}Hz".format(
                self.system_clock_frequency
            )
        )

        assert channel_word < 16, "Channel word cannot exceed 15, input was {}".format(
            channel_word
        )

        # select the chosen channels
        self._channel_select(channel)

        # calculate the fraction of the full frequency
        fraction = frequency / self.system_clock_frequency
        fraction_bin = bin(int(round(fraction * (2**32 - 1)))).lstrip(
            "0b"
        )  # full range are 32 bit
        if len(fraction_bin) < 32:
            fraction_bin = (32 - len(fraction_bin)) * "0" + fraction_bin
        closest_possible_value = (
            int(fraction_bin, base=2) / (2**32 - 1) * self.system_clock_frequency
        )
        print(
            "Setting frequency of channel {1}:{2} to closest possible value {0}MHz".format(
                closest_possible_value / 1e6, channel, channel_word
            )
        )

        # set the frequency word in the frequency register
        frequency_word = "".join(" 0" + b for b in fraction_bin)
        frequency_word = frequency_word[1:]
        if channel_word == 0:
            self._write_to_dds_register(0x04, frequency_word)
        else:
            register = channel_word + 0x0A - 1
            self._write_to_dds_register(register, frequency_word)

        # load and update I/O
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def set_phase(self, phase, channel=None):
        """Sets the phase offset for a given channel.

        :phase: float
            phase in degree, 0 < `phase` < 360
        :channels: int or list
            ID or list of IDs of the selected channels

        """
        if channel is None:
            channel = self.channel
        assert 0 <= phase <= 360, "Phase should be between 0 and 360 degree!"

        # select the channels
        self._channel_select(channel)

        # calculate the binary phase word
        phase_fraction = phase / 360
        phase_fraction_bin = bin(round(phase_fraction * 2**14)).lstrip("0b")
        if len(phase_fraction_bin) < 16:
            phase_fraction_bin = (
                16 - len(phase_fraction_bin)
            ) * "0" + phase_fraction_bin

        # construct the message for cypress chip
        phase_fraction_word = "".join(" 0" + b for b in phase_fraction_bin)
        phase_fraction_word = phase_fraction_word[1:]

        # write the phase word to the register
        self._write_to_dds_register(0x05, phase_fraction_word)

        # update I/O
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    # add the hannel function here it is important
    def toggle_clear_sweep_accumulator_enable(self):
        "sets the clear phase accumulator as 1 wherein the memory elements are asynchronously cleared"
        cfr = self._read_from_register(0x03, 24)
        print(cfr)
        cfr[20] = 1
        print(cfr)
        if cfr[13] == 1:
            sys.exit("this bit must be 0 and is predetermined in the datasheet")
        # construct the command for the cypress chip
        cfr_new_bin = "".join(" 0" + str(b) for b in cfr)
        cfr_new_bin = cfr_new_bin[1:]
        print(cfr_new_bin)
        # write the new values to the register
        self._write_to_dds_register(0x03, cfr_new_bin)
        # update I/O
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def toggle_clear_sweep_accumulator_disable(self):
        "sets the clear phase accumulator as 0 wherein the memory elements are asynchronously cleared"
        cfr = self._read_from_register(0x03, 24)
        print(cfr)
        cfr[20] = 0
        print(cfr)
        if cfr[13] == 1:
            sys.exit("this bit must be 0 and is predetermined in the datasheet")
        # construct the command for the cypress chip
        cfr_new_bin = "".join(" 0" + str(b) for b in cfr)
        cfr_new_bin = cfr_new_bin[1:]
        print(cfr_new_bin)
        # write the new values to the register
        self._write_to_dds_register(0x03, cfr_new_bin)
        # update I/O
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def toggle_autoclear_sweep_accumulations_enable(self):
        """= the phase accumulator is automatically and synchronously cleared for one cycle upon receipt of an I/O_UPDATE signal.
        cross check the bit is it 21 or 1
        """
        # load the current channel function register setting
        cfr = self._read_from_register(0x03, 24)
        print(cfr)

        # set the autoclear phase accumulator bit to 1 if old value was 0 and

        cfr[19] = 1
        print(cfr)
        if cfr[13] == 1:
            sys.exit("this bit must be 0 and is predetermined in the datasheet")
        # construct the command for the cypress chip
        cfr_new_bin = "".join(" 0" + str(b) for b in cfr)
        cfr_new_bin = cfr_new_bin[1:]
        print(cfr_new_bin)

        # write new values to register
        self._write_to_dds_register(0x03, cfr_new_bin)

        # update I/O
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def toggle_autoclear_sweep_accumulations_disable(self):
        """= tthe current state of the phase accumulator is not impacted by receipt of an I/O_UPDATE signal (default).
        cross check the bit is it 21 or 1
        """
        # load the current channel function register setting
        cfr = self._read_from_register(0x03, 24)
        print(cfr)

        # set the autoclear phase accumulator bit to 1 if old value was 0 and

        cfr[19] = 0
        print(cfr)
        if cfr[13] == 1:
            sys.exit("this bit must be 0 and is predetermined in the datasheet")
        # construct the command for the cypress chip
        cfr_new_bin = "".join(" 0" + str(b) for b in cfr)
        cfr_new_bin = cfr_new_bin[1:]
        print(cfr_new_bin)

        # write new values to register
        self._write_to_dds_register(0x03, cfr_new_bin)

        # update I/O
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def toggle_clear_phase_accumulator_enable(self):
        "sets the clear phase accumulator as 1 wherein the memory elements are asynchronously cleared"
        cfr = self._read_from_register(0x03, 24)
        print(cfr)
        cfr[22] = 1
        print(cfr)
        if cfr[13] == 1:
            sys.exit("this bit must be 0 and is predetermined in the datasheet")
        # construct the command for the cypress chip
        cfr_new_bin = "".join(" 0" + str(b) for b in cfr)
        cfr_new_bin = cfr_new_bin[1:]
        print(cfr_new_bin)
        # write the new values to the register
        self._write_to_dds_register(0x03, cfr_new_bin)
        # update I/O
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def toggle_clear_phase_accumulator_disable(self):
        "sets the clear phase accumulator as 1 wherein the memory elements are asynchronously cleared"
        cfr = self._read_from_register(0x03, 24)
        print(cfr)
        cfr[22] = 0
        print(cfr)
        if cfr[13] == 1:
            sys.exit("this bit must be 0 and is predetermined in the datasheet")
        # construct the command for the cypress chip
        cfr_new_bin = "".join(" 0" + str(b) for b in cfr)
        cfr_new_bin = cfr_new_bin[1:]
        print(cfr_new_bin)
        # write the new values to the register
        self._write_to_dds_register(0x03, cfr_new_bin)
        # update I/O
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def toggle_autoclear_phase_accumulations_enable(self):
        """= the phase accumulator is automatically and synchronously cleared for one cycle upon receipt of an I/O_UPDATE signal.
        cross check the bit is it 21 or 1
        """
        # load the current channel function register setting
        cfr = self._read_from_register(0x03, 24)

        print(cfr)

        # set the autoclear phase accumulator bit to 1 if old value was 0 and

        cfr[21] = 1
        print(cfr)
        if cfr[13] == 1:
            sys.exit("this bit must be 0 and is predetermined in the datasheet")
        # construct the command for the cypress chip
        cfr_new_bin = "".join(" 0" + str(b) for b in cfr)
        cfr_new_bin = cfr_new_bin[1:]
        print(cfr_new_bin)

        # write new values to register
        self._write_to_dds_register(0x03, cfr_new_bin)

        # update I/O
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def toggle_autoclear_phase_accumulations_disable(self):
        """= tthe current state of the phase accumulator is not impacted by receipt of an I/O_UPDATE signal (default).
        cross check the bit is it 21 or 1
        """
        # load the current channel function register setting
        cfr = self._read_from_register(0x03, 24)
        print(cfr)

        # set the autoclear phase accumulator bit to 1 if old value was 0 and

        cfr[21] = 0
        print(cfr)
        if cfr[13] == 1:
            sys.exit("this bit must be 0 and is predetermined in the datasheet")
        # construct the command for the cypress chip
        cfr_new_bin = "".join(" 0" + str(b) for b in cfr)
        cfr_new_bin = cfr_new_bin[1:]
        print(cfr_new_bin)

        # write new values to register
        self._write_to_dds_register(0x03, cfr_new_bin)

        # update I/O
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def toggle_sine_wave_output_enable(self):
        "switchesd the output waveform on or off"
        # load the current channel function register setting to 1 if you want the output to be in the form of the sine wave. The default value is cosine
        cfr = self._read_from_register(0x03, 24)
        print(cfr)
        # enable the sine wave output bit
        cfr[23] = 1
        print(cfr)
        # construct the command for the cypress chip
        if cfr[13] == 1:
            sys.exit("this bit must be 0 and is predetermined in the datasheet")
        cfr_new_bin = "".join(" 0" + str(b) for b in cfr)
        cfr_new_bin = cfr_new_bin[1:]
        print(cfr_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x03, cfr_new_bin)
        # update I/O
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def toggle_cosine_wave_output_enable(self):
        "switchesd the output waveform on or off"
        # load the current channel function register setting to 1 if you want the output to be in the form of the sine wave. The default value is cosine
        cfr = self._read_from_register(0x03, 24)
        print(cfr)
        # disable the sine wave output bit, reset it to the default value of 0
        cfr[23] = 0
        print(cfr)
        if cfr[13] == 1:
            sys.exit("this bit must be 0 and is predetermined in the datasheet")
        # construct the command for the cypress chip
        cfr_new_bin = "".join(" 0" + str(b) for b in cfr)
        cfr_new_bin = cfr_new_bin[1:]
        print(cfr_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x03, cfr_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def toggle_matched_pipe_delay_enable(self):
        "enable the matched pipe delay IMPORTANT ADD THE EXCEPTION TO CHECK IF IT IS IN THE SINGLE TONE MODE ONLY"
        # = matched pipe delay mode is active
        cfr = self._read_from_register(0x03, 24)
        print(cfr)
        # disable the sine wave output bit, reset it to the default value of 0
        cfr[18] = 1
        print(cfr)
        if cfr[13] == 1:
            sys.exit("this bit must be 0 and is predetermined in the datasheet")

        cfr_new_bin = "".join(" 0" + str(b) for b in cfr)
        cfr_new_bin = cfr_new_bin[1:]
        print(cfr_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x03, cfr_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def toggle_matched_pipe_delay_disable(self):
        "enable the matched pipe delay IMPORTANT ADD THE EXCEPTION TO CHECK IF IT IS IN THE SINGLE TONE MODE ONLY"
        # = matched pipe delay mode is deactivated
        cfr = self._read_from_register(0x03, 24)
        print(cfr)
        # disable the sine wave output bit, reset it to the default value of 0
        cfr[18] = 0
        print(cfr)
        if cfr[13] == 1:
            sys.exit("this bit must be 0 and is predetermined in the datasheet")
        # construct the command for the cypress chip
        cfr_new_bin = "".join(" 0" + str(b) for b in cfr)
        cfr_new_bin = cfr_new_bin[1:]
        print(cfr_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x03, cfr_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def toggle_DAC_power_down_disable(self):
        "the DAC is disabled and is in its lowest power dissipation state"
        cfr = self._read_from_register(0x03, 24)
        print(cfr)
        # bit to perform the required function
        cfr[17] = 1
        print(cfr)
        # construct the command for the cypress chip
        if cfr[13] == 1:
            sys.exit("this bit must be 0 and is predetermined in the datasheet")
        cfr_new_bin = "".join(" 0" + str(b) for b in cfr)
        cfr_new_bin = cfr_new_bin[1:]
        print(cfr_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x03, cfr_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def toggle_DAC_power_down_enable(self):
        "DAC is enabled for operation (default)."
        cfr = self._read_from_register(0x03, 24)
        print(cfr)
        # bit to perform the required function
        cfr[17] = 0
        print(cfr)
        if cfr[13] == 1:
            sys.exit("this bit must be 0 and is predetermined in the datasheet")
        # construct the command for the cypress chip
        cfr_new_bin = "".join(" 0" + str(b) for b in cfr)
        cfr_new_bin = cfr_new_bin[1:]
        print(cfr_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x03, cfr_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def toggle_digital_power_down_disable(self):
        "the digital core is disabled and is in its lowest power dissipation state."
        cfr = self._read_from_register(0x03, 24)
        print(cfr)
        # bit to perform the required function
        cfr[16] = 1
        print(cfr)
        if cfr[13] == 1:
            sys.exit("this bit must be 0 and is predetermined in the datasheet")
        # construct the command for the cypress chip
        cfr_new_bin = "".join(" 0" + str(b) for b in cfr)
        cfr_new_bin = cfr_new_bin[1:]
        print(cfr_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x03, cfr_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def toggle_digital_power_down_enable(self):
        "the digital core is enabled for operation (default)."
        cfr = self._read_from_register(0x03, 24)
        print(cfr)
        # bit to perform the required function
        cfr[16] = 0
        print(cfr)
        if cfr[13] == 1:
            sys.exit("this bit must be 0 and is predetermined in the datasheet")
        # construct the command for the cypress chip
        cfr_new_bin = "".join(" 0" + str(b) for b in cfr)
        cfr_new_bin = cfr_new_bin[1:]
        print(cfr_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x03, cfr_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    # need to modify this
    def DAC_full_scale_control(self, select_mode=None):
        "the digital core is enabled for operation (default). Select 3/None for default: full scale, 1 for half scale 2 for quarter scale, 0 for eight scale"
        cfr = self._read_from_register(0x03, 24)
        print(cfr)
        # shift it to the default function if no input is entered, which in this case is 11 largest LSB
        if (select_mode == None) | (select_mode == 3):
            cfr[15] = 1
            cfr[14] = 1
        elif select_mode == 1:
            cfr[15] = 1
            cfr[14] = 0
        elif select_mode == 2:
            cfr[15] = 0
            cfr[14] = 1
        elif select_mode == 0:
            cfr[15] = 0
            cfr[14] = 0
        else:
            print("invalid value")

        print(cfr)
        if cfr[13] == 1:
            sys.exit("this bit must be 0 and is predetermined in the datasheet")
        # construct the command for the cypress chip
        cfr_new_bin = "".join(" 0" + str(b) for b in cfr)
        cfr_new_bin = cfr_new_bin[1:]
        print(cfr_new_bin)
        # write new values to the register
        self._write_to_dds_register(0x03, cfr_new_bin)
        # update I/O
        self._load_IO()
        if self._update_IO():
            self._update_IO

    def _enable_channel_modulation(self, modulation_type, channel=None):
        """Enables frequency modulation for selected channel(s).
        :channel: int or list
            channel ID or list of channel IDs that are selected
        :modulation_type: str
            can be 'frequency', 'phase' or 'amplitude'. select 0 for frequency, 1 for phase and 2 for amplitude
        :disable: bool
            when True, modulation for this channel(s) is disabled.

        """
        if channel is None:
            channel = self.channel
        if np.issubdtype(type(channel), np.integer):
            channel = [channel]

        # we need to iterate over all channels, as the channel's individual function registers
        # might have different content
        for ch in channel:
            ch1 = int(ch)
            self._channel_select(ch1)

            # the modulation type of the channel is encoded in register 0x03[23:22].
            # 00 disables modulation, 10 is frequency modulation.

            if modulation_type == "frequency":
                modulation_type_bin = "10"
            elif modulation_type == "phase":
                modulation_type_bin == "11"
            elif modulation_type == "amplitude":
                modulation_type_bin = "01"
            elif modulation_type == "disable":
                modulation_type_bin = "00"
                print("modulation has been disabled")
            else:
                modulation_type_bin = "00"
                print("invalid input")

            # 1. read the old CFR content
            cfr_old = self._read_from_register(0x03, 24)
            cfr_old_bin = "".join(str(b) for b in cfr_old)
            if cfr_old[13] == "1":
                sys.exit("this bit must be 0 and is predetermined in the datasheet")
            # 2. replace the modulation type
            cfr_new_bin = modulation_type_bin + cfr_old_bin[2:]

            cfr_word_new = "".join(" 0" + b for b in cfr_new_bin)
            cfr_word_new = cfr_word_new[1:]

            self._write_to_dds_register(0x03, cfr_word_new)

            self._load_IO()
            if self.auto_update:
                self._update_IO()

    def phase_modulation(self, channel=None, phase=0, channel_word=1):
        """ "
        this is used to control the phase angle of the modulated waveform
        the channel: 0,1,2 and 3
        angle is from 0 to 360 degrees
        """
        if channel is None:
            channel = self.channel
        assert 0 <= phase <= 360, "Phase should be between 0 and 360 degrees"
        assert channel_word < 16, "Channel word cannot exceed 15, input was {}".format(
            channel_word
        )
        self._channel_select(channel)
        phase_fraction = phase / 360
        phase_fraction_bin = bin(round(phase_fraction * 2**14)).lstrip("0b")
        if len(phase_fraction_bin) < 32:
            phase_fraction_bin = (
                32 - len(phase_fraction_bin)
            ) * "0" + phase_fraction_bin
        register = channel_word - 1 + 0x0A
        #        register = "0x{:02x}".format(register)
        # self._read_from_register(register, 32)
        phase_word = "".join(" 0" + b for b in phase_fraction_bin)
        phase_word = phase_word[1:]
        self._write_to_dds_register(register, phase_word)
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def frequency_modulation(self, channel=None, frequency=0, channel_word=1):
        """ "
        this is used to control the phase angle of the modulated waveform
        the channel: 0,1,2 and 3
        angle is from 0 to 360 degrees
        """
        if channel is None:
            channel = self.channel
        assert frequency <= self.system_clock_frequency, (
            "Frequency should not"
            + " exceed system clock frequency! System clock frequency is {}Hz".format(
                self.system_clock_frequency
            )
        )
        assert channel_word < 16, "Channel word cannot exceed 15, input was {}".format(
            channel_word
        )
        self._channel_select(channel)
        fraction = frequency / self.system_clock_frequency
        fraction_bin = bin(round(fraction * (2**32 - 1))).lstrip(
            "0b"
        )  # full range are 32 bit
        if len(fraction_bin) < 32:
            fraction_bin = (32 - len(fraction_bin)) * "0" + fraction_bin
        closest_possible_value = (
            int(fraction_bin, base=2) / (2**32 - 1) * self.system_clock_frequency
        )
        print(
            "Frequency of channel {1} encoded as closest possible value {0}MHz".format(
                closest_possible_value / 1e6, channel
            )
        )
        frequency_word = "".join(" 0" + b for b in fraction_bin)
        frequency_word = frequency_word[1:]

        register = channel_word - 1 + 0x0A
        #        register = "0x{:02x}".format(register)
        # self._read_from_register(register, 32)

        self._write_to_dds_register(register, frequency_word)
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def amplitude_modulation(self, channel=None, asf=0, channel_word=1):
        """ "
        this is used to control the phase angle of the modulated waveform
        the channel: 0,1,2 and 3
        angle is from 0 to 360 degrees
        """
        if channel is None:
            channel = self.channel
        assert asf <= 1, "amplitude should be between 0 and 1"
        assert channel_word < 16, "Channel word cannot exceed 15, input was {}".format(
            channel_word
        )
        self._channel_select(channel)
        asf_bin = bin(round(asf * (2**10))).lstrip("0b")
        if len(asf_bin) < 32:
            asf_bin = (32 - len(asf_bin)) * "0" + asf_bin
        print(asf_bin)
        amplitude_word = "".join(" 0" + b for b in asf_bin)
        amplitude_word = amplitude_word[1:]

        register = channel_word - 1 + 0x0A
        #        register = "0x{:02x}".format(register)
        # self._read_from_register(register, 32)

        self._write_to_dds_register(register, amplitude_word)
        self._load_IO()
        if self.auto_update:
            self._update_IO()

    def enable_modulation(
        self, modulation_type="frequency", level=2, active_channels=None
    ):
        """This method chooses the modulation level and type.

        :level: int
            Can be either 2, 4 or 16. The level determines the number of registers from
            which active channels can choose.
        :active_channels: int or list
            In 4- and 16-level modulation this determines which channels can be modulated.
            Note that as there is only a 4 bit input (P0-P3), in 4-level modulation only 2 channels
            can be modulated, in 16-level modulation only one.
        :modulation_type: str
            'frequency', 'amplitude' or 'phase'

        """

        if active_channels is None:
            active_channels = self.channel
        if np.issubdtype(type(active_channels), np.integer):
            active_channels = [active_channels]
        active_channels.sort()
        self._enable_channel_linear_sweep(active_channels, 1)

        # 1. get the current content of (global) function register 1
        fr_old = self._read_from_register(0x01, 24)
        fr_old_bin = "".join(str(b) for b in fr_old)

        # 2. set the modulation level
        level_bin = "00"
        if level == 4:
            level_bin = "01"
        elif level == 16:
            level_bin = "11"
        # 3. replace the old level
        fr_new_level = fr_old_bin[:14] + level_bin + fr_old_bin[16:]

        # 3.1 if the level is 4 or 16, also the PPC bits need to be updated
        if level != 2:
            # mappings are taken from the manual of the AD9959
            if level == 4:
                configurations = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
                ppcs_combinations = [bin(i)[2:] for i in range(6)]

            elif level == 16:
                configurations = [[i] for i in range(4)]
                ppcs_combinations = [bin(i)[2:] for i in range(4)]

            i = configurations.index(active_channels)
            PPC_bin = ppcs_combinations[i]
            if len(PPC_bin) < 3:
                PPC_bin = "0" * (3 - len(PPC_bin)) + PPC_bin

            # update PPC word
            fr_new_level = fr_new_level[:9] + PPC_bin + fr_new_level[12:]

        # write the new FR1 word to the register
        fr_new_word = "".join(" 0" + b for b in fr_new_level)

        fr_new_word = fr_new_word[1:]
        # return fr_new_word, fr_new_level

        self._write_to_dds_register(0x01, fr_new_word)

        self._load_IO()
        if self.auto_update:
            self._update_IO()

        # we also make sure that the active channels are in correct modulation mode
        for ch in active_channels:
            ch1 = int(ch)
            print(ch1)
            self._enable_channel_modulation(modulation_type, ch1)

    def _enable_channel_linear_sweep(self, channels=None, disable=False):
        """TODO: Docstring for _enable_channel_linear_sweep.

        :channel: int or list
            channel ID or list of channel IDs that are selected
        :disable: bool
            when True, modulation for this channel(s) is disabled.

        """
        if channels is None:
            channels = self.channel
        if np.issubdtype(type(channels), np.integer):
            channels = [channels]

        # the modulation type of the channel is encoded in CFR 0x03[14].
        # 0 disables linear sweep, 1 enables
        if not disable:
            ls_enable_bin = "1"
        else:
            ls_enable_bin = "0"

        # we need to iterate over all channels, as the channel's individual function registers
        # might have different content
        for ch in channels:
            self._channel_select(ch)

            # 1. read the old CFR content
            cfr_old = self._read_from_register(0x03, 24)
            cfr_old_bin = "".join(str(b) for b in cfr_old)

            # 2. replace the CFR by one with updated LS enable bit
            cfr_new_bin = cfr_old_bin[:9] + ls_enable_bin + cfr_old_bin[10:]
            if cfr_new_bin[13] == "1":
                sys.exit("this bit must be 0 and is predetermined in the datasheet")
            cfr_word_new = "".join(" 0" + b for b in cfr_new_bin)  # translate to bytes
            cfr_word_new = cfr_word_new[1:]  # crop the first white space

            self._write_to_dds_register(0x03, cfr_word_new)

            self._load_IO()
            if self.auto_update:
                self._update_IO()

            # print summary message
            mes = ["Disabled", "Enabled"][int(ls_enable_bin)]
            mes += " linear sweep for channel {}.".format(ch)
            print(mes)
            print(cfr_old_bin, len(cfr_old_bin))
            print(cfr_new_bin, len(cfr_new_bin))

        return

    def configure_linear_sweep(
        self, channels=None, rsrr=0, fsrr=0, rdw=0, fdw=0, disable=False
    ):
        """Configure the linear frequency sweep parameters for selected channels.

        The linear sweep ramp rate (lsrr) specifies the timestep of the rising ramp, falling sweep ramp rate
        (fsrr) works accordingly.
        Rising delta word specifies the rising frequency stepsize, falling delta works respectively.

        :channels: int or list
            Channel ID(s) for channels to configure.
        :lsrr: float
            Timestep (in seconds) of the rising sweep. Can be 1-256 times the inverse SYNC_CLK frequency.
            SYNC_CLK frequency is the SYSCLK divided by 4.
        :fsrr: float
            Same as :lsrr:
        :rdw: float
            Frequency step (in Hertz) of the rising sweep. Can be chosen similar to the channel frequency.
        :fdw: float
            Same as :rdw:
        :disable: bool
            If True, disable linear sweep for selected channels.
        :returns: TODO

        """
        if channels is None:
            channels = self.channel
        if np.issubdtype(type(channels), np.integer):
            channels = [channels]
        channels.sort()

        # If desired, disable selected channels and return.
        self._enable_channel_linear_sweep(channels, disable=disable)
        if disable:
            return

        # All linear sweep properties are in individual channel registers, so we
        # can write all channels in one go
        self._channel_select(channels)

        ######################################################
        # 1. Set the new falling and rising sweep ramp rate
        ramp_rate_word = ""
        rr_name = ["Falling", "Rising"]
        for i, rr in enumerate([fsrr, rsrr]):
            # 1.1 Compute  word in binary
            rr_time_step = 4 / self.system_clock_frequency
            fraction_bin = round(rr / rr_time_step)

            # 1.2 Check for correct bounds
            if fraction_bin < 1:
                print("Ramp rate below lower limit, choosing lowest possible value.")
                fraction_bin = 1
            elif fraction_bin > 256:
                print("Ramp rate above upper limit, choosing highest possible value.")
                fraction_bin = 256

            # align the fraction_bin with binary representation
            print(
                "Setting {} sweep ramp rate to {:1.3e} s".format(
                    rr_name[i], fraction_bin * rr_time_step
                )
            )
            fraction_bin -= 1
            rrw_bin = bin(fraction_bin)[2:]
            if len(rrw_bin) < 8:
                rrw_bin = (8 - len(rrw_bin)) * "0" + rrw_bin
            ramp_rate_word += rrw_bin
            print("Len RRW", len(ramp_rate_word))

        # write the new ramp rate word to the RR register
        ramp_rate_word = "".join(" 0" + b for b in ramp_rate_word)
        ramp_rate_word = ramp_rate_word[1:]
        print("RRW: {}".format(ramp_rate_word), len(ramp_rate_word))
        self._write_to_dds_register(0x07, ramp_rate_word)
        print(self._read_from_register(0x07, 16))

        ###############################################
        # 2. Set the falling and rising delta words.
        # calculate the fraction of the full frequency
        delta_word_registers = [0x09, 0x08]
        delta_words = [fdw, rdw]
        for i, dw in enumerate(delta_words):
            fraction = dw / self.system_clock_frequency
            fraction_bin = bin(int(round(fraction * (2**32 - 1)))).lstrip(
                "0b"
            )  # full range are 32 bit
            if len(fraction_bin) < 32:
                fraction_bin = (32 - len(fraction_bin)) * "0" + fraction_bin
            closest_possible_value = (
                int(fraction_bin, base=2) / (2**32 - 1) * self.system_clock_frequency
            )
            print(
                "Setting {2} delta word of channel {1} to closest possible value {0}MHz".format(
                    closest_possible_value / 1e6, channels, rr_name[i]
                )
            )

            # set the frequency word in the frequency register
            frequency_word = "".join(" 0" + b for b in fraction_bin)
            frequency_word = frequency_word[1:]
            self._write_to_dds_register(delta_word_registers[i], frequency_word)
            print(frequency_word)
            print(self._read_from_register(delta_word_registers[i], 32))
        return
