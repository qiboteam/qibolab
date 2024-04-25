# pylint: disable=too-many-lines
"""Instrument for using the Keysight M8195A AWG."""

import os.path
import socket


class SocketInstrumentError(Exception):
    """A class to handle errors related to socket instrument."""

    pass  # pylint: disable=unnecessary-pass


class GranularityError(Exception):
    """Waveform Granularity Exception class."""

    def __init__(self):
        pass

    def __str__(self):
        return "Must be a multiplication of Granularity"


class M8195Connection:
    """A class related to M8195A's connection."""

    def __init__(self, ip_address, port=5025, time_out=10):
        """Opens up a socket connection between an instrument and your PC
        :param ip_address: ip address of the instrument :param port: [Optional]
        socket port of the instrument (default 5025) :return: Returns the
        socket session."""
        self.open_session()  # initially was self.open_session
        self.port = port
        self.ip_address = ip_address
        self.time_out = time_out

        if ip_address.ip_address(self.ip_address):
            print(f"connecting to IPv4 address: {self.ip_address}")
        else:
            raise ValueError("Invalid IP address")

        self.open_session = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.open_session.settimeout(self.time_out)

    def open_session(self):  # pylint: disable=method-hidden
        """Opens the socket connection :return:"""
        print("Opening socket session and connection ...")
        print("connecting to M8195A ...")

        try:
            self.open_session.connect((self.ip_address, self.port))
            print("connected to M8195A ...")
        except OSError:
            print("Failed to connect to the instrument, please check your IP address")

        #  setblocking(1) = socket blocking -> it will wait for the operation to complete
        #  setblocking(0) = socket non-blocking -> it will never wait for the operation to complete
        self.open_session.setblocking(True)

        print("*IDN?: ")
        inst_query_idn = self.query("*idn?", error_check=False)
        print(inst_query_idn)
        if "Keysight Technologies,M8195A" in inst_query_idn:
            print("success!")
        else:
            self.close_session()
            raise NameError(
                "could not communicate with device, or not a Keysight Technologies, M8195A"
            )

    def close_session(self):
        """Closes the socket connection :return: TCPIP socket connection."""
        print("Closing socket session and connection ...")
        self.open_session.shutdown(socket.SHUT_RDWR)
        self.open_session.close()

    def error_check(self):
        """Checks an instrument for errors, print them out, and clears error
        queue.

        Raises SocketInstrumentError with the info of the error
        encountered.
        :return: Returns True if any errors are encountered
        """
        err = []
        response = self.query("SYST:ERR?", error_check=False).strip()

        while "0" not in response:
            err.append(response)
            response = self.query("SYST:ERR?", error_check=False).strip()

        if err:
            raise SocketInstrumentError(err)

        return response

    def query(self, command, error_check=False):
        """Sends a query to an instrument and reads the output buffer
        immediately afterward :param command: text containing an instrument
        command (Documented SCPI); Should end with "?" :param error_check:
        [Optional] Check for instrument errors (default False) :return: Returns
        the query response."""

        if not isinstance(command, str):
            raise SocketInstrumentError("command must be a string.")

        if "?" not in command:
            raise SocketInstrumentError('Query must end with "?"')

        try:
            self.open_session.sendall(str.encode(command + "\n"))
            response = self.read()
            if error_check:
                err = self.error_check()
                if err:
                    response = "<Error>"
                    print(f"Query - local: {error_check}, command: {command}")

        except socket.timeout:
            print("Query error:")
            self.error_check()
            response = "<Timeout Error>"

        return response

    def read(self):
        """Reads from a socket until a newline is read :return: Returns the
        data read."""
        response = b""
        while response[-1:] != b"\n":
            response += self.open_session.recv(4096)

        return response.decode().strip()

    def write(self, command, error_check=False):
        """Write a command to an instrument :param command: text containing an
        instrument command; i.e. Documented SCPI command :param error_check:
        [Optional] Check for instrument errors (default False) :return:"""
        if not isinstance(command, str):
            raise SocketInstrumentError("Argument must be a string.")

        command = f"{command}\n"
        self.open_session.sendall(command.encode())

        if error_check:
            print(f"Send - local: {error_check}, command: {command}")
            self.error_check()


class M8195AConfiguration:  # pylint: disable=too-many-public-methods
    """This is a class to configure M8195A."""

    def __init__(self):
        """"""
        self.established_connection = M8195Connection(ip_address="0.0.0.0", port=5025)
        self._min_max_list = ("MIN", "MAX", "MINimum", "MAXimum")
        self._on_list = (1, "1", "on", "ON", True)
        self._off_list = (0, "0", "off", "OFF", False)
        self._channel_list = (1, 2, 3, 4)

    def open_io_session(self):
        """Open IO session :return:"""
        self.established_connection.open_session()

    def close_io_session(self):
        """Close IO session :return:"""
        self.established_connection.close_session()

    #####################################################################
    # 6.5 System Related Commands (SYSTem Subsystem) ####################
    #####################################################################
    def set_event_in_trigger_out_switch(self, switch):
        """The Event In and Trigger Out functionality use a shared connector on
        the front panel.

        This command switches
        between trigger output and event input functionality. When Trigger Out functionality is active, Event In
        functionality is disabled and vice versa.
        Note: Trigger Out is for future use. There are no plans to support Trigger Out functionality directly from
        M8195A firmware. Trigger Out is tentatively supported by 81195A optical modulation generator software (V2.1
        or later).
        :param switch: 'EIN', 'TOUT'
        :return:
        """
        if switch in ("EIN", "TOUT"):
            self.established_connection.write(
                f":SYST:EIN:MODE {switch}", error_check=True
            )
        else:
            raise NameError("M8195A: Invalid switch in set_event_in_trigger_out_switch")

    def get_event_in_trigger_out_switch(self):
        """The Event In and Trigger Out functionality use a shared connector on
        the front panel.

        This command switches
        between trigger output and event input functionality. When Trigger Out functionality is active, Event In
        functionality is disabled and vice versa.
        Note: Trigger Out is for future use. There are no plans to support Trigger Out functionality directly from
        M8195A firmware. Trigger Out is tentatively supported by 81195A optical modulation generator software (V2.1
        or later).
        :return: 'EIN', 'TOUT'
        """
        self.established_connection.query(":SYST:EIN:MODE?", error_check=True)

    def get_error(self):
        """Read and clear one error from the instrument’s error queue. A record
        of up to 30 command syntax or hardware errors can be stored in the
        error queue. Errors are retrieved in first-in-first-out (FIFO) order.
        The first error returned is the first error that was stored. Errors are
        cleared as you read them. If more than 30 errors have occurred, the
        last error stored in the queue (the most recent error) is replaced with
        “Queue overflow”. No additional errors are stored until you remove
        errors from the queue.

        *CLS command: The error queue is cleared, when the power is
        cycled, or when the Soft Front Panel is re-started. *RST
        command: The error queue is not cleared by a reset.

        :return: The error messages have the following format (the error
            string may contain up to 255 characters): error
            number,”Description”, e.g. -113,"Undefined header" If no
            errors have occurred when you read the error queue, the
            instrument responds with 0,“No error”.
        """
        self.established_connection.query(":SYST:ERR?", error_check=True)

    def get_scpi_list(self):
        """The HEADers?

        query returns all SCPI commands and queries and IEEE 488.2 common commands and common queries
        implemented by the instrument.
        :return: The response is a <DEFINITE LENGTH ARBITRARY BLOCK RESPONSE DATA> element. The full path for every
        command and query is returned separated by linefeeds. The syntax of the response is defined as: The
        <nonzero digit> and sequence of <digit> follow the rules in IEEE 488.2, Section 8.7.9. A <SCPI header> is
        defined as: It contains all the nodes from the root. The <SCPI program mnemonic> contains the node in standard
        SCPI format. The short form uses uppercase characters while the additional characters for the long form are
        in lowercase characters. Default nodes are surrounded by square brackets ([]).
        """
        self.established_connection.query(":SYST:HELP:HEAD?", error_check=True)

    def get_license(self):
        """This query lists the licenses installed.

        :return:
        """
        self.established_connection.query(":SYST:LIC:EXT:LIST?", error_check=True)

    def set_instrument_setting(self, binary_data):
        """In set form, the block data must be a complete instrument set-up
        read using the query form of the command.

        The data is in a binary format.
        :return:
        """
        self.established_connection.write(f":SYST:SET {binary_data}", error_check=True)

    def get_instrument_setting(self):
        """In query form, the command reads a block of data containing the
        instrument’s complete set-up.

        The set-up information includes all parameter and mode settings,
        but does not include the contents of the instrument setting
        memories or the status group registers. The data is in a binary
        format, not ASCII, and cannot be edited. This command has the
        same functionality as the *LRN command.
        """
        self.established_connection.query(":SYST:SET?", error_check=True)

    def get_scpi_version(self):
        """Query SCPI version number :return: a formatted numeric value
        corresponding to the SCPI version number for which the instrument
        complies."""
        self.established_connection.query(":SYST:VERS?", error_check=True)

    def get_softfrontpanel_connections(self):
        """These queries return information about the instrument Soft Front
        Panel’s available connections.

        This is only useful if there is more than one Keysight module
        connected to a PC, otherwise one would normally use the default
        connections (HiSLIP and VXI-11 instrument number 0, socket port
        5025, telnet port 5024). One can never be sure if a socket port
        is already in use, so one could e.g. specify a HiSLIP number on
        the command line (AgM8195SFP.exe /AutoID /Inst5 /FallBack /r …)
        and let the Soft Front Panel find an unused socket port. Then
        this socket port can be queried using the HiSLIP connection.
        :return: If a connection is not available, the returned value is
            -1.
        """
        self.established_connection.query(":SYST:COMM:*?", error_check=True)

    def get_softfrontpanel_vxi11number(self):
        """Query VXI-11 instrument number :return: This query returns the
        VXI-11 instrument number used by the Soft Front Panel."""
        self.established_connection.query(":SYST:COMM:INST?", error_check=True)

    def get_softfrontpanel_hislipnumber(self):
        """Query HiSLIP number :return: This query returns the HiSLIP number
        used by the Soft Front Panel."""
        self.established_connection.query(":SYST:COMM:HISL?", error_check=True)

    def get_softfrontpanel_socketport(self):
        """Query socket port :return: This query returns the socket port used
        by the Soft Front Panel."""
        self.established_connection.query(":SYST:COMM:SOCK?", error_check=True)

    def get_softfrontpanel_telnetport(self):
        """Query telnet port :return: This query returns the telnet port used
        by the Soft Front Panel."""
        self.established_connection.query(":SYST:COMM:TELN?", error_check=True)

    def get_softfrontpanel_tcpport(self):
        """

        :return: This query returns the port number of the control connection. You can use the control port to send
        control commands (for example “Device Clear”) to the instrument.
        """
        self.established_connection.query(":SYST:COMM:TCP:CONT?", error_check=True)

    #####################################################################
    # 6.6 Common Command List ###########################################
    #####################################################################
    def instrument_id(self):
        """Read the instrument’s identification string which contains four
        fields separated by commas.

        The first field is
        the manufacturer’s name, the second field is the model number, the third field is the serial number, and the
        fourth field is a revision code which contains four numbers separated dots and a fifth number separated by a
        dash:
        Keysight Technologies, M8195A,<serial number>, x.x.x.x-h
        x.x.x.x= Soft Front Panel revision number, e.g. 2.0.0.0
        h= Hardware revision number
        :return:
        """
        self.established_connection.query("*IDN?", error_check=True)

    def clear_event_register(self):
        """Clear the event register in all register groups.

        This command also clears the error queue and cancels a *OPC
        operation. It doesn't clear the enable register.
        :return:
        """
        self.established_connection.write("*CLS", error_check=True)

    def set_status_register_bit5(self):
        """Enable bits in the Standard Event Status Register to be reported in
        the Status Byte.

        The selected bits are summarized in the “Standard Event” bit
        (bit 5) of the Status Byte Register. These bits are not cleared
        by a *CLS command. Value Range: 0–255.
        :return:
        """
        self.established_connection.write("*ESE", error_check=True)

    def get_status_register_bit5(self):
        """The *ESE?

        query returns a value which corresponds to the binary-weighted
        sum of all bits enabled decimal by the *ESE command. Value
        Range: 0–255.
        :return:
        """
        self.established_connection.query("*ESE?", error_check=True)

    def get_standard_event_status_register(self):
        """Query the Standard Event Status Register.

        Once a bit is set, it remains set until cleared by a *CLS (clear
        status) command or queried by this command. A query of this
        register returns a decimal value which corresponds to the
        binary-weighted sum of all bits set in the register.
        :return:
        """
        self.established_connection.query("ESR?", error_check=True)

    def set_operation_complete(self):
        """Set the “Operation Complete” bit (bit 0) in the Standard Event
        register after the previous commands have been completed.

        :return:
        """
        self.established_connection.write("*OPC", error_check=True)

    def get_operation_complete(self):
        """Return "1" to the output buffer after the previous commands have
        been completed.

        Other commands cannot be executed until this command completes.
        :return:
        """
        self.established_connection.query("*OPC?", error_check=True)

    def get_installed_options(self):
        """Read the installed options.

        The response consists of any number of fields separated by
        commas.
        :return:
        """
        self.established_connection.query("*OPT?", error_check=True)

    def instrument_reset(self):
        """Reset instrument to its factory default state.

        :return:
        """
        self.established_connection.write("*RST", error_check=True)

    def set_service_request_enable_bits(self, bits):
        """Enable bits in the Status Byte to generate a Service Request.

        To enable specific bits, you must write a decimal value which
        corresponds to the binary-weighted sum of the bits in the
        register. The selected bits are summarized in the “Master
        Summary” bit (bit 6) of the Status Byte Register. If any of the
        selected bits change from “0” to “1”, a Service Request signal
        is generated.
        :return:
        """
        if bits.isdecimal():
            self.established_connection.write(f"*SRE {bits}", error_check=True)
        else:
            raise ValueError(
                "M8195A: Invalid bits (not decimal) in set_service_request_enable_bits"
            )

    def get_service_request_enable_bits(self):
        """The *SRE?

        query returns a decimal value which corresponds to the binary-
        weighted sum of all bits enabled by the *SRE command.
        :return:
        """
        self.established_connection.query("*SRE?", error_check=True)

    def get_status_byte_register(self):
        """Query the summary (status byte condition) register in this register
        group.

        This command is similar to a Serial Poll, but it is processed
        like any other instrument command. This command returns the same
        result as a Serial Poll but the “Master Summary” bit (bit 6) is
        not cleared by the *STB? command.
        :return:
        """
        self.established_connection.query("*STB?", error_check=True)

    def get_self_test(self):
        """Execute Self Tests.

        If self-tests pass, a 0 is returned. A number lager than 0
        indicates the number of failed tests. To get actual messages,
        use :TEST:TST?
        :return:
        """
        self.established_connection.query("*TST?", error_check=True)

    def get_instrument_setting_learn(self):
        """Query the instrument and return a binary block of data containing
        the current settings (learn string).

        You can then send the string back to the instrument to restore
        this state later. For proper operation, do not modify the
        returned string before sending it to the instrument. Use
        :SYST:SET to send the learn string. See :SYSTem:SET[?].
        :return:
        """
        self.established_connection.query("*LRN?", error_check=True)

    def get_wait_current_command_execution(self):
        """Prevents the instrument from executing any further commands until
        the current command has finished executing.

        :return:
        """
        self.established_connection.query("*WAI?", error_check=True)

    #####################################################################
    # 6.7 Status Model ##################################################
    #####################################################################
    # 6.7.1 :STATus:PRESet ##############################################
    def clear_status_group_event_registers(self):
        """Clears all status group event registers.

        Presets the status group enables PTR and NTR registers as follows:
        ENABle = 0x0000, PTR = 0xffff, NTR = 0x0000
        :return:
        """
        self.established_connection.write(":STAT:PRES", error_check=True)

    # 6.7.3 Questionable Data Register Command Subsystem ################
    # 6.7.5 Voltage Status Subsystem ####################################
    # 6.7.6 Frequency Status Subsystem ##################################
    # 6.7.7 Sequence Status Subsystem ###################################
    # 6.7.8 DUC Status Subsystem ########################################
    # 6.7.9 Connection Status Subsystem #################################
    def get_questionable_status(self, event=None, sub_register=None):
        """Reads the event register in the questionable status group. It’s a
        read-only register. Once a bit is set, it remains set until cleared by
        this command or the *CLS command. A query of the register returns a
        decimal value which corresponds to the binary-weighted sum of all bits
        set in the register. :param event: event register in the questionable
        status group.

        :param sub_register: 'VOLT', 'FREQ', 'CONN', 'SEQ', or 'DUC'
            - The Voltage Status register contains the voltage conditions of the individual channels. Check
            "6.7.5 Voltage Status Subsystem" and "Table 26: Voltage status register" in Keysight M8195A AWG Revision 2
            - The Frequency Status register contains the frequency conditions of the module. Check "6.7.6 Frequency
            Status Subsystem" and "Table 27: Frequency status register" in Keysight M8195A AWG Revision 2
            - The Connection Status register contains the state of the USB connection to the M8195A module. Check
            "6.7.9 Connection Status Subsystem" and "Table 30: Connection status register" in Keysight M8195A AWG
            Revision 2
            - The Sequence Status register is used to indicate errors in the sequence table data provided by the user.
            Check "6.7.7 Sequence Status Subsystem" and "Table 28: Sequence status register" in Keysight M8195A AWG
            Revision 2
            - The DUC Status register contains the conditions after up-conversion of an imported file for the individual
            channels. Check "6.7.8 DUC Status Subsystem" and "Table 29: DUC status register" in Keysight M8195A AWG
            Revision 2
        :return:
        """
        if sub_register is None:
            if event is None:
                self.established_connection.query(":STAT:QUES?", error_check=True)
            else:
                self.established_connection.query(":STAT:QUES:EVEN?", error_check=True)
        elif sub_register in ("VOLT", "FREQ", "CONN", "SEQ", "DUC"):
            if event is None:
                self.established_connection.query(
                    f":STAT:QUES:{sub_register}?", error_check=True
                )
            else:
                self.established_connection.query(
                    f":STAT:QUES:{sub_register}:EVEN?", error_check=True
                )
        else:
            raise NameError("M8195A: Invalid sub_register in get_questionable_status")

    def get_questionable_status_condition(self, sub_register=None):
        """Reads the condition register in the questionable status group. It’s
        a read-only register and bits are not cleared when you read the
        register. A query of the register returns a decimal value which
        corresponds to the binary-weighted sum of all bits set in the register.

        :param sub_register: 'VOLT', 'FREQ', 'CONN', 'SEQ', or 'DUC'
            - The Voltage Status register contains the voltage conditions of the individual channels. Check
            "6.7.5 Voltage Status Subsystem" and "Table 26: Voltage status register" in Keysight M8195A AWG Revision 2
            - The Frequency Status register contains the frequency conditions of the module. Check "6.7.6 Frequency
            Status Subsystem" and "Table 27: Frequency status register" in Keysight M8195A AWG Revision 2
            - The Connection Status register contains the state of the USB connection to the M8195A module. Check
            "6.7.9 Connection Status Subsystem" and "Table 30: Connection status register" in Keysight M8195A AWG
            Revision 2
            - The Sequence Status register is used to indicate errors in the sequence table data provided by the user.
            Check "6.7.7 Sequence Status Subsystem" and "Table 28: Sequence status register" in Keysight M8195A AWG
            Revision 2
            - The DUC Status register contains the conditions after up-conversion of an imported file for the individual
            channels. Check "6.7.8 DUC Status Subsystem" and "Table 29: DUC status register" in Keysight M8195A AWG
            Revision 2
        :return:
        """
        if sub_register is None:
            self.established_connection.query(":STAT:QUES:COND?", error_check=True)
        elif sub_register in ("VOLT", "FREQ", "CONN", "SEQ", "DUC"):
            self.established_connection.query(
                f":STAT:QUES:{sub_register}:COND?", error_check=True
            )
        else:
            raise NameError(
                "M8195A: Invalid sub_register in get_questionable_status_condition"
            )

    def set_questionable_status_enable(self, decimal_value, sub_register=None):
        """Sets the enable register in the questionable status group. The
        selected bits are then reported to the status Byte. A *CLS will not
        clear the enable register, but it does clear all bits in the event
        register. To enable bits in the enable register, you must write a
        decimal value which corresponds to the binary-weighted sum of the bits
        you wish to enable in the register. :param decimal_value:

        :param sub_register: 'VOLT', 'FREQ', 'CONN', 'SEQ', or 'DUC'
            - The Voltage Status register contains the voltage conditions of the individual channels. Check
            "6.7.5 Voltage Status Subsystem" and "Table 26: Voltage status register" in Keysight M8195A AWG Revision 2
            - The Frequency Status register contains the frequency conditions of the module. Check "6.7.6 Frequency
            Status Subsystem" and "Table 27: Frequency status register" in Keysight M8195A AWG Revision 2
            - The Connection Status register contains the state of the USB connection to the M8195A module. Check
            "6.7.9 Connection Status Subsystem" and "Table 30: Connection status register" in Keysight M8195A AWG
            Revision 2
            - The Sequence Status register is used to indicate errors in the sequence table data provided by the user.
            Check "6.7.7 Sequence Status Subsystem" and "Table 28: Sequence status register" in Keysight M8195A AWG
            Revision 2
            - The DUC Status register contains the conditions after up-conversion of an imported file for the individual
            channels. Check "6.7.8 DUC Status Subsystem" and "Table 29: DUC status register" in Keysight M8195A AWG
            Revision 2
        :return:
        """
        if sub_register is None:
            self.established_connection.write(
                f":STAT:QUES:ENAB {decimal_value}", error_check=True
            )
        elif sub_register in ("VOLT", "FREQ", "CONN", "SEQ", "DUC"):
            self.established_connection.write(
                f":STAT:QUES:{sub_register}:ENAB {decimal_value}", error_check=True
            )
        else:
            raise NameError(
                "M8195A: Invalid sub_register in set_questionable_status_enable"
            )

    def get_questionable_status_enable(self, sub_register=None):
        """Queries the enable register in the questionable status group. The
        selected bits are then reported to the Status Byte. A *CLS will not
        clear the enable register, but it does clear all bits in the event
        register.

        :param sub_register: 'VOLT', 'FREQ', 'CONN', 'SEQ', or 'DUC'
            - The Voltage Status register contains the voltage conditions of the individual channels. Check
            "6.7.5 Voltage Status Subsystem" and "Table 26: Voltage status register" in Keysight M8195A AWG Revision 2
            - The Frequency Status register contains the frequency conditions of the module. Check "6.7.6 Frequency
            Status Subsystem" and "Table 27: Frequency status register" in Keysight M8195A AWG Revision 2
            - The Connection Status register contains the state of the USB connection to the M8195A module. Check
            "6.7.9 Connection Status Subsystem" and "Table 30: Connection status register" in Keysight M8195A AWG
            Revision 2
            - The Sequence Status register is used to indicate errors in the sequence table data provided by the user.
            Check "6.7.7 Sequence Status Subsystem" and "Table 28: Sequence status register" in Keysight M8195A AWG
            Revision 2
            - The DUC Status register contains the conditions after up-conversion of an imported file for the individual
            channels. Check "6.7.8 DUC Status Subsystem" and "Table 29: DUC status register" in Keysight M8195A AWG
            Revision 2
        :return:
        """
        if sub_register is None:
            self.established_connection.query(":STAT:QUES:ENAB?", error_check=True)
        elif sub_register in ("VOLT", "FREQ", "CONN", "SEQ", "DUC"):
            self.established_connection.query(
                f":STAT:QUES:{sub_register}:ENAB?", error_check=True
            )
        else:
            raise NameError(
                "M8195A: Invalid sub_register in get_questionable_status_enable"
            )

    def set_questionable_status_neg_transition(self, allow, sub_register=None):
        """Sets the negative-transition register in the questionable status
        group. A negative transition filter allows event to be reported when a
        condition changes from true to false. Setting both positive/negative
        filters true allows an event to be reported anytime the condition
        changes. Clearing both filters disable event reporting. The contents of
        transition filters are unchanged by *CLS and *RST. :param allow: True
        of False.

        :param sub_register: 'VOLT', 'FREQ', 'CONN', 'SEQ', or 'DUC'
            - The Voltage Status register contains the voltage conditions of the individual channels. Check
            "6.7.5 Voltage Status Subsystem" and "Table 26: Voltage status register" in Keysight M8195A AWG Revision 2
            - The Frequency Status register contains the frequency conditions of the module. Check "6.7.6 Frequency
            Status Subsystem" and "Table 27: Frequency status register" in Keysight M8195A AWG Revision 2
            - The Connection Status register contains the state of the USB connection to the M8195A module. Check
            "6.7.9 Connection Status Subsystem" and "Table 30: Connection status register" in Keysight M8195A AWG
            Revision 2
            - The Sequence Status register is used to indicate errors in the sequence table data provided by the user.
            Check "6.7.7 Sequence Status Subsystem" and "Table 28: Sequence status register" in Keysight M8195A AWG
            Revision 2
            - The DUC Status register contains the conditions after up-conversion of an imported file for the individual
            channels. Check "6.7.8 DUC Status Subsystem" and "Table 29: DUC status register" in Keysight M8195A AWG
            Revision 2
        :return:
        """
        if allow in (True, False):
            if sub_register is None:
                self.established_connection.write(
                    f":STAT:QUES:NTR {allow}", error_check=True
                )
            elif sub_register in ("VOLT", "FREQ", "CONN", "SEQ", "DUC"):
                self.established_connection.write(
                    f":STAT:QUES:{sub_register}:NTR {allow}", error_check=True
                )
            else:
                raise NameError(
                    "M8195A: Invalid sub_register in set_questionable_status_neg_transition"
                )
        else:
            raise NameError(
                "M8195A: Invalid allow in set_questionable_status_neg_transition"
            )

    def get_questionable_status_neg_transition(self, sub_register=None):
        """Queries the negative-transition register in the questionable status
        group. A negative transition filter allows event to be reported when a
        condition changes from true to false. Setting both positive/negative
        filters true allows an event to be reported anytime the condition
        changes. Clearing both filters disable event reporting. The contents of
        transition filters are unchanged by *CLS and *RST.

        :param sub_register: 'VOLT', 'FREQ', 'CONN', 'SEQ', or 'DUC'
            - The Voltage Status register contains the voltage conditions of the individual channels. Check
            "6.7.5 Voltage Status Subsystem" and "Table 26: Voltage status register" in Keysight M8195A AWG Revision 2
            - The Frequency Status register contains the frequency conditions of the module. Check "6.7.6 Frequency
            Status Subsystem" and "Table 27: Frequency status register" in Keysight M8195A AWG Revision 2
            - The Connection Status register contains the state of the USB connection to the M8195A module. Check
            "6.7.9 Connection Status Subsystem" and "Table 30: Connection status register" in Keysight M8195A AWG
            Revision 2
            - The Sequence Status register is used to indicate errors in the sequence table data provided by the user.
            Check "6.7.7 Sequence Status Subsystem" and "Table 28: Sequence status register" in Keysight M8195A AWG
            Revision 2
            - The DUC Status register contains the conditions after up-conversion of an imported file for the
            individual channels. Check "6.7.8 DUC Status Subsystem" and "Table 29: DUC status register" in Keysight
            M8195A AWG Revision 2
        :return:
        """
        if sub_register is None:
            self.established_connection.query(":STAT:QUES:NTR?", error_check=True)
        elif sub_register in ("VOLT", "FREQ", "CONN", "SEQ", "DUC"):
            self.established_connection.query(
                f":STAT:QUES:{sub_register}:NTR?", error_check=True
            )
        else:
            raise NameError(
                "M8195A: Invalid sub_register in get_questionable_status_neg_transition"
            )

    def set_questionable_status_pos_transition(self, allow, sub_register=None):
        """Set the positive-transition register in the questionable status
        group. A positive transition filter allows an event to be reported when
        a condition changes from false to true. Setting both positive/negative
        filters true allows an event to be reported anytime the condition
        changes. Clearing both filters disable event reporting. The contents of
        transition filters are unchanged by *CLS and *RST. :param allow: True
        or False.

        :param sub_register: 'VOLT', 'FREQ', 'CONN', 'SEQ', or 'DUC'
            - The Voltage Status register contains the voltage conditions of the individual channels. Check
            "6.7.5 Voltage Status Subsystem" and "Table 26: Voltage status register" in Keysight M8195A AWG Revision 2
            - The Frequency Status register contains the frequency conditions of the module. Check "6.7.6 Frequency
            Status Subsystem" and "Table 27: Frequency status register" in Keysight M8195A AWG Revision 2
            - The Connection Status register contains the state of the USB connection to the M8195A module. Check
            "6.7.9 Connection Status Subsystem" and "Table 30: Connection status register" in Keysight M8195A AWG
            Revision 2
            - The Sequence Status register is used to indicate errors in the sequence table data provided by the user.
            Check "6.7.7 Sequence Status Subsystem" and "Table 28: Sequence status register" in Keysight M8195A AWG
            Revision 2
            - The DUC Status register contains the conditions after up-conversion of an imported file for the individual
            channels. Check "6.7.8 DUC Status Subsystem" and "Table 29: DUC status register" in Keysight M8195A AWG
            Revision 2
        :return:
        """
        if allow in (True, False):
            if sub_register is None:
                self.established_connection.write(
                    f":STAT:QUES:PTR {allow}", error_check=True
                )
            elif sub_register in ("VOLT", "FREQ", "CONN", "SEQ", "DUC"):
                self.established_connection.write(
                    f":STAT:QUES:{sub_register}:PTR", error_check=True
                )
            else:
                raise NameError(
                    "M8195A: Invalid sub_register in set_questionable_status_pos_transition"
                )
        else:
            raise NameError(
                "M8195A: Invalid allow in set_questionable_status_pos_transition"
            )

    def get_questionable_status_pos_transition(self, sub_register=None):
        """Queries the positive-transition register in the questionable status
        group. A positive transition filter allows an event to be reported when
        a condition changes from false to true. Setting both positive/negative
        filters true allows an event to be reported anytime the condition
        changes. Clearing both filters disable event reporting. The contents of
        transition filters are unchanged by *CLS and *RST.

        :param sub_register: 'VOLT', 'FREQ', 'CONN', 'SEQ', or 'DUC'
            - The Voltage Status register contains the voltage conditions of the individual channels. Check
            "6.7.5 Voltage Status Subsystem" and "Table 26: Voltage status register" in Keysight M8195A AWG Revision 2
            - The Frequency Status register contains the frequency conditions of the module. Check "6.7.6 Frequency
            Status Subsystem" and "Table 27: Frequency status register" in Keysight M8195A AWG Revision 2
            - The Connection Status register contains the state of the USB connection to the M8195A module. Check
            "6.7.9 Connection Status Subsystem" and "Table 30: Connection status register" in Keysight M8195A AWG
            Revision 2
            - The Sequence Status register is used to indicate errors in the sequence table data provided by the user.
            Check "6.7.7 Sequence Status Subsystem" and "Table 28: Sequence status register" in Keysight M8195A AWG
            Revision 2
            - The DUC Status register contains the conditions after up-conversion of an imported file for the individual
            channels. Check "6.7.8 DUC Status Subsystem" and "Table 29: DUC status register" in Keysight M8195A AWG
            Revision 2
        :return:
        """
        if sub_register is None:
            self.established_connection.query(":STAT:QUES:PTR?", error_check=True)
        elif sub_register in ("VOLT", "FREQ", "CONN", "SEQ", "DUC"):
            self.established_connection.query(
                f":STAT:QUES:{sub_register}:PTR?", error_check=True
            )
        else:
            raise NameError(
                "M8195A: Invalid sub_register in get_questionable_status_pos_transition"
            )

    # 6.7.4 Operation Status Subsystem ##################################
    # 6.7.10 Run Status Subsystem #######################################
    def get_operation_status(self, event=None, sub_register=None):
        """Reads the event register in the operation status group. It’s a read-
        only register. Once a bit is set, it remains set until cleared by this
        command or *CLS command. A query of the register returns a decimal
        value which corresponds to the binary-weighted sum of all bits set in
        the register. :param event: event register in the questionable status
        group.

        :param sub_register: 'RUN'
            - The Run Status register contains the run status conditions of the individual channels. Check
            "6.7.10 Run Status Subsystem" and "Table 31: Run status register" in Keysight M8195A AWG Revision 2
        :return:
        """
        if sub_register is None:
            if event is None:
                self.established_connection.query(":STAT:OPER?", error_check=True)
            else:
                self.established_connection.query(":STAT:OPER:EVEN?", error_check=True)
        elif sub_register == "RUN":
            if event is None:
                self.established_connection.query(
                    f":STAT:OPER:{sub_register}?", error_check=True
                )
            else:
                self.established_connection.query(
                    f":STAT:OPER:{sub_register}:EVEN?", error_check=True
                )
        else:
            raise NameError("M8195A: Invalid sub_register in get_operation_status")

    def get_operation_status_condition(self, sub_register=None):
        """Reads the condition register in the operation status group. It’s a
        read-only register and bits are not cleared when you read the register.
        A query of the register returns a decimal value which corresponds to
        the binary-weighted sum of all bits set in the register.

        :param sub_register: 'RUN'
            - The Run Status register contains the run status conditions of the individual channels. Check
            "6.7.10 Run Status Subsystem" and "Table 31: Run status register" in Keysight M8195A AWG Revision 2
        :return:
        """
        if sub_register is None:
            self.established_connection.query(":STAT:OPER:COND?", error_check=True)
        elif sub_register == "RUN":
            self.established_connection.query(
                f":STAT:OPER:{sub_register}:COND?", error_check=True
            )
        else:
            raise NameError(
                "M8195A: Invalid sub_register in get_operation_status_condition"
            )

    def set_operation_status_enable(self, decimal_value, sub_register=None):
        """Sets the enable register in the operation status group. The selected
        bits are then reported to the Status Byte. A *CLS will not clear the
        enable register, but it does clear all bits in the event register. To
        enable bits in the enable register, you must write a decimal value
        which corresponds to the binary-weighted sum of the bits you wish to
        enable in the register. :param decimal_value:

        :param sub_register: 'RUN'
            - The Run Status register contains the run status conditions of the individual channels. Check
            "6.7.10 Run Status Subsystem" and "Table 31: Run status register" in Keysight M8195A AWG Revision 2
        :return:
        """
        if sub_register is None:
            self.established_connection.write(
                f":STAT:OPER:ENAB {decimal_value}", error_check=True
            )
        elif sub_register == "RUN":
            self.established_connection.write(
                f":STAT:OPER:{sub_register}:ENAB {decimal_value}", error_check=True
            )
        else:
            raise NameError(
                "M8195A: Invalid sub_register in set_operation_status_enable"
            )

    def get_operation_status_enable(self, sub_register=None):
        """Queries the enable register in the operation status group. The
        selected bits are then reported to the Status Byte. A *CLS will not
        clear the enable register, but it does clear all bits in the event
        register.

        :param sub_register: 'RUN'
            - The Run Status register contains the run status conditions of the individual channels. Check
            "6.7.10 Run Status Subsystem" and "Table 31: Run status register" in Keysight M8195A AWG Revision 2
        :return:
        """
        if sub_register is None:
            self.established_connection.query(":STAT:OPER:ENAB?", error_check=True)
        elif sub_register == "RUN":
            self.established_connection.query(
                f":STAT:OPER:{sub_register}:ENAB?", error_check=True
            )
        else:
            raise NameError(
                "M8195A: Invalid sub_register in get_operation_status_enable"
            )

    def set_operation_status_neg_transition(self, allow, sub_register=None):
        """Sets the negative-transition register in the operation status group.
        A negative transition filter allows an event to be reported when a
        condition changes from true to false. Setting both positive/negative
        filters true allows an event to be reported anytime the condition
        changes. Clearing both filters disable event reporting. The contents of
        transition filters are unchanged by *CLS and *RST. :param allow: True
        of False.

        :param sub_register: 'RUN'
            - The Run Status register contains the run status conditions of the individual channels. Check
            "6.7.10 Run Status Subsystem" and "Table 31: Run status register" in Keysight M8195A AWG Revision 2
        :return:
        """
        if allow in (True, False):
            if sub_register is None:
                self.established_connection.write(
                    f":STAT:OPER:NTR {allow}", error_check=True
                )
            elif sub_register == "RUN":
                self.established_connection.write(
                    f":STAT:OPER:{sub_register}:NTR {allow}", error_check=True
                )
            else:
                raise NameError(
                    "M8195A: Invalid sub_register in set_operation_status_neg_transition"
                )
        else:
            raise NameError(
                "M8195A: Invalid allow in set_operation_status_neg_transition"
            )

    def get_operation_status_neg_transition(self, sub_register=None):
        """Queries the negative-transition register in the operation status
        group. A negative transition filter allows an event to be reported when
        a condition changes from true to false. Setting both positive/negative
        filters true allows an event to be reported anytime the condition
        changes. Clearing both filters disable event reporting. The contents of
        transition filters are unchanged by *CLS and *RST.

        :param sub_register: 'RUN'
            - The Run Status register contains the run status conditions of the individual channels. Check
            "6.7.10 Run Status Subsystem" and "Table 31: Run status register" in Keysight M8195A AWG Revision 2
        :return:
        """
        if sub_register is None:
            self.established_connection.query(":STAT:OPER:NTR?", error_check=True)
        elif sub_register == "RUN":
            self.established_connection.query(
                f":STAT:OPER:{sub_register}:NTR?", error_check=True
            )
        else:
            raise NameError(
                "M8195A: Invalid sub_register in get_operation_status_neg_transition"
            )

    def set_operation_status_pos_transition(self, allow, sub_register=None):
        """Set the positive-transition register in the operation status group.
        A positive transition filter allows an event to be reported when a
        condition changes from false to true. Setting both positive/negative
        filters true allows an event to be reported anytime the condition
        changes. Clearing both filters disable event reporting. The contents of
        transition filters are unchanged by *CLS and *RST. :param allow:

        :param sub_register: 'RUN'
            - The Run Status register contains the run status conditions of the individual channels. Check
            "6.7.10 Run Status Subsystem" and "Table 31: Run status register" in Keysight M8195A AWG Revision 2
        :return:
        """
        if allow in (True, False):
            if sub_register is None:
                self.established_connection.write(
                    f":STAT:OPER:PTR {allow}", error_check=True
                )
            elif sub_register == "RUN":
                self.established_connection.write(
                    f":STAT:OPER:{sub_register}:PTR", error_check=True
                )
            else:
                raise NameError(
                    "M8195A: Invalid sub_register in set_operation_status_pos_transition"
                )
        else:
            raise NameError(
                "M8195A: Invalid allow in set_operation_status_pos_transition"
            )

    def get_operation_status_pos_transition(self, sub_register=None):
        """Set the positive-transition register in the operation status group.
        A positive transition filter allows an event to be reported when a
        condition changes from false to true. Setting both positive/negative
        filters true allows an event to be reported anytime the condition
        changes. Clearing both filters disable event reporting. The contents of
        transition filters are unchanged by *CLS and *RST.

        :param sub_register: 'RUN'
            - The Run Status register contains the run status conditions of the individual channels. Check
            "6.7.10 Run Status Subsystem" and "Table 31: Run status register" in Keysight M8195A AWG Revision 2
        :return:
        """
        if sub_register is None:
            self.established_connection.query(":STAT:OPER:PTR?", error_check=True)
        elif sub_register == "RUN":
            self.established_connection.query(
                f":STAT:OPER:{sub_register}:PTR?", error_check=True
            )
        else:
            raise NameError(
                "M8195A: Invalid sub_register in get_operation_status_pos_transition"
            )

    #####################################################################
    # 6.8 Arm/Trigger Subsystem #########################################
    #####################################################################
    def signal_generation_channel_stop(self):
        """Stop signal generation on all channels.

        The channel suffix is ignored.
        :return:
        """
        self.established_connection.write(":ABOR", error_check=True)

    def set_module_delay(self, delay):
        """Set the module delay settings (see section 1.5.3) .

        The unit is in seconds.
        (This field specifies the module delay for all the channels. The range is 0 to 10 ns.)
        Parameter Suffix: [s|ms|us|ns|ps]
        :param delay: 'MIN', 'MAX', 'MINimum', 'MAXimum'
        :return:
        """
        if isinstance(delay, int):
            self.established_connection.write(f":ARM:MDEL {delay}", error_check=True)
        elif delay in self._min_max_list:
            self.established_connection.write(f":ARM:MDEL {delay}", error_check=True)
        else:
            raise ValueError(
                "M8195A: delay is neither integer nor proper string in set_module_delay"
            )

    def get_module_delay(self):
        """Query the module delay settings (see section 1.5.3) .

        The unit is in seconds.
        (This field specifies the module delay for all the channels. The range is 0 to 10 ns.)
        Parameter Suffix: [s|ms|us|ns|ps]
        :return:
        """
        self.established_connection.query(":ARM:MDEL?", error_check=True)

    def set_sample_clock_delay(self, delay, channel):
        """Set the channel-specific sample delay in integral DAC sample clock
        periods.

        The range is 0..95
        (sample clock delay individually per channel as an integral number of DAC sample clocks. The range is 0..95
        DAC sample clocks.)
        DAC Sample Frequency: The DAC Sample frequency is always in the range of (53.76...65) GHz. As the DAC sample
        frequency references to a clock, the unit of the sample frequency is Hz.
        :param channel: 1|2|3|4
        :param delay:
        :return:
        """
        if channel is None:
            self.established_connection.write(f":ARM:SDEL {delay}", error_check=True)
        elif channel in self._channel_list:
            if isinstance(delay, int):
                self.established_connection.write(
                    f":ARM:SDEL{channel} {delay}", error_check=True
                )
            elif delay in self._min_max_list:
                self.established_connection.write(
                    f":ARM:SDEL{channel} {delay}", error_check=True
                )
            else:
                raise ValueError(
                    "M8195A: delay is neither integer nor proper string in set_sample_clock_delay"
                )
        else:
            raise ValueError(
                "M8195A: Invalid channel in set_sample_clock_delay"
            )  # ChannelSampleDelaySet?

    def get_sample_clock_delay(self, channel):
        """Query the channel-specific sample delay in integral DAC sample clock
        periods.

        The range is 0..95
        :param channel: 1|2|3|4
        :return:
        """
        self.established_connection.query(f":ARM:SDEL{channel}?", error_check=True)

    def set_arm_mode(self, arm_mode):
        """Set the arming mode.

        :param arm_mode: 'SELF', 'ARMed'
        :return:
        """
        arm_mode_list = ("SELF", "ARMed")
        if arm_mode in arm_mode_list:
            self.established_connection.write(
                f":INIT:CONT:ENAB {arm_mode}", error_check=True
            )
        else:
            raise NameError("M8195A: Invalid arm_mode in set_arm_mode")

    def get_arm_mode(self):
        """Query the arming mode.

        :return:
        """
        self.established_connection.query(":INIT:CONT:ENAB?", error_check=True)

    def set_continuous_mode(self, continuous_mode):
        """Set the continuous mode. This command must be used together with
        INIT:GATE to set the trigger mode. Check "6.8.6 :INITiate:GATE" and
        "Table 32: Trigger mode settings" for more info on the output.

        :param continuous_mode:
            - 0/OFF – Continuous mode is off. If gate mode is off, the trigger mode is “triggered”, else it is “gated”.
            - 1/ON – Continuous mode is on. Trigger mode is “automatic”. The value of gate mode is not relevant.
        :return:
        """
        if continuous_mode in self._on_list:
            self.established_connection.write(":INIT:CONT:STAT ON", error_check=True)
        elif continuous_mode in self._off_list:
            self.established_connection.write(":INIT:CONT:STAT OFF", error_check=True)
        else:
            raise NameError("M8195A: Invalid continuous_mode in set_continuous_mode")

    def get_continuous_mode(self):
        """Query the continuous mode. This command must be used together with
        INIT:GATE to set the trigger mode. Check "6.8.6 :INITiate:GATE" and
        "Table 32: Trigger mode settings" for more info on the output.

        :return:
            - 0/OFF – Continuous mode is off. If gate mode is off, the trigger mode is “triggered”, else it is “gated”.
            - 1/ON – Continuous mode is on. Trigger mode is “automatic”. The value of gate mode is not relevant.
        """
        self.established_connection.query(":INIT:CONT:STAT?", error_check=True)

    def set_gate_mode(self, gate_mode):
        """Set the gate mode. This command must be used together with INIT:CONT
        to set the trigger mode. Check "6.8.6 :INITiate:GATE" and "Table 32:
        Trigger mode settings" for more info on the output.

        :param gate_mode:
            - 0/OFF – Gate mode is off.
            - 1/ON – Gate mode is on. If continuous mode is off, the trigger mode is “gated”.
        :return:
        """
        if gate_mode in self._on_list:
            self.established_connection.write(":INIT:GATE:STAT ON", error_check=True)
        elif gate_mode in self._off_list:
            self.established_connection.write(":INIT:GATE:STAT OFF", error_check=True)
        else:
            raise NameError("M8195A: Invalid gate_mode in set_gate_mode")

    def get_gate_mode(self):
        """Query the gate mode. This command must be used together with
        INIT:CONT to set the trigger mode. Check "6.8.6 :INITiate:GATE" and
        "Table 32: Trigger mode settings" for more info on the output.

        :return:
            - 0/OFF – Gate mode is off.
            - 1/ON – Gate mode is on. If continuous mode is off, the trigger mode is “gated”.
        """
        self.established_connection.query(":INIT:GATE:STAT?", error_check=True)

    def signal_generation_start(self):
        """Start signal generation on all channels.

        The channel suffix is ignored. :INIT:IMM[1|2|3|4]
        :return:
        """
        self.established_connection.write(":INIT:IMM", error_check=True)

    def set_trigger_input_threshold_level(self, level):
        """Set the trigger input threshold level.

        :param level: Threshold level voltage
        :return:
        """
        if isinstance(level, int):
            self.established_connection.write(
                f":ARM:TRIG:LEV {level}", error_check=True
            )
        elif level in self._min_max_list:
            self.established_connection.write(
                f":ARM:TRIG:LEV {level}", error_check=True
            )
        else:
            raise ValueError(
                "M8195A: level is neither integer nor proper string in set_trigger_input_threshold_level"
            )

    def get_trigger_input_threshold_level(self):
        """Query the trigger input threshold level.

        :return:
        """
        self.established_connection.query(":ARM:TRIG:LEV?", error_check=True)

    def set_trigger_input_slope(self, slope):
        """Set the trigger input slope.

        :param slope:
            - POSitive: rising edge
            - NEGative: falling edge
            - EITHer: both
        :return:
        """
        slope_list = ("POS", "POSitive", "NEG", "NEGative", "EITH", "EITHer")
        if slope in slope_list:
            self.established_connection.write(
                f":ARM:TRIG:SLOP {slope}", error_check=True
            )
        else:
            raise NameError("M8195A: Invalid slope in set_trigger_input_slope")

    def get_trigger_input_slope(self):
        """Query the trigger input slope.

        :return:
            - POSitive: rising edge
            - NEGative: falling edge
            - EITHer: both
        """
        self.established_connection.query(":ARM:TRIG:SLOP?", error_check=True)

    def set_trigger_function_source(self, source):
        """Set the source for the trigger function.

        :param source:
            - TRIGger: trigger input
            - EVENt: event input
            - INTernal: internal trigger generator
        :return:
        """
        source_list = ("TRIG", "TRIGger", "EVEN", "EVENt", "INT", "INTernal")
        if source in source_list:
            self.established_connection.write(
                f":ARM:TRIG:SOUR {source}", error_check=True
            )
        else:
            raise NameError("M8195A: Invalid source in set_trigger_function_source")

    def get_trigger_function_source(self):
        """Query the source for the trigger function.

        :return:
            - TRIGger: trigger input
            - EVENt: event input
            - INTernal: internal trigger generator
        """
        self.established_connection.query(":ARM:TRIG:SOUR?", error_check=True)

    def set_internal_trigger_freq(self, frequency):
        """Set the frequency of the internal trigger generator.

        :param frequency: internal trigger frequency
        :return:
        """
        if isinstance(frequency, int):
            self.established_connection.write(
                f":ARM:TRIG:FREQ {frequency}", error_check=True
            )
        elif frequency in self._min_max_list:
            self.established_connection.write(
                f":ARM:TRIG:FREQ {frequency}", error_check=True
            )
        else:
            raise ValueError(
                "M8195A: frequency is neither integer nor proper string in set_internal_trigger_freq"
            )

    def get_internal_trigger_freq(self):
        """Query the frequency of the internal trigger generator.

        :return:
        """
        self.established_connection.query(":ARM:TRIG:FREQ?", error_check=True)

    def set_input_trigger_event_operation_mode(self, operation_mode):
        """Set the operation mode for the trigger and event input.

        :param operation_mode:
            - ASYNchronous: asynchronous operation (see section 1.5.2)
            - SYNChronous: synchronous operation (see section 1.5.2)
        :return:
        """
        if operation_mode in ("ASYN", "ASYNchronous", "SYNC", "SYNChronous"):
            self.established_connection.write(
                f":ARM:TRIG:OPER {operation_mode}", error_check=True
            )
        else:
            raise NameError(
                "M8195A: Invalid operation_mode in set_internal_trigger_freq"
            )

    def get_input_trigger_event_operation_mode(self):
        """Query the operation mode for the trigger and event input.

        :return:
            - ASYNchronous: asynchronous operation (see section 1.5.2)
            - SYNChronous: synchronous operation (see section 1.5.2)
        """
        self.established_connection.query(":ARM:TRIG:OPER?", error_check=True)

    def set_event_in_threshold_level(self, level):
        """Set the input threshold level.

        :param level: Threshold level voltage
        :return:
        """
        if isinstance(level, int):
            self.established_connection.write(
                f":ARM:EVEN:LEV {level}", error_check=True
            )
        elif level in self._min_max_list:
            self.established_connection.write(
                f":ARM:EVEN:LEV {level}", error_check=True
            )
        else:
            raise ValueError(
                "M8195A: level is neither integer nor proper string in InputThresholdLevelSet"
            )

    def get_event_in_threshold_level(self):
        """Query the input threshold level.

        :return:
        """
        self.established_connection.query(":ARM:EVEN:LEV?", error_check=True)

    def set_event_input_slope(self, slope):
        """Set the event input slope.

        :param slope:
            - POSitive: rising edge
            - NEGative: falling edge
            - EITHer: both
        :return:
        """
        if slope in ("POS", "POSitive", "NEG", "NEGative", "EITH", "EITHer"):
            self.established_connection.write(
                f":ARM:EVEN:SLOP {slope}", error_check=True
            )
        else:
            raise NameError("M8195A: Invalid slope in set_event_input_slope")

    def get_event_input_slope(self):
        """Query the event input slope.

        :return:
            - POSitive: rising edge
            - NEGative: falling edge
            - EITHer: both
        """
        self.established_connection.query(":ARM:EVEN:SLOP?", error_check=True)

    def set_enable_event_source(self, source):
        """Set the source for the enable event.

        :param source:
            - TRIGger: trigger input
            - EVENt: event input
        :return:
        """
        if source in ("TRIG", "TRIGger", "EVEN", "EVENt"):
            self.established_connection.write(
                f":TRIG:SOUR:ENAB {source}", error_check=True
            )
        else:
            raise NameError("M8195A: Invalid source in set_enable_event_source")

    def get_enable_event_source(self):
        """Query the source for the enable event.

        :return:
            - TRIGger: trigger input
            - EVENt: event input
        """
        self.established_connection.query(":TRIG:SOUR:ENAB?", error_check=True)

    def set_hw_input_disable_state_enable_function(self, state):
        """Set the hardware input disable state for the enable function.

        When the hardware input is disabled, an enable event can only be
        generated using the
        :TRIGger[:SEQuence][:STARt]:ENABle[:IMMediate] command. When the
        hardware input is enabled, an enable event can be generated by
        command or by a signal present at the trigger or event input.
        :param state: 0|1|OFF|ON
        :return:
        """
        if state in self._on_list:
            self.established_connection.write(":TRIG:ENAB:HWD ON", error_check=True)
        elif state in self._off_list:
            self.established_connection.write(":TRIG:ENAB:HWD OFF", error_check=True)
        else:
            raise NameError(
                "M8195A: Invalid state in set_hw_input_disable_state_enable_function"
            )

    def get_hw_input_disable_state_enable_function(self):
        """Query the hardware input disable state for the enable function.

        When the hardware input is disabled, an enable event can only be
        generated using the
        :TRIGger[:SEQuence][:STARt]:ENABle[:IMMediate] command. When the
        hardware input is enabled, an enable event can be generated by
        command or by a signal present at the trigger or event input.
        :return: OFF|ON
        """
        self.established_connection.query(":TRIG:ENAB:HWD?", error_check=True)

    def set_hw_input_disable_state_trigger_function(self, state):
        """Set the hardware input disable state for the trigger function.

        When the hardware input is disabled, a trigger can only be
        generated using the
        :TRIGger[:SEQuence][:STARt]:BEGin[:IMMediate] command. When the
        hardware input is enabled, a trigger can be generated by
        command, by a signal present at the trigger input or the
        internal trigger generator.
        :param state: 0|1|OFF|ON
        :return:
        """
        if state in self._on_list:
            self.established_connection.write(":TRIG:BEG:HWD ON", error_check=True)
        elif state in self._off_list:
            self.established_connection.write(":TRIG:BEG:HWD OFF", error_check=True)
        else:
            raise NameError(
                "M8195A: Invalid state in set_hw_input_disable_state_trigger_function"
            )

    def get_hw_input_disable_state_trigger_function(self):
        """Query the hardware input disable state for the trigger function.

        When the hardware input is disabled, a trigger can only be
        generated using the
        :TRIGger[:SEQuence][:STARt]:BEGin[:IMMediate] command. When the
        hardware input is enabled, a trigger can be generated by
        command, by a signal present at the trigger input or the
        internal trigger generator.
        :return: OFF|ON
        """
        self.established_connection.query(":TRIG:BEG:HWD?", error_check=True)

    def set_hw_input_disable_state_advance_function(self, state):
        """Set the hardware input disable state for the advancement function.

        When the hardware input is disabled, an advancement event can
        only be generated using the
        :TRIGger[:SEQuence][:STARt]:ADVance[:IMMediate] command. When
        the hardware input is enabled, an advancement event can be
        generated by command or by a signal present at the trigger or
        event input.
        :param state: 0|1|OFF|ON
        :return:
        """
        if state in self._on_list:
            self.established_connection.write(":TRIG:ADV:HWD ON", error_check=True)
        elif state in self._off_list:
            self.established_connection.write(":TRIG:ADV:HWD OFF", error_check=True)
        else:
            raise NameError(
                "M8195A: Invalid state in set_hw_input_disable_state_advance_function"
            )

    def get_hw_input_disable_state_advance_function(self):
        """Query the hardware input disable state for the advancement function.

        When the hardware input is disabled, an advancement event can
        only be generated using the
        :TRIGger[:SEQuence][:STARt]:ADVance[:IMMediate] command. When
        the hardware input is enabled, an advancement event can be
        generated by command or by a signal present at the trigger or
        event input.
        :return: OFF|ON
        """
        self.established_connection.query(":TRIG:ADV:HWD?", error_check=True)

    #####################################################################
    # 6.9 Trigger - Trigger Input #######################################
    #####################################################################
    def set_advance_event_source(self, source):
        """Set the source for the advancement event.

        :param source:
            - TRIGger: trigger input
            - EVENt: event input
            - INTernal: internal trigger generator
        :return:
        """
        if source in ("TRIG", "TRIGger", "EVEN", "EVENt", "INT", "INTernal"):
            self.established_connection.write(
                f":TRIG:SOUR:ADV {source}", error_check=True
            )
        else:
            raise NameError("8195A: Invalid source in set_advance_event_source")

    def get_advance_event_source(self):
        """Query the source for the advancement event.

        :return:
            - TRIGger: trigger input
            - EVENt: event input
            - INTernal: internal trigger generator
        """
        self.established_connection.query(":TRIG:SOUR:ADV?", error_check=True)

    def trigger_enable_event(self):
        """Send the enable event to a channel.

        :return:
        """
        self.established_connection.write(":TRIG:ENAB", error_check=True)

    def trigger_begin_event(self):
        """In triggered mode send the start/begin event to a channel.

        :return:
        """
        self.established_connection.write(":TRIG:BEG", error_check=True)

    def set_trigger_gated_mode(self, stat):
        """In gated mode send a "gate open" (ON|1) or "gate close" (OFF|0) to a
        channel.

        :param stat: OFF|ON|0|1
        :return:
        """
        if stat in self._off_list:
            self.established_connection.write(":TRIG:BEG:GATE OFF", error_check=True)
        elif stat in self._on_list:
            self.established_connection.write(":TRIG:BEG:GATE ON", error_check=True)
        else:
            raise NameError("M8195A: Invalid stat in set_trigger_gated_mode")

    def get_trigger_gated_mode(self):
        """In gated mode send a "gate open" (ON|1) or "gate close" (OFF|0) to a
        channel.

        :return: OFF|ON|0|1
        """
        self.established_connection.query(":TRIG:BEG:GATE?", error_check=True)

    def trigger_advance_event(self):
        """Send the advancement event to a channel.

        :return:
        """
        self.established_connection.write(":TRIG:ADV", error_check=True)

    #####################################################################
    # 6.10 Format Subsystem #############################################
    #####################################################################
    def set_format_byte_order(self, order):
        """Byte ORDer. Controls whether binary data is transferred in normal
        (“big endian”) or swapped (“little endian”)

        byte order. Affects:
                            - [:SOURce]:STABle:DATA
                            - OUTPut:FILTer:FRATe
                            - OUTPut:FILTer:HRATe
                            - OUTPut:FILTer:QRATe
        :param order: NORMal|SWAPped
        :return:
        """
        if order in ("NORM", "NORMal", "SWAP", "SWAPped"):
            self.established_connection.write(f":FORM:BORD {order}", error_check=True)
        else:
            raise NameError("M8195A: Invalid order in set_format_byte_order")

    def get_format_byte_order(self):
        """Byte ORDer. Query whether binary data is transferred in normal (“big
        endian”) or swapped (“little endian”)

        byte order. Affects
                            - [:SOURce]:STABle:DATA
                            - OUTPut:FILTer:FRATe
                            - OUTPut:FILTer:HRATe
                            - OUTPut:FILTer:QRATe
        :return: NORMal|SWAPped
        """
        self.established_connection.query(":FORM:BORD?", error_check=True)

    #####################################################################
    # 6.11 Instrument Subsystem #########################################
    #####################################################################
    def get_instrument_slot_number(self):
        """Query the instrument’s slot number in its AXIe frame.

        :return:
        """
        self.established_connection.query(":INST:SLOT?", error_check=True)

    def instrument_access_led_start(self, seconds=False):
        """Identify the instrument by flashing the green "Access" LED on the
        front panel for a certain time.

        :param seconds: optional length of the flashing interval,
            default is 10 seconds.
        :return:
        """
        if isinstance(seconds, int):
            self.established_connection.write(f":INST:IDEN {seconds}", error_check=True)
        elif seconds is None:
            self.established_connection.write(":INST:IDEN", error_check=True)
        else:
            raise ValueError("M8195A: Invalid seconds in instrument_access_led_start")

    def instrument_access_led_stop(self):
        """Stop the flashing of the green "Access" LED before the flashing
        interval has elapsed.

        :return:
        """
        self.established_connection.write(":INST:IDEN:STOP", error_check=True)

    def hw_revision_number(self):
        """Returns the M8195A hardware revision number.

        :return:
        """
        self.established_connection.query(":INST:HWR?", error_check=True)

    def set_dac_operation_mode(self, dac_mode):
        """Use this command to set the operation mode of the DAC.

        The value of the operation mode determines, to which channels
        waveforms can be transferred and the format of the waveform
        data. In operation mode SINGle, DUAL, DCDuplicate, or FOUR the
        data consists of 1-byte waveform samples only. In operation mode
        MARKer or DCMarker the data loaded to channel 1 consists of
        interleaved 1-byte waveform and 1-byte marker samples (see
        section :TRACe Subsystem). In operation mode DCDuplicate
        waveforms can only be loaded to channels 1 and 2.
        :param dac_mode: SINGle – channel 1 can generate a signal DUAL –
            channels 1 and 4 can generate a signal, channels 2 and 3 are
            unused FOUR – channels 1, 2, 3, and 4 can generate a signal
            MARKer – channel 1 with two markers output on channel 3 and
            4 DCDuplicate – dual channel duplicate: channels 1, 2, 3,
            and 4 can generate a signal. channel 3 generates the same
            signal as channel 1. channel 4 generates the same signal as
            channel 2. DCMarker – dual channel with marker: channels 1
            and 2 can generate a signal. channel 1 has two markers
            output on channel 3 and 4. channel 2 can generate signals
            without markers.
        :return:
        """
        dac_mode_list = (
            "SINGle",
            "SING",
            "DUAL",
            "FOUR",
            "MARK",
            "MARKer",
            "DCD",
            "DCDuplicate",
            "DCM",
            "DCMarker",
        )
        if dac_mode in dac_mode_list:
            self.established_connection.write(
                f":INST:DACM {dac_mode}", error_check=True
            )
        else:
            raise NameError("M8195A: Invalid dac_mode in DACOperationMode")

    def get_dac_operation_mode(self):
        """Check set_dac_operation_mode for more information :return:

        SINGle – channel 1 can generate a signal DUAL – channels 1 and 4
        can generate a signal, channels 2 and 3 are unused FOUR –
        channels 1, 2, 3, and 4 can generate a signal MARKer – channel 1
        with two markers output on channel 3 and 4 DCDuplicate – dual
        channel duplicate: channels 1, 2, 3, and 4 can generate a
        signal. channel 3 generates             the same signal as
        channel 1. channel 4 generates the same signal as channel 2.
        DCMarker – dual channel with marker: channels 1 and 2 can
        generate a signal. channel 1 has two markers             output
        on channel 3 and 4. channel 2 can generate signals without
        markers.
        """
        self.established_connection.query(":INST:DACM?", error_check=True)

    def set_extended_mem_sample_rate_divider(self, divider):
        """Use this command to set the Sample Rate divider of the Extended
        Memory.

        This value determines also the amount of available Extended
        Memory for each channel (see section 1.5.5).
        :param divider: 1|2|4
        :return:
        """
        if divider in (1, 2, 4):
            self.established_connection.write(
                f":INST:MEM:EXT:RDIV DIV{divider}", error_check=True
            )
        else:
            raise ValueError(
                "M8195A: Invalid divider in set_extended_mem_sample_rate_divider"
            )

    def get_extended_mem_sample_rate_divider(self):
        """Use this query to get the Sample Rate divider of the Extended
        Memory.

        This value determines also the amount of available Extended
        Memory for each channel (see section 1.5.5).
        :return: DIV1|DIV2|DIV4
        """
        self.established_connection.query(":INST:MEM:EXT:RDIV?", error_check=True)

    def get_multi_module_config_mode(self):
        """This query returns the state of the multimodule configuration mode.

        :return: 0: disabled, 1: enabled
        """
        self.established_connection.query(":INST:MMOD:CONF?", error_check=True)

    def get_multi_module_mode(self):
        """This query returns the multi-module mode.

        :return:
            - NORMal: Module does not belong to a multi-module group.
            - SLAVe: Module is a slave in a multi-module group
        """
        self.established_connection.query(":INST:MMOD:MODE?", error_check=True)

    #####################################################################
    # 6.12 :Memory Subsystem ############################################
    #####################################################################
    # MMEM commands requiring <directory_name> assume the current directory if a relative path or no path is provided.
    # If an absolute path is provided, then it is ignored.
    def get_disk_usage_info(self):
        """
        Query disk usage information (drive capacity, free space available) and obtain a list of files and directories
        in a specified directory in the following format:
        <numeric_value>,<numeric_value>,{<file_entry>}
        This command returns two numeric parameters and as many strings as there are files and directories. The first
        parameter indicates the total amount of storage currently used in bytes. The second parameter indicates the
        total amount of storage available, also in bytes. The <file_entry> is a string. Each <file_entry> indicates
        the name, type, and size of one file in the directory list:
        <file_name>,<file_type>,<file_size>
        As the Windows file system has an extension that indicates file type, <file_type> is always empty. <file_size>
        provides the size of the file in bytes. In case of directories, <file_entry> is surrounded by square brackets
        and both <file_type> and <file_size> are empty.
        :return:
        """
        self.established_connection.query(":MMEM:CAT?", error_check=True)

    def set_default_dir(self, path):
        r"""Changes the default directory for a mass memory file system.

        The "path" parameter is a string. If no parameter is specified,
        the directory is set to the *RST value. At *RST, this value is
        set to the default user data storage area, that is defined as
        System.Environment.SpecialFolder.Personal: e.g.
        C:\\Users\\Name\\Documents :MMEM:CDIR
        C:\\Users\\reza-\\Documents
        :param path:
        :return:
        """
        is_directory = os.path.isdir(path)
        if is_directory is True:
            self.established_connection.write(f":MMEM:CDIR {path}", error_check=True)
        elif is_directory is False:
            raise OSError("M8195A: path is not a directory in set_default_dir")
        elif path is None:
            rst_path = r"""C:\Users\reza-\Documents"""
            self.established_connection.write(
                f":MMEM:CDIR {rst_path}", error_check=True
            )
            rst_path = r"C:\Users\reza-\Documents"
            self.established_connection.write(
                f":MMEM:CDIR {rst_path}", error_check=True
            )
        else:
            raise NameError("M8195A: Unknown error in set_default_dir")

    def get_default_dir(self):
        """MMEMory:CDIRectory?

        — Query returns full path of the default directory.
        :return:
        """
        self.established_connection.query(":MMEM:CDIR?", error_check=True)

    def set_copy_file_or_dir(self, src, dst):
        """Copies an existing file to a new file or an existing directory to a
        new directory.

        Two forms of parameters are allowed. In this form, the first
        parameter specifies the source, and the second parameter
        specifies the destination. (<Source>,<Destination>)
        :param src: File/Directory name in the source
        :param dst: File/Directory name in the destination
        :return:
        """
        src_is_file = os.path.isfile(src)
        dst_is_file = os.path.isfile(dst)
        src_is_dir = os.path.isdir(src)
        dst_is_dir = os.path.isdir(dst)
        if ((src_is_file is True) and (dst_is_file is True)) or (
            (src_is_dir is True) and (dst_is_dir is True)
        ):
            self.established_connection.write(
                f":MMEM:COPY {src}, {dst}", error_check=True
            )
        else:
            raise OSError(
                "M8195A: src/dst is(are) neither file(s) not directory(s) in set_copy_file_or_dir"
            )

    def set_copy_file_and_dir(self, src_file, src_dir, dst_file, dst_dir):
        """Copies an existing file to a new file or an existing directory to a
        new directory.

        Two forms of parameters are allowed. In this form, the first and
        third parameters specify the file names. The second and fourth
        parameters specify the directories. The first pair of parameters
        specifies the source. The second pair specifies the destination.
        An error is generated if the source doesn't exist or the
        destination file already exists. (<Source: file_name>,<Source:
        directory>,<Destination: file_name>,<Destination: directory>)
        :param src_file: File name in the source
        :param src_dir: Directory of the source
        :param dst_file: File name in the destination
        :param dst_dir: Directory of the destination
        :return:
        """
        src_is_file = os.path.isfile(src_file)
        dst_is_file = os.path.isfile(dst_file)
        src_is_dir = os.path.isdir(src_dir)
        dst_is_dir = os.path.isdir(dst_dir)
        if (src_is_dir is True) and (dst_is_dir is True):
            if (src_is_file is True) and (dst_is_file is True):
                self.established_connection.write(
                    f":MMEM:COPY {src_file}, {src_dir},{dst_file}, {dst_dir}",
                    error_check=True,
                )
            else:
                raise OSError(
                    "M8195A: src_file/dst_file is (are) not file(s) in set_copy_file_and_dir"
                )
        else:
            raise OSError(
                "M8195A: src_dir/dst_dir is (are) not directory(s) in set_copy_file_and_dir"
            )

    def set_remove_file(self, file, directory=None):
        """Removes a file from the specified directory.

        :param file: It specifies the file to be removed.
        :param directory:
        :return:
        """
        is_file = os.path.isfile(file)
        if is_file is True:
            if directory is None:
                self.established_connection.write(f":MMEM:DEL {file}", error_check=True)
            else:
                is_dir = os.path.isdir(directory)
                if is_dir is True:
                    self.established_connection.write(
                        f":MMEM:DEL {file}, {directory}", error_check=True
                    )
                else:
                    raise OSError(
                        "M8195A: directory is not a directory in set_remove_file"
                    )
        else:
            raise OSError("M8195A: file is not a file in set_remove_file")

    def load_data_in_file(self, file, data):
        """The command form is MMEMory:DATA <file>,<data>.

        It loads "data" into the file "file".
        Regarding 488.2 block format:
        https://rfmw.em.keysight.com/wireless/helpfiles/n5106a/scpi_commands_mmem.htm
            #ABC:
                #: This character indicates the beginning of the data block.
                A: Number of decimal digits present in B.
                B: Decimal number specifying the number of data bytes to follow in C.
                C: Actual binary waveform data.
        :param file: "file" is string data.
        :param data: "data" is in 488.2 block format.
        :return:
        """
        is_file = os.path.isfile(file)
        if is_file is True:
            self.established_connection.write(
                f":MMEM:DATA {file}, {data}", error_check=True
            )
        else:
            raise OSError("M8195A: file is not a file in load_data_in_file")

    def get_data_in_file(self, file):
        """The query form is MMEMory:DATA?

        <file> with the response being the associated <data> in block
        format.
        :param file:
        :return: <data> in block format
        """
        is_file = os.path.isfile(file)
        if is_file is True:
            self.established_connection.query(f":MMEM:DATA? {file}", error_check=True)
        else:
            raise OSError("M8195A: file is not a file in get_data_in_file")

    def dir_create(self, directory):
        """Creates a new directory.

        The <dir> parameter specifies the name to be created.
        :param directory:
        :return:
        """
        is_dir = os.path.isdir(directory)
        if is_dir is True:
            self.established_connection.write(
                f":MMEM:MDIR {directory}", error_check=True
            )
        else:
            raise OSError("M8195A: directory is not a Directory in dir_create")

    def set_move_file_or_dir(self, src, dst):
        """Moves an existing file to a new file or an existing directory to a
        new directory.

        Two forms of parameters are allowed. In this form, the first
        parameter specifies the source, and the second parameter
        specifies the destination. (<Source>,<Destination>)
        :param src: File/Directory name in the source
        :param dst: File/Directory name in the destination
        :return:
        """
        src_is_file = os.path.isfile(src)
        dst_is_file = os.path.isfile(dst)
        src_is_dir = os.path.isdir(src)
        dst_is_dir = os.path.isdir(dst)
        if ((src_is_file is True) and (dst_is_file is True)) or (
            (src_is_dir is True) and (dst_is_dir is True)
        ):
            self.established_connection.write(
                f":MMEM:MOVE {src}, {dst}", error_check=True
            )
        else:
            raise OSError(
                "M8195A: src/dst is(are) neither file(s) not directory(s) in set_move_file_or_dir"
            )

    def set_move_file_and_dir(self, src_file, src_dir, dst_file, dst_dir):
        """Moves an existing file to a new file or an existing directory to a
        new directory.

        Two forms of parameters are allowed. In this form, the first and
        third parameters specify the file names. The second and fourth
        parameters specify the directories. The first pair of parameters
        specifies the source. The second pair specifies the destination.
        An error is generated if the source doesn't exist or the
        destination file already exists. (<Source: file_name>,<Source:
        directory>,<Destination: file_name>,<Destination: directory>)
        :param src_file: File name in the source
        :param src_dir: Directory of the source
        :param dst_file: File name in the destination
        :param dst_dir: Directory of the destination
        :return:
        """
        src_is_file = os.path.isfile(src_file)
        dst_is_file = os.path.isfile(dst_file)
        src_is_dir = os.path.isdir(src_dir)
        dst_is_dir = os.path.isdir(dst_dir)
        if (src_is_dir is True) and (dst_is_dir is True):
            if (src_is_file is True) and (dst_is_file is True):
                self.established_connection.write(
                    f":MMEM:MOVE {src_file}, {src_dir},{dst_file}, {dst_dir}",
                    error_check=True,
                )
            else:
                raise OSError(
                    "M8195A: src_file/dst_file is (are) not file(s) in set_move_file_and_dir"
                )
        else:
            raise OSError(
                "M8195A: src_dir/dst_dir is (are) not directory(s) in set_move_file_and_dir"
            )

    def dir_remove(self, directory):
        """Removes a directory.

        The <dir> parameter specifies the directory name to be removed.
        All files and directories under the specified directory are also
        removed.
        :param directory:
        :return:
        """
        is_dir = os.path.isdir(directory)
        if is_dir is True:
            self.established_connection.write(
                f":MMEM:RDIR {directory}", error_check=True
            )
        else:
            raise OSError("M8195A: directory is not a directory in dir_remove")

    def load_instrument_state(self, file):
        """Current state of instrument is loaded from a file.

        :param file:
        :return:
        """
        is_file = os.path.isfile(file)
        if is_file is True:
            self.established_connection.write(
                f":MMEM:LOAD:CST {file}", error_check=True
            )
        else:
            raise OSError("M8195A: file is not a file in load_instrument_state")

    def set_instrument_state(self, file):
        """Current state of instrument is stored to a file.

        :param file:
        :return:
        """
        is_file = os.path.isfile(file)
        if is_file is True:
            self.established_connection.write(
                f":MMEM:STOR:CST {file}", error_check=True
            )
        else:
            raise OSError("M8195A: file is not a file in set_instrument_state")

    #####################################################################
    # 6.13 :OUTPut Subsystem ############################################
    #####################################################################
    def set_output_amplifier(self, channel, state):
        """Switch the amplifier of the output path for a channel on or off.

        Check "Figure 14: Output tab" page 48 Keysight M8195A AWG
        Revision 2
        :param channel: 1|2|3|4
        :param state: OFF|ON|0|1
        :return:
        """
        if channel is None:
            if state in self._on_list:
                self.established_connection.write(":OUTP ON", error_check=True)
            elif state in self._off_list:
                self.established_connection.write(":OUTP OFF", error_check=True)
            else:
                raise NameError("M8195A: Invalid state in set_output_amplifier")
        elif channel in self._channel_list:
            if state in self._on_list:
                self.established_connection.write(
                    f":OUTP{channel} ON", error_check=True
                )
            elif state in self._off_list:
                self.established_connection.write(
                    f":OUTP{channel} OFF", error_check=True
                )
            else:
                raise NameError("M8195A: Invalid state in set_output_amplifier")
        else:
            raise ValueError("M8195A: Invalid channel in set_output_amplifier")

    def get_output_amplifier(self, channel):
        """Query the amplifier of the output path for a channel on or off.

        Check "Figure 14: Output tab" page 48 Keysight M8195A AWG
        Revision 2
        :param channel: 1|2|3|4
        :return:
        """
        if channel in self._channel_list:
            self.established_connection.query(f":OUTP{channel}?", error_check=True)
        else:
            raise ValueError("M8195A: Invalid channel in set_output_amplifier")

    def set_output_clock_source(self, source):
        """Select which signal source is routed to the reference clock output.
        Check "Figure 13: Clock tab" page 46.

        Keysight M8195A AWG Revision 2
            - INTernal: the module internal reference oscillator (100 MHz)
            - EXTernal: the external reference clock from REF CLK IN with two variable dividers (divider n and m)
            - SCLK1: DAC sample clock with variable divider and variable delay
            - SCLK2: DAC sample clock with fixed divider (32 and 8)
        :return:
        """
        if source in ("INTernal", "INT", "EXTernal", "EXT", "SCLK1", "SCLK2"):
            self.established_connection.write(
                f":OUTP:ROSC:SOUR {source}", error_check=True
            )
        else:
            raise NameError("M8195A: Invalid source in set_output_clock_source")

    def get_output_clock_source(self):
        """Query which signal source is routed to the reference clock output.
        Check "Figure 13: Clock tab" page 46 Keysight M8195A AWG Revision 2.

        :return:
            - INTernal: the module internal reference oscillator (100 MHz)
            - EXTernal: the external reference clock from REF CLK IN with two variable dividers (divider n and m)
            - SCLK1: DAC sample clock with variable divider and variable delay
            - SCLK2: DAC sample clock with fixed divider (32 and 8)
        """
        self.established_connection.query(":OUTP:ROSC:SOUR?", error_check=True)

    def set_dac_sample_freq_divider(self, divider):
        """Set the divider of the DAC sample clock signal routed to the
        reference clock output.

        Check page 46 "Figure 13: Clock tab" Keysight M8195A AWG
        Revision 2
        :param divider:
        :return:
        """
        if isinstance(divider, int):
            self.established_connection.write(
                f":OUTP:ROSC:SCD {divider}", error_check=True
            )
        elif divider in self._min_max_list:
            self.established_connection.write(
                f":OUTP:ROSC:SCD {divider}", error_check=True
            )
        else:
            raise ValueError(
                "M8195A: divider is neither integer not string in set_dac_sample_freq_divider"
            )

    def get_dac_sample_freq_divider(self):
        """Query the divider of the DAC sample clock signal routed to the
        reference clock output.

        Check page 46 "Figure 13: Clock tab" Keysight M8195A AWG
        Revision 2 :param
        :return: divider
        """
        self.established_connection.query(":OUTP:ROSC:SCD?", error_check=True)

    def set_ref_clock_freq_divider1(self, divider1):
        """Set the first divider of the reference clock signal routed to the
        reference clock output.

        Check page 46 "Figure 13: Clock tab" Keysight M8195A AWG
        Revision 2
        :param divider1:
        :return:
        """
        if isinstance(divider1, int):
            self.established_connection.write(
                f":OUTP:ROSC:RCD1 {divider1}", error_check=True
            )
        elif divider1 in self._min_max_list:
            self.established_connection.write(
                f":OUTP:ROSC:RCD1 {divider1}", error_check=True
            )
        else:
            raise ValueError(
                "M8195A: divider1 is neither integer not string in set_ref_clock_freq_divider1"
            )

    def get_ref_clock_freq_divider1(self):
        """Query the first divider of the reference clock signal routed to the
        reference clock output.

        Check page 46 "Figure 13: Clock tab" Keysight M8195A AWG
        Revision 2
        :return:
        """
        self.established_connection.query(":OUTP:ROSC:RCD1?", error_check=True)

    def set_ref_clock_freq_divider2(self, divider2):
        """Set the first divider of the reference clock signal routed to the
        reference clock output.

        Check page 46 "Figure 13: Clock tab" Keysight M8195A AWG
        Revision 2
        :param divider2:
        :return:
        """
        if isinstance(divider2, int):
            self.established_connection.write(
                f":OUTP:ROSC:RCD2 {divider2}", error_check=True
            )
        elif divider2 in self._min_max_list:
            self.established_connection.write(
                f":OUTP:ROSC:RCD2 {divider2}", error_check=True
            )
        else:
            raise ValueError(
                "M8195A: divider2 is neither integer not string in set_ref_clock_freq_divider2"
            )

    def get_ref_clock_freq_divider2(self):
        """Set the first divider of the reference clock signal routed to the
        reference clock output.

        Check page 46 "Figure 13: Clock tab" Keysight M8195A AWG
        Revision 2
        :return:
        """
        self.established_connection.query(":OUTP:ROSC:RCD2?", error_check=True)

    def differential_offset(self, channel, value):
        """
        Differential Offset: The hardware can compensate for little offset differences between the normal and
        complement output. “<value>” is the offset to the calibrated optimum DAC value, so the minimum and maximum
        depend on the result of the calibration. Check below pages in Keysight M8195A AWG Revision 2:
            - page 49, "Figure 14: Output tab"
            - page 224, "Table 33: Differential offset"
        :param channel: 1|2|3|4
        :param value: <value>|MINimum|MAXimum
        :return:
        """
        if channel is None:
            if isinstance(value, int):
                self.established_connection.write(
                    f":OUTP:DIOF {value}", error_check=True
                )
            elif value in self._min_max_list:
                self.established_connection.write(
                    f":OUTP:DIOF {value}", error_check=True
                )
            else:
                raise ValueError(
                    "M8195A: value is neither integer nor string in differential_offset"
                )
        elif channel in self._channel_list:
            if isinstance(value, int):
                self.established_connection.write(
                    f":OUTP{channel}:DIOF {value}", error_check=True
                )
            elif value in self._min_max_list:
                self.established_connection.write(
                    f":OUTP{channel}:DIOF {value}", error_check=True
                )
            else:
                raise ValueError(
                    "M8195A: value is neither integer nor string in differential_offset"
                )
        else:
            raise ValueError("M8195A: Invalid channel in differential_offset")

    def get_differential_offset(self, channel):
        """
        Query the Differential Offset: The hardware can compensate for little offset differences between the normal and
        complement output. “<value>” is the offset to the calibrated optimum DAC value, so the minimum and maximum
        depend on the result of the calibration. Check below pages in Keysight M8195A AWG Revision 2:
            - page 49, "Figure 14: Output tab"
            - page 224, "Table 33: Differential offset"
        :param channel: 1|2|3|4
        :return: <value>|MINimum|MAXimum
        """
        if channel in self._channel_list:
            self.established_connection.query(f":OUTP{channel}:DIOF?", error_check=True)
        else:
            raise ValueError("M8195A: Invalid channel in get_differential_offset")

    def sample_rate_divider(self, value):
        """The speed of operation of the extended memory is adjustable using
        the parameter ‘Sample Rate Divider (Extended Memory)’ which can be
        changed by the user.

        Possible values are 1, 2, and 4. The Sample Rate Divider is
        identical for all channels that are sourced from extended
        memory. In case the Sample Rate Divider is adjusted to two or
        four, the FIR filters are used as interpolation filters by
        factors of two or four. The interpolation is necessary as the
        DAC always operates in the range 53.76 GSa/s … 65 GSa/s. Check
        Table in page 26, and "Figure 4" in page 27
        :param value: 1|2|4
        :return:
        """
        if value == 1:
            code = "FRAT"
        elif value == 2:
            code = "HRAT"
        elif value == 4:
            code = "QRAT"
        else:
            raise ValueError("M8195A: Invalid value in sample_rate_divider")
        return code

    def set_fir_coefficient(self, channel, divider, value):
        """Set the FIR filter coefficients for a channel to be used when the
        Sample Rate Divider for the Extended Memory.

        is 1|2|4 ('FRAT', 'HRAT', 'QRAT'). The number of coefficients is 16|32|64 and the values are doubles between
        -2 and 2. The coefficients can only be set using this command, when the predefined FIR filter type is set to
        USER.
        The number of filter coefficients depends on the Sample Rate Divider; 16, 32, or 64 filter coefficients are
        available if the Sample Rate Divider is set to 1, 2 or, 4 respectively. In case the Sample Rate Divider is
        changed, the FIR filter coefficients of each channel sourced from extended memory are loaded to operate as a
        by one or by two or by four interpolation filter.
        :param channel: 1|2|3|4
        :param divider: 1|2|4
        :param value: They can be given as a list of comma-separated values or as IEEE binary block data of doubles.
            1 -> <value0>, <value1>…<value15> |<block>
            2 -> <value0>, <value1>…<value31>|<block>
            4 -> <value0>, <value1>…<value63>|<block>
        :return:
        """
        code = self.sample_rate_divider(divider)
        if channel in self._channel_list:
            self.established_connection.write(
                f":OUTP{channel}:FILT:{code}:{value}", error_check=True
            )
        else:
            raise ValueError("M8195A: Invalid channel in set_fir_coefficient")

    def get_fir_coefficient(self, channel, divider):
        """Get the FIR filter coefficients for a channel to be used when the
        Sample Rate Divider for the Extended Memory is 1|2|4 ('FRAT', 'HRAT',
        'QRAT').

        :param channel: 1|2|3|4
        :param divider: 1|2|4
        :return: FIR filter coefficients for a channel. The number of
            coefficients is: 16|32|64
        """
        code = self.sample_rate_divider(divider)
        if channel in self._channel_list:
            self.established_connection.query(
                f":OUTP{channel}:FILT:{code}?", error_check=True
            )
        else:
            raise ValueError("M8195A: Invalid channel in get_fir_coefficient")

    def set_fir_type(self, channel, divider, types):
        """Set the predefined FIR filter type for a channel to be used when the
        Sample Rate Divider for the Extended Memory is 1|2|4 ('FRAT', 'HRAT',
        'QRAT'). The command form modifies the FIR filter coefficients
        according to the set filter type, except for type USER. :param channel:
        1|2|3|4 :param divider: 1|2|4 :param types:

            if divider is 1:
                - LOWPass: equiripple lowpass filter with a passband edge at 75% of Nyquist
                - ZOH: Zero-order hold filter
                - USER: User-defined filter
            if divider is 2|4:
                - NYQuist: Nyquist filter (half-band|quarter-band filter) with rolloff factor 0.2
                - LINear: Linear interpolation filter
                - ZOH: Zero-order hold filter
                - USER: User-defined filter
        :return:
        """
        code = self.sample_rate_divider(divider)
        if channel in self._channel_list:
            if divider == 1:
                if types in ("LOWPass", "LOWP", "ZOH", "USER"):
                    self.established_connection.write(
                        f":OUTP:FILT:{code}:TYPE {types}", error_check=True
                    )
                else:
                    raise NameError(
                        "M8195A: Invalid types for divider=1 in set_fir_type"
                    )
            elif divider == (2 or 4):
                if types in ("NYQuist", "NYQ", "LINear", "ZOH", "USER"):
                    self.established_connection.write(
                        f":OUTP:FILT:{code}:TYPE {types}", error_check=True
                    )
                else:
                    raise NameError(
                        "M8195A: Invalid types for divider=1|2 in set_fir_type"
                    )
            else:
                raise ValueError("M8195A: Invalid divider in set_fir_type")
        else:
            raise ValueError("M8195A: Invalid channel in set_fir_type")

    def get_fir_type(self, channel, divider):
        """Get the predefined FIR filter type for a channel to be used when the
        Sample Rate Divider for the Extended Memory is 1|2|4 ('FRAT', 'HRAT',
        'QRAT'). :param channel: 1|2|3|4 :param divider: 1|2|4 :return: Type:

        if divider is 1:
            - LOWPass: equiripple lowpass filter with a passband edge at 75% of Nyquist
            - ZOH: Zero-order hold filter
            - USER: User-defined filter
        if divider is 2|4:
            - NYQuist: Nyquist filter (half-band|quarter-band filter) with rolloff factor 0.2
            - LINear: Linear interpolation filter
            - ZOH: Zero-order hold filter
            - USER: User-defined filter
        """
        code = self.sample_rate_divider(divider)
        if channel in self._channel_list:
            self.established_connection.query(
                f":OUTP{channel}:FILT:{code}:TYPE?", error_check=True
            )
        else:
            raise ValueError("M8195A: Invalid channel in get_fir_type")

    def set_fir_scaling_factor(self, channel, divider, scale):
        """Set the FIR filter scaling factor for a channel to be used when the
        Sample Rate Divider for the Extended Memory is 1|2|4.

        The range is between 0 and 1.
        :param channel: 1|2|3|4
        :param divider: 1|2|4 ('FRAT', 'HRAT', 'QRAT')
        :param scale: <scale>|MINimum|MAXimum
        :return:
        """
        code = self.sample_rate_divider(divider)
        if channel in self._channel_list:
            if 0 <= scale <= 1:
                self.established_connection.write(
                    f":OUTP{channel}:FILT:{code}:SCAL {scale}", error_check=True
                )
            elif scale in self._min_max_list:
                self.established_connection.write(
                    f":OUTP{channel}:FILT:{code}:SCAL {scale}", error_check=True
                )
            else:
                raise ValueError(
                    "M8195A: scale is neither proper integer nor proper string in set_fir_scaling_factor"
                )
        else:
            raise ValueError("M8195A: Invalid channel in set_fir_scaling_factor")

    def get_fir_scaling_factor(self, channel, divider):
        """Get the FIR filter scaling factor for a channel to be used when the
        Sample Rate Divider for the Extended Memory is 1|2|4.

        :param channel: 1|2|3|4
        :param divider: 1|2|4 ('FRAT', 'HRAT', 'QRAT')
        :return: The scale; the range is between 0 and 1 or
            MINimum|MAXimum.
        """
        code = self.sample_rate_divider(divider)
        if channel in self._channel_list:
            self.established_connection.query(
                f":OUTP{channel}:FILT:{code}:SCAL?", error_check=True
            )
        else:
            raise ValueError("M8195A: Invalid channel in get_fir_scaling_factor")

    def set_fir_delay(self, channel, divider, delay):
        """Set the FIR filter delay for a channel to be used when the Sample
        Rate Divider for the Extended Memory is 1|2|4.

        The range is:
            1 ('FRAT') -> -50ps..+50ps
            2 ('HRAT')-> -100ps..+100ps
            4 ('QRAT')-> -200ps..+200ps.
        The delay value has only effect for filter type:
            1 ('FRAT') -> LOWPass
            2, 4 ('HRAT', 'QRAT')-> NYQuist and LINear.
        The command form modifies the FIR filter coefficients according to the set delay value.
        Parameter Suffix -> [s|ms|us|ns|ps]
        :param channel: 1|2|3|4
        :param divider: 1|2|4 ('FRAT', 'HRAT', 'QRAT')
        :param delay: <delay>|MINimum|MAXimum
        :return:
        """
        code = self.sample_rate_divider(divider)
        if channel not in self._channel_list:
            raise ValueError("M8195A: Invalid channel in set_fir_delay")

        # if divider == 1 and abs(delay) > 50:
        #     raise Exception("M8195A: Invalid delay for divider=1 in set_fir_delay")
        # if divider == 2 and abs(delay) > 100:
        #     raise Exception("M8195A: Invalid delay for divider=2 in set_fir_delay")
        # if divider == 4 and abs(delay) > 200:
        #     raise Exception("M8195A: Invalid delay for divider=4 in set_fir_delay")
        if divider in (1, 2, 4) and abs(delay) > 50 * divider:
            raise ValueError("M8195A: Invalid delay for divider in set_fir_delay")

        self.established_connection.write(
            f"OUTP{channel}:FILT:{code}:DEL {delay}ps", error_check=True
        )

    def get_fir_delay(self, channel, divider):
        """Set the FIR filter delay for a channel to be used when the Sample
        Rate Divider for the Extended Memory is 1|2|4.

        :param channel: 1|2|3|4
        :param divider: 1|2|4 ('FRAT', 'HRAT', 'QRAT')
        :return: The delay range is:
            1 ('FRAT') -> -50ps..+50ps
            2 ('HRAT')-> -100ps..+100ps
            4 ('QRAT')-> -200ps..+200ps.
        The delay value has only effect for filter type:
            1 ('FRAT') -> LOWPass
            2, 4 ('HRAT', 'QRAT')-> NYQuist and LINear.
        """
        code = self.sample_rate_divider(divider)
        if channel in self._channel_list:
            self.established_connection.query(
                f"OUTP{channel}:FILT:{code}:DEL?", error_check=True
            )
        else:
            raise ValueError("M8195A: Invalid channel in get_fir_delay")

    #####################################################################
    # 6.14 Sampling Frequency Commands ##################################
    #####################################################################
    def set_dac_sample_freq(self, frequency):
        """Set the sample frequency of the output DAC.

        :param frequency:
        :return:
        """
        if isinstance(frequency, int):
            self.established_connection.write(
                f":FREQ:RAST {frequency}", error_check=True
            )
        elif frequency in self._min_max_list:
            self.established_connection.write(
                f":FREQ:RAST {frequency}", error_check=True
            )
        else:
            raise ValueError(
                "M8195A: frequency is neither integer nor proper string in DACSampleFreq"
            )

    def get_dac_sample_freq(self):
        """Query the sample frequency of the output DAC.

        :return:
        """
        self.established_connection.query(":FREQ:RAST?", error_check=True)

    #####################################################################
    # 6.15 Reference Oscillator Commands ################################
    #####################################################################
    def set_ref_clock_source(self, source):
        """Set the reference clock source. Command not supported with Revision
        1 hardware. Check "Figure 13: Clock tab" page 46, Keysight M8195A AWG
        Revision 2.

        :param source:
                    - EXTernal: reference is taken from REF CLK IN.
                    - AXI: reference is taken from AXI backplane.
                    - INTernal: reference is taken from module internal reference oscillator. May not be available with
                     every hardware.
        :return:
        """
        if source in ("EXTernal", "EXT", "AXI", "INTernal", "INT"):
            self.established_connection.write(f":ROSC:SOUR {source}", error_check=True)
        else:
            raise NameError("M8195A: Invalid source in set_ref_clock_source")

    def get_ref_clock_source(self):
        """Query the reference clock source. Check "Figure 13: Clock tab" page
        46, Keysight M8195A AWG Revision 2.

        :return:
            - EXTernal: reference is taken from REF CLK IN.
            - AXI: reference is taken from AXI backplane.
            - INTernal: reference is taken from module internal reference oscillator. May not be available with every
                        hardware.
        """
        self.established_connection.query(":ROSC:SOUR?", error_check=True)

    def ref_clock_source_availability(self, source):
        """Check if a reference clock source is available.

        Returns 1 if it is available and 0 if not.
        :param source: EXTernal|AXI|INTernal
        :return: 1 if reference clock source is available and 0 if not.
        """
        if source in ("EXTernal", "EXT", "AXI", "INTernal", "INT"):
            self.established_connection.query(
                f":ROSC:SOUR:CHEC? {source}", error_check=True
            )
        else:
            raise NameError("M8195A: Invalid source in ref_clock_source_availability")

    def set_external_clock_source_freq(self, freq):
        """Set the expected reference clock frequency, if the external
        reference clock source is selected.

        :param freq: <frequency>|MINimum|MAXimum
        :return:
        """
        if self.get_ref_clock_source() == ("EXT" or "EXTernal"):
            if isinstance(freq, int):
                self.established_connection.write(
                    f":ROSC:FREQ {freq}", error_check=True
                )
            elif freq in self._min_max_list:
                self.established_connection.write(
                    f":ROSC:FREQ {freq}", error_check=True
                )
            else:
                raise ValueError(
                    "M8195A: frequency is neither integer nor proper string in set_external_clock_source_freq"
                )
        else:
            raise ValueError(
                'M8195A: Reference clock source is not "EXTernal" in set_external_clock_source_freq'
            )

    def get_external_clock_source_freq(self):
        """Query the expected reference clock frequency, if the external
        reference clock source is selected.

        :return: Frequency (<frequency>|MINimum|MAXimum)
        """
        if self.get_ref_clock_source() == ("EXT" or "EXTernal"):
            self.established_connection.query(":ROSC:FREQ?", error_check=True)
        else:
            raise ValueError(
                'M8195A: Reference clock source is not "EXTernal" in get_external_clock_source_freq'
            )

    def set_external_clock_source_range(self, ranges):
        """Set the reference clock frequency range, if the external reference
        clock source is selected.

        :param ranges:
                    - RANG1: 10…300 MHz
                    - RANG2: 210MHz…17GHz
        :return:
        """
        if self.get_ref_clock_source() == ("EXT" or "EXTernal"):
            if ranges in ("RANG1", "RANG2"):
                self.established_connection.write(
                    f":ROSC:RANG {ranges}", error_check=True
                )
            else:
                raise ValueError(
                    "M8195A: Invalid ranges in set_external_clock_source_range"
                )
        else:
            raise ValueError(
                'M8195A: Reference clock source is not "EXTernal" in set_external_clock_source_range'
            )

    def get_external_clock_source_range(self):
        """Query the reference clock frequency range, if the external reference
        clock source is selected.

        :return: Range:
                    - RANG1: 10…300 MHz
                    - RANG2: 210MHz…17GHz
        """
        if self.get_ref_clock_source() == ("EXT" or "EXTernal"):
            self.established_connection.query(":ROSC:RANG?", error_check=True)
        else:
            raise ValueError(
                'M8195A: Reference clock source is not "EXTernal" in get_external_clock_source_range'
            )

    def set_external_clock_source_range_freq(self, ranges, freq):
        """Set the reference clock frequency for a specific reference clock
        range.

        Current range remains unchanged.
        :param ranges: RNG1|RNG2
        :param freq: <frequency>|MINimum|MAXimum • RNG1: 10…300 MHz •
            RNG2: 210MHz…17GHz
        :return:
        """
        if self.get_ref_clock_source() == ("EXT" or "EXTernal"):
            if ranges in ("RNG1", "RNG2"):
                if isinstance(freq, int):
                    self.established_connection.write(
                        f":ROSC:{ranges}:FREQ {freq}", error_check=True
                    )
                elif freq in self._min_max_list:
                    self.established_connection.write(
                        f":ROSC:{ranges}:FREQ {freq}", error_check=True
                    )
                else:
                    raise ValueError(
                        "M8195A: frequency is neither integer nor proper string in "
                        "set_external_clock_source_range_freq"
                    )
            else:
                raise ValueError(
                    "M8195A: Invalid ranges in set_external_clock_source_range_freq"
                )
        else:
            raise ValueError(
                'M8195A: Reference clock source is not "EXTernal" in set_external_clock_source_range_freq'
            )

    def get_external_clock_source_range_freq(self, ranges):
        """Query the reference clock frequency for a specific reference clock
        range.

        Current range remains unchanged.
        :param ranges: RNG1|RNG2
        :return: freq: <frequency>|MINimum|MAXimum • RNG1: 10…300 MHz •
            RNG2: 210MHz…17GHz
        """
        if self.get_ref_clock_source() == ("EXT" or "EXTernal"):
            if ranges in ("RNG1", "RNG2"):
                self.established_connection.query(
                    f":ROSC:{ranges}:FREQ?", error_check=True
                )
            else:
                raise ValueError(
                    "M8195A: Invalid ranges in ExternalClockSourceRangeFreqSet"
                )
        else:
            raise ValueError(
                'M8195A: Reference clock source is not "EXTernal" in get_external_clock_source_range_freq'
            )

    #####################################################################
    # 6.16 :VOLTage Subsystem ###########################################
    #####################################################################

    def set_output_amplitude(self, channel, level):
        """Set the output amplitude.

        :param channel: 1|2|3|4
        :param level: <level>|MINimum|MAXimum
        :return:
        """
        if channel in self._channel_list:
            if isinstance(level, int):
                self.established_connection.write(
                    f":VOLT{channel} {level}", error_check=True
                )
            elif level in self._min_max_list:
                self.established_connection.write(
                    f":VOLT{channel} {level}", error_check=True
                )
            else:
                raise ValueError(
                    "M8195A: level is neither integer nor proper string in set_output_amplitude"
                )
        else:
            raise ValueError("M8195A: Invalid channel in set_output_amplitude")

    def get_output_amplitude(self, channel):
        """Query the output amplitude.

        :param channel: 1|2|3|4
        :return: level: <level>|MINimum|MAXimum
        """
        if channel in self._channel_list:
            self.established_connection.query(f":VOLT{channel}?", error_check=True)
        else:
            raise ValueError("M8195A: Invalid channel in set_output_amplitude")

    def output_offset(self, channel, offset):
        """Set the output offset.

        :param channel: 1|2|3|4
        :param offset:
        :return:
        """
        if channel in self._channel_list:
            if isinstance(offset, int):
                self.established_connection.write(
                    f":VOLT{channel}:OFFS {offset}", error_check=True
                )
            elif offset in self._min_max_list:
                self.established_connection.write(
                    f":VOLT{channel}:OFFS {offset}", error_check=True
                )
            else:
                raise ValueError(
                    "M8195A: offset is neither integer nor proper string in output_offset"
                )
        else:
            raise ValueError("M8195A: Invalid channel in output_offset")

    def get_output_offset(self, channel):
        """Query the output offset.

        :param channel: 1|2|3|4
        :return: offset
        """
        if channel in self._channel_list:
            self.established_connection.query(f":VOLT{channel}:OFFS?", error_check=True)
        else:
            raise ValueError("M8195A: Invalid channel in get_output_offset")

    def set_output_high_level(self, channel, high_level):
        """Set the output high level.

        :param channel: 1|2|3|4
        :param high_level:
        :return:
        """
        if channel in self._channel_list:
            if isinstance(high_level, int):
                self.established_connection.write(
                    f":VOLT{channel}:HIGH {high_level}", error_check=True
                )
            elif high_level in self._min_max_list:
                self.established_connection.write(
                    f":VOLT{channel}:HIGH {high_level}", error_check=True
                )
            else:
                raise ValueError(
                    "M8195A: high_level is neither integer nor proper string in set_output_high_level"
                )
        else:
            raise ValueError("M8195A: Invalid channel in set_output_high_level")

    def get_output_high_level(self, channel):
        """Query the output high level.

        :param channel: 1|2|3|4
        :return: high_level
        """
        if channel in self._channel_list:
            self.established_connection.query(f":VOLT{channel}:HIGH?", error_check=True)
        else:
            raise ValueError("M8195A: Invalid channel in get_output_high_level")

    def set_output_low_level(self, channel, low_level):
        """Set the output low level.

        :param channel: 1|2|3|4
        :param low_level:
        :return:
        """
        if channel in self._channel_list:
            if isinstance(low_level, int):
                self.established_connection.write(
                    f":VOLT{channel}:LOW {low_level}", error_check=True
                )
            elif low_level in self._min_max_list:
                self.established_connection.write(
                    f":VOLT{channel}:LOW {low_level}", error_check=True
                )
            else:
                raise ValueError(
                    "M8195A: low_level is neither integer nor proper string in set_output_low_level"
                )
        else:
            raise ValueError("M8195A: Invalid channel in set_output_low_level")

    def get_output_low_level(self, channel):
        """Query the output low level.

        :param channel: 1|2|3|4
        :return: low_level
        """
        if channel in self._channel_list:
            self.established_connection.query(f":VOLT{channel}:LOW?", error_check=True)
        else:
            raise ValueError("M8195A: Invalid channel in get_output_low_level")

    def set_termination_voltage(self, channel, level):
        """Set the termination voltage level.

        :param channel: 1|2|3|4
        :param level:
        :return:
        """
        if channel in self._channel_list:
            if isinstance(level, int):
                self.established_connection.write(
                    f":VOLT{channel}:TERM {level}", error_check=True
                )
            elif level in self._min_max_list:
                self.established_connection.write(
                    f":VOLT{channel}:TERM {level}", error_check=True
                )
            else:
                raise ValueError(
                    "M8195A: level is neither integer nor proper string in set_termination_voltage"
                )
        else:
            raise ValueError("M8195A: Invalid channel in set_termination_voltage")

    def get_termination_voltage(self, channel):
        """Set the termination voltage level.

        :param channel: 1|2|3|4
        :return: level
        """
        if channel in self._channel_list:
            self.established_connection.query(f":VOLT{channel}:TERM?", error_check=True)
        else:
            raise ValueError("M8195A: Invalid channel in get_termination_voltage")

    #####################################################################
    # 6.17 Source:Function:MODE #########################################
    #####################################################################
    def set_waveform_type(self, types):
        """Use this command to set the type of waveform that will be generated
        on the channels that use the extended memory.

        The channels that use internal memory are always in ARBitrary
        mode.
        :param types: [ARB, ARBitrary]: arbitrary waveform segment [STS,
            STSequence]: sequence [STSC, STSCenario]: scenario
        :return:
        """
        if types in ("ARB", "ARBitrary", "STS", "STSequence", "STSC", "STSCenario"):
            self.established_connection.write(f":FUNC:MODE {types}", error_check=True)
        else:
            raise ValueError("M8195A: Invalid types in set_waveform_type")

    def get_waveform_type(self):
        """Use this command to query the type of waveform that will be
        generated on the channels that use the extended memory.

        The channels that use internal memory are always in ARBitrary
        mode.
        :return: type of waveform: [ARB, ARBitrary]: arbitrary waveform
            segment [STS, STSequence]: sequence [STSC, STSCenario]:
            scenario
        """
        self.established_connection.query(":FUNC:MODE?", error_check=True)

    #####################################################################
    # 6.18 :STABle Subsystem ############################################
    #####################################################################
    # Use the Sequence Table subsystem to prepare the instrument for sequence and scenario generation. The Sequencing
    # capabilities can only be used by the channels sourced from Extended Memory. These channels share a common
    # Sequence Table and execute the same sequence or scenario. The channels sourced from Internal Memory play only
    # one waveform. Follow these steps for all function modes:
    # - First create waveform data segments in the module memory like described in the “Arbitrary Waveform Generation”
    # paragraph of the “TRACe subsystem”.
    # - Create sequence table entries that refer to the waveform segments using the STAB:DATA command.
    def reset_all_sequence_table(self):
        """Reset all sequence table entries to default values.

        :return:
        """
        self.established_connection.write(":STAB:RES", error_check=True)

    def set_sequence_data(
        self,
        index,
        segm_id,
        segm_adv_mode="SING",
        seq_adv_mode="SING",
        marker_enab=False,
        marker_seq_init=False,
        marker_scen_end=False,
        marker_seq_end=False,
        seq_loop=1,
        segm_loop=1,
        segm_offset_start=0,
        segm_offset_end="#hFFFFFFFF",
    ):  # pylint: disable=too-many-arguments too-many-branches
        """The command form writes directly into the sequencer memory. Writing
        is possible, when signal generation is stopped or when signal
        generation is started in dynamic mode. The sequencer memory has
        16,777,215 (16M – 1) entries. With this command entries can be directly
        manipulated using 6 32-bit words per entry. Individual entries or
        multiple entries at once can be manipulated. The data can be given in
        IEEE binary block format or in comma-separated list of 32-bit values.

        6 32-bit words: Control, Sequence Loop Count, Segment Loop Count, Segment ID, Segment Start Offset,
                        Segment End Offset

        :param index: index of the sequence table entry to be accessed (16,777,215 (16M – 1) = 2**24-1 entries)
        :param segm_id:
                        7 bit (31:25) -> Reserved
                        25 bit (24:0) -> Segment id (1 .. 16M)
        Control:
            Reserved: 16 bit (15:0)
            :param segm_adv_mode: 4 bit (19:16),
                                0 or (0000): Auto
                                1 or (0001): Conditional
                                2 or (0010): Repeat
                                3 or (0011): Single
                                4 – 15 or (0100 - 1111): Reserved
            :param seq_adv_mode: 4 bit (23:20),
                                0 or (0000): Auto
                                1 or (0001): Conditional
                                2 or (0010): Repeat
                                3 or (0011): Single
                                4 – 15 or (0100 - 1111): Reserved
            :param marker_enab: 1 bit (24)
            :param marker_seq_init: 1 bit (28)
            :param marker_scen_end: 1 bit (29)
            :param marker_seq_end: 1 bit (30)

        :param seq_loop: 32 bit, Number of sequence iterations (1..4G-1), only applicable in the first entry of a
        sequence (31:0)
        :param segm_loop: 32 bit, Number of segment iterations (1..4G-1) (31:0)
        :param segm_offset_start: 32 bit, Allows specifying a segment start address in samples, if only part of a segment
        loaded into waveform data memory is to be used. The value must be a multiple of twice the granularity of the
        selected waveform output mode.
        :param segm_offset_end: 32 bit, Allows specifying a segment end address in samples, if only part of a segment
        loaded into waveform data memory is to be used. The value must obey the granularity of the selected waveform
        output mode. You can use the value ffffffff, if the segment end address equals the last sample in the segment.
            ffffffff = 4G-1 = 2**32-1 = 4,294,967,295
        :return:
        """
        control = 0

        if segm_adv_mode == "AUTO":
            control = control  # pylint: disable=self-assigning-variable
        elif segm_adv_mode == "COND":
            # 0001 = 65,536
            control += 1 * (2**16)
        elif segm_adv_mode == "REP":
            # 0010 = 2 * (0001) = 131,072
            control += 2 * (2**16)
        elif segm_adv_mode == "SING":
            # 0011 = 3 * (0001) = 196,608
            control += 3 * (2**16)
        else:
            raise ValueError("M8195A: Invalid segment advancement mode")

        if seq_adv_mode == "AUTO":
            control = control  # pylint: disable=self-assigning-variable
        elif seq_adv_mode == "COND":
            # 0001 = 1,048,576
            control += 1 * (2**20)
        elif seq_adv_mode == "REP":
            # 0010 = 2 * (0001) = 2,097,152
            control += 2 * (2**20)
        elif seq_adv_mode == "SING":
            # 0011 = 3 * (0001) = 3,145,728
            control += 3 * (2**20)
        else:
            raise ValueError("M8195A: Invalid sequence advancement mode")

        if marker_enab is True:
            control += 2**24

        if marker_seq_init is True:
            control += 2**28

        if marker_scen_end is True:
            control += 2**29

        if marker_seq_end is True:
            control += 2**30

        self.established_connection.write(
            f":STAB:DATA {index}, {control}, {seq_loop}, {segm_loop}, {segm_id}, {segm_offset_start}, {segm_offset_end}",
            error_check=True,
        )

    def set_sequence_idle(self, index, seq_loop, idle_sample=1, idle_delay=0):
        """The command form writes directly into the sequencer memory. Writing
        is possible, when signal generation is stopped or when signal
        generation is started in dynamic mode. The sequencer memory has
        16,777,215 (16M – 1) entries. With this command entries can be directly
        manipulated using 6 32-bit words per entry. Individual entries or
        multiple entries at once can be manipulated. The data can be given in
        IEEE binary block format or in comma-separated list of 32-bit values.

        {Software front panel, Sequence Table, page 120}
        Idle: Idle entry allows setting a pause between segments in a granularity that is smaller than the sync clock
        granularity. You can specify the sample to be played during the pause. A minimum length of this pause is
        required. The idle command segment is treated as a segment within sequences or scenarios. There is no segment
        loop count but a sequence loop counter value is required for cases where the idle command segment is the first
        segment of a sequence.
        Idle Delay: The field is enabled only when the Entry type is chosen as “Idle”. It is used to insert a numeric
        idle delay value into the sequence.
        Idle Sample: Idle Sample is the sample played during the pause time. The field is enabled only when the Entry
        type is chosen as “Idle”. It is used to insert a numeric idle sample value into the sequence. In case of
        interpolated mode, there are two idle sample values corresponding to I and Q data, respectively. So, for
        interpolated mode there will be two columns for idle samples i.e. Idle Samp. I and Idle Samp. Q.

        {3.10 Idle Command Segments, page 155}
        For some waveform types, like e.g. Radar pulses, huge pause segments with a static output are required between
        the real waveform segments. The gap between the real segments should be adjustable in a fine granularity. The
        idle command segment allows setting a pause between segments in a granularity that is smaller than the sync
        clock granularity. A minimum length of this pause is required (see section 6.18.2). The idle command segment is
        treated as a segment within sequences or scenarios. There is no segment loop count, but a sequence loop counter
        value is required for cases where the idle command segment is the first segment of a sequence. The granularity
        of the idle delay is equal to the waveform sample rate. The following table shows the granularity of the idle
        delay in DAC samples: (Table 20: Idle delay granularity, page 155)
        Mode | Idle Delay Granularity
        Sample Clock Divider = 1 | 1 DAC Output Sample,
        Sample Clock Divider = 2 | 2 DAC Output Samples,
        Sample Clock Divider = 4 | 4 DAC Output Samples
        Limitations: The logic that executes idle command segments uses some elements, which are not in sync clock
        granularity. To guarantee the trigger to sample output delay or the advancement event to sample output delay,
        these elements need to be reset before accepting new trigger or advancement events. This requires the waveform
        generation to be stopped for at least 3 sync clock cycles before being restarted by a trigger or an advancement
        event. A violation of this requirement leads to an unexpected output behavior for some sync clock cycles.
        Multiple adjacent idle command segments are not allowed. If the playtime of one idle command segment is not
        sufficient, the overall required idle length can be separated into multiple idle command segments where a
        normal data segment providing the static idle value is put in between. Even this wouldn't be really necessary.
        One idle command segment (delay of up to 2^24 sync clock cycles) and one additional small segment (e.g.
        length: 10 * segment vectors, loop count: up to 2^32) would provide an idle delay of more than 165 seconds in
        high speed mode at 64 GSa/s and should be sufficient for most applications.

        {3.11.1 Segment Length and Linear Playtime, page 156}
        Idle delay segments are also considered in computing the playtime. The corresponding playtime in sample vectors
        is computed from the idle delay value. When the data segments before and after the idle delay segment are
        adjacent in memory the playtime is computed as the sum of all three segments.

        6 32-bit words: control, Sequence Loop Count, Command Code, Idle Sample, Idle Delay, 0

        Fixed parameters:
        control >> Data/command selection (Bit = 31), Table 35: control, page 242
                - Bit = 31 (0: Data, 1: Command (type of command is selected by command code))
                - 2**31 = 2,147,483,648 (Decimal value), 0x80000000 (Hex value, equivalent to:
                            1000 0000 0000 0000 0000 0000 0000 0000)
        Command Code >> 16+16 bit, Reserved (31:16) + Command code (15:0). Table 41: Command Code, page 243
                - 0: Idle Delay

        Non-fixed parameters:
        :param index: index of the sequence table entry to be accessed (16,777,215 (16M – 1) = 2**24-1 entries)
        :param seq_loop: 32 bit, Number of sequence iterations (31:0). Table 36: Sequence loop count, page 242
                - 1..4G-1 = 2**32-1 = 4,294,967,295, only applicable in the first entry of a sequence.
        :param idle_sample: 24+8 bit, Reserved (31:8) + Sample to be played during pause (7:0).
                            Table 42: Idle sample, page 243
                - Bits 7:0 contain the DAC value.
        :param idle_delay: 32 bit, Idle delay in Waveform Sample Clocks (31:0). Table 43: Idle delay, page 244
                - Sample Rate Divider | Min | Max
                    - 1 | 10*256 | (2**24-1)*256+255
                    - 2 | 10*128 | (2**24-1)*128+127
                    - 4 | 10*64 | (2**24-1)*64+63
        :return:
        """
        if (self.get_extended_mem_sample_rate_divider() == 1) and (
            idle_delay in range(10 * 256, (2**24 - 1) * 256 + 256)
        ):
            pass
        elif (self.get_extended_mem_sample_rate_divider() == 2) and (
            idle_delay in range(10 * 128, (2**24 - 1) * 128 + 128)
        ):
            pass
        elif (self.get_extended_mem_sample_rate_divider() == 4) and (
            idle_delay in range(10 * 64, (2**24 - 1) * 64 + 64)
        ):
            pass
        else:
            raise ValueError("M8195A: Invalid idle_delay in set_sequence_idle")

        self.established_connection.write(
            f"STAB:DATA {index},2147483648,{seq_loop},0,{idle_sample},{idle_delay},0",
            error_check=True,
        )

    def get_sequence_data(self, index, length):
        """The query form reads the data from the sequencer memory, if all
        segments are read-write.

        The query returns an error, if at least one write-only segment
        in the waveform memory exists. Reading is only possible, when
        the signal generation is stopped. This query returns the same
        data as the “:STAB:DATA:BLOC?” query, but in comma-separated
        list of 32-bit values
        :param index: <sequence_table_index>: index of the sequence
            table entry to be accessed
        :param length: <length>: number of entries to be read
        :return: Return Data as comma-separated list of 32-bit values
        """
        self.established_connection.query(
            f":STAB:DATA? {index}, {length}", error_check=True
        )

    def get_sequence_data_binary_block_format(self, index, length):
        """The query form reads the data from the sequencer memory, if all
        segments are read-write.

        The query returns an error, if at least one write-only segment
        in the waveform memory exists. Reading is only possible, when
        the signal generation is stopped. This query returns the same
        data as the “:STAB:DATA?” query, but in IEEE binary block
        format.
        :param index: <sequence_table_index>: index of the sequence
            table entry to be accessed
        :param length: <length>: number of entries to be read
        :return: Return Data as IEEE binary block format
        """
        self.established_connection.query(
            f":STAB:DATA:BLOC? {index}, {length}", error_check=True
        )

    def set_sequence_starting_index(self, index):
        """Select where in the sequence table the sequence starts in STSequence
        mode.

        In dynamic sequence selection mode select the sequence that is
        played before the first sequence is dynamically selected.
        :param index: <sequence_table_index>|MINimum|MAXimum
        :return:
        """
        if self.get_waveform_type() == ("STS" or "STSequence"):
            if isinstance(index, int):
                self.established_connection.write(
                    f":STAB:SEQ:SEL {index}", error_check=True
                )
            elif index in self._min_max_list:
                self.established_connection.write(
                    f":STAB:SEQ:SEL {index}", error_check=True
                )
            else:
                raise ValueError(
                    "M8195A: index is neither integer nor proper string in set_sequence_starting_index"
                )
        else:
            raise ValueError(
                "M8195A: get_waveform_type is not STSequence in set_sequence_starting_index"
            )

    def get_sequence_starting_index(self):
        """Query where in the sequence table the sequence starts in STSequence
        mode.

        :return: index -> <sequence_table_index>|MINimum|MAXimum
        """
        if self.get_waveform_type() == ("STS" or "STSequence"):
            self.established_connection.query(":STAB:SEQ:SEL?", error_check=True)
        else:
            raise ValueError(
                "M8195A: get_waveform_type is not STSequence in get_sequence_starting_index"
            )

    def sequence_executation_state_and_index_entry(self):
        """This query returns an integer value containing the sequence
        execution state and the currently executed sequence table entry.

        Check page 246, "Table 44: Returned sequence state", Keysight
        M8195A AWG Revision 2
        :return: Bit | Width | Meaning: 31:27 | 5 | Reserved 26:25 | 2 |
            Sequence execution state (0: Idle, 1: Waiting for Trigger,
            2: Running, 3: Waiting for Advancement Event) 24:0 | 25 |
            Index of currently executed sequence table entry. In Idle
            state the value is undefined.
        """
        self.established_connection.query(":STAB:SEQ:STAT?", error_check=True)

    def set_dynamic_mode(self, mode):
        """Use this command to enable or disable dynamic mode. If dynamic mode
        is switched off, segments or sequences can only be switched in program
        mode, that is signal.

        generation must be stopped.
                - In arbitrary mode use TRACe[1|2|3|4]:SELect to switch to a new segment.
                - In sequence mode use [:SOURce]:STABle:SEQuence:SELect to switch to a new sequence.
        If dynamic mode is switched on, segments or sequences can be switched dynamically when signal generation is
        active. The next segment or sequence is either selected by:
                - the command [:SOURce]:STABle:DYNamic:SELect or by
                - a signal fed into the dynamic port of the M8197 module.
        The external input values select sequence table entries with corresponding indices.
        :param mode: OFF|ON|0|1
        :return:
        """
        if mode in self._on_list:
            self.established_connection.write(":STAB:DYN ON", error_check=True)
        elif mode in self._off_list:
            self.established_connection.write(":STAB:DYN OFF", error_check=True)
        else:
            raise ValueError("M8195A: Invalid Mode in set_dynamic_mode")

    def get_dynamic_mode(self):
        """Use this command to query whether the dynamic mode is enabled or
        disabled.

        Check the description in set_dynamic_mode.
        :return:
        """
        self.established_connection.query(":STAB:DYN?", error_check=True)

    def set_dynamic_starting_index(self, index):
        """When the dynamic mode for segments or sequences is active, set the
        sequence table entry to be executed next.

        :param index: <sequence_table_index>
        :return:
        """
        if self.get_dynamic_mode() is True:
            if isinstance(index, int):
                self.established_connection.write(
                    f":STAB:DYN:SEL {index}", error_check=True
                )
            else:
                raise ValueError("M8195A: Invalid index in set_dynamic_starting_index")
        else:
            raise ValueError(
                "M8195A: get_dynamic_mode is not enabled in set_dynamic_starting_index"
            )

    def set_scenario_starting_index(self, index):
        """Select where in the sequence table the scenario starts in STSCenario
        mode.

        :param index: <sequence_table_index>|MINimum|MAXimum
        :return:
        """
        if self.get_waveform_type() == ("STSC" or "STSCenario"):
            if isinstance(index, int):
                self.established_connection.write(
                    f":STAB:SCEN:SEL {index}", error_check=True
                )
            elif index in self._min_max_list:
                self.established_connection.write(
                    f":STAB:SCEN:SEL {index}", error_check=True
                )
            else:
                raise ValueError(
                    "M8195A: index is neither integer nor proper string in set_scenario_starting_index"
                )
        else:
            raise ValueError(
                "M8195A: get_waveform_type is not STSCenario in set_scenario_starting_index"
            )

    def get_scenario_starting_index(self):
        """Query where in the sequence table the scenario starts in STSCenario
        mode.

        :return: Index: Sequence table index
        """
        if self.get_waveform_type() == ("STSC" or "STSCenario"):
            self.established_connection.query(":STAB:SCEN:SEL?", error_check=True)
        else:
            raise ValueError(
                "M8195A: get_waveform_type is not STSCenario in get_scenario_starting_index"
            )

    def set_advancement_mode_scenario(self, mode):
        """Set the advancement mode for scenarios.

        :param mode: AUTO | COND | REP | SING
        :return:
        """
        if mode in ("AUTO", "COND", "REP", "SING"):
            self.established_connection.write(
                f":STAB:SCEN:ADV {mode}", error_check=True
            )
        else:
            raise ValueError("M8195A: Invalid mode in set_advancement_mode_scenario")

    def get_advancement_mode_scenario(self):
        """Query the advancement mode for scenarios.

        :return: AUTO | COND | REP | SING
        """
        self.established_connection.query(":STAB:SCEN:ADV?", error_check=True)

    def set_scenario_loop_count(self, count):
        """Set the loop count for scenarios.

        :param count: <count>|MINimum|MAXimum
                        - <count> – 1..4G-1: number of times the scenario is repeated. (4G-1 = 2**32-1 = 4,294,967,295)
        :return:
        """
        if 1 <= count <= ((2**32) - 1):
            self.established_connection.write(
                f":STAB:SCEN:COUN {count}", error_check=True
            )
        elif count in self._min_max_list:
            self.established_connection.write(
                f":STAB:SCEN:COUN {count}", error_check=True
            )
        else:
            raise ValueError(
                "M8195A: count is neither proper integer nor proper string in set_scenario_loop_count"
            )

    def get_scenario_loop_count(self):
        """Query the loop count for scenarios.

        :return: <count>|MINimum|MAXimum
                        - <count> – 1..4G-1: number of times the scenario is repeated. (4G-1 = 2**32-1 = 4,294,967,295)
        """
        self.established_connection.query(":STAB:SCEN:COUN?", error_check=True)

    #####################################################################
    # 6.19 Frequency and Phase Response Data Access #####################
    #####################################################################
    def get_freq_and_phase_resp_data(self, channel):
        """Query the frequency and phase response data for a channel. The query
        returns the data for the AWG sample frequency and output amplitude
        passed as parameters as a string of comma-separated values. If the
        sample frequency or both parameters are omitted, the currently
        configured AWG sample frequency and output amplitude are used. The
        frequency and phase response includes the sin x/ x roll-off of the
        currently configured AWG sample frequency. As a result the query
        delivers different results when performed at e.g. 60GSa/s or 65 GSa/s.
        To achieve optimum frequency and phase compensation results, the
        frequency and phase response has been characterized individually per
        channel and for different output amplitudes. As a result, the query
        delivers different results when performed at e.g. 500 mV or 800 mV. The
        frequency and phase response refers to the 2.92 mm connector. In case
        external cables from the 2.92 mm connector to the Device Under Test
        (DUT) shall be mathematically compensated for as well, the
        corresponding S-Parameter of that cable must be taken into account
        separately.

        :param channel: 1|2|3|4
        :return:
                    - <amplitude> the output amplitude
                    - <sample_frequency> the sample frequency
        Format: The first three values are output frequency 1 in Hz, corresponding relative magnitude in linear scale,
        corresponding phase in radians. The next three values are output frequency 2, corresponding relative magnitude,
        corresponding phase, and so on.
        """
        if channel in self._channel_list:
            self.established_connection.query(f":CHAR{channel}?", error_check=True)
        else:
            raise ValueError("M8195A: Invalid channel in get_freq_and_phase_resp_data")

    #####################################################################
    # 6.21 :TRACe Subsystem #############################################
    #####################################################################
    # Use the :TRACe subsystem to control the arbitrary waveforms and their respective parameters:
    #               - Create waveform segments of arbitrary size with optional initialization.
    #               - Download waveform data with or without marker data into the segments.
    #               - Delete one or all waveform segments from the waveform memory.
    def set_waveform_mem_source(self, channel, source):
        """Use this command to set the source of the waveform samples for a
        channel.

        There are dependencies between
        this parameter, the same parameter for other channels, the memory sample rate divider and the instrument mode
        (number of channels). The tables in section 1.5.5 show the available combinations. The value of this parameter
        for each channel determines the target (Internal/Extended Memory) of the waveform transfer operation using the
        TRAC:DATA command.
        Note: It is recommended to set these parameters in one transaction. (Check above explanation)
        :param channel: 1|2|3|4
        :param source:
                        INTernal – the channel uses Internal Memory
                        EXTended – the channel uses Extended Memory
                        NONE – the channel is not used in this configuration (query only)
        :return:
        """
        if channel in self._channel_list:
            if source in ("INT", "EXT", "INTernal", "EXTernal"):
                self.established_connection.write(
                    f":TRAC{channel}:MMOD {source}", error_check=True
                )
            else:
                raise ValueError("M8195A: Invalid source in set_waveform_mem_source")
        else:
            raise ValueError("M8195A: Invalid channel in set_waveform_mem_source")

    def get_waveform_mem_source(self, channel):
        """Check description for 'def set_waveform_mem_source' :param channel:
        1|2|3|4 :return: Source:

        INTernal – the channel uses Internal Memory EXTended – the
        channel uses Extended Memory NONE – the channel is not used in
        this configuration (query only)
        """
        if channel in self._channel_list:
            self.established_connection.query(f":TRAC{channel}:MMOD?", error_check=True)
        else:
            raise ValueError(
                "M8195A: Invalid channel in channelMemoryModeQuery"
            )  # channelMemoryModeQuery?

    def set_waveform_mem_segment(
        self, channel, segm_id, length, init_value=0, write_only=False
    ):  # pylint: disable=too-many-arguments
        """Use this command to define the size of a waveform memory segment.

        If init_value is specified, all values in the segment are
        initialized. If not specified, memory is only allocated but not
        initialized. [Note] If the channel is sourced from Extended
        Memory, the same segment is defined on all other channels
        sourced from Extended Memory.
        :param channel: channel number (1|2|3|4)
        :param segm_id: ID of the segment
        :param length: length of the segment in samples, marker samples
            do not count
        :param init_value: [Optional] optional initialization DAC value
        :param write_only: The segment will be flagged write-only, so it
            cannot be read back or stored.
        :return:
        """
        if channel in self._channel_list and all(
            isinstance(i, int) for i in [segm_id, length]
        ):
            if init_value:
                if isinstance(init_value, int):
                    if write_only:
                        self.established_connection.write(
                            f":TRAC{channel}:DEF:WONL {segm_id},{length},{init_value}",
                            error_check=True,
                        )
                    else:
                        self.established_connection.write(
                            f":TRAC{channel}:DEF {segm_id},{length},{init_value}",
                            error_check=True,
                        )
                else:
                    raise ValueError(
                        "M8195A: Invalid init_value in set_waveform_mem_segment"
                    )
            else:
                if write_only:
                    self.established_connection.write(
                        f":TRAC{channel}:DEF:WONL {segm_id},{length}", error_check=True
                    )
                else:
                    self.established_connection.write(
                        f":TRAC{channel}:DEF {segm_id},{length}", error_check=True
                    )
        else:
            raise ValueError(
                "M8195A: Invalid (channel, segm_id, length) in set_waveform_mem_segment"
            )

    def set_waveform_mem_new_segment(
        self, channel, length, init_value=0, write_only=False
    ):
        """Use this query to define the size of a waveform memory segment.

        If init_value is specified, all values in the segment are
        initialized. If not specified, memory is only allocated but not
        initialized. If the channel is sourced from Extended Memory, the
        same segment is defined on all other channels sourced from
        Extended Memory.
        :param channel: channel number (1|2|3|4)
        :param length: length of the segment in samples, marker samples
            do not count
        :param init_value: [Optional] optional initialization DAC value
        :param write_only: The segment will be flagged write-only, so it
            cannot be read back or stored.
        :return: If the query was successful, a new segm_id will be
            returned.
        """
        if channel in self._channel_list and isinstance(length, int):
            if init_value:
                if isinstance(init_value, int):
                    if write_only:
                        self.established_connection.query(
                            f":TRAC{channel}:DEF:WONL:NEW? {length},{init_value}",
                            error_check=True,
                        )
                    else:
                        self.established_connection.query(
                            f":TRAC{channel}:DEF:NEW? {length},{init_value}",
                            error_check=True,
                        )
                else:
                    raise ValueError(
                        "M8195A: Invalid init_value in set_waveform_mem_new_segment"
                    )
            else:
                if write_only:
                    self.established_connection.query(
                        f":TRAC{channel}:DEF:WONL:NEW? {length}", error_check=True
                    )
                else:
                    self.established_connection.query(
                        f":TRAC{channel}:DEF:NEW? {length}", error_check=True
                    )
        else:
            raise ValueError(
                "M8195A: Invalid (channel, length) in set_waveform_mem_new_segment"
            )

    def set_waveform_data_in_mem(self, channel, segm_id, offset, value):
        """Use this command to load waveform data into the module memory. If
        segm_id is already filled with data, the new values overwrite the
        current values. If length is exceeded error -223 (too much data) is
        reported.

        Reading is only possible, when the signal generation is stopped.
        Writing is possible, when signal generation is stopped or when signal generation is started in dynamic mode.

        The target (Internal/Extended Memory) of the waveform transfer is given by the value set by the TRAC:MMOD
        command for the channel. The data format (waveform samples only or interleaved waveform and marker samples)
        is given by the DAC Mode set by the INST:DACM command.

        When transferring data to Extended Memory, the parameter <offset> must contain a value corresponding to an even
        number of memory vectors. The number of samples in a memory vector equals the waveform memory granularity.
        This limitation does not exist for transferring data to Internal Memory.

        [Note] If the segment is split in smaller sections, the sections have to be written in order of ascending
        <Offset> values. If modification of the segment contents is necessary, the whole segment with all sections must
        be rewritten. If segments are created and deleted in arbitrary order, their position and order in memory cannot
        be controlled by the user, because the M8195 reuses the memory space of deleted segments for newly created
        segments. To fulfill the streaming and minimum linear playtime requirements the only way to control the position
        of the first downloaded segment and the order of the following segments is to delete all segments from memory
        (:TRACe[1|2|3|4]:DELete:ALL) and then creating the segments in the order in which they should be placed in
        memory.

        :param channel: 1|2|3|4
        :param segm_id: ID of the segment
        :param offset: offset from segment start in samples (marker samples do not count) to allow splitting the
        transfer in smaller portions
        :param value:
            Block: waveform data samples in the data format described above in IEEE binary block format
            NumVal: waveform data samples in the data format described above in comma separated list format
        :return:
        """
        if all(isinstance(i, int) for i in [channel, segm_id, offset, value]):
            self.established_connection.write(
                f":TRAC{channel}:DATA {segm_id},{offset},{value}", error_check=True
            )
        else:
            raise ValueError(
                "M8195A: Invalid (channel, segm_id, Offset, value) in set_waveform_data_in_mem"
            )

    def get_waveform_data_in_mem(
        self, channel, segm_id, offset, length, bloc=False
    ):  # pylint: disable=too-many-arguments
        """Check description for 'def set_waveform_data_in_mem'.

        :param channel: 1|2|3|4
        :param segm_id: ID of the segment
        :param offset: offset from segment start in samples (marker
            samples do not count) to allow splitting the transfer in
            smaller portions
        :param length: number of samples to read in the query case
        :param bloc: if True, returns the data as the “:TRAC:DATA?”
            query, but in IEEE binary block format.
        :return:
        """
        if all(isinstance(i, int) for i in [channel, segm_id, offset, length]):
            if bloc is False:
                self.established_connection.query(
                    f":TRAC{channel}:DATA? {segm_id},{offset},{length}",
                    error_check=True,
                )
            elif bloc is True:
                self.established_connection.query(
                    f":TRAC{channel}:DATA:BLOC? {segm_id},{offset},{length}",
                    error_check=True,
                )
        else:
            raise ValueError(
                "M8195A: Invalid (channel, segm_id, offset, length) in get_waveform_data_in_mem"
            )

    def waveform_data_from_file_import(
        self,
        channel,
        segm_id,
        file_name,
        file_type,
        data_type,
        marker_flag,
        padding,
        init_value,
        ignore_header_parameters,
    ):  # pylint: disable=too-many-arguments
        """Use this command to import waveform data from a file and write it to
        the waveform memory. You can fill an already existing segment or a new
        segment can also be created. This command can be used to import real-
        only waveform data as well as complex I/Q data. This command supports
        different file formats.

        :param channel: 1|2|3|4
        :param segm_id: This is the number of the segment, into which the data will be written.
        :param file_name: This is the complete path of the file.
        :param file_type: TXT|BIN|BIN8|IQBIN|BIN6030|BIN5110|LICensed |MAT89600|DSA90000|CSV
        :param data_type: This parameter is only used, if the file contains complex I/Q data. It selects, if the values
        of I or Q are imported.
                            − IONLy: Import I values.
                            − QONLy: Import Q values.
                            − BOTH: Import I and Q values and up-convert them to the carrier frequency set by the
                            CARR:FREQ command. This selection is only supported for the LICensed file type.
        :param marker_flag: This flag is applicable to BIN5110 format only, which can either consists of full 16 bit DAC
        values without markers or 14 bit DAC values and marker bits in the 2 LSBs.
                            − ON|1: The imported data will be interpreted as 14 bit DAC values and marker bits in the
                            2 LSBs.
                            − OFF|0: The imported data will be interpreted as 16 bit DAC values without marker bits.
        :param padding: This parameter is optional and specifies the padding type. The parameter is ignored for the
        LICensed file type.
                            - ALENgth: Automatically determine the required length of the segment. If the segment does
                            not exist, it is created. After execution the segment has exactly the length of the pattern
                            in file or a multiple of this length to fulfill granularity and minimum segment length
                            requirements. This is the default behavior.
                            − FILL: The segment must exist, otherwise an error is returned. If the pattern in the file
                            is larger than the defined segment length, excessive samples are ignored. If the pattern in
                            the file is smaller than the defined segment length, remaining samples are filled with the
                            value specified by the <init_value> parameter.
        :param init_value: This is an optional initialization value used when FILL is selected as padding type. For
        real-only formats this is a DAC value. For complex I/Q file formats this is the I-part or Q-part of an I/Q
        sample pair in binary format (int8). Defaults to 0 if not specified.
        :param ignore_header_parameters: This flag is optional and used to specify if the header parameters from the
        file need to be set in the instrument or ignored. This flag is applicable to formats CSV and MAT89600, which
        can contain header parameters.
                            − ON|1: Header parameters will be ignored.
                            − OFF|0: Header parameters will be set. This is the default.
        :return:
        """

        if file_type in (
            "TXT",
            "BIN",
            "BIN8",
            "IQBIN",
            "BIN6030",
            "BIN5110",
            "LICensed",
            "MAT89600",
            "DSA90000",
            "CSV",
        ):
            pass
        else:
            raise ValueError(
                "M8195A: Invalid file_type in waveform_data_from_file_import"
            )

        if data_type in ("IONL", "IONLy", "QONL", "QONLy", "BOTH"):
            pass
        else:
            raise ValueError(
                "M8195A: Invalid data_type in waveform_data_from_file_import"
            )

        if marker_flag in self._on_list or self._off_list:
            pass
        else:
            raise ValueError(
                "M8195A: Invalid marker_flag in waveform_data_from_file_import"
            )

        if padding in ("ALEN", "ALENgth", "FILL"):
            pass
        else:
            raise ValueError(
                "M8195A: Invalid padding in waveform_data_from_file_import"
            )

        if ignore_header_parameters in self._on_list or self._off_list:
            pass
        else:
            raise ValueError(
                "M8195A: Invalid marker_flag in waveform_data_from_file_import"
            )

        self.established_connection.write(
            f":TRAC{channel}:IMP {segm_id},{file_name},{file_type},{data_type},"
            f"{marker_flag},{padding},{init_value},{ignore_header_parameters}",
            error_check=True,
        )

    def waveform_data_from_bin_import(self, channel, segm_id, file_name):
        """Check description for waveform_data_from_file_import :param channel:
        1|2|3|4 :param segm_id:

        :param file_name:
        :return:
        """
        self.established_connection.write(
            f":TRAC{channel}:IMP {segm_id},{file_name}, BIN8, IONLY, ON, ALEN",
            error_check=True,
        )

    def set_file_import_scaling_state(self, channel, state):
        """Set the scaling state for the file import.

        If scaling is disabled, an imported waveform is not scaled. If
        the waveform contains out-of-range values, the import command
        returns an error. If scaling is enabled, an imported waveform is
        scaled, so that it uses the whole DAC range. This also allows
        importing waveforms with out-of-range values. The scaling
        affects all file formats. But for files of type LICensed, if
        scaling is disabled, the value set by the CARR:SCAL command is
        used. If scaling is enabled, CARR:SCAL is ignored and an optimal
        scaling factor is calculated, so that the whole DAC range is
        used.
        :param channel: 1|2|3|4
        :param state: OFF|ON|0|1
        :return:
        """
        if channel in self._channel_list:
            if state in self._on_list:
                self.established_connection.write(
                    f":TRAC{channel}:IMP:SCAL ON", error_check=True
                )
            elif state in self._off_list:
                self.established_connection.write(
                    f":TRAC{channel}:IMP:SCAL OFF", error_check=True
                )
            else:
                raise ValueError(
                    "M8195A: Invalid state in set_file_import_scaling_state"
                )
        else:
            raise ValueError("M8195A: Invalid channel in set_file_import_scaling_state")

    def get_file_import_scaling_state(self, channel):
        """Query the scaling state for the file import.

        Read the description for set_file_import_scaling_state
        :param channel: 1|2|3|4
        :return:
        """
        self.established_connection.query(
            f":TRAC{channel}:IMP:SCAL:STAT?", error_check=True
        )

    def delete_segment(self, channel, segm_id):
        """Delete a segment.

        The command can only be used in program mode. If the channel is
        sourced from Extended Memory, the same segment is deleted on all
        other channels sourced from Extended Memory.
        :param channel: 1|2|3|4
        :param segm_id:
        :return:
        """
        if channel in self._channel_list:
            if isinstance(segm_id, int):
                self.established_connection.write(
                    f":TRAC{channel}:DEL {segm_id}", error_check=True
                )
            else:
                raise ValueError(
                    "M8195A: Invalid segm_id (segment ID) in delete_segment"
                )
        else:
            raise ValueError("M8195A: Invalid channel in delete_segment")

    def delete_all_segment(self, channel):
        """Delete all segments.

        The command can only be used in program mode. If the channel is
        sourced from Extended Memory, the same segment is deleted on all
        other channels sourced from Extended Memory.
        :param channel: 1|2|3|4
        :return:
        """
        if channel in self._channel_list:
            self.established_connection.write(
                f":TRAC{channel}:DEL:ALL", error_check=True
            )
        else:
            raise ValueError("M8195A: Invalid channel in delete_all_segment")

    def get_segments_id_length(self, channel):
        """The query returns a comma-separated list of segment-ids that are
        defined and the length of each segment.

        So
        first number is a segment id, next length ...
        If no segment is defined, “0, 0” is returned.
        :param channel: 1|2|3|4
        :return:
        """
        if channel in self._channel_list:
            self.established_connection.query(f":TRAC{channel}:CAT?", error_check=True)
        else:
            raise ValueError("M8195A: Invalid channel in get_segments_id_length")

    def mem_space_waveform_data(self, channel):
        """
        The query returns the amount of memory space available for waveform data in the following form:
        <bytes available>, <bytes in use>, < contiguous bytes available>.
        :param channel: 1|2|3|4
        :return:
        """
        if channel in self._channel_list:
            self.established_connection.query(f":TRAC{channel}:FREE?", error_check=True)
        else:
            raise ValueError("M8195A: Invalid channel in mem_space_waveform_data")

    def set_segment_name(self, channel, segm_id, name):
        """This command associates a name to a segment.

        :param channel: 1|2|3|4
        :param segm_id: the number of the segment
        :param name: string of at most 32 characters
        :return:
        """
        if channel in self._channel_list:
            if isinstance(segm_id, int):
                if isinstance(name, str) and len(name) <= 32:
                    self.established_connection.write(
                        f":TRAC{channel}:NAME {segm_id},{name}", error_check=True
                    )
                else:
                    raise ValueError(
                        "M8195A: Invalid name (not string or improper length) in set_segment_name"
                    )
            else:
                raise ValueError(
                    "M8195A: Invalid segm_id (Segment ID) in set_segment_name"
                )
        else:
            raise ValueError("M8195A: Invalid channel in set_segment_name")

    def get_segment_name(self, channel, segm_id):
        """The query gets the name for a segment.

        :param channel: 1|2|3|4
        :param segm_id: the number of the segment
        :return:
        """
        if channel in self._channel_list:
            if isinstance(segm_id, int):
                self.established_connection.query(
                    f":TRAC{channel}:NAME? {segm_id}", error_check=True
                )
            else:
                raise ValueError(
                    "M8195A: Invalid segm_id (Segment ID) in get_segment_name"
                )
        else:
            raise ValueError("M8195A: Invalid channel in get_segment_name")

    def set_segment_comment(self, channel, segm_id, comment):
        """This command associates a comment to a segment.

        :param channel: 1|2|3|4
        :param segm_id: the number of the segment
        :param comment: string of at most 256 characters
        :return:
        """
        if channel in self._channel_list:
            if isinstance(segm_id, int):
                if isinstance(comment, str) and len(comment) <= 256:
                    self.established_connection.write(
                        f":TRAC{channel}:COMM {segm_id}, {comment}", error_check=True
                    )
                else:
                    raise ValueError(
                        "M8195A: Invalid comment (not string or improper length) in set_segment_comment"
                    )
            else:
                raise ValueError(
                    "M8195A: Invalid segm_id (Segment ID) in set_segment_comment"
                )
        else:
            raise ValueError("M8195A: Invalid channel in set_segment_comment")

    def get_segment_comment(self, channel, segm_id):
        """The query gets the comment for a segment.

        :param channel: 1|2|3|4
        :param segm_id: the number of the segment
        :return:
        """
        if channel in self._channel_list:
            if isinstance(segm_id, int):
                self.established_connection.query(
                    f":TRAC{channel}:COMM? {segm_id}", error_check=True
                )
            else:
                raise ValueError(
                    "M8195A: Invalid segm_id (Segment ID) in get_segment_comment"
                )
        else:
            raise ValueError("M8195A: Invalid channel in get_segment_comment")

    def set_segment_select(self, channel, segm_id):
        """Selects the segment, which is output by the instrument in arbitrary
        function mode.

        The command has only effect, If the channel is sourced from
        Extended Memory. In this case the same value is used for all
        other channels sourced from Extended Memory.
        :param channel: 1|2|3|4
        :param segm_id: the number of the segment,
            <segment_id>|MINimum|MAXimum
        :return:
        """
        if channel in self._channel_list:
            if isinstance(segm_id, int):
                self.established_connection.write(
                    f":TRAC{channel}:SEL {segm_id}", error_check=True
                )
            elif segm_id in self._min_max_list:
                self.established_connection.write(
                    f":TRAC{channel}:SEL {segm_id}", error_check=True
                )
            else:
                raise ValueError(
                    "M8195A: segm_id is neither integer nor proper string in set_segment_select"
                )
        else:
            raise ValueError("M8195A: Invalid channel in set_segment_select")

    def get_segment_select(self, channel):
        """Query the selected segment, which is output by the instrument in
        arbitrary function mode.

        The command has only effect, If the channel is sourced from
        Extended Memory. In this case the same value is used for all
        other channels sourced from Extended Memory.
        :param channel: 1|2|3|4
        :return:
        """
        if channel in self._channel_list:
            self.established_connection.query(f":TRAC{channel}:SEL?", error_check=True)
        else:
            raise ValueError("M8195A: Invalid channel in get_segment_select")

    def set_advancement_mode_segment(self, channel, mode):
        """Use this command to set the advancement mode for the selected
        segment.

        The advancement mode is used, if the segment is played in
        arbitrary mode. The command has only effect, If the channel is
        sourced from Extended Memory. In this case the same value is
        used for all other channels sourced from Extended Memory.
        :param channel: 1|2|3|4
        :param mode: AUTO|COND|REP|SING
        :return:
        """
        if channel in self._channel_list:
            if mode in ("AUTO", "COND", "REP", "SING"):
                self.established_connection.write(
                    f":TRAC{channel}:ADV {mode}", error_check=True
                )
            else:
                raise ValueError("M8195A: Invalid Mode in set_advancement_mode_segment")
        else:
            raise ValueError("M8195A: Invalid channel in set_advancement_mode_segment")

    def get_advancement_mode_segment(self, channel):
        """Use this query to get the advancement mode for the selected segment.

        The advancement mode is used, if the segment is played in
        arbitrary mode. The command has only effect, If the channel is
        sourced from Extended Memory. In this case the same value is
        used for all other channels sourced from Extended Memory.
        :param channel: 1|2|3|4
        :return: Mode -> AUTO|COND|REP|SING
        """
        if channel in self._channel_list:
            self.established_connection.query(f":TRAC{channel}:ADV?", error_check=True)
        else:
            raise ValueError("M8195A: Invalid channel in get_advancement_mode_segment")

    def set_selected_segment_loop_count(self, channel, count):
        """Use this command to set the segment loop count for the selected
        segment.

        The segment loop count is used, if the
        segment is played in arbitrary mode.
        The command has only effect, If the channel is sourced from Extended Memory. In this case the same value is
        used for all other channels sourced from Extended Memory.
        :param channel: 1|2|3|4
        :param count: number of times the selected segment is repeated (1..4G-1).
        :return:
        """
        if channel in self._channel_list:
            if isinstance(count, int):
                self.established_connection.write(
                    f":TRAC{channel}:COUN {count}", error_check=True
                )
            elif count in self._min_max_list:
                self.established_connection.write(
                    f":TRAC{channel}:COUN {count}", error_check=True
                )
            else:
                raise ValueError(
                    "M8195A: count is neither integer nor proper string in set_selected_segment_loop_count"
                )
        else:
            raise ValueError(
                "M8195A: Invalid channel in set_selected_segment_loop_count"
            )

    def get_selected_segment_loop_count(self, channel):
        """Use this query to get the segment loop count for the selected
        segment.

        The segment loop count is used, if the
        segment is played in arbitrary mode.
        The command has only effect, If the channel is sourced from Extended Memory. In this case the same value is
        used for all other channels sourced from Extended Memory.
        :param channel: 1|2|3|4
        :return: Count: number of times the selected segment is repeated (1..4G-1). (4G-1 = 2**32-1 = 4,294,967,295)
        """
        if channel in self._channel_list:
            self.established_connection.query(f":TRAC{channel}:COUN?", error_check=True)
        else:
            raise ValueError(
                "M8195A: Invalid channel in get_selected_segment_loop_count"
            )

    def set_selected_segment_marker_state(self, channel, state):
        """Use this command to enable or disable markers for the selected
        segment.

        The command has only effect, If the channel is sourced from
        Extended Memory. In this case the same value is used for all
        other channels sourced from Extended Memory.
        :param channel: 1|2|3|4
        :param state: OFF|ON|0|1
        :return:
        """
        if channel in self._channel_list:
            if state in self._on_list:
                self.established_connection.write(
                    f":TRAC{channel}:MARK ON", error_check=True
                )
            elif state in self._off_list:
                self.established_connection.write(
                    f":TRAC{channel}:MARK OFF", error_check=True
                )
            else:
                raise ValueError(
                    "M8195A: Invalid state in set_selected_segment_marker_state"
                )
        else:
            raise ValueError(
                "M8195A: Invalid channel in set_selected_segment_marker_state"
            )

    def get_selected_segment_marker_state(self, channel):
        """The query form gets the current marker state.

        Read the description of set_selected_segment_marker_state
        :param channel: 1|2|3|4
        :return: State: OFF|ON|0|1
        """
        if channel in self._channel_list:
            self.established_connection.query(f":TRAC{channel}:MARK?", error_check=True)
        else:
            raise ValueError(
                "M8195A: Invalid channel in get_selected_segment_marker_state"
            )

    #####################################################################
    # 6.22 :TEST Subsystem ##############################################
    #####################################################################
    def get_self_tests_power_result(self):
        """Return the results of the power on self-tests.

        :return:
        """
        self.established_connection.query(":TEST:PON?", error_check=True)

    def get_self_tests_power_result_messsage(self):
        """Same as *TST?

        but the actual test messages are returned. Currently same as
        :TEST:PON?
        :return:
        """
        self.established_connection.query(":TEST:TST?", error_check=True)


#################################


if __name__ == "__main__":
    M8195Connection(ip_address="0.0.0.0", port=5025)  # ip_address='0.0.0.0', port=5025
