"""Instrument for using the Keysight M8195A AWG."""

import ipaddress
import os.path
import socket


class SocketInstrumentError(Exception):
    pass


class GranularityError(Exception):
    """Waveform Granularity Exception class."""

    def __init__(self):
        pass

    def __str__(self):
        return f"Must be a multiplication of Granularity"


class M8195Connection:
    def __init__(self, IPAddress, port=5025, TimeOut=10):
        """Opens up a socket connection between an instrument and your PC
        :param IPAddress: ip address of the instrument :param port: [Optional]
        socket port of the instrument (default 5025) :return: Returns the
        socket session."""
        self.OpenSession = []
        self.port = port
        self.IPAddress = IPAddress
        self.TimeOut = TimeOut

        if ipaddress.ip_address(self.IPAddress):
            print(f"connecting to IPv4 address: {self.IPAddress}")
        else:
            raise ValueError(f"Invalid IP address")

        self.OpenSession = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.OpenSession.settimeout(self.TimeOut)

    def OpenSession(self):
        """Opens the socket connection :return:"""
        print("Opening socket session and connection ...")
        print(f"connecting to M8195A ...")

        try:
            self.OpenSession.connect((self.IPAddress, self.port))
            print(f"connected to M8195A ...")
        except OSError:
            print(f"Failed to connect to the instrument, please check your IP address")

        #  setblocking(1) = socket blocking -> it will wait for the operation to complete
        #  setblocking(0) = socket non-blocking -> it will never wait for the operation to complete
        self.OpenSession.setblocking(True)

        print(f"*IDN?: ")
        InstQuery_IDN = self.Query("*idn?", error_check=False)
        print(InstQuery_IDN)
        if "Keysight Technologies,M8195A" in InstQuery_IDN:
            print(f"success!")
        else:
            self.CloseSession()
            raise NameError(
                f"could not communicate with device, or not a Keysight Technologies, M8195A"
            )

    def CloseSession(self):
        """Closes the socket connection :return: TCPIP socket connection."""
        print("Closing socket session and connection ...")
        self.OpenSession.shutdown(socket.SHUT_RDWR)
        self.OpenSession.close()

    def ErrorCheck(self):
        """Checks an instrument for errors, print them out, and clears error
        queue.

        Raises SocketInstrumentError with the info of the error
        encountered.
        :return: Returns True if any errors are encountered
        """
        Err = []
        response = self.Query("SYST:ERR?", error_check=False).strip()

        while "0" not in response:
            Err.append(response)
            response = self.Query("SYST:ERR?", error_check=False).strip()

        if Err:
            raise SocketInstrumentError(Err)

    def Query(self, command, error_check=False):
        """Sends a query to an instrument and reads the output buffer
        immediately afterward :param command: text containing an instrument
        command (Documented SCPI); Should end with "?" :param error_check:

        [Optional] Check for instrument errors (default False)
        :return: Returns the query response.
        """

        if not isinstance(command, str):
            raise SocketInstrumentError(f"command must be a string.")

        if "?" not in command:
            raise SocketInstrumentError(f'Query must end with "?"')

        try:
            self.OpenSession.sendall(str.encode(command + "\n"))
            response = self.Read()
            if error_check:
                Err = self.ErrorCheck()
                if Err:
                    response = "<Error>"
                    print(f"Query - local: {error_check}, command: {command}")

        except socket.timeout:
            print(f"Query error:")
            self.ErrorCheck()
            response = "<Timeout Error>"

        return response

    def Read(self):
        """Reads from a socket until a newline is read :return: Returns the
        data read."""
        response = b""
        while response[-1:] != b"\n":
            response += self.OpenSession.recv(4096)

        return response.decode().strip()

    def Write(self, command, error_check=False):
        """Write a command to an instrument :param command: text containing an
        instrument command; i.e. Documented SCPI command :param error_check:

        [Optional] Check for instrument errors (default False)
        :return:
        """
        if not isinstance(command, str):
            raise SocketInstrumentError(f"Argument must be a string.")

        command = "{}\n".format(command)
        self.OpenSession.sendall(command.encode())

        if error_check:
            print(f"Send - local: {error_check}, command: {command}")
            self.ErrorCheck()


class M8195AConfiguration:
    def __init__(self):
        """"""
        self.EstablishedConnection = M8195Connection(IPAddress="0.0.0.0", port=5025)
        self._MinMaxList = ("MIN", "MAX", "MINimum", "MAXimum")
        self._OnList = (1, "1", "on", "ON", True)
        self._OffList = (0, "0", "off", "OFF", False)
        self._ChannelList = (1, 2, 3, 4)

    def OpenIOSession(self):
        """Open IO session :return:"""
        self.EstablishedConnection.OpenSession()

    def CloseIOSession(self):
        """Close IO session :return:"""
        self.EstablishedConnection.CloseSession()

    #####################################################################
    # 6.5 System Related Commands (SYSTem Subsystem) ####################
    #####################################################################
    def EventInTriggerOutSwitchSet(self, Switch):
        """The Event In and Trigger Out functionality use a shared connector on
        the front panel.

        This command switches
        between trigger output and event input functionality. When Trigger Out functionality is active, Event In
        functionality is disabled and vice versa.
        Note: Trigger Out is for future use. There are no plans to support Trigger Out functionality directly from
        M8195A firmware. Trigger Out is tentatively supported by 81195A optical modulation generator software (V2.1
        or later).
        :param Switch: 'EIN', 'TOUT'
        :return:
        """
        if Switch in ("EIN", "TOUT"):
            self.EstablishedConnection.Write(
                f":SYST:EIN:MODE {Switch}", error_check=True
            )
        else:
            raise Exception("M8195A: Invalid Switch in EventInTriggerOutSwitchSet")

    def EventInTriggerOutSwitchQuery(self):
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
        self.EstablishedConnection.Query(f":SYST:EIN:MODE?", error_check=True)

    def ErrorQuery(self):
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
        self.EstablishedConnection.Query(f":SYST:ERR?", error_check=True)

    def SCPIListQuery(self):
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
        self.EstablishedConnection.Query(f":SYST:HELP:HEAD?", error_check=True)

    def LicenseQuery(self):
        """This query lists the licenses installed.

        :return:
        """
        self.EstablishedConnection.Query(f":SYST:LIC:EXT:LIST?", error_check=True)

    def InstrumentSettingSet(self, BinaryData):
        """In set form, the block data must be a complete instrument set-up
        read using the query form of the command.

        The data is in a binary format.
        :return:
        """
        self.EstablishedConnection.Write(f":SYST:SET {BinaryData}", error_check=True)

    def InstrumentSettingQuery(self):
        """In query form, the command reads a block of data containing the
        instrument’s complete set-up.

        The set-up information includes all parameter and mode settings,
        but does not include the contents of the instrument setting
        memories or the status group registers. The data is in a binary
        format, not ASCII, and cannot be edited. This command has the
        same functionality as the *LRN command.
        """
        self.EstablishedConnection.Query(f":SYST:SET?", error_check=True)

    def SCPIVersionQuery(self):
        """
        Query SCPI version number
        :return: a formatted numeric value corresponding to the SCPI version number for which the instrument complies.
        """
        self.EstablishedConnection.Query(f":SYST:VERS?", error_check=True)

    def SoftFrontPanelConnectionsQuery(self):
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
        self.EstablishedConnection.Query(f":SYST:COMM:*?", error_check=True)

    def SoftFrontPanelVXI11NumberQuery(self):
        """
        Query VXI-11 instrument number
        :return: This query returns the VXI-11 instrument number used by the Soft Front Panel.
        """
        self.EstablishedConnection.Query(f":SYST:COMM:INST?", error_check=True)

    def SoftFrontPanelHiSLIPNumberQuery(self):
        """
        Query HiSLIP number
        :return: This query returns the HiSLIP number used by the Soft Front Panel.
        """
        self.EstablishedConnection.Query(f":SYST:COMM:HISL?", error_check=True)

    def SoftFrontPanelSocketPortQuery(self):
        """
        Query socket port
        :return: This query returns the socket port used by the Soft Front Panel.
        """
        self.EstablishedConnection.Query(f":SYST:COMM:SOCK?", error_check=True)

    def SoftFrontPanelTelnetPortQuery(self):
        """
        Query telnet port
        :return: This query returns the telnet port used by the Soft Front Panel.
        """
        self.EstablishedConnection.Query(f":SYST:COMM:TELN?", error_check=True)

    def SoftFrontPanelTCPPortQuery(self):
        """

        :return: This query returns the port number of the control connection. You can use the control port to send
        control commands (for example “Device Clear”) to the instrument.
        """
        self.EstablishedConnection.Query(f":SYST:COMM:TCP:CONT?", error_check=True)

    #####################################################################
    # 6.6 Common Command List ###########################################
    #####################################################################
    def InstrumentID(self):
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
        self.EstablishedConnection.Query(f"*IDN?", error_check=True)

    def ClearEventRegister(self):
        """Clear the event register in all register groups.

        This command also clears the error queue and cancels a *OPC
        operation. It doesn't clear the enable register.
        :return:
        """
        self.EstablishedConnection.Write(f"*CLS", error_check=True)

    def StatusRegisterBit5Set(self):
        """Enable bits in the Standard Event Status Register to be reported in
        the Status Byte.

        The selected bits are summarized in the “Standard Event” bit
        (bit 5) of the Status Byte Register. These bits are not cleared
        by a *CLS command. Value Range: 0–255.
        :return:
        """
        self.EstablishedConnection.Write(f"*ESE", error_check=True)

    def StatusRegisterBit5Query(self):
        """The *ESE?

        query returns a value which corresponds to the binary-weighted
        sum of all bits enabled decimal by the *ESE command. Value
        Range: 0–255.
        :return:
        """
        self.EstablishedConnection.Query(f"*ESE?", error_check=True)

    def StandardEventStatusRegisterQuery(self):
        """Query the Standard Event Status Register.

        Once a bit is set, it remains set until cleared by a *CLS (clear
        status) command or queried by this command. A query of this
        register returns a decimal value which corresponds to the
        binary-weighted sum of all bits set in the register.
        :return:
        """
        self.EstablishedConnection.Query(f"ESR?", error_check=True)

    def OperationCompleteSet(self):
        """Set the “Operation Complete” bit (bit 0) in the Standard Event
        register after the previous commands have been completed.

        :return:
        """
        self.EstablishedConnection.Write(f"*OPC", error_check=True)

    def OperationCompleteQuery(self):
        """Return "1" to the output buffer after the previous commands have
        been completed.

        Other commands cannot be executed until this command completes.
        :return:
        """
        self.EstablishedConnection.Query(f"*OPC?", error_check=True)

    def InstalledOptionsQuery(self):
        """Read the installed options.

        The response consists of any number of fields separated by
        commas.
        :return:
        """
        self.EstablishedConnection.Query(f"*OPT?", error_check=True)

    def InstrumentReset(self):
        """Reset instrument to its factory default state.

        :return:
        """
        self.EstablishedConnection.Write(f"*RST", error_check=True)

    def ServiceRequestEnableBitsSet(self, Bits):
        """Enable bits in the Status Byte to generate a Service Request.

        To enable specific bits, you must write a decimal value which
        corresponds to the binary-weighted sum of the bits in the
        register. The selected bits are summarized in the “Master
        Summary” bit (bit 6) of the Status Byte Register. If any of the
        selected bits change from “0” to “1”, a Service Request signal
        is generated.
        :return:
        """
        if Bits.isdecimal():
            self.EstablishedConnection.Write(f"*SRE {Bits}", error_check=True)
        else:
            raise Exception(
                f"M8195A: Invalid Bits (not decimal) in ServiceRequestEnableBitsSet"
            )

    def ServiceRequestEnableBitsQuery(self):
        """The *SRE?

        query returns a decimal value which corresponds to the binary-
        weighted sum of all bits enabled by the *SRE command.
        :return:
        """
        self.EstablishedConnection.Query(f"*SRE?", error_check=True)

    def StatusByteRegisterQuery(self):
        """Query the summary (status byte condition) register in this register
        group.

        This command is similar to a Serial Poll, but it is processed
        like any other instrument command. This command returns the same
        result as a Serial Poll but the “Master Summary” bit (bit 6) is
        not cleared by the *STB? command.
        :return:
        """
        self.EstablishedConnection.Query(f"*STB?", error_check=True)

    def SelfTestQuery(self):
        """Execute Self Tests.

        If self-tests pass, a 0 is returned. A number lager than 0
        indicates the number of failed tests. To get actual messages,
        use :TEST:TST?
        :return:
        """
        self.EstablishedConnection.Query(f"*TST?", error_check=True)

    def InstrumentSettingLearnQuery(self):
        """Query the instrument and return a binary block of data containing
        the current settings (learn string).

        You can then send the string back to the instrument to restore
        this state later. For proper operation, do not modify the
        returned string before sending it to the instrument. Use
        :SYST:SET to send the learn string. See :SYSTem:SET[?].
        :return:
        """
        self.EstablishedConnection.Query(f"*LRN?", error_check=True)

    def WaitCurrentCommandExecutionQuery(self):
        """Prevents the instrument from executing any further commands until
        the current command has finished executing.

        :return:
        """
        self.EstablishedConnection.Query(f"*WAI?", error_check=True)

    #####################################################################
    # 6.7 Status Model ##################################################
    #####################################################################
    # 6.7.1 :STATus:PRESet ##############################################
    def StatusGroupEventRegistersClear(self):
        """Clears all status group event registers.

        Presets the status group enables PTR and NTR registers as follows:
        ENABle = 0x0000, PTR = 0xffff, NTR = 0x0000
        :return:
        """
        self.EstablishedConnection.Write(f":STAT:PRES", error_check=True)

    # 6.7.3 Questionable Data Register Command Subsystem ################
    # 6.7.5 Voltage Status Subsystem ####################################
    # 6.7.6 Frequency Status Subsystem ##################################
    # 6.7.7 Sequence Status Subsystem ###################################
    # 6.7.8 DUC Status Subsystem ########################################
    # 6.7.9 Connection Status Subsystem #################################
    def QuestionableStatusQuery(self, Event=None, SubRegister=None):
        """Reads the event register in the questionable status group. It’s a
        read-only register. Once a bit is set, it remains set until cleared by
        this command or the *CLS command. A query of the register returns a
        decimal value which corresponds to the binary-weighted sum of all bits
        set in the register. :param Event: event register in the questionable
        status group.

        :param SubRegister: 'VOLT', 'FREQ', 'CONN', 'SEQ', or 'DUC'
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
        if SubRegister is None:
            if Event is None:
                self.EstablishedConnection.Query(f":STAT:QUES?", error_check=True)
            else:
                self.EstablishedConnection.Query(f":STAT:QUES:EVEN?", error_check=True)
        elif SubRegister in ("VOLT", "FREQ", "CONN", "SEQ", "DUC"):
            if Event is None:
                self.EstablishedConnection.Query(
                    f":STAT:QUES:{SubRegister}?", error_check=True
                )
            else:
                self.EstablishedConnection.Query(
                    f":STAT:QUES:{SubRegister}:EVEN?", error_check=True
                )
        else:
            raise Exception("M8195A: Invalid SubRegister in QuestionableStatusQuery")

    def QuestionableStatusConditionQuery(self, SubRegister=None):
        """Reads the condition register in the questionable status group. It’s
        a read-only register and bits are not cleared when you read the
        register. A query of the register returns a decimal value which
        corresponds to the binary-weighted sum of all bits set in the register.

        :param SubRegister: 'VOLT', 'FREQ', 'CONN', 'SEQ', or 'DUC'
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
        if SubRegister is None:
            self.EstablishedConnection.Query(f":STAT:QUES:COND?", error_check=True)
        elif SubRegister in ("VOLT", "FREQ", "CONN", "SEQ", "DUC"):
            self.EstablishedConnection.Query(
                f":STAT:QUES:{SubRegister}:COND?", error_check=True
            )
        else:
            raise Exception(
                "M8195A: Invalid SubRegister in QuestionableStatusConditionQuery"
            )

    def QuestionableStatusEnableSet(self, DecimalValue, SubRegister=None):
        """Sets the enable register in the questionable status group. The
        selected bits are then reported to the status Byte. A *CLS will not
        clear the enable register, but it does clear all bits in the event
        register. To enable bits in the enable register, you must write a
        decimal value which corresponds to the binary-weighted sum of the bits
        you wish to enable in the register. :param DecimalValue:

        :param SubRegister: 'VOLT', 'FREQ', 'CONN', 'SEQ', or 'DUC'
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
        if SubRegister is None:
            self.EstablishedConnection.Write(
                f":STAT:QUES:ENAB {DecimalValue}", error_check=True
            )
        elif SubRegister in ("VOLT", "FREQ", "CONN", "SEQ", "DUC"):
            self.EstablishedConnection.Write(
                f":STAT:QUES:{SubRegister}:ENAB {DecimalValue}", error_check=True
            )
        else:
            raise Exception(
                "M8195A: Invalid SubRegister in QuestionableStatusEnableSet"
            )

    def QuestionableStatusEnableQuery(self, SubRegister=None):
        """Queries the enable register in the questionable status group. The
        selected bits are then reported to the Status Byte. A *CLS will not
        clear the enable register, but it does clear all bits in the event
        register.

        :param SubRegister: 'VOLT', 'FREQ', 'CONN', 'SEQ', or 'DUC'
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
        if SubRegister is None:
            self.EstablishedConnection.Query(f":STAT:QUES:ENAB?", error_check=True)
        elif SubRegister in ("VOLT", "FREQ", "CONN", "SEQ", "DUC"):
            self.EstablishedConnection.Query(
                f":STAT:QUES:{SubRegister}:ENAB?", error_check=True
            )
        else:
            raise Exception(
                "M8195A: Invalid SubRegister in QuestionableStatusEnableQuery"
            )

    def QuestionableStatusNegTransitionSet(self, Allow, SubRegister=None):
        """Sets the negative-transition register in the questionable status
        group. A negative transition filter allows event to be reported when a
        condition changes from true to false. Setting both positive/negative
        filters true allows an event to be reported anytime the condition
        changes. Clearing both filters disable event reporting. The contents of
        transition filters are unchanged by *CLS and *RST. :param Allow: True
        of False.

        :param SubRegister: 'VOLT', 'FREQ', 'CONN', 'SEQ', or 'DUC'
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
        if Allow in (True, False):
            if SubRegister is None:
                self.EstablishedConnection.Write(
                    f":STAT:QUES:NTR {Allow}", error_check=True
                )
            elif SubRegister in ("VOLT", "FREQ", "CONN", "SEQ", "DUC"):
                self.EstablishedConnection.Write(
                    f":STAT:QUES:{SubRegister}:NTR {Allow}", error_check=True
                )
            else:
                raise Exception(
                    "M8195A: Invalid SubRegister in QuestionableStatusNegTransitionSet"
                )
        else:
            raise Exception(
                "M8195A: Invalid Allow in QuestionableStatusNegTransitionSet"
            )

    def QuestionableStatusNegTransitionQuery(self, SubRegister=None):
        """Queries the negative-transition register in the questionable status
        group. A negative transition filter allows event to be reported when a
        condition changes from true to false. Setting both positive/negative
        filters true allows an event to be reported anytime the condition
        changes. Clearing both filters disable event reporting. The contents of
        transition filters are unchanged by *CLS and *RST.

        :param SubRegister: 'VOLT', 'FREQ', 'CONN', 'SEQ', or 'DUC'
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
        if SubRegister is None:
            self.EstablishedConnection.Query(f":STAT:QUES:NTR?", error_check=True)
        elif SubRegister in ("VOLT", "FREQ", "CONN", "SEQ", "DUC"):
            self.EstablishedConnection.Query(
                f":STAT:QUES:{SubRegister}:NTR?", error_check=True
            )
        else:
            raise Exception(
                "M8195A: Invalid SubRegister in QuestionableStatusNegTransitionQuery"
            )

    def QuestionableStatusPosTransitionSet(self, Allow, SubRegister=None):
        """Set the positive-transition register in the questionable status
        group. A positive transition filter allows an event to be reported when
        a condition changes from false to true. Setting both positive/negative
        filters true allows an event to be reported anytime the condition
        changes. Clearing both filters disable event reporting. The contents of
        transition filters are unchanged by *CLS and *RST. :param Allow: True
        or False.

        :param SubRegister: 'VOLT', 'FREQ', 'CONN', 'SEQ', or 'DUC'
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
        if Allow in (True, False):
            if SubRegister is None:
                self.EstablishedConnection.Write(
                    f":STAT:QUES:PTR {Allow}", error_check=True
                )
            elif SubRegister in ("VOLT", "FREQ", "CONN", "SEQ", "DUC"):
                self.EstablishedConnection.Write(
                    f":STAT:QUES:{SubRegister}:PTR", error_check=True
                )
            else:
                raise Exception(
                    "M8195A: Invalid SubRegister in QuestionableStatusPosTransitionSet"
                )
        else:
            raise Exception(
                "M8195A: Invalid Allow in QuestionableStatusPosTransitionSet"
            )

    def QuestionableStatusPosTransitionQuery(self, SubRegister=None):
        """Queries the positive-transition register in the questionable status
        group. A positive transition filter allows an event to be reported when
        a condition changes from false to true. Setting both positive/negative
        filters true allows an event to be reported anytime the condition
        changes. Clearing both filters disable event reporting. The contents of
        transition filters are unchanged by *CLS and *RST.

        :param SubRegister: 'VOLT', 'FREQ', 'CONN', 'SEQ', or 'DUC'
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
        if SubRegister is None:
            self.EstablishedConnection.Query(f":STAT:QUES:PTR?", error_check=True)
        elif SubRegister in ("VOLT", "FREQ", "CONN", "SEQ", "DUC"):
            self.EstablishedConnection.Query(
                f":STAT:QUES:{SubRegister}:PTR?", error_check=True
            )
        else:
            raise Exception(
                "M8195A: Invalid SubRegister in QuestionableStatusPosTransitionQuery"
            )

    # 6.7.4 Operation Status Subsystem ##################################
    # 6.7.10 Run Status Subsystem #######################################
    def OperationStatusQuery(self, Event=None, SubRegister=None):
        """Reads the event register in the operation status group. It’s a read-
        only register. Once a bit is set, it remains set until cleared by this
        command or *CLS command. A query of the register returns a decimal
        value which corresponds to the binary-weighted sum of all bits set in
        the register. :param Event: event register in the questionable status
        group.

        :param SubRegister: 'RUN'
            - The Run Status register contains the run status conditions of the individual channels. Check
            "6.7.10 Run Status Subsystem" and "Table 31: Run status register" in Keysight M8195A AWG Revision 2
        :return:
        """
        if SubRegister is None:
            if Event is None:
                self.EstablishedConnection.Query(f":STAT:OPER?", error_check=True)
            else:
                self.EstablishedConnection.Query(f":STAT:OPER:EVEN?", error_check=True)
        elif SubRegister == "RUN":
            if Event is None:
                self.EstablishedConnection.Query(
                    f":STAT:OPER:{SubRegister}?", error_check=True
                )
            else:
                self.EstablishedConnection.Query(
                    f":STAT:OPER:{SubRegister}:EVEN?", error_check=True
                )
        else:
            raise Exception("M8195A: Invalid SubRegister in OperationStatusQuery")

    def OperationStatusConditionQuery(self, SubRegister=None):
        """Reads the condition register in the operation status group. It’s a
        read-only register and bits are not cleared when you read the register.
        A query of the register returns a decimal value which corresponds to
        the binary-weighted sum of all bits set in the register.

        :param SubRegister: 'RUN'
            - The Run Status register contains the run status conditions of the individual channels. Check
            "6.7.10 Run Status Subsystem" and "Table 31: Run status register" in Keysight M8195A AWG Revision 2
        :return:
        """
        if SubRegister is None:
            self.EstablishedConnection.Query(f":STAT:OPER:COND?", error_check=True)
        elif SubRegister == "RUN":
            self.EstablishedConnection.Query(
                f":STAT:OPER:{SubRegister}:COND?", error_check=True
            )
        else:
            raise Exception(
                "M8195A: Invalid SubRegister in OperationStatusConditionQuery"
            )

    def OperationStatusEnableSet(self, DecimalValue, SubRegister=None):
        """Sets the enable register in the operation status group. The selected
        bits are then reported to the Status Byte. A *CLS will not clear the
        enable register, but it does clear all bits in the event register. To
        enable bits in the enable register, you must write a decimal value
        which corresponds to the binary-weighted sum of the bits you wish to
        enable in the register. :param DecimalValue:

        :param SubRegister: 'RUN'
            - The Run Status register contains the run status conditions of the individual channels. Check
            "6.7.10 Run Status Subsystem" and "Table 31: Run status register" in Keysight M8195A AWG Revision 2
        :return:
        """
        if SubRegister is None:
            self.EstablishedConnection.Write(
                f":STAT:OPER:ENAB {DecimalValue}", error_check=True
            )
        elif SubRegister == "RUN":
            self.EstablishedConnection.Write(
                f":STAT:OPER:{SubRegister}:ENAB {DecimalValue}", error_check=True
            )
        else:
            raise Exception("M8195A: Invalid SubRegister in OperationStatusEnableSet")

    def OperationStatusEnableQuery(self, SubRegister=None):
        """Queries the enable register in the operation status group. The
        selected bits are then reported to the Status Byte. A *CLS will not
        clear the enable register, but it does clear all bits in the event
        register.

        :param SubRegister: 'RUN'
            - The Run Status register contains the run status conditions of the individual channels. Check
            "6.7.10 Run Status Subsystem" and "Table 31: Run status register" in Keysight M8195A AWG Revision 2
        :return:
        """
        if SubRegister is None:
            self.EstablishedConnection.Query(f":STAT:OPER:ENAB?", error_check=True)
        elif SubRegister == "RUN":
            self.EstablishedConnection.Query(
                f":STAT:OPER:{SubRegister}:ENAB?", error_check=True
            )
        else:
            raise Exception("M8195A: Invalid SubRegister in OperationStatusEnableQuery")

    def OperationStatusNegTransitionSet(self, Allow, SubRegister=None):
        """Sets the negative-transition register in the operation status group.
        A negative transition filter allows an event to be reported when a
        condition changes from true to false. Setting both positive/negative
        filters true allows an event to be reported anytime the condition
        changes. Clearing both filters disable event reporting. The contents of
        transition filters are unchanged by *CLS and *RST. :param Allow: True
        of False.

        :param SubRegister: 'RUN'
            - The Run Status register contains the run status conditions of the individual channels. Check
            "6.7.10 Run Status Subsystem" and "Table 31: Run status register" in Keysight M8195A AWG Revision 2
        :return:
        """
        if Allow in (True, False):
            if SubRegister is None:
                self.EstablishedConnection.Write(
                    f":STAT:OPER:NTR {Allow}", error_check=True
                )
            elif SubRegister == "RUN":
                self.EstablishedConnection.Write(
                    f":STAT:OPER:{SubRegister}:NTR {Allow}", error_check=True
                )
            else:
                raise Exception(
                    "M8195A: Invalid SubRegister in OperationStatusNegTransitionSet"
                )
        else:
            raise Exception("M8195A: Invalid Allow in OperationStatusNegTransitionSet")

    def OperationStatusNegTransitionQuery(self, SubRegister=None):
        """Queries the negative-transition register in the operation status
        group. A negative transition filter allows an event to be reported when
        a condition changes from true to false. Setting both positive/negative
        filters true allows an event to be reported anytime the condition
        changes. Clearing both filters disable event reporting. The contents of
        transition filters are unchanged by *CLS and *RST.

        :param SubRegister: 'RUN'
            - The Run Status register contains the run status conditions of the individual channels. Check
            "6.7.10 Run Status Subsystem" and "Table 31: Run status register" in Keysight M8195A AWG Revision 2
        :return:
        """
        if SubRegister is None:
            self.EstablishedConnection.Query(f":STAT:OPER:NTR?", error_check=True)
        elif SubRegister == "RUN":
            self.EstablishedConnection.Query(
                f":STAT:OPER:{SubRegister}:NTR?", error_check=True
            )
        else:
            raise Exception(
                "M8195A: Invalid SubRegister in OperationStatusNegTransitionQuery"
            )

    def OperationStatusPosTransitionSet(self, Allow, SubRegister=None):
        """Set the positive-transition register in the operation status group.
        A positive transition filter allows an event to be reported when a
        condition changes from false to true. Setting both positive/negative
        filters true allows an event to be reported anytime the condition
        changes. Clearing both filters disable event reporting. The contents of
        transition filters are unchanged by *CLS and *RST. :param Allow:

        :param SubRegister: 'RUN'
            - The Run Status register contains the run status conditions of the individual channels. Check
            "6.7.10 Run Status Subsystem" and "Table 31: Run status register" in Keysight M8195A AWG Revision 2
        :return:
        """
        if Allow in (True, False):
            if SubRegister is None:
                self.EstablishedConnection.Write(
                    f":STAT:OPER:PTR {Allow}", error_check=True
                )
            elif SubRegister == "RUN":
                self.EstablishedConnection.Write(
                    f":STAT:OPER:{SubRegister}:PTR", error_check=True
                )
            else:
                raise Exception(
                    "M8195A: Invalid SubRegister in OperationStatusPosTransitionSet"
                )
        else:
            raise Exception("M8195A: Invalid Allow in OperationStatusPosTransitionSet")

    def OperationStatusPosTransitionQuery(self, SubRegister=None):
        """Set the positive-transition register in the operation status group.
        A positive transition filter allows an event to be reported when a
        condition changes from false to true. Setting both positive/negative
        filters true allows an event to be reported anytime the condition
        changes. Clearing both filters disable event reporting. The contents of
        transition filters are unchanged by *CLS and *RST.

        :param SubRegister: 'RUN'
            - The Run Status register contains the run status conditions of the individual channels. Check
            "6.7.10 Run Status Subsystem" and "Table 31: Run status register" in Keysight M8195A AWG Revision 2
        :return:
        """
        if SubRegister is None:
            self.EstablishedConnection.Query(f":STAT:OPER:PTR?", error_check=True)
        elif SubRegister == "RUN":
            self.EstablishedConnection.Query(
                f":STAT:OPER:{SubRegister}:PTR?", error_check=True
            )
        else:
            raise Exception(
                "M8195A: Invalid SubRegister in OperationStatusPosTransitionQuery"
            )

    #####################################################################
    # 6.8 Arm/Trigger Subsystem #########################################
    #####################################################################
    def SignalGenerationChannelStop(self):
        """Stop signal generation on all channels.

        The channel suffix is ignored.
        :return:
        """
        self.EstablishedConnection.Write(f":ABOR", error_check=True)

    def ModuleDelaySet(self, Delay):
        """Set the module delay settings (see section 1.5.3) .

        The unit is in seconds.
        (This field specifies the module delay for all the channels. The range is 0 to 10 ns.)
        Parameter Suffix: [s|ms|us|ns|ps]
        :param Delay: 'MIN', 'MAX', 'MINimum', 'MAXimum'
        :return:
        """
        if isinstance(Delay, int):
            self.EstablishedConnection.Write(f":ARM:MDEL {Delay}", error_check=True)
        elif Delay in self._MinMaxList:
            self.EstablishedConnection.Write(f":ARM:MDEL {Delay}", error_check=True)
        else:
            raise Exception(
                "M8195A: Delay is neither integer nor proper string in ModuleDelaySet"
            )

    def ModuleDelayQuery(self):
        """Query the module delay settings (see section 1.5.3) .

        The unit is in seconds.
        (This field specifies the module delay for all the channels. The range is 0 to 10 ns.)
        Parameter Suffix: [s|ms|us|ns|ps]
        :return:
        """
        self.EstablishedConnection.Query(f":ARM:MDEL?", error_check=True)

    def SampleClockDelaySet(self, Delay, Channel):
        """Set the channel-specific sample delay in integral DAC sample clock
        periods.

        The range is 0..95
        (sample clock delay individually per channel as an integral number of DAC sample clocks. The range is 0..95
        DAC sample clocks.)
        DAC Sample Frequency: The DAC Sample frequency is always in the range of (53.76...65) GHz. As the DAC sample
        frequency references to a clock, the unit of the sample frequency is Hz.
        :param Channel: 1|2|3|4
        :param Delay:
        :return:
        """
        if Channel is None:
            self.EstablishedConnection.Write(f":ARM:SDEL {Delay}", error_check=True)
        elif Channel in self._ChannelList:
            if isinstance(Delay, int):
                self.EstablishedConnection.Write(
                    f":ARM:SDEL{Channel} {Delay}", error_check=True
                )
            elif Delay in self._MinMaxList:
                self.EstablishedConnection.Write(
                    f":ARM:SDEL{Channel} {Delay}", error_check=True
                )
            else:
                raise Exception(
                    "M8195A: Delay is neither integer nor proper string in ChannelSampleDelaySet"
                )
        else:
            raise Exception("M8195A: Invalid Channel in ChannelSampleDelaySet")

    def SampleClockDelayQuery(self, Channel):
        """Query the channel-specific sample delay in integral DAC sample clock
        periods.

        The range is 0..95
        :param Channel: 1|2|3|4
        :return:
        """
        self.EstablishedConnection.Query(f":ARM:SDEL{Channel}?", error_check=True)

    def ArmModeSet(self, ArmMode):
        """Set the arming mode.

        :param ArmMode: 'SELF', 'ARMed'
        :return:
        """
        ArmModeList = ("SELF", "ARMed")
        if ArmMode in ArmModeList:
            self.EstablishedConnection.Write(
                f":INIT:CONT:ENAB {ArmMode}", error_check=True
            )
        else:
            raise Exception("M8195A: Invalid ArmMode in ArmModeSet")

    def ArmModeQuery(self):
        """Query the arming mode.

        :return:
        """
        self.EstablishedConnection.Query(f":INIT:CONT:ENAB?", error_check=True)

    def ContinuousModeSet(self, ContinuousMode):
        """Set the continuous mode. This command must be used together with
        INIT:GATE to set the trigger mode. Check "6.8.6 :INITiate:GATE" and
        "Table 32: Trigger mode settings" for more info on the output.

        :param ContinuousMode:
            - 0/OFF – Continuous mode is off. If gate mode is off, the trigger mode is “triggered”, else it is “gated”.
            - 1/ON – Continuous mode is on. Trigger mode is “automatic”. The value of gate mode is not relevant.
        :return:
        """
        if ContinuousMode in self._OnList:
            self.EstablishedConnection.Write(f":INIT:CONT:STAT ON", error_check=True)
        elif ContinuousMode in self._OffList:
            self.EstablishedConnection.Write(f":INIT:CONT:STAT OFF", error_check=True)
        else:
            raise Exception("M8195A: Invalid ContinuousMode in ContinuousModeSet")

    def ContinuousModeQuery(self):
        """Query the continuous mode. This command must be used together with
        INIT:GATE to set the trigger mode. Check "6.8.6 :INITiate:GATE" and
        "Table 32: Trigger mode settings" for more info on the output.

        :return:
            - 0/OFF – Continuous mode is off. If gate mode is off, the trigger mode is “triggered”, else it is “gated”.
            - 1/ON – Continuous mode is on. Trigger mode is “automatic”. The value of gate mode is not relevant.
        """
        self.EstablishedConnection.Query(f":INIT:CONT:STAT?", error_check=True)

    def GateModeSet(self, GateMode):
        """Set the gate mode. This command must be used together with INIT:CONT
        to set the trigger mode. Check "6.8.6 :INITiate:GATE" and "Table 32:
        Trigger mode settings" for more info on the output.

        :param GateMode:
            - 0/OFF – Gate mode is off.
            - 1/ON – Gate mode is on. If continuous mode is off, the trigger mode is “gated”.
        :return:
        """
        if GateMode in self._OnList:
            self.EstablishedConnection.Write(f":INIT:GATE:STAT ON", error_check=True)
        elif GateMode in self._OffList:
            self.EstablishedConnection.Write(f":INIT:GATE:STAT OFF", error_check=True)
        else:
            raise Exception("M8195A: Invalid GateMode in GateModeSet")

    def GateModeQuery(self):
        """Query the gate mode. This command must be used together with
        INIT:CONT to set the trigger mode. Check "6.8.6 :INITiate:GATE" and
        "Table 32: Trigger mode settings" for more info on the output.

        :return:
            - 0/OFF – Gate mode is off.
            - 1/ON – Gate mode is on. If continuous mode is off, the trigger mode is “gated”.
        """
        self.EstablishedConnection.Query(f":INIT:GATE:STAT?", error_check=True)

    def SignalGenerationStart(self):
        """Start signal generation on all channels.

        The channel suffix is ignored. :INIT:IMM[1|2|3|4]
        :return:
        """
        self.EstablishedConnection.Write(f":INIT:IMM", error_check=True)

    def TriggerInputThresholdLevelSet(self, Level):
        """Set the trigger input threshold level.

        :param Level: Threshold level voltage
        :return:
        """
        if isinstance(Level, int):
            self.EstablishedConnection.Write(f":ARM:TRIG:LEV {Level}", error_check=True)
        elif Level in self._MinMaxList:
            self.EstablishedConnection.Write(f":ARM:TRIG:LEV {Level}", error_check=True)
        else:
            raise Exception(
                "M8195A: Level is neither integer nor proper string in TriggerInputThresholdLevelSet"
            )

    def TriggerInputThresholdLevelQuery(self):
        """Query the trigger input threshold level.

        :return:
        """
        self.EstablishedConnection.Query(f":ARM:TRIG:LEV?", error_check=True)

    def TriggerInputSlopeSet(self, Slope):
        """Set the trigger input slope.

        :param Slope:
            - POSitive: rising edge
            - NEGative: falling edge
            - EITHer: both
        :return:
        """
        SlopeList = ("POS", "POSitive", "NEG", "NEGative", "EITH", "EITHer")
        if Slope in SlopeList:
            self.EstablishedConnection.Write(
                f":ARM:TRIG:SLOP {Slope}", error_check=True
            )
        else:
            raise Exception("M8195A: Invalid Slope in TriggerInputSlopeSet")

    def TriggerInputSlopeQuery(self):
        """Query the trigger input slope.

        :return:
            - POSitive: rising edge
            - NEGative: falling edge
            - EITHer: both
        """
        self.EstablishedConnection.Query(f":ARM:TRIG:SLOP?", error_check=True)

    def TriggerFunctionSourceSet(self, Source):
        """Set the source for the trigger function.

        :param Source:
            - TRIGger: trigger input
            - EVENt: event input
            - INTernal: internal trigger generator
        :return:
        """
        SourceList = ("TRIG", "TRIGger", "EVEN", "EVENt", "INT", "INTernal")
        if Source in SourceList:
            self.EstablishedConnection.Write(
                f":ARM:TRIG:SOUR {Source}", error_check=True
            )
        else:
            raise Exception("M8195A: Invalid Source in TriggerFunctionSourceSet")

    def TriggerFunctionSourceQuery(self):
        """Query the source for the trigger function.

        :return:
            - TRIGger: trigger input
            - EVENt: event input
            - INTernal: internal trigger generator
        """
        self.EstablishedConnection.Query(f":ARM:TRIG:SOUR?", error_check=True)

    def InternalTriggerFrequencySet(self, Frequency):
        """Set the frequency of the internal trigger generator.

        :param Frequency: internal trigger frequency
        :return:
        """
        if isinstance(Frequency, int):
            self.EstablishedConnection.Write(
                f":ARM:TRIG:FREQ {Frequency}", error_check=True
            )
        elif Frequency in self._MinMaxList:
            self.EstablishedConnection.Write(
                f":ARM:TRIG:FREQ {Frequency}", error_check=True
            )
        else:
            raise Exception(
                "M8195A: Frequency is neither integer nor proper string in InternalTriggerFrequencySet"
            )

    def InternalTriggerFrequencyQuery(self):
        """Query the frequency of the internal trigger generator.

        :return:
        """
        self.EstablishedConnection.Query(f":ARM:TRIG:FREQ?", error_check=True)

    def InputTriggerEventOperationModeSet(self, OperationMode):
        """Set the operation mode for the trigger and event input.

        :param OperationMode:
            - ASYNchronous: asynchronous operation (see section 1.5.2)
            - SYNChronous: synchronous operation (see section 1.5.2)
        :return:
        """
        if OperationMode in ("ASYN", "ASYNchronous", "SYNC", "SYNChronous"):
            self.EstablishedConnection.Write(
                f":ARM:TRIG:OPER {OperationMode}", error_check=True
            )
        else:
            raise Exception(
                "M8195A: Invalid OperationMode in InternalTriggerFrequencySet"
            )

    def InputTriggerEventOperationModeQuery(self):
        """Query the operation mode for the trigger and event input.

        :return:
            - ASYNchronous: asynchronous operation (see section 1.5.2)
            - SYNChronous: synchronous operation (see section 1.5.2)
        """
        self.EstablishedConnection.Query(f":ARM:TRIG:OPER?", error_check=True)

    def EventInputThresholdLevelSet(self, Level):
        """Set the input threshold level.

        :param Level: Threshold level voltage
        :return:
        """
        if isinstance(Level, int):
            self.EstablishedConnection.Write(f":ARM:EVEN:LEV {Level}", error_check=True)
        elif Level in self._MinMaxList:
            self.EstablishedConnection.Write(f":ARM:EVEN:LEV {Level}", error_check=True)
        else:
            raise Exception(
                "M8195A: Level is neither integer nor proper string in InputThresholdLevelSet"
            )

    def EventInputThresholdLevelQuery(self):
        """Query the input threshold level.

        :return:
        """
        self.EstablishedConnection.Query(f":ARM:EVEN:LEV?", error_check=True)

    def EventInputSlopeSet(self, Slope):
        """Set the event input slope.

        :param Slope:
            - POSitive: rising edge
            - NEGative: falling edge
            - EITHer: both
        :return:
        """
        if Slope in ("POS", "POSitive", "NEG", "NEGative", "EITH", "EITHer"):
            self.EstablishedConnection.Write(
                f":ARM:EVEN:SLOP {Slope}", error_check=True
            )
        else:
            raise Exception("M8195A: Invalid Slope in EventInputSlopeSet")

    def EventInputSlopeQuery(self):
        """Query the event input slope.

        :return:
            - POSitive: rising edge
            - NEGative: falling edge
            - EITHer: both
        """
        self.EstablishedConnection.Query(f":ARM:EVEN:SLOP?", error_check=True)

    def EnableEventSourceSet(self, Source):
        """Set the source for the enable event.

        :param Source:
            - TRIGger: trigger input
            - EVENt: event input
        :return:
        """
        if Source in ("TRIG", "TRIGger", "EVEN", "EVENt"):
            self.EstablishedConnection.Write(
                f":TRIG:SOUR:ENAB {Source}", error_check=True
            )
        else:
            raise Exception("M8195A: Invalid Source in EnableEventSourceSet")

    def EnableEventSourceQuery(self):
        """Query the source for the enable event.

        :return:
            - TRIGger: trigger input
            - EVENt: event input
        """
        self.EstablishedConnection.Query(f":TRIG:SOUR:ENAB?", error_check=True)

    def HardwareInputDisableStateEnableFunctionSet(self, State):
        """Set the hardware input disable state for the enable function.

        When the hardware input is disabled, an enable event can only be
        generated using the
        :TRIGger[:SEQuence][:STARt]:ENABle[:IMMediate] command. When the
        hardware input is enabled, an enable event can be generated by
        command or by a signal present at the trigger or event input.
        :param State: 0|1|OFF|ON
        :return:
        """
        if State in self._OnList:
            self.EstablishedConnection.Write(f":TRIG:ENAB:HWD ON", error_check=True)
        elif State in self._OffList:
            self.EstablishedConnection.Write(f":TRIG:ENAB:HWD OFF", error_check=True)
        else:
            raise Exception(
                "M8195A: Invalid State in HardwareInputDisableStateEnableFunctionSet"
            )

    def HardwareInputDisableStateEnableFunctionQuery(self):
        """Query the hardware input disable state for the enable function.

        When the hardware input is disabled, an enable event can only be
        generated using the
        :TRIGger[:SEQuence][:STARt]:ENABle[:IMMediate] command. When the
        hardware input is enabled, an enable event can be generated by
        command or by a signal present at the trigger or event input.
        :return: OFF|ON
        """
        self.EstablishedConnection.Query(f":TRIG:ENAB:HWD?", error_check=True)

    def HardwareInputDisableStateTriggerFunctionSet(self, State):
        """Set the hardware input disable state for the trigger function.

        When the hardware input is disabled, a trigger can only be
        generated using the
        :TRIGger[:SEQuence][:STARt]:BEGin[:IMMediate] command. When the
        hardware input is enabled, a trigger can be generated by
        command, by a signal present at the trigger input or the
        internal trigger generator.
        :param State: 0|1|OFF|ON
        :return:
        """
        if State in self._OnList:
            self.EstablishedConnection.Write(f":TRIG:BEG:HWD ON", error_check=True)
        elif State in self._OffList:
            self.EstablishedConnection.Write(f":TRIG:BEG:HWD OFF", error_check=True)
        else:
            raise Exception(
                "M8195A: Invalid State in HardwareInputDisableStateTriggerFunctionSet"
            )

    def HardwareInputDisableStateTriggerFunctionQuery(self):
        """Query the hardware input disable state for the trigger function.

        When the hardware input is disabled, a trigger can only be
        generated using the
        :TRIGger[:SEQuence][:STARt]:BEGin[:IMMediate] command. When the
        hardware input is enabled, a trigger can be generated by
        command, by a signal present at the trigger input or the
        internal trigger generator.
        :return: OFF|ON
        """
        self.EstablishedConnection.Query(f":TRIG:BEG:HWD?", error_check=True)

    def HardwareInputDisableStateAdvanceFunctionSet(self, State):
        """Set the hardware input disable state for the advancement function.

        When the hardware input is disabled, an advancement event can
        only be generated using the
        :TRIGger[:SEQuence][:STARt]:ADVance[:IMMediate] command. When
        the hardware input is enabled, an advancement event can be
        generated by command or by a signal present at the trigger or
        event input.
        :param State: 0|1|OFF|ON
        :return:
        """
        if State in self._OnList:
            self.EstablishedConnection.Write(f":TRIG:ADV:HWD ON", error_check=True)
        elif State in self._OffList:
            self.EstablishedConnection.Write(f":TRIG:ADV:HWD OFF", error_check=True)
        else:
            raise Exception(
                "M8195A: Invalid State in HardwareInputDisableStateAdvanceFunctionSet"
            )

    def HardwareInputDisableStateAdvanceFunctionQuery(self):
        """Query the hardware input disable state for the advancement function.

        When the hardware input is disabled, an advancement event can
        only be generated using the
        :TRIGger[:SEQuence][:STARt]:ADVance[:IMMediate] command. When
        the hardware input is enabled, an advancement event can be
        generated by command or by a signal present at the trigger or
        event input.
        :return: OFF|ON
        """
        self.EstablishedConnection.Query(f":TRIG:ADV:HWD?", error_check=True)

    #####################################################################
    # 6.9 Trigger - Trigger Input #######################################
    #####################################################################
    def AdvanceEventSourceSet(self, Source):
        """Set the source for the advancement event.

        :param Source:
            - TRIGger: trigger input
            - EVENt: event input
            - INTernal: internal trigger generator
        :return:
        """
        if Source in ("TRIG", "TRIGger", "EVEN", "EVENt", "INT", "INTernal"):
            self.EstablishedConnection.Write(
                f":TRIG:SOUR:ADV {Source}", error_check=True
            )
        else:
            raise Exception("M8195A: Invalid Source in AdvanceEventSourceSet")

    def AdvanceEventSourceQuery(self):
        """Query the source for the advancement event.

        :return:
            - TRIGger: trigger input
            - EVENt: event input
            - INTernal: internal trigger generator
        """
        self.EstablishedConnection.Query(f":TRIG:SOUR:ADV?", error_check=True)

    def TriggerEnableEvent(self):
        """Send the enable event to a channel.

        :return:
        """
        self.EstablishedConnection.Write(f":TRIG:ENAB", error_check=True)

    def TriggerBeginEvent(self):
        """In triggered mode send the start/begin event to a channel.

        :return:
        """
        self.EstablishedConnection.Write(f":TRIG:BEG", error_check=True)

    def TriggerGatedModeSet(self, Stat):
        """In gated mode send a "gate open" (ON|1) or "gate close" (OFF|0) to a
        channel.

        :param Stat: OFF|ON|0|1
        :return:
        """
        if Stat in self._OffList:
            self.EstablishedConnection.Write(f":TRIG:BEG:GATE OFF", error_check=True)
        elif Stat in self._OnList:
            self.EstablishedConnection.Write(f":TRIG:BEG:GATE ON", error_check=True)
        else:
            raise Exception("M8195A: Invalid Stat in TriggerGatedModeSet")

    def TriggerGatedModeQuery(self):
        """In gated mode send a "gate open" (ON|1) or "gate close" (OFF|0) to a
        channel.

        :return: OFF|ON|0|1
        """
        self.EstablishedConnection.Query(f":TRIG:BEG:GATE?", error_check=True)

    def TriggerAdvanceEvent(self):
        """Send the advancement event to a channel.

        :return:
        """
        self.EstablishedConnection.Write(f":TRIG:ADV", error_check=True)

    #####################################################################
    # 6.10 Format Subsystem #############################################
    #####################################################################
    def FormatByteOrderSet(self, Order):
        """Byte ORDer. Controls whether binary data is transferred in normal
        (“big endian”) or swapped (“little endian”)

        byte order. Affects:
                            - [:SOURce]:STABle:DATA
                            - OUTPut:FILTer:FRATe
                            - OUTPut:FILTer:HRATe
                            - OUTPut:FILTer:QRATe
        :param Order: NORMal|SWAPped
        :return:
        """
        if Order in ("NORM", "NORMal", "SWAP", "SWAPped"):
            self.EstablishedConnection.Write(f":FORM:BORD {Order}", error_check=True)
        else:
            raise Exception("M8195A: Invalid Order in FormatByteOrderSet")

    def FormatByteOrderQuery(self):
        """Byte ORDer. Query whether binary data is transferred in normal (“big
        endian”) or swapped (“little endian”)

        byte order. Affects
                            - [:SOURce]:STABle:DATA
                            - OUTPut:FILTer:FRATe
                            - OUTPut:FILTer:HRATe
                            - OUTPut:FILTer:QRATe
        :return: NORMal|SWAPped
        """
        self.EstablishedConnection.Query(f":FORM:BORD?", error_check=True)

    #####################################################################
    # 6.11 Instrument Subsystem #########################################
    #####################################################################
    def InstrumentSlotNumberQuery(self):
        """Query the instrument’s slot number in its AXIe frame.

        :return:
        """
        self.EstablishedConnection.Query(f":INST:SLOT?", error_check=True)

    def InstrumentAccessLEDStart(self, Seconds=False):
        """Identify the instrument by flashing the green "Access" LED on the
        front panel for a certain time.

        :param Seconds: optional length of the flashing interval,
            default is 10 seconds.
        :return:
        """
        if isinstance(Seconds, int):
            self.EstablishedConnection.Write(f":INST:IDEN {Seconds}", error_check=True)
        elif Seconds is None:
            self.EstablishedConnection.Write(f":INST:IDEN", error_check=True)
        else:
            raise Exception("M8195A: Invalid Seconds in InstrumentAccessLEDStart")

    def InstrumentAccessLEDStop(self):
        """Stop the flashing of the green "Access" LED before the flashing
        interval has elapsed.

        :return:
        """
        self.EstablishedConnection.Write(f":INST:IDEN:STOP", error_check=True)

    def HardwareRevisionNumber(self):
        """Returns the M8195A hardware revision number.

        :return:
        """
        self.EstablishedConnection.Query(f":INST:HWR?", error_check=True)

    def DACOperationModeSet(self, DACMode):
        """Use this command to set the operation mode of the DAC.

        The value of the operation mode determines, to which channels
        waveforms can be transferred and the format of the waveform
        data. In operation mode SINGle, DUAL, DCDuplicate, or FOUR the
        data consists of 1-byte waveform samples only. In operation mode
        MARKer or DCMarker the data loaded to channel 1 consists of
        interleaved 1-byte waveform and 1-byte marker samples (see
        section :TRACe Subsystem). In operation mode DCDuplicate
        waveforms can only be loaded to channels 1 and 2.
        :param DACMode: SINGle – Channel 1 can generate a signal DUAL –
            Channels 1 and 4 can generate a signal, channels 2 and 3 are
            unused FOUR – Channels 1, 2, 3, and 4 can generate a signal
            MARKer – Channel 1 with two markers output on channel 3 and
            4 DCDuplicate – dual channel duplicate: Channels 1, 2, 3,
            and 4 can generate a signal. Channel 3 generates the same
            signal as channel 1. Channel 4 generates the same signal as
            channel 2. DCMarker – dual channel with marker: Channels 1
            and 2 can generate a signal. Channel 1 has two markers
            output on channel 3 and 4. Channel 2 can generate signals
            without markers.
        :return:
        """
        DACModeList = (
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
        if DACMode in DACModeList:
            self.EstablishedConnection.Write(f":INST:DACM {DACMode}", error_check=True)
        else:
            raise Exception("M8195A: Invalid DACMode in DACOperationMode")

    def DACOperationModeQuery(self):
        """Check DACOperationModeSet for more information :return:

        SINGle – Channel 1 can generate a signal DUAL – Channels 1 and 4
        can generate a signal, channels 2 and 3 are unused FOUR –
        Channels 1, 2, 3, and 4 can generate a signal MARKer – Channel 1
        with two markers output on channel 3 and 4 DCDuplicate – dual
        channel duplicate: Channels 1, 2, 3, and 4 can generate a
        signal. Channel 3 generates             the same signal as
        channel 1. Channel 4 generates the same signal as channel 2.
        DCMarker – dual channel with marker: Channels 1 and 2 can
        generate a signal. Channel 1 has two markers             output
        on channel 3 and 4. Channel 2 can generate signals without
        markers.
        """
        self.EstablishedConnection.Query(f":INST:DACM?", error_check=True)

    def ExtendedMemorySampleRateDividerSet(self, Divider):
        """Use this command to set the Sample Rate Divider of the Extended
        Memory.

        This value determines also the amount of available Extended
        Memory for each channel (see section 1.5.5).
        :param Divider: 1|2|4
        :return:
        """
        if Divider in (1, 2, 4):
            self.EstablishedConnection.Write(
                f":INST:MEM:EXT:RDIV DIV{Divider}", error_check=True
            )
        else:
            raise Exception(
                "M8195A: Invalid Divider in ExtendedMemorySampleRateDividerSet"
            )

    def ExtendedMemorySampleRateDividerQuery(self):
        """Use this query to get the Sample Rate Divider of the Extended
        Memory.

        This value determines also the amount of available Extended
        Memory for each channel (see section 1.5.5).
        :return: DIV1|DIV2|DIV4
        """
        self.EstablishedConnection.Query(f":INST:MEM:EXT:RDIV?", error_check=True)

    def MultiModuleConfigurationModeQuery(self):
        """This query returns the state of the multimodule configuration mode.

        :return: 0: disabled, 1: enabled
        """
        self.EstablishedConnection.Query(f":INST:MMOD:CONF?", error_check=True)

    def MultiModuleModeQuery(self):
        """This query returns the multi-module mode.

        :return:
            - NORMal: Module does not belong to a multi-module group.
            - SLAVe: Module is a slave in a multi-module group
        """
        self.EstablishedConnection.Query(f":INST:MMOD:MODE?", error_check=True)

    #####################################################################
    # 6.12 :Memory Subsystem ############################################
    #####################################################################
    # MMEM commands requiring <directory_name> assume the current directory if a relative path or no path is provided.
    # If an absolute path is provided, then it is ignored.
    def DiskUsageInformationQuery(self):
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
        self.EstablishedConnection.Query(f":MMEM:CAT?", error_check=True)

    def DefaultDirectorySet(self, Path):
        """Changes the default directory for a mass memory file system.

        The "path" parameter is a string. If no parameter is specified,
        the directory is set to the *RST value. At *RST, this value is
        set to the default user data storage area, that is defined as
        System.Environment.SpecialFolder.Personal: e.g.
        C:\\Users\\Name\\Documents :MMEM:CDIR
        "C:\\Users\\reza-\\Documents"
        :param Path:
        :return:
        """
        isDirectory = os.path.isdir(Path)
        if isDirectory is True:
            self.EstablishedConnection.Write(f":MMEM:CDIR {Path}", error_check=True)
        elif isDirectory is False:
            raise Exception("M8195A: Path is not a directory in DefaultDirectorySet")
        elif Path is None:
            RSTPath = r"""C:\Users\reza-\Documents"""
            self.EstablishedConnection.Write(f':MMEM:CDIR {RSTPath}', error_check=True)
            RSTPath = r"C:\Users\reza-\Documents"
            self.EstablishedConnection.Write(f":MMEM:CDIR {RSTPath}", error_check=True)
        else:
            raise Exception("M8195A: Unknown error in DefaultDirectorySet")

    def DefaultDirectoryQuery(self):
        """MMEMory:CDIRectory?

        — Query returns full path of the default directory.
        :return:
        """
        self.EstablishedConnection.Query(f":MMEM:CDIR?", error_check=True)

    def CopyFileOrDirSet(self, Src, Dst):
        """Copies an existing file to a new file or an existing directory to a
        new directory.

        Two forms of parameters are allowed. In this form, the first
        parameter specifies the source, and the second parameter
        specifies the destination. (<Source>,<Destination>)
        :param Src: File/Directory name in the source
        :param Dst: File/Directory name in the destination
        :return:
        """
        SrcIsFile = os.path.isfile(Src)
        DstIsFile = os.path.isfile(Dst)
        SrcIsDir = os.path.isdir(Src)
        DstIsDir = os.path.isdir(Dst)
        if ((SrcIsFile is True) and (DstIsFile is True)) or (
            (SrcIsDir is True) and (DstIsDir is True)
        ):
            self.EstablishedConnection.Write(
                f":MMEM:COPY {Src}, {Dst}", error_check=True
            )
        else:
            raise Exception(
                "M8195A: Src/Dst is(are) neither file(s) not directory(s) in CopyFileOrDirSet"
            )

    def CopyFileAndDirSet(self, SrcFile, SrcDir, DstFile, DstDir):
        """Copies an existing file to a new file or an existing directory to a
        new directory.

        Two forms of parameters are allowed. In this form, the first and
        third parameters specify the file names. The second and fourth
        parameters specify the directories. The first pair of parameters
        specifies the source. The second pair specifies the destination.
        An error is generated if the source doesn't exist or the
        destination file already exists. (<Source: file_name>,<Source:
        directory>,<Destination: file_name>,<Destination: directory>)
        :param SrcFile: File name in the source
        :param SrcDir: Directory of the source
        :param DstFile: File name in the destination
        :param DstDir: Directory of the destination
        :return:
        """
        SrcIsFile = os.path.isfile(SrcFile)
        DstIsFile = os.path.isfile(DstFile)
        SrcIsDir = os.path.isdir(SrcDir)
        DstIsDir = os.path.isdir(DstDir)
        if (SrcIsDir is True) and (DstIsDir is True):
            if (SrcIsFile is True) and (DstIsFile is True):
                self.EstablishedConnection.Write(
                    f":MMEM:COPY {SrcFile}, {SrcDir},{DstFile}, {DstDir}",
                    error_check=True,
                )
            else:
                raise Exception(
                    "M8195A: SrcFile/DstFile is (are) not file(s) in CopyFileAndDirSet"
                )
        else:
            raise Exception(
                "M8195A: SrcDir/DstDir is (are) not directory(s) in CopyFileAndDirSet"
            )

    def RemoveFileSet(self, File, Dir=None):
        """Removes a file from the specified directory.

        :param File: It specifies the file to be removed.
        :param Dir:
        :return:
        """
        isFile = os.path.isfile(File)
        if isFile is True:
            if Dir is None:
                self.EstablishedConnection.Write(f":MMEM:DEL {File}", error_check=True)
            else:
                isDir = os.path.isdir(Dir)
                if isDir is True:
                    self.EstablishedConnection.Write(
                        f":MMEM:DEL {File}, {Dir}", error_check=True
                    )
                else:
                    raise Exception("M8195A: Dir is not a directory in RemoveFileSet")
        else:
            raise Exception("M8195A: File is not a file in RemoveFileSet")

    def DataInFileLoad(self, File, Data):
        """The command form is MMEMory:DATA <File>,<Data>.

        It loads "Data" into the file "File".
        Regarding 488.2 block format:
        https://rfmw.em.keysight.com/wireless/helpfiles/n5106a/scpi_commands_mmem.htm
            #ABC:
                #: This character indicates the beginning of the data block.
                A: Number of decimal digits present in B.
                B: Decimal number specifying the number of data bytes to follow in C.
                C: Actual binary waveform data.
        :param File: "File" is string data.
        :param Data: "Data" is in 488.2 block format.
        :return:
        """
        isFile = os.path.isfile(File)
        if isFile is True:
            self.EstablishedConnection.Write(
                f":MMEM:DATA {File}, {Data}", error_check=True
            )
        else:
            raise Exception("M8195A: File is not a file in DataInFileLoad")

    def DataInFileQuery(self, File):
        """The query form is MMEMory:DATA?

        <File> with the response being the associated <data> in block
        format.
        :param File:
        :return: <data> in block format
        """
        isFile = os.path.isfile(File)
        if isFile is True:
            self.EstablishedConnection.Query(f":MMEM:DATA? {File}", error_check=True)
        else:
            raise Exception("M8195A: File is not a file in DataInFileQuery")

    def DirectoryCreate(self, Dir):
        """Creates a new directory.

        The <Dir> parameter specifies the name to be created.
        :param Dir:
        :return:
        """
        isDir = os.path.isdir(Dir)
        if isDir is True:
            self.EstablishedConnection.Write(f":MMEM:MDIR {Dir}", error_check=True)
        else:
            raise Exception("M8195A: Dir is not a Directory in DirectoryCreate")

    def MoveFileOrDirSet(self, Src, Dst):
        """Moves an existing file to a new file or an existing directory to a
        new directory.

        Two forms of parameters are allowed. In this form, the first
        parameter specifies the source, and the second parameter
        specifies the destination. (<Source>,<Destination>)
        :param Src: File/Directory name in the source
        :param Dst: File/Directory name in the destination
        :return:
        """
        SrcIsFile = os.path.isfile(Src)
        DstIsFile = os.path.isfile(Dst)
        SrcIsDir = os.path.isdir(Src)
        DstIsDir = os.path.isdir(Dst)
        if ((SrcIsFile is True) and (DstIsFile is True)) or (
            (SrcIsDir is True) and (DstIsDir is True)
        ):
            self.EstablishedConnection.Write(
                f":MMEM:MOVE {Src}, {Dst}", error_check=True
            )
        else:
            raise Exception(
                "M8195A: Src/Dst is(are) neither file(s) not directory(s) in MoveFileOrDirSet"
            )

    def MoveFileAndDirSet(self, SrcFile, SrcDir, DstFile, DstDir):
        """Moves an existing file to a new file or an existing directory to a
        new directory.

        Two forms of parameters are allowed. In this form, the first and
        third parameters specify the file names. The second and fourth
        parameters specify the directories. The first pair of parameters
        specifies the source. The second pair specifies the destination.
        An error is generated if the source doesn't exist or the
        destination file already exists. (<Source: file_name>,<Source:
        directory>,<Destination: file_name>,<Destination: directory>)
        :param SrcFile: File name in the source
        :param SrcDir: Directory of the source
        :param DstFile: File name in the destination
        :param DstDir: Directory of the destination
        :return:
        """
        SrcIsFile = os.path.isfile(SrcFile)
        DstIsFile = os.path.isfile(DstFile)
        SrcIsDir = os.path.isdir(SrcDir)
        DstIsDir = os.path.isdir(DstDir)
        if (SrcIsDir is True) and (DstIsDir is True):
            if (SrcIsFile is True) and (DstIsFile is True):
                self.EstablishedConnection.Write(
                    f":MMEM:MOVE {SrcFile}, {SrcDir},{DstFile}, {DstDir}",
                    error_check=True,
                )
            else:
                raise Exception(
                    "M8195A: SrcFile/DstFile is (are) not file(s) in MoveFileAndDirSet"
                )
        else:
            raise Exception(
                "M8195A: SrcDir/DstDir is (are) not directory(s) in MoveFileAndDirSet"
            )

    def DirectoryRemove(self, Dir):
        """Removes a directory.

        The <Dir> parameter specifies the directory name to be removed.
        All files and directories under the specified directory are also
        removed.
        :param Dir:
        :return:
        """
        isDir = os.path.isdir(Dir)
        if isDir is True:
            self.EstablishedConnection.Write(f":MMEM:RDIR {Dir}", error_check=True)
        else:
            raise Exception("M8195A: Dir is not a directory in DirectoryRemove")

    def InstrumentStateLoad(self, File):
        """Current state of instrument is loaded from a file.

        :param File:
        :return:
        """
        isFile = os.path.isfile(File)
        if isFile is True:
            self.EstablishedConnection.Write(f":MMEM:LOAD:CST {File}", error_check=True)
        else:
            raise Exception("M8195A: File is not a file in InstrumentStateLoad")

    def InstrumentStateSet(self, File):
        """Current state of instrument is stored to a file.

        :param File:
        :return:
        """
        isFile = os.path.isfile(File)
        if isFile is True:
            self.EstablishedConnection.Write(f":MMEM:STOR:CST {File}", error_check=True)
        else:
            raise Exception("M8195A: File is not a file in InstrumentStateSet")

    #####################################################################
    # 6.13 :OUTPut Subsystem ############################################
    #####################################################################
    def OutputAmplifierSet(self, Channel, State):
        """Switch the amplifier of the output path for a channel on or off.

        Check "Figure 14: Output tab" page 48 Keysight M8195A AWG
        Revision 2
        :param Channel: 1|2|3|4
        :param State: OFF|ON|0|1
        :return:
        """
        if Channel is None:
            if State in self._OnList:
                self.EstablishedConnection.Write(f":OUTP ON", error_check=True)
            elif State in self._OffList:
                self.EstablishedConnection.Write(f":OUTP OFF", error_check=True)
            else:
                raise Exception("M8195A: Invalid State in OutputAmplifierSet")
        elif Channel in self._ChannelList:
            if State in self._OnList:
                self.EstablishedConnection.Write(f":OUTP{Channel} ON", error_check=True)
            elif State in self._OffList:
                self.EstablishedConnection.Write(
                    f":OUTP{Channel} OFF", error_check=True
                )
            else:
                raise Exception("M8195A: Invalid State in OutputAmplifierSet")
        else:
            raise Exception("M8195A: Invalid Channel in OutputAmplifierSet")

    def OutputAmplifierQuery(self, Channel):
        """Query the amplifier of the output path for a channel on or off.

        Check "Figure 14: Output tab" page 48 Keysight M8195A AWG
        Revision 2
        :param Channel: 1|2|3|4
        :return:
        """
        if Channel in self._ChannelList:
            self.EstablishedConnection.Query(f":OUTP{Channel}?", error_check=True)
        else:
            raise Exception("M8195A: Invalid Channel in OutputAmplifierSet")

    def OutputClockSourceSet(self, Source):
        """Select which signal source is routed to the reference clock output.
        Check "Figure 13: Clock tab" page 46.

        Keysight M8195A AWG Revision 2
            - INTernal: the module internal reference oscillator (100 MHz)
            - EXTernal: the external reference clock from REF CLK IN with two variable dividers (Divider n and m)
            - SCLK1: DAC sample clock with variable divider and variable delay
            - SCLK2: DAC sample clock with fixed divider (32 and 8)
        :return:
        """
        if Source in ("INTernal", "INT", "EXTernal", "EXT", "SCLK1", "SCLK2"):
            self.EstablishedConnection.Write(
                f":OUTP:ROSC:SOUR {Source}", error_check=True
            )
        else:
            raise Exception("M8195A: Invalid Source in OutputClockSourceSet")

    def OutputClockSourceQuery(self):
        """Query which signal source is routed to the reference clock output.
        Check "Figure 13: Clock tab" page 46 Keysight M8195A AWG Revision 2.

        :return:
            - INTernal: the module internal reference oscillator (100 MHz)
            - EXTernal: the external reference clock from REF CLK IN with two variable dividers (Divider n and m)
            - SCLK1: DAC sample clock with variable divider and variable delay
            - SCLK2: DAC sample clock with fixed divider (32 and 8)
        """
        self.EstablishedConnection.Query(f":OUTP:ROSC:SOUR?", error_check=True)

    def DACSampleFreqDividerSet(self, Divider):
        """Set the divider of the DAC sample clock signal routed to the
        reference clock output.

        Check page 46 "Figure 13: Clock tab" Keysight M8195A AWG
        Revision 2
        :param Divider:
        :return:
        """
        if isinstance(Divider, int):
            self.EstablishedConnection.Write(
                f":OUTP:ROSC:SCD {Divider}", error_check=True
            )
        elif Divider in self._MinMaxList:
            self.EstablishedConnection.Write(
                f":OUTP:ROSC:SCD {Divider}", error_check=True
            )
        else:
            raise Exception(
                "M8195A: Divider is neither integer not string in DACSampleFreqDividerSet"
            )

    def DACSampleFreqDividerQuery(self):
        """Query the divider of the DAC sample clock signal routed to the
        reference clock output.

        Check page 46 "Figure 13: Clock tab" Keysight M8195A AWG
        Revision 2 :param
        :return: Divider
        """
        self.EstablishedConnection.Query(f":OUTP:ROSC:SCD?", error_check=True)

    def RefClockFreqDivider1Set(self, Divider1):
        """Set the first divider of the reference clock signal routed to the
        reference clock output.

        Check page 46 "Figure 13: Clock tab" Keysight M8195A AWG
        Revision 2
        :param Divider1:
        :return:
        """
        if isinstance(Divider1, int):
            self.EstablishedConnection.Write(
                f":OUTP:ROSC:RCD1 {Divider1}", error_check=True
            )
        elif Divider1 in self._MinMaxList:
            self.EstablishedConnection.Write(
                f":OUTP:ROSC:RCD1 {Divider1}", error_check=True
            )
        else:
            raise Exception(
                "M8195A: Divider1 is neither integer not string in RefClockFreqDivider1Set"
            )

    def RefClockFreqDivider1Query(self):
        """Query the first divider of the reference clock signal routed to the
        reference clock output.

        Check page 46 "Figure 13: Clock tab" Keysight M8195A AWG
        Revision 2
        :return:
        """
        self.EstablishedConnection.Query(f":OUTP:ROSC:RCD1?", error_check=True)

    def RefClockFreqDivider2Set(self, Divider2):
        """Set the first divider of the reference clock signal routed to the
        reference clock output.

        Check page 46 "Figure 13: Clock tab" Keysight M8195A AWG
        Revision 2
        :param Divider2:
        :return:
        """
        if isinstance(Divider2, int):
            self.EstablishedConnection.Write(
                f":OUTP:ROSC:RCD2 {Divider2}", error_check=True
            )
        elif Divider2 in self._MinMaxList:
            self.EstablishedConnection.Write(
                f":OUTP:ROSC:RCD2 {Divider2}", error_check=True
            )
        else:
            raise Exception(
                "M8195A: Divider2 is neither integer not string in RefClockFreqDivider2Set"
            )

    def RefClockFreqDivider2Query(self):
        """Set the first divider of the reference clock signal routed to the
        reference clock output.

        Check page 46 "Figure 13: Clock tab" Keysight M8195A AWG
        Revision 2
        :return:
        """
        self.EstablishedConnection.Query(f":OUTP:ROSC:RCD2?", error_check=True)

    def DifferentialOffset(self, Channel, Value):
        """
        Differential Offset: The hardware can compensate for little offset differences between the normal and
        complement output. “<Value>” is the offset to the calibrated optimum DAC value, so the minimum and maximum
        depend on the result of the calibration. Check below pages in Keysight M8195A AWG Revision 2:
            - page 49, "Figure 14: Output tab"
            - page 224, "Table 33: Differential offset"
        :param Channel: 1|2|3|4
        :param Value: <value>|MINimum|MAXimum
        :return:
        """
        if Channel is None:
            if isinstance(Value, int):
                self.EstablishedConnection.Write(
                    f":OUTP:DIOF {Value}", error_check=True
                )
            elif Value in self._MinMaxList:
                self.EstablishedConnection.Write(
                    f":OUTP:DIOF {Value}", error_check=True
                )
            else:
                raise Exception(
                    "M8195A: Value is neither integer nor string in DifferentialOffset"
                )
        elif Channel in self._ChannelList:
            if isinstance(Value, int):
                self.EstablishedConnection.Write(
                    f":OUTP{Channel}:DIOF {Value}", error_check=True
                )
            elif Value in self._MinMaxList:
                self.EstablishedConnection.Write(
                    f":OUTP{Channel}:DIOF {Value}", error_check=True
                )
            else:
                raise Exception(
                    "M8195A: Value is neither integer nor string in DifferentialOffset"
                )
        else:
            raise Exception("M8195A: Invalid Channel in DifferentialOffset")

    def DifferentialOffsetQuery(self, Channel):
        """
        Query the Differential Offset: The hardware can compensate for little offset differences between the normal and
        complement output. “<Value>” is the offset to the calibrated optimum DAC value, so the minimum and maximum
        depend on the result of the calibration. Check below pages in Keysight M8195A AWG Revision 2:
            - page 49, "Figure 14: Output tab"
            - page 224, "Table 33: Differential offset"
        :param Channel: 1|2|3|4
        :return: <value>|MINimum|MAXimum
        """
        if Channel in self._ChannelList:
            self.EstablishedConnection.Query(f":OUTP{Channel}:DIOF?", error_check=True)
        else:
            raise Exception("M8195A: Invalid Channel in DifferentialOffsetQuery")

    def SampleRateDivider(self, Value):
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
        :param Value: 1|2|4
        :return:
        """
        if Value == 1:
            Code = "FRAT"
        elif Value == 2:
            Code = "HRAT"
        elif Value == 4:
            Code = "QRAT"
        else:
            raise Exception("M8195A: Invalid Value in SampleRateDivider")
        return Code

    def FIRCoefficientSet(self, Channel, Divider, Value):
        """Set the FIR filter coefficients for a channel to be used when the
        Sample Rate Divider for the Extended Memory.

        is 1|2|4 ('FRAT', 'HRAT', 'QRAT'). The number of coefficients is 16|32|64 and the values are doubles between
        -2 and 2. The coefficients can only be set using this command, when the predefined FIR filter type is set to
        USER.
        The number of filter coefficients depends on the Sample Rate Divider; 16, 32, or 64 filter coefficients are
        available if the Sample Rate Divider is set to 1, 2 or, 4 respectively. In case the Sample Rate Divider is
        changed, the FIR filter coefficients of each channel sourced from extended memory are loaded to operate as a
        by one or by two or by four interpolation filter.
        :param Channel: 1|2|3|4
        :param Divider: 1|2|4
        :param Value: They can be given as a list of comma-separated values or as IEEE binary block data of doubles.
            1 -> <value0>, <value1>…<value15> |<block>
            2 -> <value0>, <value1>…<value31>|<block>
            4 -> <value0>, <value1>…<value63>|<block>
        :return:
        """
        Code = self.SampleRateDivider(Divider)
        if Channel in self._ChannelList:
            self.EstablishedConnection.Write(
                f":OUTP{Channel}:FILT:{Code}:{Value}", error_check=True
            )
        else:
            raise Exception("M8195A: Invalid Channel in FIRCoefficientSet")

    def FIRCoefficientQuery(self, Channel, Divider):
        """Get the FIR filter coefficients for a channel to be used when the
        Sample Rate Divider for the Extended Memory is 1|2|4 ('FRAT', 'HRAT',
        'QRAT').

        :param Channel: 1|2|3|4
        :param Divider: 1|2|4
        :return: FIR filter coefficients for a channel. The number of
            coefficients is: 16|32|64
        """
        Code = self.SampleRateDivider(Divider)
        if Channel in self._ChannelList:
            self.EstablishedConnection.Query(
                f":OUTP{Channel}:FILT:{Code}?", error_check=True
            )
        else:
            raise Exception("M8195A: Invalid Channel in FIRCoefficientQuery")

    def FIRTypeSet(self, Channel, Divider, Type):
        """Set the predefined FIR filter type for a channel to be used when the
        Sample Rate Divider for the Extended Memory is 1|2|4 ('FRAT', 'HRAT',
        'QRAT'). The command form modifies the FIR filter coefficients
        according to the set filter type, except for type USER. :param Channel:
        1|2|3|4 :param Divider: 1|2|4 :param Type:

            if Divider is 1:
                - LOWPass: equiripple lowpass filter with a passband edge at 75% of Nyquist
                - ZOH: Zero-order hold filter
                - USER: User-defined filter
            if Divider is 2|4:
                - NYQuist: Nyquist filter (half-band|quarter-band filter) with rolloff factor 0.2
                - LINear: Linear interpolation filter
                - ZOH: Zero-order hold filter
                - USER: User-defined filter
        :return:
        """
        Code = self.SampleRateDivider(Divider)
        if Channel in self._ChannelList:
            if Divider == 1:
                if Type in ("LOWPass", "LOWP", "ZOH", "USER"):
                    self.EstablishedConnection.Write(
                        f":OUTP:FILT:{Code}:TYPE {Type}", error_check=True
                    )
                else:
                    raise Exception("M8195A: Invalid Type for Divider=1 in FIRTypeSet")
            elif Divider == 2 or 4:
                if Type in ("NYQuist", "NYQ", "LINear", "ZOH", "USER"):
                    self.EstablishedConnection.Write(
                        f":OUTP:FILT:{Code}:TYPE {Type}", error_check=True
                    )
                else:
                    raise Exception(
                        "M8195A: Invalid Type for Divider=1|2 in FIRTypeSet"
                    )
            else:
                raise Exception("M8195A: Invalid Divider in FIRTypeSet")
        else:
            raise Exception("M8195A: Invalid Channel in FIRTypeSet")

    def FIRTypeQuery(self, Channel, Divider):
        """Get the predefined FIR filter type for a channel to be used when the
        Sample Rate Divider for the Extended Memory is 1|2|4 ('FRAT', 'HRAT',
        'QRAT'). :param Channel: 1|2|3|4 :param Divider: 1|2|4 :return: Type:

        if Divider is 1:
            - LOWPass: equiripple lowpass filter with a passband edge at 75% of Nyquist
            - ZOH: Zero-order hold filter
            - USER: User-defined filter
        if Divider is 2|4:
            - NYQuist: Nyquist filter (half-band|quarter-band filter) with rolloff factor 0.2
            - LINear: Linear interpolation filter
            - ZOH: Zero-order hold filter
            - USER: User-defined filter
        """
        Code = self.SampleRateDivider(Divider)
        if Channel in self._ChannelList:
            self.EstablishedConnection.Query(
                f":OUTP{Channel}:FILT:{Code}:TYPE?", error_check=True
            )
        else:
            raise Exception("M8195A: Invalid Channel in FIRTypeQuery")

    def FIRScalingFactorSet(self, Channel, Divider, Scale):
        """Set the FIR filter scaling factor for a channel to be used when the
        Sample Rate Divider for the Extended Memory is 1|2|4.

        The range is between 0 and 1.
        :param Channel: 1|2|3|4
        :param Divider: 1|2|4 ('FRAT', 'HRAT', 'QRAT')
        :param Scale: <scale>|MINimum|MAXimum
        :return:
        """
        Code = self.SampleRateDivider(Divider)
        if Channel in self._ChannelList:
            if 0 <= Scale <= 1:
                self.EstablishedConnection.Write(
                    f":OUTP{Channel}:FILT:{Code}:SCAL {Scale}", error_check=True
                )
            elif Scale in self._MinMaxList:
                self.EstablishedConnection.Write(
                    f":OUTP{Channel}:FILT:{Code}:SCAL {Scale}", error_check=True
                )
            else:
                raise Exception(
                    "M8195A: Scale is neither proper integer nor proper string in FIRScalingFactorSet"
                )
        else:
            raise Exception("M8195A: Invalid Channel in FIRScalingFactorSet")

    def FIRScalingFactorQuery(self, Channel, Divider):
        """Get the FIR filter scaling factor for a channel to be used when the
        Sample Rate Divider for the Extended Memory is 1|2|4.

        :param Channel: 1|2|3|4
        :param Divider: 1|2|4 ('FRAT', 'HRAT', 'QRAT')
        :return: The scale; the range is between 0 and 1 or
            MINimum|MAXimum.
        """
        Code = self.SampleRateDivider(Divider)
        if Channel in self._ChannelList:
            self.EstablishedConnection.Query(
                f":OUTP{Channel}:FILT:{Code}:SCAL?", error_check=True
            )
        else:
            raise Exception("M8195A: Invalid Channel in FIRScalingFactorQuery")

    def FIRDelaySet(self, Channel, Divider, Delay):
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
        :param Channel: 1|2|3|4
        :param Divider: 1|2|4 ('FRAT', 'HRAT', 'QRAT')
        :param Delay: <delay>|MINimum|MAXimum
        :return:
        """
        Code = self.SampleRateDivider(Divider)
        if Channel in self._ChannelList:
            if Divider == 1 and abs(Delay) > 50:
                raise Exception("M8195A: Invalid Delay for Divider=1 in FIRDelaySet")
            elif Divider == 2 and abs(Delay) > 100:
                raise Exception("M8195A: Invalid Delay for Divider=2 in FIRDelaySet")
            elif Divider == 4 and abs(Delay) > 200:
                raise Exception("M8195A: Invalid Delay for Divider=4 in FIRDelaySet")
            else:
                self.EstablishedConnection.Write(
                    f"OUTP{Channel}:FILT:{Code}:DEL {Delay}ps", error_check=True
                )
        else:
            raise Exception("M8195A: Invalid Channel in FIRDelaySet")

    def FIRDelayQuery(self, Channel, Divider):
        """Set the FIR filter delay for a channel to be used when the Sample
        Rate Divider for the Extended Memory is 1|2|4.

        :param Channel: 1|2|3|4
        :param Divider: 1|2|4 ('FRAT', 'HRAT', 'QRAT')
        :return: The delay range is:
            1 ('FRAT') -> -50ps..+50ps
            2 ('HRAT')-> -100ps..+100ps
            4 ('QRAT')-> -200ps..+200ps.
        The delay value has only effect for filter type:
            1 ('FRAT') -> LOWPass
            2, 4 ('HRAT', 'QRAT')-> NYQuist and LINear.
        """
        Code = self.SampleRateDivider(Divider)
        if Channel in self._ChannelList:
            self.EstablishedConnection.Query(
                f"OUTP{Channel}:FILT:{Code}:DEL?", error_check=True
            )
        else:
            raise Exception("M8195A: Invalid Channel in FIRDelayQuery")

    #####################################################################
    # 6.14 Sampling Frequency Commands ##################################
    #####################################################################
    def DACSampleFreqSet(self, Frequency):
        """Set the sample frequency of the output DAC.

        :param Frequency:
        :return:
        """
        if isinstance(Frequency, int):
            self.EstablishedConnection.Write(
                f":FREQ:RAST {Frequency}", error_check=True
            )
        elif Frequency in self._MinMaxList:
            self.EstablishedConnection.Write(
                f":FREQ:RAST {Frequency}", error_check=True
            )
        else:
            raise Exception(
                "M8195A: Frequency is neither integer nor proper string in DACSampleFreq"
            )

    def DACSampleFreqQuery(self):
        """Query the sample frequency of the output DAC.

        :return:
        """
        self.EstablishedConnection.Query(f":FREQ:RAST?", error_check=True)

    #####################################################################
    # 6.15 Reference Oscillator Commands ################################
    #####################################################################
    def ReferenceClockSourceSet(self, Source):
        """Set the reference clock source. Command not supported with Revision
        1 hardware. Check "Figure 13: Clock tab" page 46, Keysight M8195A AWG
        Revision 2.

        :param Source:
                    - EXTernal: reference is taken from REF CLK IN.
                    - AXI: reference is taken from AXI backplane.
                    - INTernal: reference is taken from module internal reference oscillator. May not be available with
                     every hardware.
        :return:
        """
        if Source in ("EXTernal", "EXT", "AXI", "INTernal", "INT"):
            self.EstablishedConnection.Write(f":ROSC:SOUR {Source}", error_check=True)
        else:
            raise Exception("M8195A: Invalid Source in ReferenceClockSourceSet")

    def ReferenceClockSourceQuery(self):
        """Query the reference clock source. Check "Figure 13: Clock tab" page
        46, Keysight M8195A AWG Revision 2.

        :return:
            - EXTernal: reference is taken from REF CLK IN.
            - AXI: reference is taken from AXI backplane.
            - INTernal: reference is taken from module internal reference oscillator. May not be available with every
                        hardware.
        """
        self.EstablishedConnection.Query(f":ROSC:SOUR?", error_check=True)

    def ReferenceClockSourceAvailability(self, Source):
        """Check if a reference clock source is available.

        Returns 1 if it is available and 0 if not.
        :param Source: EXTernal|AXI|INTernal
        :return: 1 if reference clock source is available and 0 if not.
        """
        if Source in ("EXTernal", "EXT", "AXI", "INTernal", "INT"):
            self.EstablishedConnection.Query(
                f":ROSC:SOUR:CHEC? {Source}", error_check=True
            )
        else:
            raise Exception(
                "M8195A: Invalid Source in ReferenceClockSourceAvailability"
            )

    def ExternalClockSourceFreqSet(self, Freq):
        """Set the expected reference clock frequency, if the external
        reference clock source is selected.

        :param Freq: <frequency>|MINimum|MAXimum
        :return:
        """
        if self.ReferenceClockSourceQuery() == "EXT" or "EXTernal":
            if isinstance(Freq, int):
                self.EstablishedConnection.Write(f":ROSC:FREQ {Freq}", error_check=True)
            elif Freq in self._MinMaxList:
                self.EstablishedConnection.Write(f":ROSC:FREQ {Freq}", error_check=True)
            else:
                raise Exception(
                    "M8195A: Frequency is neither integer nor proper string in ExternalClockSourceFreqSet"
                )
        else:
            raise Exception(
                'M8195A: Reference clock source is not "EXTernal" in ExternalClockSourceFreqSet'
            )

    def ExternalClockSourceFreqQuery(self):
        """Query the expected reference clock frequency, if the external
        reference clock source is selected.

        :return: Frequency (<frequency>|MINimum|MAXimum)
        """
        if self.ReferenceClockSourceQuery() == "EXT" or "EXTernal":
            self.EstablishedConnection.Query(f":ROSC:FREQ?", error_check=True)
        else:
            raise Exception(
                'M8195A: Reference clock source is not "EXTernal" in ExternalClockSourceFreqQuery'
            )

    def ExternalClockSourceRangeSet(self, Range):
        """Set the reference clock frequency range, if the external reference
        clock source is selected.

        :param Range:
                    - RANG1: 10…300 MHz
                    - RANG2: 210MHz…17GHz
        :return:
        """
        if self.ReferenceClockSourceQuery() == "EXT" or "EXTernal":
            if Range in ("RANG1", "RANG2"):
                self.EstablishedConnection.Write(
                    f":ROSC:RANG {Range}", error_check=True
                )
            else:
                raise Exception("M8195A: Invalid Range in ExternalClockSourceRangeSet")
        else:
            raise Exception(
                'M8195A: Reference clock source is not "EXTernal" in ExternalClockSourceRangeSet'
            )

    def ExternalClockSourceRangeQuery(self):
        """Query the reference clock frequency range, if the external reference
        clock source is selected.

        :return: Range:
                    - RANG1: 10…300 MHz
                    - RANG2: 210MHz…17GHz
        """
        if self.ReferenceClockSourceQuery() == "EXT" or "EXTernal":
            self.EstablishedConnection.Query(f":ROSC:RANG?", error_check=True)
        else:
            raise Exception(
                'M8195A: Reference clock source is not "EXTernal" in ExternalClockSourceRangeQuery'
            )

    def ExternalClockSourceRangeFreqSet(self, Range, Freq):
        """Set the reference clock frequency for a specific reference clock
        range.

        Current range remains unchanged.
        :param Range: RNG1|RNG2
        :param Freq: <frequency>|MINimum|MAXimum • RNG1: 10…300 MHz •
            RNG2: 210MHz…17GHz
        :return:
        """
        if self.ReferenceClockSourceQuery() == "EXT" or "EXTernal":
            if Range in ("RNG1", "RNG2"):
                if isinstance(Freq, int):
                    self.EstablishedConnection.Write(
                        f":ROSC:{Range}:FREQ {Freq}", error_check=True
                    )
                elif Freq in self._MinMaxList:
                    self.EstablishedConnection.Write(
                        f":ROSC:{Range}:FREQ {Freq}", error_check=True
                    )
                else:
                    raise Exception(
                        "M8195A: Frequency is neither integer nor proper string in "
                        "ExternalClockSourceRangeFreqSet"
                    )
            else:
                raise Exception(
                    "M8195A: Invalid Range in ExternalClockSourceRangeFreqSet"
                )
        else:
            raise Exception(
                'M8195A: Reference clock source is not "EXTernal" in ExternalClockSourceRangeFreqSet'
            )

    def ExternalClockSourceRangeFreqQuery(self, Range):
        """Query the reference clock frequency for a specific reference clock
        range.

        Current range remains unchanged.
        :param Range: RNG1|RNG2
        :return: Freq: <frequency>|MINimum|MAXimum • RNG1: 10…300 MHz •
            RNG2: 210MHz…17GHz
        """
        if self.ReferenceClockSourceQuery() == "EXT" or "EXTernal":
            if Range in ("RNG1", "RNG2"):
                self.EstablishedConnection.Query(
                    f":ROSC:{Range}:FREQ?", error_check=True
                )
            else:
                raise Exception(
                    "M8195A: Invalid Range in ExternalClockSourceRangeFreqSet"
                )
        else:
            raise Exception(
                'M8195A: Reference clock source is not "EXTernal" in ExternalClockSourceRangeFreqQuery'
            )

    #####################################################################
    # 6.16 :VOLTage Subsystem ###########################################
    #####################################################################

    def OutputAmplitudeSet(self, Channel, Level):
        """Set the output amplitude.

        :param Channel: 1|2|3|4
        :param Level: <level>|MINimum|MAXimum
        :return:
        """
        if Channel in self._ChannelList:
            if isinstance(Level, int):
                self.EstablishedConnection.Write(
                    f":VOLT{Channel} {Level}", error_check=True
                )
            elif Level in self._MinMaxList:
                self.EstablishedConnection.Write(
                    f":VOLT{Channel} {Level}", error_check=True
                )
            else:
                raise Exception(
                    "M8195A: Level is neither integer nor proper string in OutputAmplitudeSet"
                )
        else:
            raise Exception("M8195A: Invalid Channel in OutputAmplitudeSet")

    def OutputAmplitudeQuery(self, Channel):
        """Query the output amplitude.

        :param Channel: 1|2|3|4
        :return: Level: <level>|MINimum|MAXimum
        """
        if Channel in self._ChannelList:
            self.EstablishedConnection.Query(f":VOLT{Channel}?", error_check=True)
        else:
            raise Exception("M8195A: Invalid Channel in OutputAmplitudeSet")

    def OutputOffset(self, Channel, Offset):
        """Set the output offset.

        :param Channel: 1|2|3|4
        :param Offset:
        :return:
        """
        if Channel in self._ChannelList:
            if isinstance(Offset, int):
                self.EstablishedConnection.Write(
                    f":VOLT{Channel}:OFFS {Offset}", error_check=True
                )
            elif Offset in self._MinMaxList:
                self.EstablishedConnection.Write(
                    f":VOLT{Channel}:OFFS {Offset}", error_check=True
                )
            else:
                raise Exception(
                    "M8195A: Offset is neither integer nor proper string in OutputOffset"
                )
        else:
            raise Exception("M8195A: Invalid Channel in OutputOffset")

    def OutputOffsetQuery(self, Channel):
        """Query the output offset.

        :param Channel: 1|2|3|4
        :return: Offset
        """
        if Channel in self._ChannelList:
            self.EstablishedConnection.Query(f":VOLT{Channel}:OFFS?", error_check=True)
        else:
            raise Exception("M8195A: Invalid Channel in OutputOffsetQuery")

    def OutputHighLevelSet(self, Channel, HighLevel):
        """Set the output high level.

        :param Channel: 1|2|3|4
        :param HighLevel:
        :return:
        """
        if Channel in self._ChannelList:
            if isinstance(HighLevel, int):
                self.EstablishedConnection.Write(
                    f":VOLT{Channel}:HIGH {HighLevel}", error_check=True
                )
            elif HighLevel in self._MinMaxList:
                self.EstablishedConnection.Write(
                    f":VOLT{Channel}:HIGH {HighLevel}", error_check=True
                )
            else:
                raise Exception(
                    "M8195A: HighLevel is neither integer nor proper string in OutputHighLevelSet"
                )
        else:
            raise Exception("M8195A: Invalid Channel in OutputHighLevelSet")

    def OutputHighLevelQuery(self, Channel):
        """Query the output high level.

        :param Channel: 1|2|3|4
        :return: HighLevel
        """
        if Channel in self._ChannelList:
            self.EstablishedConnection.Query(f":VOLT{Channel}:HIGH?", error_check=True)
        else:
            raise Exception("M8195A: Invalid Channel in OutputHighLevelQuery")

    def OutputLowLevelSet(self, Channel, LowLevel):
        """Set the output low level.

        :param Channel: 1|2|3|4
        :param LowLevel:
        :return:
        """
        if Channel in self._ChannelList:
            if isinstance(LowLevel, int):
                self.EstablishedConnection.Write(
                    f":VOLT{Channel}:LOW {LowLevel}", error_check=True
                )
            elif LowLevel in self._MinMaxList:
                self.EstablishedConnection.Write(
                    f":VOLT{Channel}:LOW {LowLevel}", error_check=True
                )
            else:
                raise Exception(
                    "M8195A: LowLevel is neither integer nor proper string in OutputLowLevelSet"
                )
        else:
            raise Exception("M8195A: Invalid Channel in OutputLowLevelSet")

    def OutputLowLevelQuery(self, Channel):
        """Query the output low level.

        :param Channel: 1|2|3|4
        :return: LowLevel
        """
        if Channel in self._ChannelList:
            self.EstablishedConnection.Query(f":VOLT{Channel}:LOW?", error_check=True)
        else:
            raise Exception("M8195A: Invalid Channel in OutputLowLevelQuery")

    def TerminationVoltageSet(self, Channel, Level):
        """Set the termination voltage level.

        :param Channel: 1|2|3|4
        :param Level:
        :return:
        """
        if Channel in self._ChannelList:
            if isinstance(Level, int):
                self.EstablishedConnection.Write(
                    f":VOLT{Channel}:TERM {Level}", error_check=True
                )
            elif Level in self._MinMaxList:
                self.EstablishedConnection.Write(
                    f":VOLT{Channel}:TERM {Level}", error_check=True
                )
            else:
                raise Exception(
                    "M8195A: Level is neither integer nor proper string in TerminationVoltageSet"
                )
        else:
            raise Exception("M8195A: Invalid Channel in TerminationVoltageSet")

    def TerminationVoltageQuery(self, Channel):
        """Set the termination voltage level.

        :param Channel: 1|2|3|4
        :return: Level
        """
        if Channel in self._ChannelList:
            self.EstablishedConnection.Query(f":VOLT{Channel}:TERM?", error_check=True)
        else:
            raise Exception("M8195A: Invalid Channel in TerminationVoltageQuery")

    #####################################################################
    # 6.17 Source:Function:MODE #########################################
    #####################################################################
    def WaveformTypeSet(self, Type):
        """Use this command to set the type of waveform that will be generated
        on the channels that use the extended memory.

        The channels that use internal memory are always in ARBitrary
        mode.
        :param Type: [ARB, ARBitrary]: arbitrary waveform segment [STS,
            STSequence]: sequence [STSC, STSCenario]: scenario
        :return:
        """
        if Type in ("ARB", "ARBitrary", "STS", "STSequence", "STSC", "STSCenario"):
            self.EstablishedConnection.Write(f":FUNC:MODE {Type}", error_check=True)
        else:
            raise Exception("M8195A: Invalid Type in WaveformTypeSet")

    def WaveformTypeQuery(self):
        """Use this command to query the type of waveform that will be
        generated on the channels that use the extended memory.

        The channels that use internal memory are always in ARBitrary
        mode.
        :return: Type of waveform: [ARB, ARBitrary]: arbitrary waveform
            segment [STS, STSequence]: sequence [STSC, STSCenario]:
            scenario
        """
        self.EstablishedConnection.Query(f":FUNC:MODE?", error_check=True)

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
    def ResetAllSequenceTable(self):
        """Reset all sequence table entries to default values.

        :return:
        """
        self.EstablishedConnection.Write(f":STAB:RES", error_check=True)

    def SequenceDataSet(
        self,
        Index,
        SegmID,
        SegmAdvMode="SING",
        SeqAdvMode="SING",
        MarkerEnab=False,
        MarkerSeqInit=False,
        MarkerScenEnd=False,
        MarkerSeqEnd=False,
        SeqLoop=1,
        SegmLoop=1,
        SegmOffsetStart=0,
        SegmOffsetEnd="#hFFFFFFFF",
    ):
        """The command form writes directly into the sequencer memory. Writing
        is possible, when signal generation is stopped or when signal
        generation is started in dynamic mode. The sequencer memory has
        16,777,215 (16M – 1) entries. With this command entries can be directly
        manipulated using 6 32-bit words per entry. Individual entries or
        multiple entries at once can be manipulated. The data can be given in
        IEEE binary block format or in comma-separated list of 32-bit values.

        6 32-bit words: Control, Sequence Loop Count, Segment Loop Count, Segment ID, Segment Start Offset,
                        Segment End Offset

        :param Index: index of the sequence table entry to be accessed (16,777,215 (16M – 1) = 2**24-1 entries)
        :param SegmID:
                        7 bit (31:25) -> Reserved
                        25 bit (24:0) -> Segment id (1 .. 16M)
        Control:
            Reserved: 16 bit (15:0)
            :param SegmAdvMode: 4 bit (19:16),
                                0 or (0000): Auto
                                1 or (0001): Conditional
                                2 or (0010): Repeat
                                3 or (0011): Single
                                4 – 15 or (0100 - 1111): Reserved
            :param SeqAdvMode: 4 bit (23:20),
                                0 or (0000): Auto
                                1 or (0001): Conditional
                                2 or (0010): Repeat
                                3 or (0011): Single
                                4 – 15 or (0100 - 1111): Reserved
            :param MarkerEnab: 1 bit (24)
            :param MarkerSeqInit: 1 bit (28)
            :param MarkerScenEnd: 1 bit (29)
            :param MarkerSeqEnd: 1 bit (30)

        :param SeqLoop: 32 bit, Number of sequence iterations (1..4G-1), only applicable in the first entry of a
        sequence (31:0)
        :param SegmLoop: 32 bit, Number of segment iterations (1..4G-1) (31:0)
        :param SegmOffsetStart: 32 bit, Allows specifying a segment start address in samples, if only part of a segment
        loaded into waveform data memory is to be used. The value must be a multiple of twice the granularity of the
        selected waveform output mode.
        :param SegmOffsetEnd: 32 bit, Allows specifying a segment end address in samples, if only part of a segment
        loaded into waveform data memory is to be used. The value must obey the granularity of the selected waveform
        output mode. You can use the value ffffffff, if the segment end address equals the last sample in the segment.
            ffffffff = 4G-1 = 2**32-1 = 4,294,967,295
        :return:
        """
        Control = 0

        if SegmAdvMode == "AUTO":
            Control = Control
        elif SegmAdvMode == "COND":
            # 0001 = 65,536
            Control += 1 * (2**16)
        elif SegmAdvMode == "REP":
            # 0010 = 2 * (0001) = 131,072
            Control += 2 * (2**16)
        elif SegmAdvMode == "SING":
            # 0011 = 3 * (0001) = 196,608
            Control += 3 * (2**16)
        else:
            raise Exception("M8195A: Invalid segment advancement mode")

        if SeqAdvMode == "AUTO":
            Control = Control
        elif SeqAdvMode == "COND":
            # 0001 = 1,048,576
            Control += 1 * (2**20)
        elif SeqAdvMode == "REP":
            # 0010 = 2 * (0001) = 2,097,152
            Control += 2 * (2**20)
        elif SeqAdvMode == "SING":
            # 0011 = 3 * (0001) = 3,145,728
            Control += 3 * (2**20)
        else:
            raise Exception("M8195A: Invalid sequence advancement mode")

        if MarkerEnab is True:
            Control += 2**24

        if MarkerSeqInit is True:
            Control += 2**28

        if MarkerScenEnd is True:
            Control += 2**29

        if MarkerSeqEnd is True:
            Control += 2**30

        self.EstablishedConnection.Write(
            f":STAB:DATA {Index}, {Control}, {SeqLoop}, {SegmLoop}, {SegmID}, {SegmOffsetStart}, {SegmOffsetEnd}",
            error_check=True,
        )

    def SequenceIdleSet(self, Index, SeqLoop, IdleSample=1, IdleDelay=0):
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
        Idle Delay: The field is enabled only when the Entry Type is chosen as “Idle”. It is used to insert a numeric
        idle delay value into the sequence.
        Idle Sample: Idle Sample is the sample played during the pause time. The field is enabled only when the Entry
        Type is chosen as “Idle”. It is used to insert a numeric idle sample value into the sequence. In case of
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

        6 32-bit words: Control, Sequence Loop Count, Command Code, Idle Sample, Idle Delay, 0

        Fixed parameters:
        Control >> Data/command selection (Bit = 31), Table 35: Control, page 242
                - Bit = 31 (0: Data, 1: Command (type of command is selected by command code))
                - 2**31 = 2,147,483,648 (Decimal value), 0x80000000 (Hex value, equivalent to:
                            1000 0000 0000 0000 0000 0000 0000 0000)
        Command Code >> 16+16 bit, Reserved (31:16) + Command code (15:0). Table 41: Command Code, page 243
                - 0: Idle Delay

        Non-fixed parameters:
        :param Index: index of the sequence table entry to be accessed (16,777,215 (16M – 1) = 2**24-1 entries)
        :param SeqLoop: 32 bit, Number of sequence iterations (31:0). Table 36: Sequence loop count, page 242
                - 1..4G-1 = 2**32-1 = 4,294,967,295, only applicable in the first entry of a sequence.
        :param IdleSample: 24+8 bit, Reserved (31:8) + Sample to be played during pause (7:0).
                            Table 42: Idle sample, page 243
                - Bits 7:0 contain the DAC value.
        :param IdleDelay: 32 bit, Idle delay in Waveform Sample Clocks (31:0). Table 43: Idle delay, page 244
                - Sample Rate Divider | Min | Max
                    - 1 | 10*256 | (2**24-1)*256+255
                    - 2 | 10*128 | (2**24-1)*128+127
                    - 4 | 10*64 | (2**24-1)*64+63
        :return:
        """
        if (self.ExtendedMemorySampleRateDividerQuery() == 1) and (
            IdleDelay in range(10 * 256, (2**24 - 1) * 256 + 256)
        ):
            pass
        elif (self.ExtendedMemorySampleRateDividerQuery() == 2) and (
            IdleDelay in range(10 * 128, (2**24 - 1) * 128 + 128)
        ):
            pass
        elif (self.ExtendedMemorySampleRateDividerQuery() == 4) and (
            IdleDelay in range(10 * 64, (2**24 - 1) * 64 + 64)
        ):
            pass
        else:
            raise Exception("M8195A: Invalid IdleDelay in SequenceIdleSet")

        self.EstablishedConnection.Write(
            f"STAB:DATA {Index},2147483648,{SeqLoop},0,{IdleSample},{IdleDelay},0",
            error_check=True,
        )

    def SequenceDataQuery(self, Index, Length):
        """The query form reads the data from the sequencer memory, if all
        segments are read-write.

        The query returns an error, if at least one write-only segment
        in the waveform memory exists. Reading is only possible, when
        the signal generation is stopped. This query returns the same
        data as the “:STAB:DATA:BLOC?” query, but in comma-separated
        list of 32-bit values
        :param Index: <sequence_table_index>: index of the sequence
            table entry to be accessed
        :param Length: <length>: number of entries to be read
        :return: Return Data as comma-separated list of 32-bit values
        """
        self.EstablishedConnection.Query(
            f":STAB:DATA? {Index}, {Length}", error_check=True
        )

    def SequenceDataBinaryBlockFormatQuery(self, Index, Length):
        """The query form reads the data from the sequencer memory, if all
        segments are read-write.

        The query returns an error, if at least one write-only segment
        in the waveform memory exists. Reading is only possible, when
        the signal generation is stopped. This query returns the same
        data as the “:STAB:DATA?” query, but in IEEE binary block
        format.
        :param Index: <sequence_table_index>: index of the sequence
            table entry to be accessed
        :param Length: <length>: number of entries to be read
        :return: Return Data as IEEE binary block format
        """
        self.EstablishedConnection.Query(
            f":STAB:DATA:BLOC? {Index}, {Length}", error_check=True
        )

    def SequenceStartingIndexSet(self, Index):
        """Select where in the sequence table the sequence starts in STSequence
        mode.

        In dynamic sequence selection mode select the sequence that is
        played before the first sequence is dynamically selected.
        :param Index: <sequence_table_index>|MINimum|MAXimum
        :return:
        """
        if self.WaveformTypeQuery() == "STS" or "STSequence":
            if isinstance(Index, int):
                self.EstablishedConnection.Write(
                    f":STAB:SEQ:SEL {Index}", error_check=True
                )
            elif Index in self._MinMaxList:
                self.EstablishedConnection.Write(
                    f":STAB:SEQ:SEL {Index}", error_check=True
                )
            else:
                raise Exception(
                    "M8195A: Index is neither integer nor proper string in SequenceStartingIndexSet"
                )
        else:
            raise Exception(
                "M8195A: WaveformTypeQuery is not STSequence in SequenceStartingIndexSet"
            )

    def SequenceStartingIndexQuery(self):
        """Query where in the sequence table the sequence starts in STSequence
        mode.

        :return: Index -> <sequence_table_index>|MINimum|MAXimum
        """
        if self.WaveformTypeQuery() == "STS" or "STSequence":
            self.EstablishedConnection.Query(f":STAB:SEQ:SEL?", error_check=True)
        else:
            raise Exception(
                "M8195A: WaveformTypeQuery is not STSequence in SequenceStartingIndexQuery"
            )

    def SequenceExecutionStateAndIndexEntry(self):
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
        self.EstablishedConnection.Query(f":STAB:SEQ:STAT?", error_check=True)

    def DynamicModeSet(self, Mode):
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
        :param Mode: OFF|ON|0|1
        :return:
        """
        if Mode in self._OnList:
            self.EstablishedConnection.Write(f":STAB:DYN ON", error_check=True)
        elif Mode in self._OffList:
            self.EstablishedConnection.Write(f":STAB:DYN OFF", error_check=True)
        else:
            raise Exception("M8195A: Invalid Mode in DynamicModeSet")

    def DynamicModeQuery(self):
        """Use this command to Query whether the dynamic mode is enabled or
        disabled.

        Check the description in DynamicModeSet.
        :return:
        """
        self.EstablishedConnection.Query(f":STAB:DYN?", error_check=True)

    def DynamicStartingIndexSet(self, Index):
        """When the dynamic mode for segments or sequences is active, set the
        sequence table entry to be executed next.

        :param Index: <sequence_table_index>
        :return:
        """
        if self.DynamicModeQuery() is True:
            if isinstance(Index, int):
                self.EstablishedConnection.Write(
                    f":STAB:DYN:SEL {Index}", error_check=True
                )
            else:
                raise Exception("M8195A: Invalid Index in DynamicStartingIndexSet")
        else:
            raise Exception(
                "M8195A: DynamicModeQuery is not enabled in DynamicStartingIndexSet"
            )

    def ScenarioStartingIndexSet(self, Index):
        """Select where in the sequence table the scenario starts in STSCenario
        mode.

        :param Index: <sequence_table_index>|MINimum|MAXimum
        :return:
        """
        if self.WaveformTypeQuery() == "STSC" or "STSCenario":
            if isinstance(Index, int):
                self.EstablishedConnection.Write(
                    f":STAB:SCEN:SEL {Index}", error_check=True
                )
            elif Index in self._MinMaxList:
                self.EstablishedConnection.Write(
                    f":STAB:SCEN:SEL {Index}", error_check=True
                )
            else:
                raise Exception(
                    "M8195A: Index is neither integer nor proper string in ScenarioStartingIndexSet"
                )
        else:
            raise Exception(
                "M8195A: WaveformTypeQuery is not STSCenario in ScenarioStartingIndexSet"
            )

    def ScenarioStartingIndexQuery(self):
        """Query where in the sequence table the scenario starts in STSCenario
        mode.

        :return: Index: Sequence table index
        """
        if self.WaveformTypeQuery() == "STSC" or "STSCenario":
            self.EstablishedConnection.Query(f":STAB:SCEN:SEL?", error_check=True)
        else:
            raise Exception(
                "M8195A: WaveformTypeQuery is not STSCenario in ScenarioStartingIndexQuery"
            )

    def AdvancementModeScenarioSet(self, Mode):
        """Set the advancement mode for scenarios.

        :param Mode: AUTO | COND | REP | SING
        :return:
        """
        if Mode in ("AUTO", "COND", "REP", "SING"):
            self.EstablishedConnection.Write(f":STAB:SCEN:ADV {Mode}", error_check=True)
        else:
            raise Exception("M8195A: Invalid Mode in AdvancementModeScenarioSet")

    def AdvancementModeScenarioQuery(self):
        """Query the advancement mode for scenarios.

        :return: AUTO | COND | REP | SING
        """
        self.EstablishedConnection.Query(f":STAB:SCEN:ADV?", error_check=True)

    def ScenarioLoopCountSet(self, Count):
        """Set the loop count for scenarios.

        :param Count: <count>|MINimum|MAXimum
                        - <count> – 1..4G-1: number of times the scenario is repeated. (4G-1 = 2**32-1 = 4,294,967,295)
        :return:
        """
        if 1 <= Count <= ((2**32) - 1):
            self.EstablishedConnection.Write(
                f":STAB:SCEN:COUN {Count}", error_check=True
            )
        elif Count in self._MinMaxList:
            self.EstablishedConnection.Write(
                f":STAB:SCEN:COUN {Count}", error_check=True
            )
        else:
            raise Exception(
                "M8195A: Count is neither proper integer nor proper string in ScenarioLoopCountSet"
            )

    def ScenarioLoopCountQuery(self):
        """Query the loop count for scenarios.

        :return: <count>|MINimum|MAXimum
                        - <count> – 1..4G-1: number of times the scenario is repeated. (4G-1 = 2**32-1 = 4,294,967,295)
        """
        self.EstablishedConnection.Query(f":STAB:SCEN:COUN?", error_check=True)

    #####################################################################
    # 6.19 Frequency and Phase Response Data Access #####################
    #####################################################################
    def FreqAndPhaseRespDataQuery(self, Channel):
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

        :param Channel: 1|2|3|4
        :return:
                    - <amplitude> the output amplitude
                    - <sample_frequency> the sample frequency
        Format: The first three values are output frequency 1 in Hz, corresponding relative magnitude in linear scale,
        corresponding phase in radians. The next three values are output frequency 2, corresponding relative magnitude,
        corresponding phase, and so on.
        """
        if Channel in self._ChannelList:
            self.EstablishedConnection.Query(f":CHAR{Channel}?", error_check=True)
        else:
            raise Exception("M8195A: Invalid Channel in FreqAndPhaseRespDataQuery")

    #####################################################################
    # 6.21 :TRACe Subsystem #############################################
    #####################################################################
    # Use the :TRACe subsystem to control the arbitrary waveforms and their respective parameters:
    #               - Create waveform segments of arbitrary size with optional initialization.
    #               - Download waveform data with or without marker data into the segments.
    #               - Delete one or all waveform segments from the waveform memory.
    def WaveformMemorySourceSet(self, Channel, Source):
        """Use this command to set the source of the waveform samples for a
        channel.

        There are dependencies between
        this parameter, the same parameter for other channels, the memory sample rate divider and the instrument mode
        (number of channels). The tables in section 1.5.5 show the available combinations. The value of this parameter
        for each channel determines the target (Internal/Extended Memory) of the waveform transfer operation using the
        TRAC:DATA command.
        Note: It is recommended to set these parameters in one transaction. (Check above explanation)
        :param Channel: 1|2|3|4
        :param Source:
                        INTernal – the channel uses Internal Memory
                        EXTended – the channel uses Extended Memory
                        NONE – the channel is not used in this configuration (query only)
        :return:
        """
        if Channel in self._ChannelList:
            if Source in ("INT", "EXT", "INTernal", "EXTernal"):
                self.EstablishedConnection.Write(
                    f":TRAC{Channel}:MMOD {Source}", error_check=True
                )
            else:
                raise Exception("M8195A: Invalid Source in WaveformMemorySourceSet")
        else:
            raise Exception("M8195A: Invalid Channel in WaveformMemorySourceSet")

    def WaveformMemorySourceQuery(self, Channel):
        """Check description for 'def WaveformMemorySourceSet' :param Channel:
        1|2|3|4 :return: Source:

        INTernal – the channel uses Internal Memory EXTended – the
        channel uses Extended Memory NONE – the channel is not used in
        this configuration (query only)
        """
        if Channel in self._ChannelList:
            self.EstablishedConnection.Query(f":TRAC{Channel}:MMOD?", error_check=True)
        else:
            raise Exception("M8195A: Invalid Channel in ChannelMemoryModeQuery")

    def WaveformMemorySegmentSet(
        self, Channel, SegmID, Len, InitValue=0, WriteOnly=False
    ):
        """Use this command to define the size of a waveform memory segment.

        If InitValue is specified, all values in the segment are
        initialized. If not specified, memory is only allocated but not
        initialized. [Note] If the channel is sourced from Extended
        Memory, the same segment is defined on all other channels
        sourced from Extended Memory.
        :param Channel: Channel number (1|2|3|4)
        :param SegmID: ID of the segment
        :param Len: Length of the segment in samples, marker samples do
            not count
        :param InitValue: [Optional] optional initialization DAC value
        :param WriteOnly: The segment will be flagged write-only, so it
            cannot be read back or stored.
        :return:
        """
        if Channel in self._ChannelList and all(
            isinstance(i, int) for i in [SegmID, Len]
        ):
            if InitValue:
                if isinstance(InitValue, int):
                    if WriteOnly:
                        self.EstablishedConnection.Write(
                            f":TRAC{Channel}:DEF:WONL {SegmID},{Len},{InitValue}",
                            error_check=True,
                        )
                    else:
                        self.EstablishedConnection.Write(
                            f":TRAC{Channel}:DEF {SegmID},{Len},{InitValue}",
                            error_check=True,
                        )
                else:
                    raise Exception(
                        "M8195A: Invalid InitValue in WaveformMemorySegmentSet"
                    )
            else:
                if WriteOnly:
                    self.EstablishedConnection.Write(
                        f":TRAC{Channel}:DEF:WONL {SegmID},{Len}", error_check=True
                    )
                else:
                    self.EstablishedConnection.Write(
                        f":TRAC{Channel}:DEF {SegmID},{Len}", error_check=True
                    )
        else:
            raise Exception(
                "M8195A: Invalid (Channel, SegmID, Len) in WaveformMemorySegmentSet"
            )

    def WaveformMemoryNewSegmentSet(self, Channel, Len, InitValue=0, WriteOnly=False):
        """Use this query to define the size of a waveform memory segment.

        If InitValue is specified, all values in the segment are
        initialized. If not specified, memory is only allocated but not
        initialized. If the channel is sourced from Extended Memory, the
        same segment is defined on all other channels sourced from
        Extended Memory.
        :param Channel: Channel number (1|2|3|4)
        :param Len: length of the segment in samples, marker samples do
            not count
        :param InitValue: [Optional] optional initialization DAC value
        :param WriteOnly: The segment will be flagged write-only, so it
            cannot be read back or stored.
        :return: If the query was successful, a new SegmID will be
            returned.
        """
        if Channel in self._ChannelList and isinstance(Len, int):
            if InitValue:
                if isinstance(InitValue, int):
                    if WriteOnly:
                        self.EstablishedConnection.Query(
                            f":TRAC{Channel}:DEF:WONL:NEW? {Len},{InitValue}",
                            error_check=True,
                        )
                    else:
                        self.EstablishedConnection.Query(
                            f":TRAC{Channel}:DEF:NEW? {Len},{InitValue}",
                            error_check=True,
                        )
                else:
                    raise Exception(
                        "M8195A: Invalid InitValue in WaveformMemoryNewSegmentSet"
                    )
            else:
                if WriteOnly:
                    self.EstablishedConnection.Query(
                        f":TRAC{Channel}:DEF:WONL:NEW? {Len}", error_check=True
                    )
                else:
                    self.EstablishedConnection.Query(
                        f":TRAC{Channel}:DEF:NEW? {Len}", error_check=True
                    )
        else:
            raise Exception(
                "M8195A: Invalid (Channel, Len) in WaveformMemoryNewSegmentSet"
            )

    def WaveformDataInMemorySet(self, Channel, SegmID, Offset, Value):
        """Use this command to load waveform data into the module memory. If
        SegmID is already filled with data, the new values overwrite the
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

        :param Channel: 1|2|3|4
        :param SegmID: ID of the segment
        :param Offset: offset from segment start in samples (marker samples do not count) to allow splitting the
        transfer in smaller portions
        :param Value:
            Block: waveform data samples in the data format described above in IEEE binary block format
            NumVal: waveform data samples in the data format described above in comma separated list format
        :return:
        """
        if all(isinstance(i, int) for i in [Channel, SegmID, Offset, Value]):
            self.EstablishedConnection.Write(
                f":TRAC{Channel}:DATA {SegmID},{Offset},{Value}", error_check=True
            )
        else:
            raise Exception(
                "M8195A: Invalid (Channel, SegmID, Offset, Value) in WaveformDataInMemorySet"
            )

    def WaveformDataInMemoryQuery(self, Channel, SegmID, Offset, Len, Bloc=False):
        """Check description for 'def WaveformDataInMemorySet'.

        :param Channel: 1|2|3|4
        :param SegmID: ID of the segment
        :param Offset: offset from segment start in samples (marker
            samples do not count) to allow splitting the transfer in
            smaller portions
        :param Len: number of samples to read in the query case
        :param Bloc: if True, returns the data as the “:TRAC:DATA?”
            query, but in IEEE binary block format.
        :return:
        """
        if all(isinstance(i, int) for i in [Channel, SegmID, Offset, Len]):
            if Bloc is False:
                self.EstablishedConnection.Query(
                    f":TRAC{Channel}:DATA? {SegmID},{Offset},{Len}", error_check=True
                )
            elif Bloc is True:
                self.EstablishedConnection.Query(
                    f":TRAC{Channel}:DATA:BLOC? {SegmID},{Offset},{Len}",
                    error_check=True,
                )
        else:
            raise Exception(
                "M8195A: Invalid (Channel, SegmID, Offset, Len) in WaveformDataInMemoryQuery"
            )

    def WaveformDataFromFileImport(
        self,
        Channel,
        SegmID,
        FileName,
        FileType,
        DataType,
        MarkerFlag,
        Padding,
        InitValue,
        IgnoreHeaderParameters,
    ):
        """Use this command to import waveform data from a file and write it to
        the waveform memory. You can fill an already existing segment or a new
        segment can also be created. This command can be used to import real-
        only waveform data as well as complex I/Q data. This command supports
        different file formats.

        :param Channel: 1|2|3|4
        :param SegmID: This is the number of the segment, into which the data will be written.
        :param FileName: This is the complete path of the file.
        :param FileType: TXT|BIN|BIN8|IQBIN|BIN6030|BIN5110|LICensed |MAT89600|DSA90000|CSV
        :param DataType: This parameter is only used, if the file contains complex I/Q data. It selects, if the values
        of I or Q are imported.
                            − IONLy: Import I values.
                            − QONLy: Import Q values.
                            − BOTH: Import I and Q values and up-convert them to the carrier frequency set by the
                            CARR:FREQ command. This selection is only supported for the LICensed file type.
        :param MarkerFlag: This flag is applicable to BIN5110 format only, which can either consists of full 16 bit DAC
        values without markers or 14 bit DAC values and marker bits in the 2 LSBs.
                            − ON|1: The imported data will be interpreted as 14 bit DAC values and marker bits in the
                            2 LSBs.
                            − OFF|0: The imported data will be interpreted as 16 bit DAC values without marker bits.
        :param Padding: This parameter is optional and specifies the padding type. The parameter is ignored for the
        LICensed file type.
                            - ALENgth: Automatically determine the required length of the segment. If the segment does
                            not exist, it is created. After execution the segment has exactly the length of the pattern
                            in file or a multiple of this length to fulfill granularity and minimum segment length
                            requirements. This is the default behavior.
                            − FILL: The segment must exist, otherwise an error is returned. If the pattern in the file
                            is larger than the defined segment length, excessive samples are ignored. If the pattern in
                            the file is smaller than the defined segment length, remaining samples are filled with the
                            value specified by the <InitValue> parameter.
        :param InitValue: This is an optional initialization value used when FILL is selected as padding type. For
        real-only formats this is a DAC value. For complex I/Q file formats this is the I-part or Q-part of an I/Q
        sample pair in binary format (int8). Defaults to 0 if not specified.
        :param IgnoreHeaderParameters: This flag is optional and used to specify if the header parameters from the
        file need to be set in the instrument or ignored. This flag is applicable to formats CSV and MAT89600, which
        can contain header parameters.
                            − ON|1: Header parameters will be ignored.
                            − OFF|0: Header parameters will be set. This is the default.
        :return:
        """

        if FileType in (
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
            raise Exception("M8195A: Invalid FileType in WaveformDataFromFileImport")

        if DataType in ("IONL", "IONLy", "QONL", "QONLy", "BOTH"):
            pass
        else:
            raise Exception("M8195A: Invalid DataType in WaveformDataFromFileImport")

        if MarkerFlag in self._OnList or self._OffList:
            pass
        else:
            raise Exception("M8195A: Invalid MarkerFlag in WaveformDataFromFileImport")

        if Padding in ("ALEN", "ALENgth", "FILL"):
            pass
        else:
            raise Exception("M8195A: Invalid Padding in WaveformDataFromFileImport")

        if IgnoreHeaderParameters in self._OnList or self._OffList:
            pass
        else:
            raise Exception("M8195A: Invalid MarkerFlag in WaveformDataFromFileImport")

        self.EstablishedConnection.Write(
            f":TRAC{Channel}:IMP {SegmID},{FileName},{FileType},{DataType},"
            f"{MarkerFlag},{Padding},{InitValue},{IgnoreHeaderParameters}",
            error_check=True,
        )

    def WaveformDataFromBinImport(self, Channel, SegmID, FileName):
        """Check description for WaveformDataFromFileImport :param Channel:
        1|2|3|4 :param SegmID:

        :param FileName:
        :return:
        """
        self.EstablishedConnection.Write(
            f":TRAC{Channel}:IMP {SegmID},{FileName}, BIN8, IONLY, ON, ALEN",
            error_check=True,
        )

    def FileImportScalingStateSet(self, Channel, State):
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
        :param Channel: 1|2|3|4
        :param State: OFF|ON|0|1
        :return:
        """
        if Channel in self._ChannelList:
            if State in self._OnList:
                self.EstablishedConnection.Write(
                    f":TRAC{Channel}:IMP:SCAL ON", error_check=True
                )
            elif State in self._OffList:
                self.EstablishedConnection.Write(
                    f":TRAC{Channel}:IMP:SCAL OFF", error_check=True
                )
            else:
                raise Exception("M8195A: Invalid State in FileImportScalingStateSet")
        else:
            raise Exception("M8195A: Invalid Channel in FileImportScalingStateSet")

    def FileImportScalingStateQuery(self, Channel):
        """Query the scaling state for the file import.

        Read the description for FileImportScalingStateSet
        :param Channel: 1|2|3|4
        :return:
        """
        self.EstablishedConnection.Query(
            f":TRAC{Channel}:IMP:SCAL:STAT?", error_check=True
        )

    def DeleteSegment(self, Channel, SegmID):
        """Delete a segment.

        The command can only be used in program mode. If the channel is
        sourced from Extended Memory, the same segment is deleted on all
        other channels sourced from Extended Memory.
        :param Channel: 1|2|3|4
        :param SegmID:
        :return:
        """
        if Channel in self._ChannelList:
            if isinstance(SegmID, int):
                self.EstablishedConnection.Write(
                    f":TRAC{Channel}:DEL {SegmID}", error_check=True
                )
            else:
                raise Exception("M8195A: Invalid SegmID (segment ID) in DeleteSegment")
        else:
            raise Exception("M8195A: Invalid Channel in DeleteSegment")

    def DeleteAllSegment(self, Channel):
        """Delete all segments.

        The command can only be used in program mode. If the channel is
        sourced from Extended Memory, the same segment is deleted on all
        other channels sourced from Extended Memory.
        :param Channel: 1|2|3|4
        :return:
        """
        if Channel in self._ChannelList:
            self.EstablishedConnection.Write(
                f":TRAC{Channel}:DEL:ALL", error_check=True
            )
        else:
            raise Exception("M8195A: Invalid Channel in DeleteAllSegment")

    def SegmentsIDLengthQuery(self, Channel):
        """The query returns a comma-separated list of segment-ids that are
        defined and the length of each segment.

        So
        first number is a segment id, next length ...
        If no segment is defined, “0, 0” is returned.
        :param Channel: 1|2|3|4
        :return:
        """
        if Channel in self._ChannelList:
            self.EstablishedConnection.Query(f":TRAC{Channel}:CAT?", error_check=True)
        else:
            raise Exception("M8195A: Invalid Channel in SegmentsIDLengthQuery")

    def MemorySpaceWaveformData(self, Channel):
        """
        The query returns the amount of memory space available for waveform data in the following form:
        <bytes available>, <bytes in use>, < contiguous bytes available>.
        :param Channel: 1|2|3|4
        :return:
        """
        if Channel in self._ChannelList:
            self.EstablishedConnection.Query(f":TRAC{Channel}:FREE?", error_check=True)
        else:
            raise Exception("M8195A: Invalid Channel in MemorySpaceWaveformData")

    def SegmentNameSet(self, Channel, SegmID, Name):
        """This command associates a name to a segment.

        :param Channel: 1|2|3|4
        :param SegmID: the number of the segment
        :param Name: string of at most 32 characters
        :return:
        """
        if Channel in self._ChannelList:
            if isinstance(SegmID, int):
                if isinstance(Name, str) and len(Name) <= 32:
                    self.EstablishedConnection.Write(
                        f":TRAC{Channel}:NAME {SegmID},{Name}", error_check=True
                    )
                else:
                    raise Exception(
                        "M8195A: Invalid Name (not string or improper length) in SegmentNameSet"
                    )
            else:
                raise Exception("M8195A: Invalid SegmID (Segment ID) in SegmentNameSet")
        else:
            raise Exception("M8195A: Invalid Channel in SegmentNameSet")

    def SegmentNameQuery(self, Channel, SegmID):
        """The query gets the name for a segment.

        :param Channel: 1|2|3|4
        :param SegmID: the number of the segment
        :return:
        """
        if Channel in self._ChannelList:
            if isinstance(SegmID, int):
                self.EstablishedConnection.Query(
                    f":TRAC{Channel}:NAME? {SegmID}", error_check=True
                )
            else:
                raise Exception(
                    "M8195A: Invalid SegmID (Segment ID) in SegmentNameQuery"
                )
        else:
            raise Exception("M8195A: Invalid Channel in SegmentNameQuery")

    def SegmentCommentSet(self, Channel, SegmID, Comment):
        """This command associates a comment to a segment.

        :param Channel: 1|2|3|4
        :param SegmID: the number of the segment
        :param Comment: string of at most 256 characters
        :return:
        """
        if Channel in self._ChannelList:
            if isinstance(SegmID, int):
                if isinstance(Comment, str) and len(Comment) <= 256:
                    self.EstablishedConnection.Write(
                        f":TRAC{Channel}:COMM {SegmID}, {Comment}", error_check=True
                    )
                else:
                    raise Exception(
                        "M8195A: Invalid Comment (not string or improper length) in SegmentCommentSet"
                    )
            else:
                raise Exception(
                    "M8195A: Invalid SegmID (Segment ID) in SegmentCommentSet"
                )
        else:
            raise Exception("M8195A: Invalid Channel in SegmentCommentSet")

    def SegmentCommentQuery(self, Channel, SegmID):
        """The query gets the comment for a segment.

        :param Channel: 1|2|3|4
        :param SegmID: the number of the segment
        :return:
        """
        if Channel in self._ChannelList:
            if isinstance(SegmID, int):
                self.EstablishedConnection.Query(
                    f":TRAC{Channel}:COMM? {SegmID}", error_check=True
                )
            else:
                raise Exception(
                    "M8195A: Invalid SegmID (Segment ID) in SegmentCommentQuery"
                )
        else:
            raise Exception("M8195A: Invalid Channel in SegmentCommentQuery")

    def SegmentSelectSet(self, Channel, SegmID):
        """Selects the segment, which is output by the instrument in arbitrary
        function mode.

        The command has only effect, If the channel is sourced from
        Extended Memory. In this case the same value is used for all
        other channels sourced from Extended Memory.
        :param Channel: 1|2|3|4
        :param SegmID: the number of the segment,
            <segment_id>|MINimum|MAXimum
        :return:
        """
        if Channel in self._ChannelList:
            if isinstance(SegmID, int):
                self.EstablishedConnection.Write(
                    f":TRAC{Channel}:SEL {SegmID}", error_check=True
                )
            elif SegmID in self._MinMaxList:
                self.EstablishedConnection.Write(
                    f":TRAC{Channel}:SEL {SegmID}", error_check=True
                )
            else:
                raise Exception(
                    "M8195A: SegmID is neither integer nor proper string in SegmentSelectSet"
                )
        else:
            raise Exception("M8195A: Invalid Channel in SegmentSelectSet")

    def SegmentSelectQuery(self, Channel):
        """Query the selected segment, which is output by the instrument in
        arbitrary function mode.

        The command has only effect, If the channel is sourced from
        Extended Memory. In this case the same value is used for all
        other channels sourced from Extended Memory.
        :param Channel: 1|2|3|4
        :return:
        """
        if Channel in self._ChannelList:
            self.EstablishedConnection.Query(f":TRAC{Channel}:SEL?", error_check=True)
        else:
            raise Exception("M8195A: Invalid Channel in SegmentSelectQuery")

    def AdvancementModeSegmentSet(self, Channel, Mode):
        """Use this command to set the advancement mode for the selected
        segment.

        The advancement mode is used, if the segment is played in
        arbitrary mode. The command has only effect, If the channel is
        sourced from Extended Memory. In this case the same value is
        used for all other channels sourced from Extended Memory.
        :param Channel: 1|2|3|4
        :param Mode: AUTO|COND|REP|SING
        :return:
        """
        if Channel in self._ChannelList:
            if Mode in ("AUTO", "COND", "REP", "SING"):
                self.EstablishedConnection.Write(
                    f":TRAC{Channel}:ADV {Mode}", error_check=True
                )
            else:
                raise Exception("M8195A: Invalid Mode in AdvancementModeSegmentSet")
        else:
            raise Exception("M8195A: Invalid Channel in AdvancementModeSegmentSet")

    def AdvancementModeSegmentQuery(self, Channel):
        """Use this Query to get the advancement mode for the selected segment.

        The advancement mode is used, if the segment is played in
        arbitrary mode. The command has only effect, If the channel is
        sourced from Extended Memory. In this case the same value is
        used for all other channels sourced from Extended Memory.
        :param Channel: 1|2|3|4
        :return: Mode -> AUTO|COND|REP|SING
        """
        if Channel in self._ChannelList:
            self.EstablishedConnection.Query(f":TRAC{Channel}:ADV?", error_check=True)
        else:
            raise Exception("M8195A: Invalid Channel in AdvancementModeSegmentQuery")

    def SelectedSegmentLoopCountSet(self, Channel, Count):
        """Use this command to set the segment loop count for the selected
        segment.

        The segment loop count is used, if the
        segment is played in arbitrary mode.
        The command has only effect, If the channel is sourced from Extended Memory. In this case the same value is
        used for all other channels sourced from Extended Memory.
        :param Channel: 1|2|3|4
        :param Count: number of times the selected segment is repeated (1..4G-1).
        :return:
        """
        if Channel in self._ChannelList:
            if isinstance(Count, int):
                self.EstablishedConnection.Write(
                    f":TRAC{Channel}:COUN {Count}", error_check=True
                )
            elif Count in self._MinMaxList:
                self.EstablishedConnection.Write(
                    f":TRAC{Channel}:COUN {Count}", error_check=True
                )
            else:
                raise Exception(
                    "M8195A: Count is neither integer nor proper string in SelectedSegmentLoopCountSet"
                )
        else:
            raise Exception("M8195A: Invalid Channel in SelectedSegmentLoopCountSet")

    def SelectedSegmentLoopCountQuery(self, Channel):
        """Use this Query to get the segment loop count for the selected
        segment.

        The segment loop count is used, if the
        segment is played in arbitrary mode.
        The command has only effect, If the channel is sourced from Extended Memory. In this case the same value is
        used for all other channels sourced from Extended Memory.
        :param Channel: 1|2|3|4
        :return: Count: number of times the selected segment is repeated (1..4G-1). (4G-1 = 2**32-1 = 4,294,967,295)
        """
        if Channel in self._ChannelList:
            self.EstablishedConnection.Query(f":TRAC{Channel}:COUN?", error_check=True)
        else:
            raise Exception("M8195A: Invalid Channel in SelectedSegmentLoopCountQuery")

    def SelectedSegmentMarkerStateSet(self, Channel, State):
        """Use this command to enable or disable markers for the selected
        segment.

        The command has only effect, If the channel is sourced from
        Extended Memory. In this case the same value is used for all
        other channels sourced from Extended Memory.
        :param Channel: 1|2|3|4
        :param State: OFF|ON|0|1
        :return:
        """
        if Channel in self._ChannelList:
            if State in self._OnList:
                self.EstablishedConnection.Write(
                    f":TRAC{Channel}:MARK ON", error_check=True
                )
            elif State in self._OffList:
                self.EstablishedConnection.Write(
                    f":TRAC{Channel}:MARK OFF", error_check=True
                )
            else:
                raise Exception(
                    "M8195A: Invalid State in SelectedSegmentMarkerStateSet"
                )
        else:
            raise Exception("M8195A: Invalid Channel in SelectedSegmentMarkerStateSet")

    def SelectedSegmentMarkerStateQuery(self, Channel):
        """The query form gets the current marker state.

        Read the description of SelectedSegmentMarkerStateSet
        :param Channel: 1|2|3|4
        :return: State: OFF|ON|0|1
        """
        if Channel in self._ChannelList:
            self.EstablishedConnection.Query(f":TRAC{Channel}:MARK?", error_check=True)
        else:
            raise Exception(
                "M8195A: Invalid Channel in SelectedSegmentMarkerStateQuery"
            )

    #####################################################################
    # 6.22 :TEST Subsystem ##############################################
    #####################################################################
    def SelfTestsPowerResultQuery(self):
        """Return the results of the power on self-tests.

        :return:
        """
        self.EstablishedConnection.Query(f":TEST:PON?", error_check=True)

    def SelfTestsPowerResultMessageQuery(self):
        """Same as *TST?

        but the actual test messages are returned. Currently same as
        :TEST:PON?
        :return:
        """
        self.EstablishedConnection.Query(f":TEST:TST?", error_check=True)


#################################


if __name__ == "__main__":
    M8195Connection(IPAddress="0.0.0.0", port=5025)  # IPAddress='0.0.0.0', port=5025
