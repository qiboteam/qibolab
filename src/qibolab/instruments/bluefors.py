## This code was created by Adam Anderson
## https://github.com/adamanderson/pybluefors/

import datetime
import json

import websocket  # pip install websocket-client


class TemperatureController:
    """
    Class to control a Bluefors temperature controller system. Currently written
    for compatibility with an LD-400 system, although it may be compatible with
    other models of Bluefors fridges.
    """

    def __init__(self, ip_address):
        """
        Constructor. Queries all heaters and thermometer channels at startup
        to obtain names and diagnostic information.

        Parameters
        ----------
        ip_address : str
            IP address of temperature controller device
        """
        self.ip_address = ip_address
        self.heaters_info = {}
        self.thermometers_info = {}

        # get info on heaters
        ws = websocket.create_connection("ws://{}:5002/heater".format(self.ip_address), timeout=10)
        for heater_chan in [1, 2, 3, 4]:
            ws.send(json.dumps({"heater_nr": heater_chan}))
            resp = ws.recv()
            data = json.loads(resp)
            self.heaters_info[data["name"]] = data
        ws.close()

        # get info on thermometers
        ws = websocket.create_connection("ws://{}:5002/channel".format(self.ip_address), timeout=10)
        for thermometer_chan in [1, 2, 3, 4, 5, 6, 7, 8]:
            ws.send(json.dumps({"channel_nr": thermometer_chan}))
            resp = ws.recv()
            data = json.loads(resp)
            if data["name"] not in self.thermometers_info:
                self.thermometers_info[data["name"]] = data
            else:
                num_chans = len([channame for channame in self.thermometers_info if data["name"] in channame])
                new_name = "{}_{}".format(data["name"], num_chans + 1)
                self.thermometers_info[new_name] = data
        ws.close()

    def get_data(self, channel, start_time=None, stop_time=None):
        """
        Get data from the temperature controller. If no start and stop times are specified,
        the function fetches only the most recent available temperature data.

        Parameters
        ----------
        channel : int or str
            Channel number or name for which to get data

        start_time : None or float or int
            Time at which to start reporting temperature data, in seconds since the
            epoch. If None, most recent data will be returned.

        stop_time : None or float or int
            Time at which to stop reporting temperature data, in seconds since the
            epoch. If None, data up to most recent will be returned.
        """
        # parse arguments
        if type(channel) is int:
            channel_num = channel
        elif type(channel) is str:
            channel_num = self.thermometers_info[channel]["channel_nr"]
        else:
            raise ValueError("Invalid argument type: channel. Must be int or str.")

        if start_time is None:
            start_time = datetime.datetime.timestamp(datetime.datetime.now() - datetime.timedelta(minutes=5))
            return_most_recent = True
        elif not (isinstance(start_time, float) or isinstance(start_time, int)):
            raise ValueError("Invalid argument type: start_time. Must be None or int or float (seconds since epoch).")
        else:
            return_most_recent = False

        if stop_time is None:
            stop_time = datetime.datetime.timestamp(datetime.datetime.now())
        elif not (isinstance(stop_time, float) or isinstance(stop_time, int)):
            raise ValueError("Invalid argument type: stop_time. Must be None or int or float (seconds since epoch).")

        ws = websocket.create_connection("ws://{}:5002/channel/historical-data".format(self.ip_address), timeout=10)
        ws.send(
            json.dumps(
                {
                    "channel_nr": channel_num,
                    "start_time": datetime.datetime.utcfromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
                    "stop_time": datetime.datetime.utcfromtimestamp(stop_time).strftime("%Y-%m-%d %H:%M:%S"),
                    "fields": ["timestamp", "resistance", "temperature"],
                }
            )
        )
        resp = ws.recv()
        data = json.loads(resp)

        if return_most_recent:
            # throw out data from all but the most recent measurement
            for field in ["timestamp", "resistance", "temperature"]:
                data["measurements"][field] = data["measurements"][field][-1]

        ws.close()
        return data

    def set_heater(
        self,
        channel,
        active=None,
        pid_mode=None,
        power=None,
        max_power=None,
        setpoint=None,
        control_algorithm_settings=None,
    ):
        """
        Change the heater settings.

        Parameters
        ----------
        channel : int or str
            Channel number or name for which to get data
        active : bool
            Set whether heater is active
        pid_mode : int
            0 : manual mode
            1 : PID mode
        power : float
            Manual power to apply, in Watts
        max_power : float
            Max power to apply, useful during PID regulation, in Watts
        setpoint : float
            PID setpoint in units of Kelvin (?)
        control_algorithm_settings : dict
            Proportional, integral, and derivative terms for the PID
            controller. Be sure to read the Bluefors docs when setting this!
            The argument must be of the form:

            {'proportional': 0.04
             'integral': 150
             'derivative': 0}

            where the values above should be reasonable for regulating between
            20 and 100mK.
        """
        # parse arguments
        args_dict = locals()

        if type(channel) is int:
            channel_num = channel
        elif type(channel) is str:
            channel_num = self.heaters_info[channel]["heater_nr"]
        else:
            raise ValueError("Invalid argument type.")

        settings_dict = {"heater_nr": channel_num}
        args_dict.pop("self")
        args_dict.pop("channel")
        for arg in args_dict:
            if arg != "channel" and args_dict[arg] is not None:
                settings_dict[arg] = args_dict[arg]

        ws = websocket.create_connection("ws://{}:5002/heater/update".format(self.ip_address), timeout=10)
        ws.send(json.dumps(settings_dict))
        ws.close()
