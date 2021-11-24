from tkinter import *
import json, os
import qibolab

platform_file = os.path.dirname(qibolab.__file__) + '/platforms/tii_single_qubit_config.json'

class ConfigWindow:
    def __init__(self, window):

        config = open(platform_file, 'r')
        data = json.load(config)

        _settings = data["_settings"]
        #print("_settings: " + str(_settings))

        _QRM_settings = data["_QRM_settings"]
        #print("_QRM_settings: " + str(_QRM_settings))
        #print(_QRM_settings["pulses"][0])
        #print(_QRM_settings["pulses"][1])

        _LO_QRM_settings = data["_LO_QRM_settings"]
        #print(str(_LO_QRM_settings))
        _LO_QCM_settings = data["_LO_QCM_settings"]
        #print(str(_LO_QCM_settings))

        lbl=Label(window, text="Introduce data to configure Single Qubit Single Pulsar Platform", fg='blue', font=("arial", 16))
        lbl.place(x=120, y=20)
        lbl=Label(window, text="(showing actual config)", fg='blue', font=("arial", 10))
        lbl.place(x=340, y=60)

        #Data for _settings
        lbl_settings=Label(window, text="_settings", fg='black', font=("arial", 10))
        lbl_settings.place(x=100, y=120)

        lbl_data_dictionary=Label(window, text="data_folder", fg='gray', font=("arial", 10))
        lbl_data_dictionary.place(x=20, y=160)
        self.txtfld_data_dictionary=Entry(window, text="", bd=3)
        self.txtfld_data_dictionary.place(x=140, y=160)
        self.txtfld_data_dictionary.insert(0, _settings['data_folder'])

        lbl_hardware_avg=Label(window, text="hardware_avg", fg='gray', font=("arial", 10))
        lbl_hardware_avg.place(x=20, y=190)
        self.txtfld_hardware_avg=Entry(window, text="", bd=3)
        self.txtfld_hardware_avg.place(x=140, y=190)
        self.txtfld_hardware_avg.insert(0, _settings['hardware_avg'])

        lbl_sampling_rate=Label(window, text="sampling_rate", fg='gray', font=("arial", 10))
        lbl_sampling_rate.place(x=20, y=220)
        self.txtfld_sampling_rate=Entry(window, text="", bd=3)
        self.txtfld_sampling_rate.place(x=140, y=220)
        self.txtfld_sampling_rate.insert(0, _settings['sampling_rate'])

        lbl_software_averages=Label(window, text="software_averages", fg='gray', font=("arial", 10))
        lbl_software_averages.place(x=20, y=250)
        self.txtfld_software_averages=Entry(window, text="", bd=3)
        self.txtfld_software_averages.place(x=140, y=250)
        self.txtfld_software_averages.insert(0, _settings['software_averages'])


        lbl_repetition_duration=Label(window, text="repetition_duration", fg='gray', font=("arial", 10))
        lbl_repetition_duration.place(x=20, y=280)
        self.txtfld_repetition_duration=Entry(window, text="", bd=3)
        self.txtfld_repetition_duration.place(x=140, y=280)
        self.txtfld_repetition_duration.insert(0, _settings['repetition_duration'])


        #LO QRM Settings
        lbl_LO_QRM_settings=Label(window, text="LO_QRM_settings", fg='black', font=("arial", 10))
        lbl_LO_QRM_settings.place(x=390, y=120)

        lbl_LO_QRM_settings_power=Label(window, text="power", fg='gray', font=("arial", 10))
        lbl_LO_QRM_settings_power.place(x=330, y=160)
        self.txtfld_LO_QRM_settings_power=Entry(window, text="", bd=3)
        self.txtfld_LO_QRM_settings_power.place(x=400, y=160)
        self.txtfld_LO_QRM_settings_power.insert(0, _LO_QRM_settings['power'])

        lbl_LO_QRM_settings_frequency=Label(window, text="frequency", fg='gray', font=("arial", 10))
        lbl_LO_QRM_settings_frequency.place(x=330, y=190)
        self.txtfld_LO_QRM_settings_frequency=Entry(window, text="", bd=3)
        self.txtfld_LO_QRM_settings_frequency.place(x=400, y=190)
        self.txtfld_LO_QRM_settings_frequency.insert(0, _LO_QRM_settings['frequency'])


        #LO QCM Settings
        lbl_LO_QCM_settings=Label(window, text="LO_QCM_settings", fg='black', font=("arial", 10))
        lbl_LO_QCM_settings.place(x=650, y=120)

        lbl_LO_QCM_settings_power=Label(window, text="power", fg='gray', font=("arial", 10))
        lbl_LO_QCM_settings_power.place(x=590, y=160)
        self.txtfld_LO_QCM_settings_power=Entry(window, text="", bd=3)
        self.txtfld_LO_QCM_settings_power.place(x=660, y=160)
        self.txtfld_LO_QCM_settings_power.insert(0, _LO_QCM_settings['power'])

        lbl_LO_QCM_settings_frequency=Label(window, text="frequency", fg='gray', font=("arial", 10))
        lbl_LO_QCM_settings_frequency.place(x=590, y=190)
        self.txtfld_LO_QCM_settings_frequency=Entry(window, text="", bd=3)
        self.txtfld_LO_QCM_settings_frequency.place(x=660, y=190)
        self.txtfld_LO_QCM_settings_frequency.insert(0, _LO_QCM_settings['frequency'])


        #QRM Settings
        lbl_QRM_settings=Label(window, text="QRM_settings", fg='black', font=("arial", 10))
        lbl_QRM_settings.place(x=100, y=350)

        lbl_QRM_settings_gain=Label(window, text="gain", fg='gray', font=("arial", 10))
        lbl_QRM_settings_gain.place(x=20, y=390)
        self.txtfld_QRM_settings_gain=Entry(window, text="", bd=3)
        self.txtfld_QRM_settings_gain.place(x=140, y=390)
        self.txtfld_QRM_settings_gain.insert(0, _QRM_settings['gain'])

        lbl_QRM_settings_hardware_avg=Label(window, text="hardware_avg", fg='gray', font=("arial", 10))
        lbl_QRM_settings_hardware_avg.place(x=20, y=420)
        self.txtfld_QRM_settings_hardware_avg=Entry(window, text="", bd=3)
        self.txtfld_QRM_settings_hardware_avg.place(x=140, y=420)
        self.txtfld_QRM_settings_hardware_avg.insert(0, _QRM_settings['hardware_avg'])

        lbl_QRM_settings_initial_delay=Label(window, text="initial_delay", fg='gray', font=("arial", 10))
        lbl_QRM_settings_initial_delay.place(x=20, y=450)
        self.txtfld_QRM_settings_initial_delay=Entry(window, text="", bd=3)
        self.txtfld_QRM_settings_initial_delay.place(x=140, y=450)
        self.txtfld_QRM_settings_initial_delay.insert(0, _QRM_settings['initial_delay'])

        lbl_QRM_settings_repetition_duration=Label(window, text="repetition_duration", fg='gray', font=("arial", 10))
        lbl_QRM_settings_repetition_duration.place(x=20, y=480)
        self.txtfld_QRM_settings_repetition_duration=Entry(window, text="", bd=3)
        self.txtfld_QRM_settings_repetition_duration.place(x=140, y=480)
        self.txtfld_QRM_settings_repetition_duration.insert(0, _QRM_settings['repetition_duration'])


        lbl_QRM_settings_start_sample=Label(window, text="start_sample", fg='gray', font=("arial", 10))
        lbl_QRM_settings_start_sample.place(x=20, y=510)
        self.txtfld_QRM_settings_start_sample=Entry(window, text="", bd=3)
        self.txtfld_QRM_settings_start_sample.place(x=140, y=510)
        self.txtfld_QRM_settings_start_sample.insert(0, _QRM_settings['start_sample'])


        lbl_QRM_settings_integration_length=Label(window, text="integration_length", fg='gray', font=("arial", 10))
        lbl_QRM_settings_integration_length.place(x=20, y=540)
        self.txtfld_QRM_settings_integration_length=Entry(window, text="", bd=3)
        self.txtfld_QRM_settings_integration_length.place(x=140, y=540)
        self.txtfld_QRM_settings_integration_length.insert(0, _QRM_settings['integration_length'])


        lbl_QRM_settings_sampling_rate=Label(window, text="sampling_rate", fg='gray', font=("arial", 10))
        lbl_QRM_settings_sampling_rate.place(x=20, y=570)
        self.txtfld_QRM_settings_sampling_rate=Entry(window, text="", bd=3)
        self.txtfld_QRM_settings_sampling_rate.place(x=140, y=570)
        self.txtfld_QRM_settings_sampling_rate.insert(0, _QRM_settings['sampling_rate'])


        lbl_QRM_settings_mode=Label(window, text="mode", fg='gray', font=("arial", 10))
        lbl_QRM_settings_mode.place(x=20, y=600)
        self.txtfld_QRM_settings_mode=Entry(window, text="", bd=3)
        self.txtfld_QRM_settings_mode.place(x=140, y=600)
        self.txtfld_QRM_settings_mode.insert(0, _QRM_settings['mode'])


        #Pulses settings
        lbl_pulses=Label(window, text="Pulses settings", fg='black', font=("arial", 10))
        lbl_pulses.place(x=520, y=320)

        #QCM Pulse settings
        lbl_qc_pulse=Label(window, text="Pulse#0", fg='black', font=("arial", 10))
        lbl_qc_pulse.place(x=420, y=350)

        pulse0 = _QRM_settings['pulses']['qc_pulse']

        lbl_qc_pulse_freq_if=Label(window, text="freq_if", fg='gray', font=("arial", 10))
        lbl_qc_pulse_freq_if.place(x=330, y=390)
        self.txtfld_qc_pulse_freq_if=Entry(window, text="", bd=3)
        self.txtfld_qc_pulse_freq_if.place(x=400, y=390)
        self.txtfld_qc_pulse_freq_if.insert(0, pulse0['freq_if'])


        lbl_qc_pulse_amplitude=Label(window, text="amplitude", fg='gray', font=("arial", 10))
        lbl_qc_pulse_amplitude.place(x=330, y=420)
        self.txtfld_qc_pulse_amplitude=Entry(window, text="", bd=3)
        self.txtfld_qc_pulse_amplitude.place(x=400, y=420)
        self.txtfld_qc_pulse_amplitude.insert(0, pulse0['amplitude'])


        lbl_qc_pulse_start=Label(window, text="start", fg='gray', font=("arial", 10))
        lbl_qc_pulse_start.place(x=330, y=450)
        self.txtfld_qc_pulse_start=Entry(window, text="", bd=3)
        self.txtfld_qc_pulse_start.place(x=400, y=450)
        self.txtfld_qc_pulse_start.insert(0, pulse0['start'])


        lbl_qc_pulse_length=Label(window, text="length", fg='gray', font=("arial", 10))
        lbl_qc_pulse_length.place(x=330, y=480)
        self.txtfld_qc_pulse_length=Entry(window, text="", bd=3)
        self.txtfld_qc_pulse_length.place(x=400, y=480)
        self.txtfld_qc_pulse_length.insert(0, pulse0['length'])

        lbl_qc_pulse_offset_i=Label(window, text="offset_i", fg='gray', font=("arial", 10))
        lbl_qc_pulse_offset_i.place(x=330, y=510)
        self.txtfld_qc_pulse_offset_i=Entry(window, text="", bd=3)
        self.txtfld_qc_pulse_offset_i.place(x=400, y=510)
        self.txtfld_qc_pulse_offset_i.insert(0, pulse0['offset_i'])


        lbl_qc_pulse_offset_q=Label(window, text="offset_q", fg='gray', font=("arial", 10))
        lbl_qc_pulse_offset_q.place(x=330, y=540)
        self.txtfld_qc_pulse_offset_q=Entry(window, text="", bd=3)
        self.txtfld_qc_pulse_offset_q.place(x=400, y=540)
        self.txtfld_qc_pulse_offset_q.insert(0, pulse0['offset_q'])


        lbl_qc_pulse_shape=Label(window, text="shape", fg='gray', font=("arial", 10))
        lbl_qc_pulse_shape.place(x=330, y=570)
        self.txtfld_qc_pulse_shape=Entry(window, text="", bd=3)
        self.txtfld_qc_pulse_shape.place(x=400, y=570)
        self.txtfld_qc_pulse_shape.insert(0, pulse0['shape'])


        #RO Pulse settings
        pulse1 = _QRM_settings['pulses']['ro_pulse']

        lbl_ro_pulse=Label(window, text="Pulse#1", fg='black', font=("arial", 10))
        lbl_ro_pulse.place(x=700, y=350)

        lbl_ro_pulse_freq_if=Label(window, text="freq_if", fg='gray', font=("arial", 10))
        lbl_ro_pulse_freq_if.place(x=590, y=390)
        self.txtfld_ro_pulse_freq_if=Entry(window, text="", bd=3)
        self.txtfld_ro_pulse_freq_if.place(x=660, y=390)
        self.txtfld_ro_pulse_freq_if.insert(0, pulse1['freq_if'])


        lbl_ro_pulse_amplitude=Label(window, text="amplitude", fg='gray', font=("arial", 10))
        lbl_ro_pulse_amplitude.place(x=590, y=420)
        self.txtfld_ro_pulse_amplitude=Entry(window, text="", bd=3)
        self.txtfld_ro_pulse_amplitude.place(x=660, y=420)
        self.txtfld_ro_pulse_amplitude.insert(0, pulse1['amplitude'])


        lbl_ro_pulse_start=Label(window, text="start", fg='gray', font=("arial", 10))
        lbl_ro_pulse_start.place(x=590, y=450)
        self.txtfld_ro_pulse_start=Entry(window, text="", bd=3)
        self.txtfld_ro_pulse_start.place(x=660, y=450)
        self.txtfld_ro_pulse_start.insert(0, pulse1['start'])


        lbl_ro_pulse_length=Label(window, text="length", fg='gray', font=("arial", 10))
        lbl_ro_pulse_length.place(x=590, y=480)
        self.txtfld_ro_pulse_length=Entry(window, text="", bd=3)
        self.txtfld_ro_pulse_length.place(x=660, y=480)
        self.txtfld_ro_pulse_length.insert(0, pulse1['length'])


        lbl_ro_pulse_offset_i=Label(window, text="offset_i", fg='gray', font=("arial", 10))
        lbl_ro_pulse_offset_i.place(x=590, y=510)
        self.txtfld_ro_pulse_offset_i=Entry(window, text="", bd=3)
        self.txtfld_ro_pulse_offset_i.place(x=660, y=510)
        self.txtfld_ro_pulse_offset_i.insert(0, pulse1['offset_i'])


        lbl_ro_pulse_offset_q=Label(window, text="offset_q", fg='gray', font=("arial", 10))
        lbl_ro_pulse_offset_q.place(x=590, y=540)
        self.txtfld_ro_pulse_offset_q=Entry(window, text="", bd=3)
        self.txtfld_ro_pulse_offset_q.place(x=660, y=540)
        self.txtfld_ro_pulse_offset_q.insert(0, pulse1['offset_q'])


        lbl_ro_pulse_shape=Label(window, text="shape", fg='gray', font=("arial", 10))
        lbl_ro_pulse_shape.place(x=590, y=570)
        self.txtfld_ro_pulse_shape=Entry(window, text="", bd=3)
        self.txtfld_ro_pulse_shape.place(x=660, y=570)
        self.txtfld_ro_pulse_shape.insert(0, pulse1['shape'])

        #Add save button
        btn=Button(window, text="Save config", fg='blue', command=self.saveConfig)
        btn.place(x=380, y=660)

    def saveConfig(self):
        print("_____Saving new config_____")

        data = {
            "_settings":
            {
                "data_dictionary": self.txtfld_data_dictionary.get(),
                "hardware_avg": self.txtfld_hardware_avg.get(),
                "sampling_rate": self.txtfld_sampling_rate.get(),
                "software_averages": self.txtfld_software_averages.get(),
                "repetition_duration": self.txtfld_repetition_duration.get()
            },
            "_QRM_settings":
            {
                "gain": self.txtfld_QRM_settings_gain.get(),
                "hardware_avg": self.txtfld_QRM_settings_hardware_avg.get(),
                "initial_delay": self.txtfld_QRM_settings_initial_delay.get(),
                "repetition_duration": self.txtfld_QRM_settings_repetition_duration.get(),
                "start_sample": self.txtfld_QRM_settings_start_sample.get(),
                "integration_length": self.txtfld_QRM_settings_integration_length.get(),
                "sampling_rate": self.txtfld_QRM_settings_sampling_rate.get(),
                "mode": self.txtfld_QRM_settings_mode.get(),
                "pulses":
                {
                    "qc_pulse":
                    {
                        "freq_if": self.txtfld_qc_pulse_freq_if.get(),
                        "amplitude": self.txtfld_qc_pulse_amplitude.get(),
                        "start": self.txtfld_qc_pulse_start.get(),
                        "length": self.txtfld_qc_pulse_length.get(),
                        "offset_i": self.txtfld_qc_pulse_offset_i.get(),
                        "offset_q": self.txtfld_qc_pulse_offset_q.get(),
                        "shape": self.txtfld_qc_pulse_shape.get()
                    },
                    "ro_pulse":
                    {
                        "freq_if": self.txtfld_ro_pulse_freq_if.get(),
                        "amplitude": self.txtfld_ro_pulse_amplitude.get(),
                        "start": self.txtfld_ro_pulse_start.get(),
                        "length": self.txtfld_ro_pulse_length.get(),
                        "offset_i": self.txtfld_ro_pulse_offset_i.get(),
                        "offset_q": self.txtfld_ro_pulse_offset_q.get(),
                        "shape": self.txtfld_ro_pulse_shape.get()
                    }
                }
            },
            "_LO_QRM_settings":
            {
                "power": self.txtfld_LO_QRM_settings_power.get(),
                "frequency": self.txtfld_LO_QRM_settings_frequency.get()
            },
            "_LO_QCM_settings":
            {
                "power": self.txtfld_LO_QCM_settings_power.get(),
                "frequency": self.txtfld_LO_QCM_settings_frequency.get()
            }
        }

        with open(platform_file, 'w') as f:
            json.dump(data, f, indent=4)

        print("_____Config saved_____")

window=Tk()
mywin=ConfigWindow(window)
window.title('Platform config GUI')
window.geometry("850x700+10+10")
window.mainloop()
