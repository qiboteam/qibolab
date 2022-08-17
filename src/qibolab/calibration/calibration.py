# -*- coding: utf-8 -*-
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import yaml
from quantify_core.data.handling import set_datadir
from quantify_core.measurement import MeasurementControl
from quantify_core.measurement.control import Gettable, Settable
from scipy.signal import savgol_filter

from qibolab import Platform
from qibolab.calibration import fitting, utils
from qibolab.paths import qibolab_folder
from qibolab.pulses import (
    Drag,
    Gaussian,
    Pulse,
    PulseSequence,
    ReadoutPulse,
    Rectangular,
)


class Calibration:
    def __init__(self, platform: Platform, settings_file=None, show_plots=True):
        self.platform = platform
        if not settings_file:
            script_folder = pathlib.Path(__file__).parent
            settings_file = script_folder / "calibration.yml"
        self.settings_file = settings_file
        # TODO: Set mc plotting to false when auto calibrates (default = True for diagnostics)
        self.mc, self.pl, self.ins = utils.create_measurement_control(
            "Calibration", show_plots
        )

    def load_settings(self):
        # Load calibration settings
        with open(self.settings_file, "r") as file:
            self.settings = yaml.safe_load(file)
            self.software_averages = self.settings["software_averages"]
            self.software_averages_precision = self.settings[
                "software_averages_precision"
            ]
            self.max_num_plots = self.settings["max_num_plots"]

        if self.platform.settings["nqubits"] == 1:
            self.resonator_type = "3D"
        else:
            self.resonator_type = "2D"

    def reload_settings(self):
        self.load_settings()

    # --------------------------#
    # Single qubit experiments #
    # --------------------------#

    def run_resonator_spectroscopy(self, qubit=0):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc

        self.reload_settings()
        lowres_width = self.settings["resonator_spectroscopy"]["lowres_width"]
        lowres_step = self.settings["resonator_spectroscopy"]["lowres_step"]
        highres_width = self.settings["resonator_spectroscopy"]["highres_width"]
        highres_step = self.settings["resonator_spectroscopy"]["highres_step"]
        precision_width = self.settings["resonator_spectroscopy"]["precision_width"]
        precision_step = self.settings["resonator_spectroscopy"]["precision_step"]

        sequence = PulseSequence()
        ro_pulse = platform.qubit_readout_pulse(qubit, start=0)

        sequence.add(ro_pulse)

        self.pl.tuids_max_num(self.max_num_plots)
        lo_qrm_frequency = (
            platform.characterization["single_qubit"][qubit]["resonator_freq"]
            - ro_pulse.frequency
        )

        # Fast Sweep
        if self.software_averages != 0:
            scanrange = utils.variable_resolution_scanrange(
                lowres_width, lowres_step, highres_width, highres_step
            )

            mc.settables(
                settable(platform.ro_port[qubit], "lo_frequency", "Frequency", "Hz")
            )
            mc.setpoints(scanrange + lo_qrm_frequency)
            mc.gettables(ROController(platform, sequence, qubit))
            platform.start()
            dataset = mc.run(
                "Resonator Spectroscopy Fast", soft_avg=self.software_averages
            )
            platform.stop()

            if self.resonator_type == "3D":
                lo_qrm_frequency = dataset["x0"].values[dataset["y0"].argmax().values]
                avg_voltage = (
                    np.mean(dataset["y0"].values[: (lowres_width // lowres_step)]) * 1e6
                )
            elif self.resonator_type == "2D":
                lo_qrm_frequency = dataset["x0"].values[dataset["y0"].argmin().values]
                avg_voltage = (
                    np.mean(dataset["y0"].values[: (lowres_width // lowres_step)]) * 1e6
                )

        # Precision Sweep
        if self.software_averages_precision != 0:
            scanrange = np.arange(-precision_width, precision_width, precision_step)
            mc.settables(
                settable(platform.ro_port[qubit], "lo_frequency", "Frequency", "Hz")
            )
            mc.setpoints(scanrange + lo_qrm_frequency)
            mc.gettables(ROController(platform, sequence, qubit))
            platform.start()
            dataset = mc.run(
                "Resonator Spectroscopy Precision",
                soft_avg=self.software_averages_precision,
            )
            platform.stop()

        # Fitting
        if self.resonator_type == "3D":
            f0, BW, Q, peak_voltage = fitting.lorentzian_fit(
                "last", max, "Resonator_spectroscopy"
            )
            resonator_freq = int(f0 + ro_pulse.frequency)
        elif self.resonator_type == "2D":
            f0, BW, Q, peak_voltage = fitting.lorentzian_fit(
                "last", min, "Resonator_spectroscopy"
            )
            resonator_freq = int(f0 + ro_pulse.frequency)
            # TODO: Fix fitting of minimum values
        peak_voltage = peak_voltage * 1e6

        print(f"\nResonator Frequency = {resonator_freq}")
        return resonator_freq, avg_voltage, peak_voltage, dataset

    def run_resonator_punchout(self, qubit=0):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc

        self.reload_settings()
        freq_width = self.settings["resonator_punchout"]["freq_width"]
        freq_step = self.settings["resonator_punchout"]["freq_step"]
        att_min = self.settings["resonator_punchout"]["att_min"]
        att_max = self.settings["resonator_punchout"]["att_max"]
        att_step = self.settings["resonator_punchout"]["att_step"]

        sequence = PulseSequence()
        ro_pulse = platform.qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulse)

        self.pl.tuids_max_num(self.max_num_plots)
        lo_qrm_frequency = (
            platform.characterization["single_qubit"][qubit]["resonator_freq"]
            - ro_pulse.frequency
        )

        scanrange = np.arange(-freq_width, freq_width, freq_step)
        freqs = scanrange + lo_qrm_frequency
        atts = np.flip(np.arange(att_min, att_max, att_step))

        mc.setpoints_grid([freqs, atts])
        mc.settables(
            [
                settable(platform.ro_port[qubit], "lo_frequency", "Frequency", "Hz"),
                settable(platform.qrm[qubit], "attenuation", "Attenuation", "dB"),
            ]
        )
        mc.gettables(
            ROControllerNormalised(platform, sequence, qubit, platform.qrm[qubit])
        )
        platform.start()
        dataset = mc.run("Resonator Punchout", soft_avg=self.software_averages)
        platform.stop()

        # TODO: automatically extract the best attenuation setting
        # TODO: normalise results in power (not possible with meassurement control)
        return dataset

    def run_resonator_spectroscopy_flux(self, qubit=0, fluxline=0):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc

        self.reload_settings()
        freq_width = self.settings["resonator_spectroscopy_flux"]["freq_width"]
        freq_step = self.settings["resonator_spectroscopy_flux"]["freq_step"]
        current_min = self.settings["resonator_spectroscopy_flux"]["current_min"]
        current_max = self.settings["resonator_spectroscopy_flux"]["current_max"]
        current_step = self.settings["resonator_spectroscopy_flux"]["current_step"]

        sequence = PulseSequence()
        ro_pulse = platform.qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulse)

        self.pl.tuids_max_num(self.max_num_plots)
        lo_qrm_frequency = (
            platform.characterization["single_qubit"][qubit]["resonator_freq"]
            - ro_pulse.frequency
        )
        spi = platform.instruments["SPI"].device
        dacs = [
            spi.mod2.dac0,
            spi.mod1.dac0,
            spi.mod1.dac1,
            spi.mod1.dac2,
            spi.mod1.dac3,
        ]
        spi.set_dacs_zero()

        scanrange = np.arange(-freq_width, freq_width, freq_step)
        freqs = scanrange + lo_qrm_frequency
        flux = np.arange(current_min, current_max, current_step)

        mc.setpoints_grid([freqs, flux])
        mc.settables(
            [
                settable(platform.ro_port[qubit], "lo_frequency", "Frequency", "Hz"),
                dacs[fluxline].current,
            ]
        )
        mc.gettables(ROController(platform, sequence, qubit))
        platform.start()
        dataset = mc.run(name="Resonator Spectroscopy Flux")
        platform.stop()
        spi.set_dacs_zero()

        # TODO: call platform.qfm[fluxline] instead of dacs[fluxline]
        # TODO: automatically extract the sweet spot current
        # TODO: add a method to generate the matrix
        return dataset

    def run_qubit_spectroscopy(self, qubit=0):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc

        self.reload_settings()
        fast_start = self.settings["qubit_spectroscopy"]["fast_start"]
        fast_end = self.settings["qubit_spectroscopy"]["fast_end"]
        fast_step = self.settings["qubit_spectroscopy"]["fast_step"]
        precision_start = self.settings["qubit_spectroscopy"]["precision_start"]
        precision_end = self.settings["qubit_spectroscopy"]["precision_end"]
        precision_step = self.settings["qubit_spectroscopy"]["precision_step"]

        sequence = PulseSequence()
        qd_pulse = platform.qubit_drive_pulse(qubit, start=0, duration=5000)
        ro_pulse = platform.qubit_readout_pulse(qubit, start=5000)
        sequence.add(qd_pulse)
        sequence.add(ro_pulse)

        self.pl.tuids_max_num(self.max_num_plots)

        # Fast Sweep
        if self.software_averages != 0:
            lo_qcm_frequency = (
                platform.characterization["single_qubit"][qubit]["qubit_freq"]
                - qd_pulse.frequency
            )
            fast_sweep_scan_range = np.arange(fast_start, fast_end, fast_step)
            mc.settables(
                settable(platform.qd_port[qubit], "lo_frequency", "Frequency", "Hz")
            )
            mc.setpoints(fast_sweep_scan_range + lo_qcm_frequency)
            mc.gettables(ROController(platform, sequence, qubit))
            platform.start()
            dataset = mc.run("Qubit Spectroscopy Fast", soft_avg=self.software_averages)
            platform.stop()

            if self.resonator_type == "3D":
                lo_qcm_frequency = dataset["x0"].values[dataset["y0"].argmin().values]
                avg_voltage = np.mean(dataset["y0"]) * 1e6
            elif self.resonator_type == "2D":
                lo_qcm_frequency = dataset["x0"].values[dataset["y0"].argmax().values]
                avg_voltage = np.mean(dataset["y0"]) * 1e6

        # Precision Sweep
        if self.software_averages_precision != 0:
            precision_sweep_scan_range = np.arange(
                precision_start, precision_end, precision_step
            )
            mc.settables(
                settable(platform.qd_port[qubit], "lo_frequency", "Frequency", "Hz")
            )
            mc.setpoints(precision_sweep_scan_range + lo_qcm_frequency)
            mc.gettables(ROController(platform, sequence, qubit))
            platform.start()
            dataset = mc.run(
                "Qubit Spectroscopy Precision",
                soft_avg=self.software_averages_precision,
            )
            platform.stop()

        # Fitting
        if self.resonator_type == "3D":
            f0, BW, Q, peak_voltage = fitting.lorentzian_fit(
                "last", min, "Qubit_spectroscopy"
            )
            qubit_freq = int(f0 + qd_pulse.frequency)
            # TODO: Fix fitting of minimum values
        elif self.resonator_type == "2D":
            f0, BW, Q, peak_voltage = fitting.lorentzian_fit(
                "last", max, "Qubit_spectroscopy"
            )
            qubit_freq = int(f0 + qd_pulse.frequency)

        # TODO: Estimate avg_voltage correctly
        print(f"\nQubit Frequency = {qubit_freq}")
        return qubit_freq, avg_voltage, peak_voltage, dataset

    def run_qubit_spectroscopy_flux(self, qubit=0, fluxline=0):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc

        self.reload_settings()
        freq_width = self.settings["qubit_spectroscopy_flux"]["freq_width"]
        freq_step = self.settings["qubit_spectroscopy_flux"]["freq_step"]
        current_min = self.settings["qubit_spectroscopy_flux"]["current_min"]
        current_max = self.settings["qubit_spectroscopy_flux"]["current_max"]
        current_step = self.settings["qubit_spectroscopy_flux"]["current_step"]

        sequence = PulseSequence()
        qd_pulse = platform.qubit_drive_pulse(qubit, start=0, duration=5000)
        ro_pulse = platform.qubit_readout_pulse(qubit, start=5000)
        sequence.add(qd_pulse)
        sequence.add(ro_pulse)

        self.pl.tuids_max_num(self.max_num_plots)
        # Normal
        lo_qcm_frequency = (
            platform.characterization["single_qubit"][qubit]["qubit_freq"]
            - qd_pulse.frequency
        )
        # Seeping with LO
        lo_qcm_frequency = platform.characterization["single_qubit"][qubit][
            "qubit_freq"
        ]

        spi = platform.instruments["SPI"].device
        dacs = [
            spi.mod2.dac0,
            spi.mod1.dac0,
            spi.mod1.dac1,
            spi.mod1.dac2,
            spi.mod1.dac3,
        ]
        spi.set_dacs_zero()

        scanrange = np.arange(-freq_width, freq_width, freq_step)
        freqs = scanrange + lo_qcm_frequency
        flux = np.arange(current_min, current_max, current_step)

        mc.setpoints_grid([freqs, flux])
        mc.settables(
            [
                settable(platform.qd_port[qubit], "lo_frequency", "Frequency", "Hz"),
                dacs[fluxline].current,
            ]
        )
        mc.gettables(ROController(platform, sequence, qubit))
        platform.start()
        dataset = mc.run(name="Qubit Spectroscopy Flux")
        platform.stop()
        spi.set_dacs_zero()

        # use a resonator calibration matrix (generated during resonator spectroscopy flux) (cannot be done with quantify)
        # TODO: call platform.qfm[fluxline] instead of dacs[fluxline]
        # TODO: automatically extract the sweet spot current
        # TODO: add a method to generate the matrix
        return dataset

    def run_rabi_pulse_length(self, qubit=0):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc

        self.reload_settings()
        pulse_duration_start = self.settings["rabi_pulse_length"][
            "pulse_duration_start"
        ]
        pulse_duration_end = self.settings["rabi_pulse_length"]["pulse_duration_end"]
        pulse_duration_step = self.settings["rabi_pulse_length"]["pulse_duration_step"]

        sequence = PulseSequence()
        qd_pulse = platform.qubit_drive_pulse(qubit, start=0, duration=4)
        ro_pulse = platform.qubit_readout_pulse(qubit, start=4)
        sequence.add(qd_pulse)
        sequence.add(ro_pulse)

        self.pl.tuids_max_num(self.max_num_plots)

        mc.settables(QCPulseLengthParameter(ro_pulse, qd_pulse))
        mc.setpoints(
            np.arange(pulse_duration_start, pulse_duration_end, pulse_duration_step)
        )
        mc.gettables(ROController(platform, sequence, qubit))
        platform.start()
        dataset = mc.run("Rabi Pulse Length", soft_avg=self.software_averages)
        platform.stop()

        # Fitting
        pi_pulse_amplitude = qd_pulse.amplitude
        if self.resonator_type == "3D":
            (
                pi_pulse_duration,
                rabi_oscillations_pi_pulse_peak_voltage,
            ) = fitting.rabi_fit_3D(dataset)
        elif self.resonator_type == "2D":
            (
                pi_pulse_duration,
                rabi_oscillations_pi_pulse_peak_voltage,
            ) = fitting.rabi_fit_2D(dataset)

        print(f"\nPi pulse duration = {pi_pulse_duration}")
        print(f"\nPi pulse amplitude = {pi_pulse_amplitude}")
        print(
            f"\nrabi oscillation peak voltage = {rabi_oscillations_pi_pulse_peak_voltage}"
        )

        # TODO: implement some verifications to check if the returned value from fitting is correct.
        return (
            pi_pulse_duration,
            pi_pulse_amplitude,
            rabi_oscillations_pi_pulse_peak_voltage,
            dataset,
        )

    def ro_pulse_phase(self, qubit=0):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc

        self.reload_settings()
        pulse_phase_start = self.settings["rabi_pulse_phase"]["pulse_phase_start"]
        pulse_phase_end = self.settings["rabi_pulse_phase"]["pulse_phase_end"]
        pulse_phase_step = self.settings["rabi_pulse_phase"]["pulse_phase_step"]

        sequence = PulseSequence()
        qd_pulse = platform.RX_pulse(qubit, start=0)
        ro_pulse = platform.qubit_readout_pulse(qubit, start=qd_pulse.duration)
        sequence.add(qd_pulse)
        sequence.add(ro_pulse)

        self.pl.tuids_max_num(self.max_num_plots)

        mc.settables(ROPulsePhaseParameter(ro_pulse))
        mc.setpoints(np.arange(pulse_phase_start, pulse_phase_end, pulse_phase_step))
        mc.gettables(ROController(platform, sequence, qubit))
        platform.start()
        dataset = mc.run("Rabi Pulse Phase", soft_avg=self.software_averages)
        platform.stop()

        return dataset

    def run_rabi_pulse_gain(self, qubit=0):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc

        self.reload_settings()
        pulse_gain_start = self.settings["rabi_pulse_gain"]["pulse_gain_start"]
        pulse_gain_end = self.settings["rabi_pulse_gain"]["pulse_gain_end"]
        pulse_gain_step = self.settings["rabi_pulse_gain"]["pulse_gain_step"]

        sequence = PulseSequence()
        qd_pulse = platform.RX_pulse(qubit, start=0)
        ro_pulse = platform.qubit_readout_pulse(qubit, start=qd_pulse.duration)
        sequence.add(qd_pulse)
        sequence.add(ro_pulse)

        self.pl.tuids_max_num(self.max_num_plots)

        mc.settables(settable(platform.qd_port[qubit], "gain", "Gain", "dB"))
        mc.setpoints(np.arange(pulse_gain_start, pulse_gain_end, pulse_gain_step))
        mc.gettables(ROController(platform, sequence, qubit))
        platform.start()
        dataset = mc.run(
            f"Rabi Pulse Gain - for pulse duration {qd_pulse.duration} ns",
            soft_avg=self.software_averages,
        )
        platform.stop()

        # Fitting
        pi_pulse_amplitude = qd_pulse.amplitude
        pi_pulse_duration = qd_pulse.duration
        if self.resonator_type == "3D":
            (
                pi_pulse_gain,
                rabi_oscillations_pi_pulse_peak_voltage,
            ) = fitting.rabi_fit_3D(dataset)
        elif self.resonator_type == "2D":
            (
                pi_pulse_gain,
                rabi_oscillations_pi_pulse_peak_voltage,
            ) = fitting.rabi_fit_2D(dataset)

        print(f"\nPi pulse gain = {pi_pulse_gain}")
        print(f"\nPi pulse amplitude = {pi_pulse_amplitude}")
        print(f"\nPi pulse duration = {pi_pulse_duration}")
        print(
            f"\nrabi oscillation peak voltage = {rabi_oscillations_pi_pulse_peak_voltage}"
        )
        # TODO: calibrate freq for each iteration first
        # TODO: implement some verifications to check if the returned value from fitting is correct.
        return (
            pi_pulse_gain,
            pi_pulse_amplitude,
            rabi_oscillations_pi_pulse_peak_voltage,
            dataset,
        )

    def run_rabi_pulse_amplitude(self, qubit=0):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc

        self.reload_settings()
        pulse_amplitude_start = self.settings["rabi_pulse_amplitude"][
            "pulse_amplitude_start"
        ]
        pulse_amplitude_end = self.settings["rabi_pulse_amplitude"][
            "pulse_amplitude_end"
        ]
        pulse_amplitude_step = self.settings["rabi_pulse_amplitude"][
            "pulse_amplitude_step"
        ]

        sequence = PulseSequence()
        qd_pulse = platform.RX_pulse(qubit, start=0)
        ro_pulse = platform.qubit_readout_pulse(qubit, start=qd_pulse.duration)
        sequence.add(qd_pulse)
        sequence.add(ro_pulse)

        self.pl.tuids_max_num(self.max_num_plots)

        mc.settables(QCPulseAmplitudeParameter(qd_pulse))
        mc.setpoints(
            np.arange(pulse_amplitude_start, pulse_amplitude_end, pulse_amplitude_step)
        )
        mc.gettables(ROController(platform, sequence, qubit))
        platform.start()
        dataset = mc.run(
            f"Rabi Pulse Ampitude - for pulse gain {platform.qd_port[qubit].gain}",
            soft_avg=self.software_averages,
        )
        platform.stop()

        # Fitting
        pi_pulse_gain = platform.qd_port[qubit].gain
        pi_pulse_duration = qd_pulse.duration
        if self.resonator_type == "3D":
            (
                pi_pulse_amplitude,
                rabi_oscillations_pi_pulse_peak_voltage,
            ) = fitting.rabi_fit_3D(dataset)
        elif self.resonator_type == "2D":
            (
                pi_pulse_amplitude,
                rabi_oscillations_pi_pulse_peak_voltage,
            ) = fitting.rabi_fit_2D(dataset)

        print(f"\nPi pulse gain = {pi_pulse_gain}")
        print(f"\nPi pulse amplitude = {pi_pulse_amplitude}")
        print(f"\nPi pulse duration = {pi_pulse_duration}")
        print(
            f"\nrabi oscillation peak voltage = {rabi_oscillations_pi_pulse_peak_voltage}"
        )
        return (
            pi_pulse_gain,
            pi_pulse_amplitude,
            rabi_oscillations_pi_pulse_peak_voltage,
            dataset,
        )

    # T1: RX(pi) - wait t(rotates z) - readout
    def run_t1(self, qubit=0):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc

        self.reload_settings()
        self.delay_before_readout_start = self.settings["t1"][
            "delay_before_readout_start"
        ]
        self.delay_before_readout_end = self.settings["t1"]["delay_before_readout_end"]
        self.delay_before_readout_step = self.settings["t1"][
            "delay_before_readout_step"
        ]

        sequence = PulseSequence()
        RX_pulse = platform.RX_pulse(qubit, start=0)
        ro_pulse = platform.qubit_readout_pulse(qubit, start=RX_pulse.duration)
        sequence.add(RX_pulse)
        sequence.add(ro_pulse)

        self.pl.tuids_max_num(self.max_num_plots)

        mc.settables(Settable(T1WaitParameter(ro_pulse, RX_pulse)))
        mc.setpoints(
            np.arange(
                self.delay_before_readout_start,
                self.delay_before_readout_end,
                self.delay_before_readout_step,
            )
        )
        mc.gettables(Gettable(ROController(platform, sequence, qubit)))
        platform.start()
        dataset = mc.run("T1", soft_avg=self.software_averages)
        platform.stop()

        # Fitting
        if self.resonator_type == "3D":
            t1 = fitting.t1_fit_3D(dataset)
        elif self.resonator_type == "2D":
            t1 = fitting.t1_fit_2D(dataset)

        print(f"\nT1 = {t1}")

        return t1, dataset

    # Ramsey: RX(pi/2) - wait t(rotates z) - RX(pi/2) - readout
    def run_ramsey(self, qubit=0):
        platform = self.platform
        platform.reload_settings()
        sampling_rate = platform.sampling_rate
        mc = self.mc

        self.reload_settings()
        self.delay_between_pulses_start = self.settings["ramsey"][
            "delay_between_pulses_start"
        ]
        self.delay_between_pulses_end = self.settings["ramsey"][
            "delay_between_pulses_end"
        ]
        self.delay_between_pulses_step = self.settings["ramsey"][
            "delay_between_pulses_step"
        ]

        sequence = PulseSequence()
        RX90_pulse1 = platform.RX90_pulse(qubit, start=0)
        RX90_pulse2 = platform.RX90_pulse(
            qubit,
            start=RX90_pulse1.duration,
            phase=RX90_pulse1.duration
            / sampling_rate
            * 2
            * np.pi
            * RX90_pulse1.frequency,
        )
        ro_pulse = platform.qubit_readout_pulse(
            qubit, start=RX90_pulse1.duration + RX90_pulse2.duration
        )
        sequence.add(RX90_pulse1)
        sequence.add(RX90_pulse2)
        sequence.add(ro_pulse)

        self.pl.tuids_max_num(self.max_num_plots)

        mc.settables(
            Settable(RamseyWaitParameter(ro_pulse, RX90_pulse2, sampling_rate))
        )
        mc.setpoints(
            np.arange(
                self.delay_between_pulses_start,
                self.delay_between_pulses_end,
                self.delay_between_pulses_step,
            )
        )
        mc.gettables(Gettable(ROController(platform, sequence, qubit)))
        platform.start()
        dataset = mc.run("Ramsey", soft_avg=self.software_averages)
        platform.stop()

        # Fitting
        smooth_dataset, delta_frequency, t2 = fitting.ramsey_fit(dataset)
        utils.plot(smooth_dataset, dataset, "Ramsey", 1)
        print(f"\nDelta Frequency = {delta_frequency}")
        corrected_qubit_frequency = int(
            platform.settings["characterization"]["single_qubit"][qubit]["qubit_freq"]
            + delta_frequency
        )
        print(f"\nCorrected Qubit Frequency = {corrected_qubit_frequency}")
        print(f"\nT2 = {int(t2)} ns")

        # TODO: return corrected frequency
        return (
            delta_frequency,
            corrected_qubit_frequency,
            int(t2),
            smooth_dataset,
            dataset,
        )

    # Ramsey: RX(pi/2) - wait t(rotates z) - RX(pi/2) - readout
    def run_ramsey_frequency_detuned(self, qubit):
        platform = self.platform
        platform.reload_settings()
        sampling_rate = platform.sampling_rate
        mc = self.mc

        self.reload_settings()
        t_start = self.settings["ramsey_freq"]["t_start"]
        t_end = self.settings["ramsey_freq"]["t_end"]
        t_step = self.settings["ramsey_freq"]["t_step"]
        n_osc = self.settings["ramsey_freq"]["n_osc"]

        sequence = PulseSequence()
        RX90_pulse1 = platform.RX90_pulse(qubit, start=0)
        RX90_pulse2 = platform.RX90_pulse(
            qubit,
            start=RX90_pulse1.duration,
            phase=(RX90_pulse1.duration / sampling_rate)
            * (2 * np.pi)
            * (RX90_pulse1.frequency),
        )
        ro_pulse = platform.qubit_readout_pulse(
            qubit, start=RX90_pulse1.duration + RX90_pulse2.duration
        )
        sequence.add(RX90_pulse1)
        sequence.add(RX90_pulse2)
        sequence.add(ro_pulse)

        self.pl.tuids_max_num(self.max_num_plots)

        runcard_qubit_freq = platform.settings["characterization"]["single_qubit"][
            qubit
        ]["qubit_freq"]
        runcard_T2 = platform.settings["characterization"]["single_qubit"][qubit]["T2"]
        intermediate_freq = platform.settings["native_gates"]["single_qubit"][qubit][
            "RX"
        ]["frequency"]

        current_qubit_freq = runcard_qubit_freq
        current_T2 = runcard_T2

        for t_max in t_end:
            platform.qd_port[qubit].lo_frequency = (
                current_qubit_freq - intermediate_freq
            )

            offset_freq = n_osc / t_max * sampling_rate  # Hz
            t_range = np.arange(t_start, t_max, t_step)
            mc.settables(
                RamseyFreqWaitParameter(
                    ro_pulse, RX90_pulse2, offset_freq, sampling_rate
                )
            )
            mc.setpoints(t_range)
            mc.gettables(ROController(platform, sequence, qubit))
            platform.start()
            dataset = mc.run(
                "Ramsey Frequency Detuned", soft_avg=self.software_averages
            )
            platform.stop()

            # Fitting
            smooth_dataset, delta_fitting, new_t2 = fitting.ramsey_freq_fit(dataset)
            delta_phys = int((delta_fitting * sampling_rate) - offset_freq)
            corrected_qubit_freq = int(current_qubit_freq - delta_phys)

            # if ((new_t2 * 3.5) > t_max):
            if new_t2 > current_T2:
                print(
                    f"\nFound a better T2: {new_t2}, for a corrected qubit frequency: {corrected_qubit_freq}"
                )
                current_qubit_freq = corrected_qubit_freq
                current_T2 = new_t2
            else:
                print(f"\nCould not find a further improvement on T2")
                corrected_qubit_freq = current_qubit_freq
                new_t2 = current_T2
                break

        return (
            new_t2,
            (corrected_qubit_freq - runcard_qubit_freq),
            corrected_qubit_freq,
            dataset,
        )

    def run_dispersive_shift(self, qubit=0):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc

        self.reload_settings()
        freq_width = self.settings["dispersive_shift"]["freq_width"]
        freq_step = self.settings["dispersive_shift"]["freq_step"]

        sequence = PulseSequence()
        ro_pulse = platform.qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulse)

        self.pl.tuids_max_num(self.max_num_plots)

        lo_qrm_frequency = (
            platform.characterization["single_qubit"][qubit]["resonator_freq"]
            - ro_pulse.frequency
        )
        scanrange = np.arange(-freq_width, freq_width, freq_step)
        frequencies = scanrange + lo_qrm_frequency

        # Resonator Spectroscopy
        mc.settables(
            settable(platform.ro_port[qubit], "lo_frequency", "Frequency", "Hz")
        )
        mc.setpoints(frequencies)
        mc.gettables(ROController(platform, sequence, qubit))
        platform.start()
        dataset = mc.run("Resonator Spectroscopy", soft_avg=self.software_averages)
        platform.stop()

        if self.resonator_type == "3D":
            f0, BW, Q, pv = fitting.lorentzian_fit(
                "last", max, "Resonator_spectroscopy"
            )
            resonator_freq = int(f0 + ro_pulse.frequency)
        elif self.resonator_type == "2D":
            f0, BW, Q, pv = fitting.lorentzian_fit(
                "last", min, "Resonator_spectroscopy"
            )
            resonator_freq = int(f0 + ro_pulse.frequency)

        # Shifted Spectroscopy
        sequence = PulseSequence()
        RX_pulse = platform.RX_pulse(qubit, start=0)
        ro_pulse = platform.qubit_readout_pulse(qubit, start=RX_pulse.duration)
        sequence.add(RX_pulse)
        sequence.add(ro_pulse)

        mc.settables(
            settable(platform.ro_port[qubit], "lo_frequency", "Frequency", "Hz")
        )
        mc.setpoints(scanrange + lo_qrm_frequency)
        mc.gettables(ROController(platform, sequence, qubit))
        platform.start()
        dataset = mc.run(
            "Dispersive Shifted Resonator Spectroscopy", soft_avg=self.software_averages
        )
        platform.stop()

        # Fitting
        if self.resonator_type == "3D":
            f0, BW, Q, peak_voltage = fitting.lorentzian_fit(
                "last", max, "Resonator_spectroscopy"
            )
            shifted_resonator_freq = int(f0 + ro_pulse.frequency)
        elif self.resonator_type == "2D":
            f0, BW, Q, peak_voltage = fitting.lorentzian_fit(
                "last", min, "Resonator_spectroscopy"
            )
            shifted_resonator_freq = int(f0 + ro_pulse.frequency)

        dispersive_shift = shifted_resonator_freq - resonator_freq
        print(f"\nResonator Frequency = {resonator_freq}")
        print(f"\nShifted Frequency = {shifted_resonator_freq}")
        print(f"\nDispersive Shift = {dispersive_shift}")
        return shifted_resonator_freq, dispersive_shift, peak_voltage, dataset

    def run_rabi_pulse_length_and_gain(self, qubit=0):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc

        self.reload_settings()
        pulse_duration_start = self.settings["rabi_pulse_length"][
            "pulse_duration_start"
        ]
        pulse_duration_end = self.settings["rabi_pulse_length"]["pulse_duration_end"]
        pulse_duration_step = self.settings["rabi_pulse_length"]["pulse_duration_step"]
        pulse_gain_start = self.settings["rabi_pulse_gain"]["pulse_gain_start"]
        pulse_gain_end = self.settings["rabi_pulse_gain"]["pulse_gain_end"]
        pulse_gain_step = self.settings["rabi_pulse_gain"]["pulse_gain_step"]

        sequence = PulseSequence()
        qd_pulse = platform.qubit_drive_pulse(qubit, start=0, duration=4)
        ro_pulse = platform.qubit_readout_pulse(qubit, start=4)
        sequence.add(qd_pulse)
        sequence.add(ro_pulse)

        self.pl.tuids_max_num(self.max_num_plots)

        mc.settables(
            [
                QCPulseLengthParameter(ro_pulse, qd_pulse),
                settable(platform.qd_port[qubit], "gain", "Gain", "dB"),
            ]
        )
        setpoints_length = np.arange(
            pulse_duration_start, pulse_duration_end, pulse_duration_step
        )
        setpoints_gain = np.arange(pulse_gain_start, pulse_gain_end, pulse_gain_step)
        mc.setpoints_grid([setpoints_length, setpoints_gain])
        mc.gettables(ROController(platform, sequence, qubit))

        platform.start()
        dataset = mc.run("Rabi Pulse Length and Gain", soft_avg=self.software_averages)
        platform.stop()

        return dataset

    def run_rabi_pulse_length_and_amplitude(self, qubit=0):
        platform = self.platform
        platform.reload_settings()
        mc = self.mc

        self.reload_settings()
        pulse_duration_start = self.settings["rabi_pulse_length"][
            "pulse_duration_start"
        ]
        pulse_duration_end = self.settings["rabi_pulse_length"]["pulse_duration_end"]
        pulse_duration_step = self.settings["rabi_pulse_length"]["pulse_duration_step"]
        pulse_amplitude_start = self.settings["rabi_pulse_amplitude"][
            "pulse_amplitude_start"
        ]
        pulse_amplitude_end = self.settings["rabi_pulse_amplitude"][
            "pulse_amplitude_end"
        ]
        pulse_amplitude_step = self.settings["rabi_pulse_amplitude"][
            "pulse_amplitude_step"
        ]

        sequence = PulseSequence()
        qd_pulse = platform.qubit_drive_pulse(qubit, start=0, duration=4)
        ro_pulse = platform.qubit_readout_pulse(qubit, start=4)
        sequence.add(qd_pulse)
        sequence.add(ro_pulse)

        self.pl.tuids_max_num(self.max_num_plots)

        mc.settables(
            [
                QCPulseLengthParameter(ro_pulse, qd_pulse),
                QCPulseAmplitudeParameter(qd_pulse),
            ]
        )
        setpoints_length = np.arange(
            pulse_duration_start, pulse_duration_end, pulse_duration_step
        )
        setpoints_amplitude = np.arange(
            pulse_amplitude_start, pulse_amplitude_end, pulse_amplitude_step
        )
        mc.setpoints_grid([setpoints_length, setpoints_amplitude])
        mc.gettables(ROController(platform, sequence, qubit))

        platform.start()
        dataset = mc.run(
            "Rabi Pulse Length and Amplitude", soft_avg=self.software_averages
        )
        platform.stop()

        return dataset

    # Spin Echo: RX(pi/2) - wait t(rotates z) - RX(pi) - wait t(rotates z) - readout
    def run_spin_echo(self, qubit=0):
        platform = self.platform
        platform.reload_settings()
        sampling_rate = platform.sampling_rate
        mc = self.mc

        sequence = PulseSequence()
        RX90_pulse = platform.RX90_pulse(qubit, start=0)
        RX_pulse = platform.RX_pulse(
            qubit,
            start=RX90_pulse.duration,
            phase=RX90_pulse.duration
            / sampling_rate
            * 2
            * np.pi
            * RX90_pulse.frequency,
        )
        ro_pulse = platform.qubit_readout_pulse(
            qubit, start=RX_pulse.start + RX_pulse.duration
        )
        sequence.add(RX90_pulse)
        sequence.add(RX_pulse)
        sequence.add(ro_pulse)

        self.reload_settings()
        self.delay_between_pulses_start = self.settings["spin_echo"][
            "delay_between_pulses_start"
        ]
        self.delay_between_pulses_end = self.settings["spin_echo"][
            "delay_between_pulses_end"
        ]
        self.delay_between_pulses_step = self.settings["spin_echo"][
            "delay_between_pulses_step"
        ]

        self.pl.tuids_max_num(self.max_num_plots)

        mc.settables(SpinEchoWaitParameter(ro_pulse, RX_pulse, sampling_rate))
        mc.setpoints(
            np.arange(
                self.delay_between_pulses_start,
                self.delay_between_pulses_end,
                self.delay_between_pulses_step,
            )
        )
        mc.gettables(ROController(platform, sequence, qubit))
        platform.start()
        dataset = mc.run("Spin Echo", soft_avg=self.software_averages)
        platform.stop()

        # Fitting

        return dataset

    # Spin Echo 3 Pulses: RX(pi/2) - wait t(rotates z) - RX(pi) - wait t(rotates z) - RX(pi/2) - readout
    def run_spin_echo_3pulses(self, qubit=0):

        platform = self.platform
        platform.reload_settings()
        sampling_rate = platform.sampling_rate
        mc = self.mc

        sequence = PulseSequence()
        RX90_pulse1 = platform.RX90_pulse(qubit, start=0)
        RX_pulse = platform.RX_pulse(qubit, start=RX90_pulse1.duration)
        RX90_pulse2 = platform.RX90_pulse(
            qubit, start=RX_pulse.start + RX_pulse.duration
        )
        ro_pulse = platform.qubit_readout_pulse(
            qubit, start=RX90_pulse2.start + RX90_pulse2.duration
        )
        sequence.add(RX90_pulse1)
        sequence.add(RX_pulse)
        sequence.add(RX90_pulse2)
        sequence.add(ro_pulse)

        self.reload_settings()
        self.delay_between_pulses_start = self.settings["spin_echo_3pulses"][
            "delay_between_pulses_start"
        ]
        self.delay_between_pulses_end = self.settings["spin_echo_3pulses"][
            "delay_between_pulses_end"
        ]
        self.delay_between_pulses_step = self.settings["spin_echo_3pulses"][
            "delay_between_pulses_step"
        ]

        self.pl.tuids_max_num(self.max_num_plots)

        mc.settables(
            SpinEcho3PWaitParameter(ro_pulse, RX_pulse, RX90_pulse2, sampling_rate)
        )
        mc.setpoints(
            np.arange(
                self.delay_between_pulses_start,
                self.delay_between_pulses_end,
                self.delay_between_pulses_step,
            )
        )
        mc.gettables(Gettable(ROController(platform, sequence, qubit)))
        platform.start()
        dataset = mc.run("Spin Echo 3 Pulses", soft_avg=self.software_averages)
        platform.stop()

        return dataset

    def calibrate_qubit_states(self, qubit=0):
        platform = self.platform
        platform.reload_settings()

        self.reload_settings()
        self.niter = self.settings["calibrate_qubit_states"]["niter"]

        # create exc and gnd pulses
        exc_sequence = PulseSequence()
        RX_pulse = platform.RX_pulse(qubit, start=0)
        ro_pulse = platform.qubit_readout_pulse(qubit, start=RX_pulse.duration)
        exc_sequence.add(RX_pulse)
        exc_sequence.add(ro_pulse)

        platform.start()
        # Exectue niter single exc shots
        all_exc_states = []
        for i in range(self.niter):
            print(f"Starting exc state calibration {i}")
            qubit_state = platform.execute_pulse_sequence(
                exc_sequence, nshots=1
            )  # TODO: Improve the speed of this with binning
            qubit_state = list(list(qubit_state.values())[0].values())[0]
            print(f"Finished exc single shot execution  {i}")
            # Compose complex point from i, q obtained from execution
            point = complex(qubit_state[2], qubit_state[3])
            all_exc_states.append(point)
        platform.stop()

        gnd_sequence = PulseSequence()
        ro_pulse = platform.qubit_readout_pulse(qubit, start=0)
        gnd_sequence.add(ro_pulse)

        # Exectue niter single gnd shots
        platform.start()
        all_gnd_states = []
        for i in range(self.niter):
            print(f"Starting gnd state calibration  {i}")
            qubit_state = platform.execute_pulse_sequence(
                gnd_sequence, 1
            )  # TODO: Improve the speed of this with binning
            qubit_state = list(list(qubit_state.values())[0].values())[0]
            print(f"Finished gnd single shot execution  {i}")
            # Compose complex point from i, q obtained from execution
            point = complex(qubit_state[2], qubit_state[3])
            all_gnd_states.append(point)
        platform.stop()

        return (
            all_gnd_states,
            np.mean(all_gnd_states),
            all_exc_states,
            np.mean(all_exc_states),
        )

    def _get_eigenstate_from_voltage(self, readout, min_voltage, max_voltage):
        norm = max_voltage - min_voltage
        normalized_voltage = (readout[0] * 1e6 - min_voltage) / norm
        normalized_voltage = (2 * normalized_voltage) - 1
        return normalized_voltage

    def _get_sequence_from_gate_pair(self, gates, qubit, beta_param):
        platform = self.platform
        sampling_rate = platform.sampling_rate
        pulse_frequency = platform.settings["native_gates"]["single_qubit"][qubit][
            "RX"
        ]["frequency"]
        pulse_duration = platform.settings["native_gates"]["single_qubit"][qubit]["RX"][
            "duration"
        ]
        # All gates have equal pulse duration

        sequence = PulseSequence()

        sequenceDuration = 0
        pulse_start = 0

        for gate in gates:
            if gate == "I":
                # print("Transforming to sequence I gate")
                pass

            if gate == "RX(pi)":
                # print("Transforming to sequence RX(pi) gate")
                if beta_param == None:
                    RX_pulse = platform.RX_pulse(
                        qubit,
                        start=pulse_start,
                        phase=(pulse_start / sampling_rate)
                        * 2
                        * np.pi
                        * pulse_frequency,
                    )
                else:
                    RX_pulse = platform.RX_drag_pulse(
                        qubit,
                        start=pulse_start,
                        phase=(pulse_start / sampling_rate)
                        * 2
                        * np.pi
                        * pulse_frequency,
                        beta=beta_param,
                    )
                sequence.add(RX_pulse)

            if gate == "RX(pi/2)":
                # print("Transforming to sequence RX(pi/2) gate")
                if beta_param == None:
                    RX90_pulse = platform.RX90_pulse(
                        qubit,
                        start=pulse_start,
                        phase=(pulse_start / sampling_rate)
                        * 2
                        * np.pi
                        * pulse_frequency,
                    )
                else:
                    RX90_pulse = platform.RX90_drag_pulse(
                        qubit,
                        start=pulse_start,
                        phase=(pulse_start / sampling_rate)
                        * 2
                        * np.pi
                        * pulse_frequency,
                        beta=beta_param,
                    )
                sequence.add(RX90_pulse)

            if gate == "RY(pi)":
                # print("Transforming to sequence RY(pi) gate")
                if beta_param == None:
                    RY_pulse = platform.RX_pulse(
                        qubit,
                        start=pulse_start,
                        phase=(pulse_start / sampling_rate)
                        * 2
                        * np.pi
                        * pulse_frequency
                        + np.pi / 2,
                    )
                else:
                    RY_pulse = platform.RX_drag_pulse(
                        qubit,
                        start=pulse_start,
                        phase=(pulse_start / sampling_rate)
                        * 2
                        * np.pi
                        * pulse_frequency
                        + np.pi / 2,
                        beta=beta_param,
                    )
                sequence.add(RY_pulse)

            if gate == "RY(pi/2)":
                # print("Transforming to sequence RY(pi/2) gate")
                if beta_param == None:
                    RY90_pulse = platform.RX90_pulse(
                        qubit,
                        start=pulse_start,
                        phase=(pulse_start / sampling_rate)
                        * 2
                        * np.pi
                        * pulse_frequency
                        + np.pi / 2,
                    )
                else:
                    RY90_pulse = platform.RX90_drag_pulse(
                        qubit,
                        start=pulse_start,
                        phase=(pulse_start / sampling_rate)
                        * 2
                        * np.pi
                        * pulse_frequency
                        + np.pi / 2,
                        beta=beta_param,
                    )
                sequence.add(RY90_pulse)

            sequenceDuration = sequenceDuration + pulse_duration
            pulse_start = pulse_duration

        # RO pulse starting just after pair of gates
        ro_pulse = platform.qubit_readout_pulse(qubit, start=sequenceDuration + 4)
        sequence.add(ro_pulse)

        return sequence

    def run_allXY(self, qubit, beta_param=None):
        platform = self.platform
        platform.reload_settings()

        # allXY rotations
        gatelist = [
            ["I", "I"],
            ["RX(pi)", "RX(pi)"],
            ["RY(pi)", "RY(pi)"],
            ["RX(pi)", "RY(pi)"],
            ["RY(pi)", "RX(pi)"],
            ["RX(pi/2)", "I"],
            ["RY(pi/2)", "I"],
            ["RX(pi/2)", "RY(pi/2)"],
            ["RY(pi/2)", "RX(pi/2)"],
            ["RX(pi/2)", "RY(pi)"],
            ["RY(pi/2)", "RX(pi)"],
            ["RX(pi)", "RY(pi/2)"],
            ["RY(pi)", "RX(pi/2)"],
            ["RX(pi/2)", "RX(pi)"],
            ["RX(pi)", "RX(pi/2)"],
            ["RY(pi/2)", "RY(pi)"],
            ["RY(pi)", "RY(pi/2)"],
            ["RX(pi)", "I"],
            ["RY(pi)", "I"],
            ["RX(pi/2)", "RX(pi/2)"],
            ["RY(pi/2)", "RY(pi/2)"],
        ]

        results = []
        gateNumber = []
        min_voltage = platform.settings["characterization"]["single_qubit"][qubit][
            "state1_voltage"
        ]
        max_voltage = platform.settings["characterization"]["single_qubit"][qubit][
            "state0_voltage"
        ]
        n = 0
        for gates in gatelist:
            # transform gate string to pulseSequence
            seq = self._get_sequence_from_gate_pair(gates, qubit, beta_param)
            # Execute PulseSequence defined by gates
            platform.start()
            state = platform.execute_pulse_sequence(seq)
            state = list(list(state.values())[0].values())[0]
            platform.stop()
            # transform readout I and Q into probabilities
            res = self._get_eigenstate_from_voltage(state, min_voltage, max_voltage)
            results.append(res)
            gateNumber.append(n)
            n = n + 1

        return results, gateNumber

    # RO Matrix
    def run_RO_matrix(self):
        platform = self.platform
        platform.reload_settings()
        nqubits = platform.settings["nqubits"]

        # Init RO_matrix[2^5][2^5] with 0
        RO_matrix = [[0 for x in range(2**nqubits)] for y in range(2**nqubits)]
        # set niter = 1024 to collect good statistics
        self.reload_settings()
        self.niter = self.settings["RO_matrix"]["niter"]

        # for all possible states 2^5 --> |00000> ... |11111>
        for i in range(2**nqubits):
            # repeat multiqubit state sequence niter times
            for j in range(self.niter):
                # covert the multiqubit state i into binary representation
                multiqubit_state = bin(i)[2:].zfill(nqubits)
                print(f"Prepared state: {multiqubit_state}")

                # multiqubit_state = |00000>, |00001> ... |11111>
                for n in multiqubit_state:
                    # n = qubit_0 value ... qubit_4 value of a given state
                    seq = PulseSequence()
                    qubit = 0
                    if n == "1":
                        # Define sequence for qubit for Pipulse state
                        RX_pulse = platform.RX_pulse(qubit, start=0)
                        ro_pulse = platform.qubit_readout_pulse(
                            qubit, start=RX_pulse.duration
                        )
                        seq.add(RX_pulse)
                        seq.add(ro_pulse)

                    if n == "0":
                        # Define sequence for qubit Identity state
                        ro_pulse = platform.qubit_readout_pulse(qubit, start=0)
                        seq.add(ro_pulse)
                    qubit = qubit + 1

                platform.start()
                ro_multiqubit_state = platform.execute_pulse_sequence(seq, nshots=1)
                platform.stop()

                # Iterate over list of RO results
                res = ""
                for qubit in range(nqubits):
                    globals()["qubit_state_%s" % qubit] = list(
                        list(ro_multiqubit_state.values())[qubit].values()
                    )[0]
                    I = (globals()[f"qubit_state_{qubit}"])[2]
                    Q = (globals()[f"qubit_state_{qubit}"])[3]
                    point = complex(I, Q)
                    # classify state of qubit n
                    mean_gnd_states = platform.settings["characterization"][
                        "single_qubit"
                    ][qubit]["mean_gnd_states"]
                    mean_gnd = complex(mean_gnd_states)
                    mean_exc_states = platform.settings["characterization"][
                        "single_qubit"
                    ][qubit]["mean_exc_states"]
                    mean_exc = complex(mean_exc_states)
                    res += str(utils.classify(point, mean_gnd, mean_exc))

                # End of processing multiqubit state i
                # populate state i with RO results obtained
                print(f"ReadOut classified state: {res}")
                RO_matrix[i][int(res, 2)] = RO_matrix[i][int(res, 2)] + 1
            # End of repeting RO for a given state i
        # end states
        return np.array(RO_matrix) / self.niter

    def run_drag_pulse_tunning(self, qubit):
        platform = self.platform
        platform.reload_settings()
        sampling_rate = platform.sampling_rate

        res1 = []
        res2 = []
        beta_params = []

        self.reload_settings()
        self.beta_start = self.settings["drag_tunning"]["beta_start"]
        self.beta_end = self.settings["drag_tunning"]["beta_end"]
        self.beta_step = self.settings["drag_tunning"]["beta_step"]

        for beta_param in np.arange(
            self.beta_start, self.beta_end, self.beta_step
        ).round(4):
            # print(f"Executing sequence for beta parameter: {beta_param}")
            # drag pulse RX(pi/2)
            RX90_drag_pulse = platform.RX90_drag_pulse(qubit, start=0, beta=beta_param)
            # drag pulse RY(pi)
            RY_drag_pulse = platform.RX_drag_pulse(
                qubit,
                start=RX90_drag_pulse.duration,
                phase=RX90_drag_pulse.duration
                / sampling_rate
                * (2 * np.pi)
                * RX90_drag_pulse.frequency
                + np.pi / 2,
                beta=beta_param,
            )
            # RO pulse
            ro_pulse = platform.qubit_readout_pulse(
                qubit, start=RX90_drag_pulse.duration + RY_drag_pulse.duration
            )

            # Rx(pi/2) - Ry(pi) - Ro
            seq1 = PulseSequence()
            seq1.add(RX90_drag_pulse)
            seq1.add(RY_drag_pulse)
            seq1.add(ro_pulse)

            platform.start()
            state1 = platform.execute_pulse_sequence(seq1, nshots=10240)
            state1 = list(list(state1.values())[0].values())[0]
            platform.stop()

            # drag pulse RY(pi)
            RY_drag_pulse = platform.RX_drag_pulse(
                qubit, start=0, phase=np.pi / 2, beta=beta_param
            )
            # drag pulse RX(pi/2)
            RX90_drag_pulse = platform.RX90_drag_pulse(
                qubit,
                start=RY_drag_pulse.duration,
                phase=RY_drag_pulse.duration
                / sampling_rate
                * (2 * np.pi)
                * RX90_drag_pulse.frequency,
                beta=beta_param,
            )

            # Ry(pi) - Rx(pi/2) - Ro
            seq2 = PulseSequence()
            seq2.add(RY_drag_pulse)
            seq2.add(RX90_drag_pulse)
            seq2.add(ro_pulse)

            platform.start()
            state2 = platform.execute_pulse_sequence(seq2, nshots=10240)
            state2 = list(list(state2.values())[0].values())[0]
            platform.stop()

            # save IQ_module and beta param of each iteration
            res1.append(state1[0])
            res2.append(state2[0])
            beta_params.append(beta_param)

        beta_optimal = fitting.fit_drag_tunning(res1, res2, beta_params)

        return beta_optimal

    def run_flipping(self, qubit):
        platform = self.platform
        platform.reload_settings()

        self.reload_settings()
        self.niter = self.settings["flipping"]["niter"]
        self.step = self.settings["flipping"]["step"]

        sequence = PulseSequence()
        RX90_pulse = platform.RX90_pulse(qubit, start=0)
        res = []
        N = []

        # Start live plotting. Args = path where the data is going to be stored
        path = qibolab_folder / "calibration" / "data" / "buffer.npy"
        # utils.start_live_plotting(path)

        # repeat N iter times
        for i in range(0, self.niter, self.step):
            # execute sequence RX(pi/2) - [RX(pi) - Rx(pi)] from 0...i times - RO
            sequence.add(RX90_pulse)
            start1 = RX90_pulse.duration
            for j in range(i):
                RX_pulse1 = platform.RX_pulse(qubit, start=start1)
                start2 = start1 + RX_pulse1.duration
                RX_pulse2 = platform.RX_pulse(qubit, start=start2)
                sequence.add(RX_pulse1)
                sequence.add(RX_pulse2)
                start1 = start2 + RX_pulse2.duration

            # add ro pulse at the end of the sequence
            ro_pulse = platform.qubit_readout_pulse(qubit, start=start1)
            sequence.add(ro_pulse)

            # Execute PulseSequence defined by gates
            platform.start()
            state = platform.execute_pulse_sequence(sequence)
            state = list(list(state.values())[0].values())[0]
            platform.stop()
            res += [state[0]]
            N += [i]
            sequence = PulseSequence()

            # Saving data for live plotting
            # np.save(path, np.array([res, N]))

        # Fitting results to obtain epsilon
        if self.resonator_type == "3D":
            popt = fitting.flipping_fit_3D(N, res)
        elif self.resonator_type == "2D":
            popt = fitting.flipping_fit_2D(N, res)

        angle = (self.niter * 2 * np.pi / popt[2] + popt[3]) / (1 + 4 * self.niter)
        state1_voltage = (
            1e-6
            * platform.settings["characterization"]["single_qubit"][qubit][
                "state1_voltage"
            ]
        )
        state0_voltage = (
            1e-6
            * platform.settings["characterization"]["single_qubit"][qubit][
                "state0_voltage"
            ]
        )
        pi_pulse_amplitude = platform.settings["native_gates"]["single_qubit"][qubit][
            "RX"
        ]["amplitude"]
        amplitude_delta = angle * 2 / np.pi * pi_pulse_amplitude
        x = np.arange(0, self.niter, self.step)
        plt.plot(x, res)
        plt.plot(x, np.sin(x * 2 * np.pi / popt[2] + popt[3]) * popt[0] + popt[1])
        plt.ylim(
            [
                np.minimum(state0_voltage, state1_voltage),
                np.maximum(state0_voltage, state1_voltage),
            ]
        )
        plt.show()
        return amplitude_delta

    def auto_calibrate_plaform(self):
        platform = self.platform

        # backup latest platform runcard
        self.backup_config_file()
        for qubit in platform.qubits:
            # run and save cavity spectroscopy calibration
            (
                resonator_freq,
                avg_min_voltage,
                max_ro_voltage,
                smooth_dataset,
                dataset,
            ) = self.run_resonator_spectroscopy(qubit)
            self.save_config_parameter(
                "resonator_freq",
                int(resonator_freq),
                "characterization",
                "single_qubit",
                qubit,
            )
            self.save_config_parameter(
                "resonator_spectroscopy_avg_ro_voltage",
                float(avg_min_voltage),
                "characterization",
                "single_qubit",
                qubit,
            )
            self.save_config_parameter(
                "state0_voltage",
                float(max_ro_voltage),
                "characterization",
                "single_qubit",
                qubit,
            )
            lo_qrm_frequency = int(
                resonator_freq
                - platform.settings["native_gates"]["single_qubit"][qubit]["MZ"][
                    "frequency"
                ]
            )
            self.save_config_parameter(
                "frequency",
                lo_qrm_frequency,
                "instruments",
                platform.lo_qrm[qubit].name,
                "settings",
            )  # TODO: cambiar IF hardcoded

            # run and save qubit spectroscopy calibration
            (
                qubit_freq,
                min_ro_voltage,
                smooth_dataset,
                dataset,
            ) = self.run_qubit_spectroscopy(qubit)
            self.save_config_parameter(
                "qubit_freq", int(qubit_freq), "characterization", "single_qubit", qubit
            )
            RX_pulse_sequence = platform.settings["native_gates"]["single_qubit"][
                qubit
            ]["RX"]["pulse_sequence"]
            lo_qcm_frequency = int(qubit_freq + RX_pulse_sequence[0]["frequency"])
            self.save_config_parameter(
                "frequency",
                lo_qcm_frequency,
                "instruments",
                platform.lo_qcm[qubit].name,
                "settings",
            )
            self.save_config_parameter(
                "qubit_spectroscopy_min_ro_voltage",
                float(min_ro_voltage),
                "characterization",
                "single_qubit",
                qubit,
            )

            # run Rabi and save Pi pulse calibration
            (
                dataset,
                pi_pulse_duration,
                pi_pulse_amplitude,
                state1_voltage,
                t1,
            ) = self.run_rabi_pulse_length(qubit)
            RX_pulse_sequence[0]["duration"] = int(pi_pulse_duration)
            RX_pulse_sequence[0]["amplitude"] = float(pi_pulse_amplitude)
            self.save_config_parameter(
                "pulse_sequence",
                RX_pulse_sequence,
                "native_gates",
                "single_qubit",
                qubit,
                "RX",
            )
            self.save_config_parameter(
                "state1_voltage",
                float(state1_voltage),
                "characterization",
                "single_qubit",
                qubit,
            )

            # run Ramsey and save T2 calibration
            delta_frequency, t2, smooth_dataset, dataset = self.run_ramsey(qubit)
            adjusted_qubit_freq = int(
                platform.characterization["single_qubit"][qubit]["qubit_freq"]
                + delta_frequency
            )
            self.save_config_parameter(
                "qubit_freq",
                adjusted_qubit_freq,
                "characterization",
                "single_qubit",
                qubit,
            )
            self.save_config_parameter(
                "T2", float(t2), "characterization", "single_qubit", qubit
            )
            RX_pulse_sequence = platform.settings["native_gates"]["single_qubit"][
                qubit
            ]["RX"]["pulse_sequence"]
            lo_qcm_frequency = int(
                adjusted_qubit_freq + RX_pulse_sequence[0]["frequency"]
            )
            self.save_config_parameter(
                "frequency",
                lo_qcm_frequency,
                "instruments",
                platform.lo_qcm[qubit].name,
                "settings",
            )

            # run calibration_qubit_states
            (
                all_gnd_states,
                mean_gnd_states,
                all_exc_states,
                mean_exc_states,
            ) = self.calibrate_qubit_states(qubit)
            # print(mean_gnd_states)
            # print(mean_exc_states)
            # TODO: Remove plot qubit states results
            # DEBUG: auto_calibrate_platform - Plot qubit states
            utils.plot_qubit_states(all_gnd_states, all_exc_states)
            self.save_config_parameter(
                "mean_gnd_states",
                mean_gnd_states,
                "characterization",
                "single_qubit",
                qubit,
            )
            self.save_config_parameter(
                "mean_exc_states",
                mean_exc_states,
                "characterization",
                "single_qubit",
                qubit,
            )

    def backup_config_file(self):
        import shutil
        from datetime import datetime

        settings_backups_folder = (
            qibolab_folder / "calibration" / "data" / "settings_backups"
        )
        settings_backups_folder.mkdir(parents=True, exist_ok=True)

        original = str(self.platform.runcard)
        original_file_name = pathlib.Path(original).name
        timestamp = datetime.now()
        timestamp = timestamp.strftime("%Y%m%d%H%M%S")
        destination_file_name = timestamp + "_" + original_file_name
        target = str(settings_backups_folder / destination_file_name)

        shutil.copyfile(original, target)

    def get_config_parameter(self, parameter, *keys):
        import os

        calibration_path = self.platform.runcard
        with open(calibration_path) as file:
            settings = yaml.safe_load(file)
        file.close()

        node = settings
        for key in keys:
            node = node.get(key)
        return node[parameter]

    def save_config_parameter(self, parameter, value, *keys):
        calibration_path = self.platform.runcard
        with open(calibration_path, "r") as file:
            settings = yaml.safe_load(file)
        file.close()

        node = settings
        for key in keys:
            node = node.get(key)
        node[parameter] = value

        # store latest timestamp
        import datetime

        settings["timestamp"] = datetime.datetime.utcnow()

        with open(calibration_path, "w") as file:
            settings = yaml.dump(settings, file, sort_keys=False, indent=4)
        file.close()


# help classes


def settable(instance, attribute_name, label, unit):
    """
    This helper class creates (on the fly) a class that wraps around a given instance attribute
    and complies with quantify settable interface
    """

    def __init__(self, instance):
        self.instance = instance

    def set(self, value):
        setattr(self.instance, attribute_name, value)

    settable_class_instance = type(
        f"settable_{attribute_name}",
        (),
        {
            "name": attribute_name,
            "label": label,
            "unit": unit,
            "__init__": __init__,
            "set": set,
        },
    )(instance)
    return settable_class_instance


class QCPulseAmplitudeParameter:
    label = "Amplitude"
    unit = "-"
    name = "amplitude"

    def __init__(self, qd_pulse):
        self.qd_pulse = qd_pulse

    def set(self, value):
        self.qd_pulse.amplitude = value


class ROPulsePhaseParameter:

    label = "Readout Pulse Phase"
    unit = "rad"
    name = "ro_pulse_phase"

    def __init__(self, ro_pulse):
        self.ro_pulse = ro_pulse

    def set(self, value):
        self.ro_pulse.phase = value


class QCPulseLengthParameter:

    label = "Qubit Control Pulse Length"
    unit = "ns"
    name = "qd_pulse_length"

    def __init__(self, ro_pulse, qd_pulse):
        self.ro_pulse = ro_pulse
        self.qd_pulse = qd_pulse

    def set(self, value):
        self.qd_pulse.duration = int(value)
        self.ro_pulse.start = int(value)


class T1WaitParameter:
    label = "Time"
    unit = "ns"
    name = "t1_wait"
    initial_value = 0

    def __init__(self, ro_pulse, qd_pulse):
        self.ro_pulse = ro_pulse
        self.qd_pulse = qd_pulse

    def set(self, value):
        # TODO: implement following condition
        # must be >= 4ns <= 65535
        # platform.delay_before_readout = value
        self.ro_pulse.start = self.qd_pulse.duration + value


class RamseyWaitParameter:
    label = "Time"
    unit = "ns"
    name = "ramsey_wait"
    initial_value = 0

    def __init__(self, ro_pulse, qc2_pulse, sampling_rate):
        self.ro_pulse = ro_pulse
        self.qc2_pulse = qc2_pulse
        self.pulse_length = qc2_pulse.duration
        self.sampling_rate = sampling_rate

    def set(self, value):
        self.qc2_pulse.start = self.pulse_length + value
        self.qc2_pulse.phase = (
            (self.qc2_pulse.start / self.sampling_rate)
            * (2 * np.pi)
            * self.qc2_pulse.frequency
        )
        self.ro_pulse.start = self.pulse_length * 2 + value


class SpinEchoWaitParameter:
    label = "Time"
    unit = "ns"
    name = "spin_echo_wait"
    initial_value = 0

    def __init__(self, ro_pulse, qc2_pulse, sampling_rate):
        self.ro_pulse = ro_pulse
        self.qc2_pulse = qc2_pulse
        self.pulse_length = qc2_pulse.duration
        self.sampling_rate = sampling_rate

    def set(self, value):
        self.qc2_pulse.start = self.pulse_length + value
        self.qc2_pulse.phase = (
            (self.qc2_pulse.start / self.sampling_rate)
            * (2 * np.pi)
            * self.qc2_pulse.frequency
        )
        self.ro_pulse.start = 2 * self.pulse_length + 2 * value


class SpinEcho3PWaitParameter:
    label = "Time"
    unit = "ns"
    name = "spin_echo_wait"
    initial_value = 0

    def __init__(self, ro_pulse, qc2_pulse, qc3_pulse, sampling_rate):
        self.ro_pulse = ro_pulse
        self.qc2_pulse = qc2_pulse
        self.qc3_pulse = qc3_pulse
        self.pulse_length = qc2_pulse.duration
        self.sampling_rate = sampling_rate

    def set(self, value):
        self.qc2_pulse.start = self.pulse_length + value
        self.qc2_pulse.phase = (
            (self.qc2_pulse.start / self.sampling_rate)
            * (2 * np.pi)
            * self.qc2_pulse.frequency
        )
        self.qc3_pulse.start = 2 * self.pulse_length + 2 * value
        self.qc3_pulse.phase = (
            (self.qc3_pulse.start / self.sampling_rate)
            * (2 * np.pi)
            * self.qc3_pulse.frequency
        )
        self.ro_pulse.start = 3 * self.pulse_length + 2 * value


class RamseyFreqWaitParameter:
    label = "Time"
    unit = "ns"
    name = "ramsey_freq"
    initial_value = 0

    def __init__(self, ro_pulse, qc2_pulse, offset_freq, sampling_rate):
        self.ro_pulse = ro_pulse
        self.qc2_pulse = qc2_pulse
        self.pulse_length = qc2_pulse.duration
        self.offset_freq = offset_freq
        self.sampling_rate = sampling_rate

    def set(self, value):
        self.qc2_pulse.start = self.pulse_length + value
        self.qc2_pulse.phase = (
            (self.qc2_pulse.start / self.sampling_rate)
            * (2 * np.pi)
            * (self.qc2_pulse.frequency - self.offset_freq)
        )
        self.ro_pulse.start = self.pulse_length * 2 + value


class ROController:
    # Quantify Gettable Interface Implementation
    label = ["Amplitude", "Phase", "I", "Q"]
    unit = ["V", "Radians", "V", "V"]
    name = ["A", "Phi", "I", "Q"]

    def __init__(self, platform, sequence, qubit):
        self.platform = platform
        self.sequence = sequence
        self.qubit = qubit

    def get(self):
        results = self.platform.execute_pulse_sequence(self.sequence)
        return list(list(results.values())[0].values())[
            0
        ]  # TODO: Replace with the particular acquisition


class ROControllerNormalised:
    # Quantify Gettable Interface Implementation
    label = ["Normalised Amplitude"]
    unit = ["V"]
    name = ["A"]

    def __init__(self, platform, sequence, qubit, instance):
        self.platform = platform
        self.sequence = sequence
        self.qubit = qubit
        self.instance = instance

    def get(self):
        att = self.instance.device.get("out0_att")
        results = self.platform.execute_pulse_sequence(self.sequence)
        results = list(list(results.values())[0].values())[0][0] * (np.exp(att / 10))
        return results
        # TODO: Replace with the particular acquisition
