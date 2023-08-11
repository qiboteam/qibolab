from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from qm import qua
from qm.qua import declare, declare_stream, fixed
from qm.qua._dsl import _ResultSource, _Variable  # for type declaration only
from qualang_tools.addons.variables import assign_variables_to_element
from qualang_tools.units import unit

from qibolab.result import (
    AveragedIntegratedResults,
    AveragedRawWaveformResults,
    AveragedSampleResults,
    IntegratedResults,
    RawWaveformResults,
    SampleResults,
)


@dataclass
class Acquisition(ABC):
    """QUA variables used for saving of acquisition results.

    This class can be instantiated only within a QUA program scope.
    Each readout pulse is associated with its own set of acquisition variables.
    """

    serial: str
    """Serial of the readout pulse that generates this acquisition."""
    average: bool

    @abstractmethod
    def assign_element(self, element):
        """Assign acquisition variables to the corresponding QM controlled.

        Proposed to do by QM to avoid crashes.

        Args:
            element (str): Element (from ``config``) that the pulse will be applied on.
        """

    @abstractmethod
    def measure(self, operation, element):
        """Send measurement pulse and acquire results.

        Args:
            operation (str): Operation (from ``config``) corresponding to the pulse to be played.
            element (str): Element (from ``config``) that the pulse will be applied on.
        """

    @abstractmethod
    def save(self):
        """Save acquired results from variables to streams."""

    @abstractmethod
    def download(self, *dimensions):
        """Save streams to prepare for fetching from host device.

        Args:
            dimensions (int): Dimensions to use for buffer of data.
        """

    @abstractmethod
    def fetch(self):
        """Fetch downloaded streams to host device."""


@dataclass
class RawAcquisition(Acquisition):
    """QUA variables used for raw waveform acquisition."""

    adc_stream: _ResultSource = field(default_factory=lambda: declare_stream(adc_trace=True))
    """Stream to collect raw ADC data."""

    def assign_element(self, element):
        pass

    def measure(self, operation, element):
        qua.measure(operation, element, self.adc_stream)

    def save(self):
        pass

    def download(self, *dimensions):
        i_stream = self.adc_stream.input1()
        q_stream = self.adc_stream.input2()
        if self.average:
            i_stream = i_stream.average()
            q_stream = q_stream.average()
        i_stream.save(f"{self.serial}_I")
        q_stream.save(f"{self.serial}_Q")

    def fetch(self, handles):
        ires = handles.get(f"{self.serial}_I").fetch_all()
        qres = handles.get(f"{self.serial}_Q").fetch_all()
        # convert raw ADC signal to volts
        u = unit()
        ires = u.raw2volts(ires)
        qres = u.raw2volts(qres)
        if self.average:
            return AveragedRawWaveformResults(ires + 1j * qres)
        return RawWaveformResults(ires + 1j * qres)


@dataclass
class IntegratedAcquisition(Acquisition):
    """QUA variables used for integrated acquisition."""

    I: _Variable = field(default_factory=lambda: declare(fixed))
    Q: _Variable = field(default_factory=lambda: declare(fixed))
    """Variables to save the (I, Q) values acquired from a single shot."""
    I_stream: _ResultSource = field(default_factory=lambda: declare_stream())
    Q_stream: _ResultSource = field(default_factory=lambda: declare_stream())
    """Streams to collect the results of all shots."""

    def assign_element(self, element):
        assign_variables_to_element(element, self.I, self.Q)

    def measure(self, operation, element):
        qua.measure(
            operation,
            element,
            None,
            qua.dual_demod.full("cos", "out1", "sin", "out2", self.I),
            qua.dual_demod.full("minus_sin", "out1", "cos", "out2", self.Q),
        )

    def save(self):
        qua.save(self.I, self.I_stream)
        qua.save(self.Q, self.Q_stream)

    def download(self, *dimensions):
        Istream = self.I_stream
        Qstream = self.Q_stream
        for dim in dimensions:
            Istream = Istream.buffer(dim)
            Qstream = Qstream.buffer(dim)
        if self.average:
            Istream = Istream.average()
            Qstream = Qstream.average()
        Istream.save(f"{self.serial}_I")
        Qstream.save(f"{self.serial}_Q")

    def fetch(self, handles):
        ires = handles.get(f"{self.serial}_I").fetch_all()
        qres = handles.get(f"{self.serial}_Q").fetch_all()
        if self.average:
            # TODO: calculate std
            return AveragedIntegratedResults(ires + 1j * qres)
        return IntegratedResults(ires + 1j * qres)


@dataclass
class ShotsAcquisition(Acquisition):
    """QUA variables used for shot classification.

    Threshold and angle must be given in order to classify shots.
    """

    threshold: float
    """Threshold to be used for classification of single shots."""
    angle: float
    """Angle in the IQ plane to be used for classification of single shots."""

    I: _Variable = field(default_factory=lambda: declare(fixed))
    Q: _Variable = field(default_factory=lambda: declare(fixed))
    """Variables to save the (I, Q) values acquired from a single shot."""
    shot: _Variable = field(default_factory=lambda: declare(int))
    """Variable for calculating an individual shots."""
    shots: _ResultSource = field(default_factory=lambda: declare_stream())
    """Stream to collect multiple shots."""

    def __post_init__(self):
        self.cos = np.cos(self.angle)
        self.sin = np.sin(self.angle)

    def assign_element(self, element):
        assign_variables_to_element(element, self.I, self.Q, self.shot)

    def measure(self, operation, element):
        qua.measure(
            operation,
            element,
            None,
            qua.dual_demod.full("cos", "out1", "sin", "out2", self.I),
            qua.dual_demod.full("minus_sin", "out1", "cos", "out2", self.Q),
        )
        qua.assign(self.shot, qua.Cast.to_int(self.I * self.cos - self.Q * self.sin > self.threshold))

    def save(self):
        qua.save(self.shot, self.shots)

    def download(self, *dimensions):
        shots = self.shots
        for dim in dimensions:
            shots = shots.buffer(dim)
        if self.average:
            shots = shots.average()
        shots.save(f"{self.serial}_shots")

    def fetch(self, handles):
        shots = handles.get(f"{self.serial}_shots").fetch_all()
        if self.average:
            # TODO: calculate std
            return AveragedSampleResults(shots)
        return SampleResults(shots.astype(int))
