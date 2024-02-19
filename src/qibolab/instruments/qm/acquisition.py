from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from qm import qua
from qm.qua import declare, declare_stream, fixed
from qm.qua._dsl import _ResultSource, _Variable  # for type declaration only
from qualang_tools.addons.variables import assign_variables_to_element
from qualang_tools.units import unit

from qibolab.execution_parameters import AcquisitionType, AveragingMode
from qibolab.qubits import QubitId
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

    This class can be instantiated only within a QUA program scope. Each
    readout pulse is associated with its own set of acquisition
    variables.
    """

    name: str
    """Name of the acquisition used as identifier to download results from the
    instruments."""
    qubit: QubitId
    average: bool

    keys: list[str] = field(default_factory=list)

    @property
    def npulses(self):
        return len(self.keys)

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

    adc_stream: _ResultSource = field(
        default_factory=lambda: declare_stream(adc_trace=True)
    )
    """Stream to collect raw ADC data."""

    def assign_element(self, element):
        pass

    def measure(self, operation, element):
        qua.measure(operation, element, self.adc_stream)

    def download(self, *dimensions):
        istream = self.adc_stream.input1()
        qstream = self.adc_stream.input2()
        if self.average:
            istream = istream.average()
            qstream = qstream.average()
        istream.save(f"{self.name}_I")
        qstream.save(f"{self.name}_Q")

    def fetch(self, handles):
        ires = handles.get(f"{self.name}_I").fetch_all()
        qres = handles.get(f"{self.name}_Q").fetch_all()
        # convert raw ADC signal to volts
        u = unit()
        signal = u.raw2volts(ires) + 1j * u.raw2volts(qres)
        if self.average:
            return [AveragedRawWaveformResults(signal)]
        return [RawWaveformResults(signal)]


@dataclass
class IntegratedAcquisition(Acquisition):
    """QUA variables used for integrated acquisition."""

    i: _Variable = field(default_factory=lambda: declare(fixed))
    q: _Variable = field(default_factory=lambda: declare(fixed))
    """Variables to save the (I, Q) values acquired from a single shot."""
    istream: _ResultSource = field(default_factory=lambda: declare_stream())
    qstream: _ResultSource = field(default_factory=lambda: declare_stream())
    """Streams to collect the results of all shots."""

    def assign_element(self, element):
        assign_variables_to_element(element, self.i, self.q)

    def measure(self, operation, element):
        qua.measure(
            operation,
            element,
            None,
            qua.dual_demod.full("cos", "out1", "sin", "out2", self.i),
            qua.dual_demod.full("minus_sin", "out1", "cos", "out2", self.q),
        )
        qua.save(self.I, self.istream)
        qua.save(self.Q, self.qstream)

    def download(self, *dimensions):
        istream = self.istream
        qstream = self.qstream
        if self.npulses > 1:
            istream = istream.buffer(self.npulses)
            qstream = qstream.buffer(self.npulses)
        for dim in dimensions:
            istream = istream.buffer(dim)
            qstream = qstream.buffer(dim)
        if self.average:
            istream = istream.average()
            qstream = qstream.average()
        istream.save(f"{self.name}_I")
        qstream.save(f"{self.name}_Q")

    def fetch(self, handles):
        ires = handles.get(f"{self.name}_I").fetch_all()
        qres = handles.get(f"{self.name}_Q").fetch_all()
        signal = ires + 1j * qres
        if self.npulses > 1:
            if self.average:
                # TODO: calculate std
                return [
                    AveragedIntegratedResults(signal[..., i])
                    for i in range(self.npulses)
                ]
            return [IntegratedResults(signal[..., i]) for i in range(self.npulses)]
        else:
            if self.average:
                # TODO: calculate std
                return [AveragedIntegratedResults(signal)]
            return [IntegratedResults(signal)]


@dataclass
class ShotsAcquisition(Acquisition):
    """QUA variables used for shot classification.

    Threshold and angle must be given in order to classify shots.
    """

    threshold: Optional[float] = None
    """Threshold to be used for classification of single shots."""
    angle: Optional[float] = None
    """Angle in the IQ plane to be used for classification of single shots."""

    i: _Variable = field(default_factory=lambda: declare(fixed))
    q: _Variable = field(default_factory=lambda: declare(fixed))
    """Variables to save the (I, Q) values acquired from a single shot."""
    shot: _Variable = field(default_factory=lambda: declare(int))
    """Variable for calculating an individual shots."""
    shots: _ResultSource = field(default_factory=lambda: declare_stream())
    """Stream to collect multiple shots."""

    def __post_init__(self):
        self.cos = np.cos(self.angle)
        self.sin = np.sin(self.angle)

    def assign_element(self, element):
        assign_variables_to_element(element, self.i, self.q, self.shot)

    def measure(self, operation, element):
        qua.measure(
            operation,
            element,
            None,
            qua.dual_demod.full("cos", "out1", "sin", "out2", self.i),
            qua.dual_demod.full("minus_sin", "out1", "cos", "out2", self.q),
        )
        qua.assign(
            self.shot,
            qua.Cast.to_int(self.i * self.cos - self.q * self.sin > self.threshold),
        )
        qua.save(self.shot, self.shots)

    def download(self, *dimensions):
        shots = self.shots
        if self.npulses > 1:
            shots = shots.buffer(self.npulses)
        for dim in dimensions:
            shots = shots.buffer(dim)
        if self.average:
            shots = shots.average()
        shots.save(f"{self.name}_shots")

    def fetch(self, handles):
        shots = handles.get(f"{self.name}_shots").fetch_all()
        if self.npulses > 1:
            if self.average:
                # TODO: calculate std
                return [
                    AveragedSampleResults(shots[..., i]) for i in range(self.npulses)
                ]
            return [
                SampleResults(shots[..., i].astype(int)) for i in range(self.npulses)
            ]
        else:
            if self.average:
                # TODO: calculate std
                return [AveragedSampleResults(shots)]
            return [SampleResults(shots.astype(int))]


ACQUISITION_TYPES = {
    AcquisitionType.RAW: RawAcquisition,
    AcquisitionType.INTEGRATION: IntegratedAcquisition,
    AcquisitionType.DISCRIMINATION: ShotsAcquisition,
}


def declare_acquisitions(ro_pulses, qubits, options):
    """Declares variables for saving acquisition in the QUA program.

    Args:
        ro_pulses (list): List of readout pulses in the sequence.
        qubits (dict): Dictionary containing all the :class:`qibolab.qubits.Qubit`
            objects of the platform.
        options (:class:`qibolab.execution_parameters.ExecutionParameters`): Execution
            options containing acquisition type and averaging mode.

    Returns:
        Dictionary containing the different :class:`qibolab.instruments.qm.acquisition.Acquisition` objects.
    """
    acquisitions = {}
    for qmpulse in ro_pulses:
        qubit = qmpulse.pulse.qubit
        name = f"{qmpulse.operation}_{qubit}"
        if name not in acquisitions:
            average = options.averaging_mode is AveragingMode.CYCLIC
            acquisition_cls = ACQUISITION_TYPES[options.acquisition_type]
            if options.acquisition_type is AcquisitionType.DISCRIMINATION:
                threshold = qubits[qubit].threshold
                iq_angle = qubits[qubit].iq_angle
                acquisition = acquisition_cls(
                    name, qubit, average, threshold=threshold, angle=iq_angle
                )
            else:
                acquisition = acquisition_cls(name, qubit, average)

            acquisition.assign_element(qmpulse.element)
            acquisitions[name] = acquisition

        acquisitions[name].keys.append(qmpulse.pulse.serial)
        qmpulse.acquisition = acquisitions[name]
    return acquisitions


def fetch_results(result, acquisitions):
    """Fetches results from an executed experiment.

    Args:
        result: Result of the executed experiment.
        acquisition (dict): Dictionary containing :class:`qibolab.instruments.qm.acquisition.Acquisition` objects.

    Returns:
        Dictionary with the results in the format required by the platform.
    """
    handles = result.result_handles
    handles.wait_for_all_values()  # for async replace with ``handles.is_processing()``
    results = {}
    for acquisition in acquisitions.values():
        data = acquisition.fetch(handles)
        for serial, result in zip(acquisition.keys, data):
            results[acquisition.qubit] = results[serial] = result
    return results
