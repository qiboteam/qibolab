from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from qm import qua
from qm.qua import declare, declare_stream, fixed
from qm.qua._dsl import _ResultSource, _Variable  # for type declaration only
from qualang_tools.addons.variables import assign_variables_to_element
from qualang_tools.units import unit

from qibolab.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab.result import (
    AveragedIntegratedResults,
    AveragedRawWaveformResults,
    AveragedSampleResults,
    IntegratedResults,
    RawWaveformResults,
    SampleResults,
)

# TODO: Change name to operation?


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
    element: str
    """Element from QM ``config`` that the pulse will be applied on."""
    average: bool

    keys: list[str] = field(default_factory=list)

    RESULT_CLS = IntegratedResults
    """Result object type that corresponds to this acquisition type."""
    AVERAGED_RESULT_CLS = AveragedIntegratedResults
    """Averaged result object type that corresponds to this acquisition
    type."""

    @property
    def npulses(self):
        return len(self.keys)

    @abstractmethod
    def declare(self):
        """Declares QUA variables related to this acquisition.

        Assigns acquisition variables to the corresponding QM
        controller. This was proposed by QM to avoid crashes.
        """

    @abstractmethod
    def measure(self, operation):
        """Send measurement pulse and acquire results.

        Args:
            operation (str): Operation (from ``config``) corresponding to the pulse to be played.
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

    def result(self, data):
        """Creates Qibolab result object that is returned to the platform."""
        res_cls = self.AVERAGED_RESULT_CLS if self.average else self.RESULT_CLS
        if self.npulses > 1:
            return [res_cls(data[..., i]) for i in range(self.npulses)]
        return [res_cls(data)]


@dataclass
class RawAcquisition(Acquisition):
    """QUA variables used for raw waveform acquisition."""

    adc_stream: Optional[_ResultSource] = None
    """Stream to collect raw ADC data."""

    RESULT_CLS = RawWaveformResults
    AVERAGED_RESULT_CLS = AveragedRawWaveformResults

    def declare(self):
        self.adc_stream = declare_stream(adc_trace=True)

    def measure(self, operation, element):
        qua.reset_phase(element)
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
        return self.result(signal)


@dataclass
class IntegratedAcquisition(Acquisition):
    """QUA variables used for integrated acquisition."""

    i: Optional[_Variable] = None
    q: Optional[_Variable] = None
    """Variables to save the (I, Q) values acquired from a single shot."""
    istream: Optional[_ResultSource] = None
    qstream: Optional[_ResultSource] = None
    """Streams to collect the results of all shots."""

    RESULT_CLS = IntegratedResults
    AVERAGED_RESULT_CLS = AveragedIntegratedResults

    def declare(self):
        self.i = declare(fixed)
        self.q = declare(fixed)
        self.istream = declare_stream()
        self.qstream = declare_stream()
        assign_variables_to_element(self.element, self.i, self.q)

    def measure(self, operation):
        qua.measure(
            operation,
            self.element,
            None,
            qua.dual_demod.full("cos", "out1", "sin", "out2", self.i),
            qua.dual_demod.full("minus_sin", "out1", "cos", "out2", self.q),
        )
        qua.save(self.i, self.istream)
        qua.save(self.q, self.qstream)

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
        return self.result(ires + 1j * qres)


@dataclass
class ShotsAcquisition(Acquisition):
    """QUA variables used for shot classification.

    Threshold and angle must be given in order to classify shots.
    """

    threshold: Optional[float] = None
    """Threshold to be used for classification of single shots."""
    angle: Optional[float] = None
    """Angle in the IQ plane to be used for classification of single shots."""

    i: Optional[_Variable] = None
    q: Optional[_Variable] = None
    """Variables to save the (I, Q) values acquired from a single shot."""
    shot: Optional[_Variable] = None
    """Variable for calculating an individual shots."""
    shots: Optional[_ResultSource] = None
    """Stream to collect multiple shots."""

    RESULT_CLS = SampleResults
    AVERAGED_RESULT_CLS = AveragedSampleResults

    def __post_init__(self):
        self.cos = np.cos(self.angle)
        self.sin = np.sin(self.angle)

    def declare(self):
        self.i = declare(fixed)
        self.q = declare(fixed)
        self.shot = declare(int)
        self.shots = declare_stream()
        assign_variables_to_element(self.element, self.i, self.q, self.shot)

    def measure(self, operation):
        qua.measure(
            operation,
            self.element,
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
        return self.result(shots)


ACQUISITION_TYPES = {
    AcquisitionType.RAW: RawAcquisition,
    AcquisitionType.INTEGRATION: IntegratedAcquisition,
    AcquisitionType.DISCRIMINATION: ShotsAcquisition,
}


def create_acquisition(
    operation: str,
    element: str,
    options: ExecutionParameters,
    threshold: float,
    angle: float,
):
    """Create container for the variables used for saving acquisition in the
    QUA program.

    Args:
        operation (str):
        element (str):
        options (:class:`qibolab.execution_parameters.ExecutionParameters`): Execution
            options containing acquisition type and averaging mode.

    Returns:
        :class:`qibolab.instruments.qm.acquisition.Acquisition` object containing acquisition variables.
    """
    average = options.averaging_mode is AveragingMode.CYCLIC
    kwargs = {}
    if options.acquisition_type is AcquisitionType.DISCRIMINATION:
        kwargs = {"threshold": threshold, "angle": angle}
    acquisition = ACQUISITION_TYPES[options.acquisition_type](
        operation, element, average, **kwargs
    )
    return acquisition


def fetch_results(result, acquisitions):
    """Fetches results from an executed experiment.

    Args:
        result: Result of the executed experiment.
        acquisition: Dictionary containing :class:`qibolab.instruments.qm.acquisition.Acquisition` objects.

    Returns:
        Dictionary with the results in the format required by the platform.
    """
    handles = result.result_handles
    handles.wait_for_all_values()  # for async replace with ``handles.is_processing()``
    results = defaultdict(list)
    for acquisition in acquisitions:
        data = acquisition.fetch(handles)
        for serial, result in zip(acquisition.keys, data):
            results[serial].append(result)

    # collapse single element lists for back-compatibility
    return {
        key: value[0] if len(value) == 1 else value for key, value in results.items()
    }
