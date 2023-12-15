import pytest

from qibolab.instruments.abstract import Instrument
from qibolab.instruments.qblox.cluster import (
    Cluster,
    Cluster_Settings,
    ReferenceClockSource,
)

from .qblox_fixtures import (  # nopycln: import
    cluster,
    connected_cluster,
    connected_controller,
    controller,
)


def test_ReferenceClockSource():
    # Test the values of the enum match with the values required by qblox parameter
    assert ReferenceClockSource.INTERNAL.value == "internal"
    assert ReferenceClockSource.EXTERNAL.value == "external"


def test_ClusterSettings():
    # Test default value
    cs = Cluster_Settings()
    assert cs.reference_clock_source == ReferenceClockSource.INTERNAL
    # Test initialisation with all possible values
    cs = Cluster_Settings(reference_clock_source=ReferenceClockSource.INTERNAL)
    cs = Cluster_Settings(reference_clock_source=ReferenceClockSource.EXTERNAL)


def test_instrument_interface(cluster: Cluster):
    # Test compliance with :class:`qibolab.instruments.abstract.Instrument` interface
    for abstract_method in Instrument.__abstractmethods__:
        assert hasattr(cluster, abstract_method)

    for attribute in [
        "name",
        "address",
        "is_connected",
        "signature",
        "tmp_folder",
        "data_folder",
    ]:
        assert hasattr(cluster, attribute)


def test_init(cluster: Cluster):
    assert cluster.settings.reference_clock_source == ReferenceClockSource.INTERNAL
    assert cluster.device == None


def test_reference_clock_source(cluster: Cluster):
    cluster.reference_clock_source = ReferenceClockSource.EXTERNAL
    assert cluster.settings.reference_clock_source == ReferenceClockSource.EXTERNAL
    cluster.settings.reference_clock_source = ReferenceClockSource.INTERNAL
    assert cluster.reference_clock_source == ReferenceClockSource.INTERNAL


# def test_connect_error(cluster: Cluster):
#     cluster.address = "192.168.0.0"
#     with pytest.raises(InstrumentException):
#         cluster.connect()
#     cluster.address = ADDRESS


@pytest.mark.qpu
def test_connect(connected_cluster: Cluster):
    assert connected_cluster.is_connected


@pytest.mark.qpu
def test_setup(connected_cluster: Cluster):
    connected_cluster.setup()


@pytest.mark.qpu
def test_start_stop(connected_cluster: Cluster):
    connected_cluster.start()
    connected_cluster.stop()


@pytest.mark.qpu
def test_reference_clock_source_device(connected_cluster: Cluster):
    connected_cluster.reference_clock_source = ReferenceClockSource.EXTERNAL
    assert connected_cluster.device.get("reference_source") == "external"
    connected_cluster.reference_clock_source = ReferenceClockSource.INTERNAL
    assert connected_cluster.device.get("reference_source") == "internal"
