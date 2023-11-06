import pytest

from qibolab.instruments.qblox.cluster import Cluster, Cluster_Settings
from qibolab.instruments.qblox.controller import QbloxController


def get_controller(platform):
    for instrument in platform.instruments:
        if isinstance(instrument, QbloxController):
            return instrument


@pytest.fixture(scope="module")
def controller(platform):
    return get_controller(platform)


def get_cluster(controller):
    cluster = controller.cluster
    return Cluster(cluster.name, cluster.address, Cluster_Settings())


@pytest.fixture(scope="module")
def cluster(controller):
    return get_cluster(controller)


@pytest.fixture(scope="module")
def connected_controller(connected_platform):
    return get_controller(connected_platform)


@pytest.fixture(scope="module")
def connected_cluster(connected_controller):
    cluster = get_cluster(connected_controller)
    cluster.connect()
    return cluster
