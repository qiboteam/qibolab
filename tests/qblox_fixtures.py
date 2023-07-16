import pytest

from qibolab.instruments.qblox.cluster import Cluster, Cluster_Settings
from qibolab.instruments.qblox.controller import QbloxController


@pytest.fixture(scope="session")
def controller(platform):
    for instrument in platform.instruments:
        if isinstance(instrument, QbloxController):
            return instrument
    pytest.skip(f"Skipping qblox test for {platform.name}.")


@pytest.fixture(scope="session")
def cluster(controller):
    cluster = controller.cluster
    return Cluster(cluster.name, cluster.address, Cluster_Settings())
