import pytest

from qibolab.instruments.qblox.controller import QbloxController


@pytest.fixture(scope="module")
def controller(platform):
    for instrument in platform.instruments:
        if isinstance(instrument, QbloxController):
            return instrument
    pytest.skip(f"Skipping qblox test for {platform.name}.")


@pytest.fixture(scope="module")
def cluster(controller):
    cluster = controller.cluster
    return Cluster(cluster.name, cluster.address, Cluster_Settings())
