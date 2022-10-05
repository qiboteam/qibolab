# -*- coding: utf-8 -*-
import pytest


def pytest_addoption(parser):
    parser.addoption("--platforms", type=str, action="store", default="tii5q", help="qpu platforms to test on")
    parser.addoption("--skip-qpu", action="store_true", help="skip tests that require qpu")
    parser.addoption("--skip-no-qpu", action="store_true", help="skip tests that do not require qpu")


def pytest_configure(config):
    config.addinivalue_line("markers", "qpu: mark tests that require qpu")


def pytest_runtest_setup(item):
    marked_qpu = "qpu" in {mark.name for mark in item.iter_markers()}
    if item.config.getoption("--skip-qpu") and marked_qpu:
        pytest.skip("Skipping test that requires qpu.")
    elif item.config.getoption("--skip-no-qpu") and not marked_qpu:
        pytest.skip("Skipping test that does not require qpu.")


def pytest_generate_tests(metafunc):
    platforms = metafunc.config.option.platforms.split(",")
    if "platform_name" in metafunc.fixturenames:
        metafunc.parametrize("platform_name", platforms)
