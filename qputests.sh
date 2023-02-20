#!/bin/bash

# Script for running the selfhosted tests on QPUs directly from GitHub
# Tests need to be copied to /tmp/ because coverage does not work with NFS

cp -r tests /tmp/
cp pyproject.toml /tmp/
cd /tmp/tests
pytest -m qpu --platforms $PLATFORM
cd -
mv /tmp/tests/coverage.xml .
mv /tmp/tests/htmlcov .
rm -r /tmp/tests
