#!/bin/bash

# Script for running the selfhosted tests on QPUs directly from GitHub
# Tests need to be copied to /tmp/ because coverage does not work with NFS

cp -r tests /tmp/
cd /tmp/tests
pytest --cov=qibolab --cov-report=xml -m qpu --platforms $PLATFORM
cd -
mv /tmp/tests/coverage.xml .
rm -r /tmp/tests
