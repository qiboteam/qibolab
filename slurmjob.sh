#!/bin/bash

cp -r tests /tmp/
cd /tmp/tests
pytest --cov=qibolab --cov-report=xml -m qpu --platforms $PLATFORM
cd -
mv /tmp/tests/coverage.xml .
rm -r /tmp/tests
