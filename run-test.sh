#!/bin/bash

if (( $# == 0 )); then
    TEST_CASE="test-model"
else
    TEST_CASE=$1
fi

python3 test.py --case $TEST_CASE
