#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Main script for thesis test harness.

Author:  Christopher Good
Version: 1.0.0

Usage: main.py

"""
# See following link for proper docstring documentation
# https://pandas.pydata.org/docs/development/contributing_docstring.html 

### Futures ###
#TODO

### Built-in Imports ###
import datetime
import os
import time

### Other Library Imports ###
#TODO

### Local Imports ###
from test_harness.command_line_parser import test_harness_parser
from test_harness.test_harness import run_test_harness

### Environment ###
# remove abundant output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

### Constants ###
#TODO

### Definitions ###
#TODO

### Main ###

if __name__ == "__main__":
    # Start timing experiments
    test_harness_start = time.time()

    # Arguments
    parser = test_harness_parser()
    args = parser.parse_args()

    hyperparams = vars(args)

    # Run the test harness
    run_test_harness(**hyperparams)

    test_harness_end = time.time()
    test_harness_runtime = datetime.timedelta(seconds=(test_harness_end - test_harness_start))

    print(f' < Total Test Harness Runtime: {test_harness_runtime} >')

    print()
    print('   ~~~ EXITTING TEST HARNESS PROGRAM ~~~')
    print()