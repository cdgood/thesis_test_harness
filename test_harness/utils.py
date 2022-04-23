#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions module of thesis testing harness.

Author:  Christopher Good
Version: 1.0.0

Usage: utils.py

"""
# See following link for proper docstring documentation
# https://pandas.pydata.org/docs/development/contributing_docstring.html 

### Futures ###
#TODO

### Built-in Imports ###
#TODO

### Other Library Imports ###
import numpy as np
import tensorflow as tf

### Local Imports ###
from data.datasets import  get_valid_gt_indices

### Constants ###
#TODO

### Function Definitions ###

def get_device(ordinal):
    """
    Takes a GPU device identifier and, if available, returns the device,
    and if not returns the CPU device.

    Parameters
    ----------
    ordinal : int
        The Tensorflow device ordinal ID

    Returns
    -------
    device 
        A context manager for the specified device to use for newly created ops
    """
    if ordinal < 0:
        print("Computation on CPU")
        device = '/CPU:0'
    elif len(tf.config.list_physical_devices('GPU')) > 0:
        print(f'Computation on CUDA GPU device {ordinal}')
        device = f'/GPU:{ordinal}'
    else:
        print("<!> CUDA was requested but is not available! Computation will go on CPU. <!>")
        device = '/CPU:0'
    return device

def prime_generator():
    """ 
    Generate an infinite sequence of prime numbers.

    Sieve of Eratosthenes
    Code by David Eppstein, UC Irvine, 28 Feb 2002
    http://code.activestate.com/recipes/117119/
    """
    # Maps composites to primes witnessing their compositeness.
    # This is memory efficient, as the sieve is not "run forward"
    # indefinitely, but only as long as required by the current
    # number being tested.
    #
    D = {}
    
    # The running integer that's checked for primeness
    q = 2
    
    while True:
        if q not in D:
            # q is a new prime.
            # Yield it and mark its first multiple that isn't
            # already marked in previous iterations
            # 
            yield q
            D[q * q] = [q]
        else:
            # q is composite. D[q] is the list of primes that
            # divide it. Since we've reached q, we no longer
            # need it in the map, but we'll mark the next 
            # multiples of its witnesses to prepare for larger
            # numbers
            # 
            for p in D[q]:
                D.setdefault(p + q, []).append(p)
            del D[q]
        
        q += 1

def preprocess_data(data, **hyperparams):
    """
    """
    #TODO
    return data

def postprocess_data(pred_test, **hyperparams):
    """
    """
    #TODO
    return pred_test

def filter_pred_results(test_gt, pred_test, ignored_labels):
    """
    """
    # Reshape pred_test to be the same size as train_gt
    pred_test = np.reshape(pred_test, test_gt.shape)
    indices = get_valid_gt_indices(test_gt, ignored_labels=ignored_labels)
    target_test = np.array([test_gt[x, y] for x, y in indices])
    pred_test = np.array([pred_test[x, y] for x, y in indices])

    return target_test, pred_test