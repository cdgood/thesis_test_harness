#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Band selection functions module of thesis testing harness.

Author:  Christopher Good
Version: 1.0.0

Usage: band_selection.py

"""
# See following link for proper docstring documentation
# https://pandas.pydata.org/docs/development/contributing_docstring.html 

### Futures ###
#TODO

### Built-in Imports ###
import datetime
import time


### Other Library Imports ###
import numpy as np
from sklearn.decomposition import PCA, FastICA

### Local Imports ###
#TODO

### Constants ###
#TODO

### Function Definitions ###

def band_selection(data, classes, **hyperparams):
    """
    """
    band_reduction_method = hyperparams['band_reduction_method']
    n_components = hyperparams['n_components']

    if band_reduction_method == 'pca':
        print('Using PCA dimensionality reduction on data...')

        # https://towardsdatascience.com/pca-on-hyperspectral-data-99c9c5178385 
        print('Reshaping the data into two dimensions...')
        orig_rows, orig_cols, orig_channels = data.shape
        data = data.reshape(data.shape[0]*data.shape[1], -1)
        print(f'Reshaped data shape: {data.shape}')

        if n_components is None: n_components = 'mle'
        print('Fitting PCA to data...')
        pca = PCA(n_components=n_components,
                  svd_solver='auto',
                  tol=0.0,
                  iterated_power='auto',
                  random_state=hyperparams['random_seed'])
        fit_start = time.time()
        pca.fit(data)
        fit_end = time.time()
        fit_runtime = datetime.timedelta(seconds=(fit_end - fit_start))
        print(f'PCA fitting completed! Fit runtime: {fit_runtime}')

        print(f'PCA fit data to {pca.n_components_} components!')
        data = pca.transform(data)
        print(f'New data shape: {data.shape}')
        data = np.reshape(data, (orig_rows, orig_cols, data.shape[-1]))
        print(f'Reshaped new data shape: {data.shape}')
    elif band_reduction_method == 'ica':
        print('Using ICA dimensionality reduction on data...')

        print('Reshaping the data into two dimensions...')
        orig_rows, orig_cols, orig_channels = data.shape
        data = data.reshape(data.shape[0]*data.shape[1], -1)
        print(f'Reshaped data shape: {data.shape}')

        print('Fitting ICA to data...')
        ica = FastICA(n_components=n_components,
                  algorithm='parallel',
                  fun='logcosh',
                  fun_args=None,
                  max_iter=200,
                  tol=1e-4,
                  w_init=None,
                  random_state=hyperparams['random_seed'])
        fit_start = time.time()
        ica.fit(data)
        fit_end = time.time()
        fit_runtime = datetime.timedelta(seconds=(fit_end - fit_start))
        print(f'ICA fitting completed! Fit runtime: {fit_runtime}')
        print(f'ICA found {ica.n_features_in_} features while fitting!')
        data = ica.transform(data)
        print(f'New data shape: {data.shape}')
        data = np.reshape(data, (orig_rows, orig_cols, data.shape[-1]))
        print(f'Reshaped new data shape: {data.shape}')
    else:
        print('No valid band selection method chosen! Dimensionality will be unaltered...')

    return data