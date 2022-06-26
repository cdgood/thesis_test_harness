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
from band_selection.CAE_SSC import CAE_BS
from band_selection.DSC_NET import DSCBS
from band_selection.ISSC import ISSC_HSI
from band_selection.NDFS import NDFS_HSI
from band_selection.SNMF import BandSelection_SNMF
from band_selection.SpaBS import SpaBS
from band_selection.SPEC import SPEC_HSI
from band_selection.SSR import SSC_BS

### Constants ###
#TODO

### Function Definitions ###

def band_selection(data, gt, **hyperparams):
    """
    """
    band_reduction_method = hyperparams['band_reduction_method']
    n_components = hyperparams['n_components']

    orig_rows, orig_cols, orig_channels = data.shape
    bands_selected = None

    band_selection_start = time.time()

    if band_reduction_method == 'pca':
        print('Using PCA dimensionality reduction on data...')

        # https://towardsdatascience.com/pca-on-hyperspectral-data-99c9c5178385 
        print('Reshaping the data into two dimensions...')
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
        ica.fit(data.astype(np.float32))
        fit_end = time.time()
        fit_runtime = datetime.timedelta(seconds=(fit_end - fit_start))
        print(f'ICA fitting completed! Fit runtime: {fit_runtime}')
        print(f'ICA found {ica.n_features_in_} features while fitting!')
        data = ica.transform(data)
        print(f'New data shape: {data.shape}')
        data = np.reshape(data, (orig_rows, orig_cols, data.shape[-1]))
        print(f'Reshaped new data shape: {data.shape}')

    elif band_reduction_method == 'cae-ssc':
        print('Using CAE SSC dimensionality reduction on data...')
        cae_ssc = CAE_BS(n_band=n_components)
        # cae_ssc = CAE_BS(n_band=data.shape[-1])
        predict_start = time.time()
        # bands_selected = cae_ssc.predict(np.array([gt.flatten(), data.shape[-1]]), 
        #                                  data.reshape(data.shape[0]*data.shape[1], -1))
        # bands_selected = cae_ssc.predict(data.reshape(data.shape[0]*data.shape[1], -1), 
        #                                  data.reshape(data.shape[0]*data.shape[1], -1))
        reduced_data = cae_ssc.predict(data.reshape(data.shape[0]*data.shape[1], -1), 
                                         data.reshape(data.shape[0]*data.shape[1], -1))
        predict_end = time.time()
        predict_runtime = datetime.timedelta(seconds=(predict_end - predict_start))
        print(f'CAE SSC prediction completed! Prediction runtime: {predict_runtime}')
        print(f'Bands selected by CAE SSC: {bands_selected}')
        # data = data[...,bands_selected]
        data = np.reshape(reduced_data, (orig_rows, orig_cols, reduced_data.shape[-1]))
        print(f'New data shape: {data.shape}')
        
    elif band_reduction_method == 'dsc-net':
        print('Using DSC-NET dimensionality reduction on data...')
        dscnet = DSCBS(n_band=n_components, 
                       n_input=(data.shape[0]*data.shape[1], data.shape[2]), 
                       kernel_size=(3,), 
                       n_hidden=2)
        # dscnet = DSCBS(n_band=data.shape[-1])
        predict_start = time.time()
        # bands_selected = dscnet.predict(data.reshape(data.shape[0]*data.shape[1], -1))
        reduced_data = dscnet.predict(data.reshape(data.shape[0]*data.shape[1], -1))
        predict_end = time.time()
        predict_runtime = datetime.timedelta(seconds=(predict_end - predict_start))
        print(f'DSC-NET prediction completed! Prediction runtime: {predict_runtime}')
        print(f'Bands selected by DSC-NET: {bands_selected}')
        # data = data[...,bands_selected]
        data = np.reshape(reduced_data, (orig_rows, orig_cols, reduced_data.shape[-1]))
        print(f'New data shape: {data.shape}')

    elif band_reduction_method == 'issc':
        print('Using ISSC dimensionality reduction on data...')
        issc = ISSC_HSI(n_band=n_components)
        # issc = ISSC_HSI(n_band=data.shape[-1])
        predict_start = time.time()
        # bands_selected = issc.predict(data.reshape(data.shape[0]*data.shape[1], -1))
        reduced_data, bands_selected = issc.predict(data.reshape(data.shape[0]*data.shape[1], -1))
        predict_end = time.time()
        predict_runtime = datetime.timedelta(seconds=(predict_end - predict_start))

        # Sort bands
        bands_selected = sorted(bands_selected)

        print(f'ISSC prediction completed! Prediction runtime: {predict_runtime}')
        print(f'Bands selected by ISSC: {bands_selected}')
        # data = data[...,bands_selected]
        data = np.reshape(reduced_data, (orig_rows, orig_cols, reduced_data.shape[-1]))
        print(f'New data shape: {data.shape}')

    elif band_reduction_method == 'ndfs':
        print('Using NDFS dimensionality reduction on data...')
        ndfs = NDFS_HSI(n_band=data.shape[-1], n_cluster=n_components)
        # ndfs = NDFS_HSI(n_band=data.shape[-1])
        predict_start = time.time()
        # bands_selected = ndfs.predict(data.reshape(data.shape[0]*data.shape[1], -1))
        reduced_data = ndfs.predict(data.reshape(data.shape[0]*data.shape[1], -1))
        predict_end = time.time()
        predict_runtime = datetime.timedelta(seconds=(predict_end - predict_start))
        print(f'NDFS prediction completed! Prediction runtime: {predict_runtime}')
        print(f'Bands selected by NDFS: {bands_selected}')
        # data = data[...,bands_selected]
        data = np.reshape(reduced_data, (orig_rows, orig_cols, reduced_data.shape[-1]))
        print(f'New data shape: {data.shape}')

    elif band_reduction_method == 'snmf':
        print('Using SNMF dimensionality reduction on data...')
        snmf = BandSelection_SNMF(n_band=n_components)
        # snmf = BandSelection_SNMF(n_band=data.shape[-1])
        predict_start = time.time()
        # bands_selected = snmf.predict(data.reshape(data.shape[0]*data.shape[1], -1))
        reduced_data = snmf.predict(data.reshape(data.shape[0]*data.shape[1], -1))
        predict_end = time.time()
        predict_runtime = datetime.timedelta(seconds=(predict_end - predict_start))
        print(f'SNMF prediction completed! Prediction runtime: {predict_runtime}')
        print(f'Bands selected by SNMF: {bands_selected}')
        # data = data[...,bands_selected]
        data = np.reshape(reduced_data, (orig_rows, orig_cols, reduced_data.shape[-1]))
        print(f'New data shape: {data.shape}')

    elif band_reduction_method == 'spabs':
        print('Using SpaBS dimensionality reduction on data...')
        spabs = SpaBS(n_band=n_components, sparsity_level=0.5)
        # spabs = SpaBS(n_band=data.shape[-1], sparsity_level=0.5)
        predict_start = time.time()
        # bands_selected = spabs.predict(data.reshape(data.shape[0]*data.shape[1], -1))
        reduced_data = spabs.predict(data.reshape(data.shape[0]*data.shape[1], -1))
        predict_end = time.time()
        predict_runtime = datetime.timedelta(seconds=(predict_end - predict_start))
        print(f'SpaBS prediction completed! Prediction runtime: {predict_runtime}')
        print(f'Bands selected by SpaBS: {bands_selected}')
        # data = data[...,bands_selected]
        data = np.reshape(reduced_data, (orig_rows, orig_cols, reduced_data.shape[-1]))
        print(f'New data shape: {data.shape}')

    elif band_reduction_method == 'spec':
        print('Using SPEC dimensionality reduction on data...')
        spec = SPEC_HSI(n_band=n_components)
        # spec = SPEC_HSI(n_band=data.shape[-1])
        predict_start = time.time()
        # bands_selected = spec.predict(data.reshape(data.shape[0]*data.shape[1], -1))
        reduced_data = spec.predict(data.reshape(data.shape[0]*data.shape[1], -1))
        predict_end = time.time()
        predict_runtime = datetime.timedelta(seconds=(predict_end - predict_start))
        print(f'SPEC prediction completed! Prediction runtime: {predict_runtime}')
        print(f'Bands selected by SPEC: {bands_selected}')
        # data = data[...,bands_selected]
        data = np.reshape(reduced_data, (orig_rows, orig_cols, reduced_data.shape[-1]))
        print(f'New data shape: {data.shape}')

    elif band_reduction_method == 'ssr' or band_reduction_method == 'ssr-close':
        print('Using SSR (closed form solution) dimensionality reduction on data...')
        ssc = SSC_BS(n_hidden=2, n_clusters=n_components, lambda_coef=1)
        fit_start = time.time()
        data = ssc.fit_predict_close(data.reshape(data.shape[0]*data.shape[1], -1))
        fit_end = time.time()
        fit_runtime = datetime.timedelta(seconds=(fit_end - fit_start))
        print(f'SSR fitting completed! Fit runtime: {fit_runtime}')
        print(f'New data shape: {data.shape}')

    elif band_reduction_method == 'ssr-cvx':
        print('Using SSR (self-expressive representation) dimensionality reduction on data...')
        ssc = SSC_BS(n_hidden=2, n_clusters=n_components, lambda_coef=1)
        fit_start = time.time()
        data = ssc.fit_predict_cvx(data.reshape(data.shape[0]*data.shape[1], -1))
        fit_end = time.time()
        fit_runtime = datetime.timedelta(seconds=(fit_end - fit_start))
        print(f'SSR fitting completed! Fit runtime: {fit_runtime}')
        print(f'New data shape: {data.shape}')

    elif band_reduction_method == 'ssr-omp':
        print('Using SSR (orthogonal matching pursuit) dimensionality reduction on data...')
        ssc = SSC_BS(n_hidden=2, n_clusters=n_components, lambda_coef=1)
        fit_start = time.time()
        data = ssc.fit_predict_omp(data.reshape(data.shape[0]*data.shape[1], -1))
        fit_end = time.time()
        fit_runtime = datetime.timedelta(seconds=(fit_end - fit_start))
        print(f'SSR fitting completed! Fit runtime: {fit_runtime}')
        print(f'New data shape: {data.shape}')

    elif band_reduction_method == 'manual':
        print('Using manually selected bands...')
        if hyperparams['selected_bands'] is not None and type (hyperparams['selected_bands']) is list:
            # Sort bands
            bands_selected = sorted(hyperparams['selected_bands'])
            fit_start = time.time()
            data = data[...,np.array(bands_selected, dtype=int)]
            fit_end = time.time()
            fit_runtime = datetime.timedelta(seconds=(fit_end - fit_start))
            print(f'Manual fitting completed! Fit runtime: {fit_runtime}')
            print(f'Bands selected by manual selection: {bands_selected}')
            print(f'New data shape: {data.shape}')
        else:
            print('Selected bands list for manual selection is invalid! Dimensionality will be unaltered...')
    else:
        print('No valid band selection method chosen! Dimensionality will be unaltered...')

    band_selection_end = time.time()
    band_selection_time = datetime.timedelta(seconds=(band_selection_end - band_selection_start))

    return data, band_selection_time, bands_selected