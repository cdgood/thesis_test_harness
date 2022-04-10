#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test harness module for Automatic Hyperspectral Band Selection

This script is the test harness for experimenting on automatic
hyperspectral (HS) band selection for classifying hyperspectral images.

Author:  Christopher Good
Version: 1.0.0

Usage: test_harness.py

"""
# See following link for proper docstring documentation
# https://pandas.pydata.org/docs/development/contributing_docstring.html 

### Futures ###
#TODO

### Built-in Imports ###
import argparse
import datetime
from operator import truediv
import os
from pathlib import Path
import time
import traceback


### Other Library Imports ###
import cpuinfo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA, FastICA
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)

### Local Imports ###
from datasets import (
    # preprocess_data,
    sample_gt,
    create_datasets,
    load_grss_dfc_2018_uh_dataset,
    load_indian_pines_dataset,
    load_pavia_center_dataset,
    load_university_of_pavia_dataset,
)
from models import (
    fusion_fcn_model,
    fusion_fcn_v2_model,
    get_optimizer,
    densenet_2d_model,
    densenet_2d_multi_model,
    densenet_3d_model,
    cnn_2d_model,
    cnn_3d_model,
    baseline_cnn_model,
)

### Environment ###
# remove abundant output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

### Constants ###

### Definitions ###

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
    
    #TODO
    return data

def band_selection(data, classes, **hyperparams):
    
    band_reduction_method = hyperparams['band_reduction_method']
    n_components = hyperparams['n_components']

    if band_reduction_method == 'pca':
        print('Using PCA dimensionality reduction on data...')

        # https://towardsdatascience.com/pca-on-hyperspectral-data-99c9c5178385 
        print('Reshaping the data into two dimensions...')
        orig_rows, orig_cols, orig_channels = data.shape
        data = data.reshape(data.shape[0]*data.shape[1], -1)
        print(f'Reshaped data shape: {data.shape}')

        # covariance_matrix = np.cov(data.T)
        # eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        # ind=np.arange(0, len(eigen_values), 1)
        # ind=[x for _,x in sorted(zip(eigen_values,ind))]
        # ind=ind[::-1]
        # eigen_values_1 = eigen_values[ind]
        # eigen_vectors_1 = eigen_vectors[zip(eigen_values,ind)]
        # eigen_vectors_1 = eigen_vectors[:,:3]
        # y=(eigen_vectors_1.T).dot(data)
        # projection_matrix = (eigen_vectors.T[:][:3]).T
        # eigen_values_1=eigen_values_1[:10]

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

def postprocess_data(data, **hyperparams):
    
    #TODO
    return data

def run_model(model, train_dataset, val_dataset, test_dataset, target_test,
              labels, iteration = None, **hyperparams):

    # Initialize variables from the hyperparameters
    epochs = hyperparams['epochs']
    batch_size = hyperparams['batch_size']
    loss = hyperparams['loss']
    model_metrics = hyperparams['metrics']
    workers = hyperparams['workers']
    output_path = hyperparams['output_path']
    if iteration is not None:
        best_weights_path = os.path.join(output_path, 
            f'{model.name}_best_weights_experiment_{iteration+1}.hdf5')
    else:
        best_weights_path = os.path.join(output_path, 
            f'{model.name}_best_weights_experiment.hdf5')
    patience = hyperparams['patience']
    optimizer = get_optimizer(**hyperparams)
    ignored_labels = hyperparams['ignored_labels']
    filtered_labels = [label for index, label in enumerate(labels) if index not in ignored_labels]

    # Create callback to stop training early if metrics don't improve
    cb_early_stopping = EarlyStopping(monitor='val_loss', 
                                      patience=patience, 
                                      verbose=1, 
                                      mode='auto',
                                      restore_best_weights=True)

    # Create callback to save model weights if the model performs
    # better than the previously trained models
    cb_save_best_model = ModelCheckpoint(best_weights_path, 
                                         monitor='val_loss', 
                                         verbose=1, 
                                         save_best_only=False, 
                                         mode='auto')

    # Compile the model with the appropriate loss function, optimizer,
    # and metrics
    model.compile(loss=loss, 
                  optimizer=optimizer, 
                  metrics=model_metrics,
                  loss_weights=None,
                  weighted_metrics=None,
                  run_eagerly=None,
                  )
    
    # Display a summary of the model being trained
    model.summary()

    # Record start time for model training
    model_train_start = time.process_time()

    # Train the model
    model_history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            # batch_size=batch_size,
            epochs=epochs, 
            shuffle=True, 
            # use_multiprocessing=True,
            # workers=workers,
            callbacks=[cb_early_stopping, cb_save_best_model]
        )

    # Record end time for model training
    model_train_end = time.process_time()

    # Record start time for model evaluation
    model_test_start = time.process_time()

    # Evaluate the trained 3D-DenseNet
    loss_and_metrics = model.evaluate(
            test_dataset,
            # batch_size=batch_size
        )

    # Record end time for model evaluation
    model_test_end = time.process_time()

    # Get prediction values for test dataset
    pred_test = model.predict(test_dataset).argmax(axis=1)

    # Calculate training and testing times
    model_train_time = datetime.timedelta(seconds=(model_train_end - model_train_start))
    model_test_time = datetime.timedelta(seconds=(model_test_end - model_test_start))

    overall_acc = metrics.accuracy_score(target_test, pred_test)
    precision = metrics.precision_score(target_test, pred_test, average='micro')
    recall = metrics.recall_score(target_test, pred_test, average='micro')
    kappa = metrics.cohen_kappa_score(target_test, pred_test)
    confusion_matrix = metrics.confusion_matrix(target_test, pred_test)

    # Calculate average accuracy and per-class accuracies
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)


    if np.unique(pred_test).shape[0] == len(filtered_labels):
        labels = filtered_labels
    else:
        each_acc = np.insert(each_acc, ignored_labels, 0.0)
    
    # Print results
    print('---------------------------------------------------')
    if iteration is None:
        print('             MODEL EXPERIMENT RESULTS              ')
    else:
        print(f'          MODEL EXPERIMENT #{iteration} RESULTS              ')
    print('---------------------------------------------------')
    print(f'{model.name} train time: {model_train_time}')
    print(f'{model.name} test time:  {model_test_time}')
    print('...................................................')
    print(f'{model.name} test score:     {loss_and_metrics[0]}')
    print(f'{model.name} test accuracy:  {loss_and_metrics[1]}')
    print('...................................................')
    print(f'{model.name} overall accuracy:  {overall_acc}')
    print(f'{model.name} average accuracy:  {average_acc}')
    print(f'{model.name} precision score:   {precision}')
    print(f'{model.name} recall score:      {recall}')
    print(f'{model.name} cohen kappa score: {kappa}')
    print('...................................................')
    print(f'{model.name} Per-class accuracies:')
    for i, label in enumerate(labels):
        print(f'{label}: {each_acc[i]}')
    print('---------------------------------------------------')
    print()
    # print(metrics.classification_report(target_test, pred_test, 
    #                                 target_names=labels, digits=len(labels)))

    # Save confusion matrix image
    print('Creating confusion matrix plot...')
    cm_filename = os.path.join(output_path,
                f'experiment_{iteration+1}_{model.name}_confusion_matrix.png')
    cm_sum = np.sum(confusion_matrix, axis=1, keepdims=True)
    cm_perc = confusion_matrix / cm_sum.astype(float) * 100
    annot = np.empty_like(confusion_matrix).astype(str)
    nrows, ncols = confusion_matrix.shape
    for i in range(nrows):
        for j in range(ncols):
            c = confusion_matrix[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    

    cm = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=(25,25))
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.title(f'Experiment #{iteration+1} {model.name} Confusion Matrix')

    print('Saving confusion matrix plot...')
    plt.savefig(cm_filename)

    # Clear plot data for next plot
    plt.clf()

    

    # Write model history to file
    with open(os.path.join(output_path,
         f'Experiment_{iteration+1}_training_history.txt'), 'w') as hf:

        hf.write(f'EXPERIMENT #{iteration+1} MODEL HISTORY:\n')
        hf.write('-----------------------------------------------\n')
        hf.write(f'MODEL: {model.name}\n')
        hf.write('-----------------------------------------------\n')

        # Save model summary to file as well
        model.summary(print_fn=lambda x: hf.write(x + '\n'))

        # Show epoch with best validation value
        hf.write(f'Best Epoch: {cb_early_stopping.best_epoch}\n')
        
        # Get number of epochs model actually ran for
        if cb_early_stopping.stopped_epoch > 0:
            ran_epochs = cb_early_stopping.stopped_epoch
        else:
            ran_epochs = model_history.params['epochs']

        # Save info from each epoch to file
        for epoch in range(ran_epochs):
            hf.write(f'EPOCH: {epoch+1}\n')
            for key in model_history.history.keys():
                hf.write(f'  {key}: {model_history.history[key][epoch]}\n')


    results = {
        'model_name': model.name,
        'model': model,
        'train_time': model_train_time,
        'test_time': model_test_time,
        'test_score': loss_and_metrics[0],
        'test_accuracy': loss_and_metrics[1],
        'overall_accuracy': overall_acc,
        'average_accuracy': average_acc,
        'precision_score': precision,
        'recall_score': recall,
        'cohen_kappa_score': kappa,
        'confusion_matrix': confusion_matrix,
        'per_class_accuracies': each_acc,
        'labels': labels,
    }

    return results

def test_model(model, test_dataset, target_test, labels, 
               iteration=None, **hyperparams):
    pass

def show_training_results(results, **hyperparams):
    overall_acc = metrics.accuracy_score(target_test, pred_test)
    precision = metrics.precision_score(target_test, pred_test, average='micro')
    recall = metrics.recall_score(target_test, pred_test, average='micro')
    kappa = metrics.cohen_kappa_score(target_test, pred_test)
    confusion_matrix = metrics.confusion_matrix(target_test, pred_test)

    # Calculate average accuracy and per-class accuracies
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)


    if np.unique(pred_test).shape[0] == len(filtered_labels):
        labels = filtered_labels
    else:
        each_acc = np.insert(each_acc, ignored_labels, 0.0)
    
    # Print results
    print('---------------------------------------------------')
    if iteration is None:
        print('             MODEL EXPERIMENT RESULTS              ')
    else:
        print(f'          MODEL EXPERIMENT #{iteration} RESULTS              ')
    print('---------------------------------------------------')
    print(f'{model.name} train time: {model_train_time}')
    print(f'{model.name} test time:  {model_test_time}')
    print('...................................................')
    print(f'{model.name} test score:     {loss_and_metrics[0]}')
    print(f'{model.name} test accuracy:  {loss_and_metrics[1]}')
    print('...................................................')
    print(f'{model.name} overall accuracy:  {overall_acc}')
    print(f'{model.name} average accuracy:  {average_acc}')
    print(f'{model.name} precision score:   {precision}')
    print(f'{model.name} recall score:      {recall}')
    print(f'{model.name} cohen kappa score: {kappa}')
    print('...................................................')
    print(f'{model.name} Per-class accuracies:')
    for i, label in enumerate(labels):
        print(f'{label}: {each_acc[i]}')
    print('---------------------------------------------------')
    print()
    # print(metrics.classification_report(target_test, pred_test, 
    #                                 target_names=labels, digits=len(labels)))

    # Save confusion matrix image
    print('Creating confusion matrix plot...')
    cm_filename = os.path.join(output_path,
                f'experiment_{iteration+1}_{model.name}_confusion_matrix.png')
    cm_sum = np.sum(confusion_matrix, axis=1, keepdims=True)
    cm_perc = confusion_matrix / cm_sum.astype(float) * 100
    annot = np.empty_like(confusion_matrix).astype(str)
    nrows, ncols = confusion_matrix.shape
    for i in range(nrows):
        for j in range(ncols):
            c = confusion_matrix[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    

    cm = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=(25,25))
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.title(f'Experiment #{iteration+1} {model.name} Confusion Matrix')

    print('Saving confusion matrix plot...')
    plt.savefig(cm_filename)

    # Clear plot data for next plot
    plt.clf()

def test_harness_parser():
    """
    Sets up the parser for command-line flags for the test harness 
    script.

    Returns
    -------
    argparse.ArgumentParser
        An ArgumentParser object configured with the test_harness.py
        command-line arguments.
    """

    SCRIPT_DESCRIPTION = ('Test harness script for experimenting on automatic '
                      'hyperspectral (HS) band selection for the classification'
                      'of HS images.')

    parser = argparse.ArgumentParser(SCRIPT_DESCRIPTION)
    parser.add_argument(
        "--cuda",
        type=int,
        default=-1,
        help="Specify CUDA device (defaults to -1, which learns on CPU)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Specify number of workers (processes) to use when training (defaults to 1)",
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of runs (default: 1)")
    parser.add_argument(
        "--restore",
        type=str,
        default=None,
        help="Weights to use for initialization, e.g. a checkpoint",
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='./',
        help='Path to where output files should be created'
    )
    parser.add_argument(
        '--experiments_csv',
        type=str,
        default=None,
        help='Path to a CSV file with a set of experiments to run with \
            specific parameter values'
    )
    parser.add_argument(
        '--experiments_json',
        type=str,
        default=None,
        help='Path to a JSON file with a set of experiments to run with \
            specific parameter values'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='The hyperspectral or data fusion dataset to use for experiments'
    )
    parser.add_argument(
        '--reuse_last_dataset',
        action='store_true',
        help='Reuse the last dataset generator'
    )
    parser.add_argument(
        '--skip_data_preprocessing',
        action='store_true',
        help='Skip the data preprocessing step'
    )
    parser.add_argument(
        '--skip_band_selection',
        action='store_true',
        help='Skip the band selection step'
    )
    parser.add_argument(
        '--skip_data_postprocessing',
        action='store_true',
        help='Skip the data postprocessing step'
    )
    parser.add_argument(
        '--model_id',
        type=str,
        default=None,
        help='The identifier for the machine learning model to used on the dataset'
    )

    # Training options
    group_train = parser.add_argument_group("Training")
    group_train.add_argument(
        "--random_seed",
        type=int,
        help="Random number generator seed.",
    )
    group_train.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Training epochs (default = 1)",
    )
    group_train.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (default = 64)",
    )
    group_train.add_argument(
        "--patch_size",
        type=int,
        default=3,
        help="Size of the spatial neighborhood [e.g. patch_size X patch_size square] (default = 3)",
    )
    group_train.add_argument(
        "--train_split", 
        type=float, 
        default = 0.80,
        help="The amount of samples set aside for training \
              during validation split (default = 0.80)"
    )
    group_train.add_argument(
        "--split_mode", 
        type=str, 
        default = 'random',
        help="The mode by which to split datasets (random, fixed, or disjoint)"
    )
    group_train.add_argument(
        "--class_balancing",
        action="store_true",
        help="Inverse median frequency class balancing (default = False)",
    )
    group_train.add_argument(
        "--test_stride",
        type=int,
        default=1,
        help="Sliding window step stride during inference (default = 1)",
    )
    group_train.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run the model for (default = 1)",
    )
    group_train.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Number of epochs without improvement before stopping training",
    )

    # Model optimizer parameters
    group_optimizer = parser.add_argument_group('Optimizer')
    group_optimizer.add_argument(
        '--optimizer', 
        type=str,
        help="The optimizer used by the machine learning model"
    )
    group_optimizer.add_argument(
        '--lr',
        type=float, 
        help="The model's learning rate"
    )
    group_optimizer.add_argument(
        '--momentum', 
        help="The optimizer's momentum, if applicable"
    )
    group_optimizer.add_argument(
        '--epsilon',
        type=float, 
        help="The optimizer's epsilon value, if applicable"
    )
    group_optimizer.add_argument(
        '--initial_accumulator_value',
        type=float, 
        help="The optimizer's initial_accumulator_value value, if applicable"
    )
    group_optimizer.add_argument(
        '--beta',
        type=float, 
        help="The optimizer's beta value, if applicable (Ftrl only)"
    )
    group_optimizer.add_argument(
        '--beta_1',
        type=float, 
        help="The optimizer's beta_1 value, if applicable"
    )
    group_optimizer.add_argument(
        '--beta_2',
        type=float, 
        help="The optimizer's beta_2 value, if applicable"
    )
    group_optimizer.add_argument(
        '--amsgrad',
        type=bool, 
        help="The optimizer's amsgrad value, if applicable"
    )
    group_optimizer.add_argument(
        '--rho',
        type=float, 
        help="The optimizer's rho value, if applicable"
    )
    group_optimizer.add_argument(
        '--centered',
        type=bool, 
        help="The optimizer's centered value, if applicable"
    )
    group_optimizer.add_argument(
        '--nesterov',
        type=bool, 
        help="The optimizer's nesterov value, if applicable"
    )
    group_optimizer.add_argument(
        '--learning_rate_power',
        type=float, 
        help="The optimizer's learning_rate_power value, if applicable"
    )
    group_optimizer.add_argument(
        '--l1_regularization_strength',
        type=float, 
        help="The optimizer's l1_regularization_strength value, if applicable"
    )
    group_optimizer.add_argument(
        '--l2_regularization_strength',
        type=float, 
        help="The optimizer's l2_regularization_strength value, if applicable"
    )
    group_optimizer.add_argument(
        '--l2_shrinkage_regularization_strength',
        type=float, 
        help="The optimizer's l2_shrinkage_regularization_strength value, if applicable"
    )

    # Data augmentation parameters
    group_da = parser.add_argument_group("Data augmentation")
    group_da.add_argument(
        "--flip_augmentation", action="store_true", help="Random flips (if patch_size > 1)"
    )
    group_da.add_argument(
        "--radiation_augmentation",
        action="store_true",
        help="Random radiation noise (illumination)",
    )
    group_da.add_argument(
        "--mixture_augmentation", action="store_true", help="Random mixes between spectra"
    )

    # GRSS_DFC_2018 dataset parameters
    group_grss_dfc_2018 = parser.add_argument_group("GRSS_DFC_2018 Dataset")
    group_grss_dfc_2018.add_argument(
        "--use_hs_data", action="store_true", help="Use hyperspectral data"
    )
    group_grss_dfc_2018.add_argument(
        "--use_lidar_ms_data", action="store_true", help="Use lidar multispectral intensity data"
    )
    group_grss_dfc_2018.add_argument(
        "--use_lidar_ndsm_data", action="store_true", help="Use lidar NDSM data"
    )
    group_grss_dfc_2018.add_argument(
        "--use_vhr_data", action="store_true", help="Use very high resolution RGB data"
    )
    group_grss_dfc_2018.add_argument(
        "--use_all_data", action="store_true", help="Use all data sources"
    )
    group_grss_dfc_2018.add_argument(
        "--normalize_hs_data", action="store_false", help="Normalize hyperspectral data"
    )
    group_grss_dfc_2018.add_argument(
        "--normalize_lidar_ms_data", action="store_false", help="Normalize LiDAR multispectral data"
    )
    group_grss_dfc_2018.add_argument(
        "--normalize_lidar_ndsm_data", action="store_false", help="Normalize LiDAR NDSM data"
    )
    group_grss_dfc_2018.add_argument(
        "--normalize_vhr_data", action="store_false", help="Normalize VHR RGB data"
    )
    group_grss_dfc_2018.add_argument(
        '--hs_resampling',
        type=str,
        default=None,
        help='Resampling method to use on the grss_dfc_2018 hyperspectral image'
    )
    group_grss_dfc_2018.add_argument(
        '--lidar_ms_resampling',
        type=str,
        default=None,
        help='Resampling method to use on the grss_dfc_2018 LiDAR multispectral image'
    )
    group_grss_dfc_2018.add_argument(
        '--lidar_ndsm_resampling',
        type=str,
        default=None,
        help='Resampling method to use on the grss_dfc_2018 LiDAR NDSM image'
    )
    group_grss_dfc_2018.add_argument(
        '--vhr_resampling',
        type=str,
        default=None,
        help='Resampling method to use on the grss_dfc_2018 VHR RGB image'
    )
    group_grss_dfc_2018.add_argument(
        '--hs_data_filter',
        type=str,
        default=None,
        help='Filtering method to use on the grss_dfc_2018 hyperspectral image'
    )
    group_grss_dfc_2018.add_argument(
        '--lidar_ms_data_filter',
        type=str,
        default=None,
        help='Filtering method to use on the grss_dfc_2018 LiDAR multispectral image'
    )
    group_grss_dfc_2018.add_argument(
        '--lidar_dsm_data_filter',
        type=str,
        default=None,
        help='Filtering method to use on the grss_dfc_2018 LiDAR DSM image'
    )
    group_grss_dfc_2018.add_argument(
        '--lidar_dem_data_filter',
        type=str,
        default=None,
        help='Filtering method to use on the grss_dfc_2018 LiDAR DEM image'
    )
    group_grss_dfc_2018.add_argument(
        '--vhr_data_filter',
        type=str,
        default=None,
        help='Filtering method to use on the grss_dfc_2018 VHR RGB image'
    )

    group_band_selection = parser.add_argument_group("Band Selection")
    group_band_selection.add_argument(
        '--band_reduction_method',
        type=str,
        default=None,
        help='The band dimensionality reduction method to be used'
    )
    group_band_selection.add_argument(
        '--n_components',
        type=int,
        default=None,
        help='The number of components to be used with PCA or ICAa'
    )

    return parser


### Main ###

if __name__ == "__main__":
    # Start timing experiments
    test_harness_start = time.time()

    # Arguments
    parser = test_harness_parser()
    args = parser.parse_args()

    hyperparams = vars(args)

    # Get output path
    if hyperparams['output_path'] is not None:
        output_path = hyperparams['output_path']
    else:
        output_path = './'
    
    # Get number of worker processes
    workers = hyperparams['workers']

    # Get hyperparam derived variable values
    if hyperparams['experiments_json'] is not None:
        # Transpose the json dataframe, since the experiments are read
        # in as columns instead of rows
        experiments = pd.read_json(hyperparams['experiments_json']).T
        iterations = experiments.shape[0]
        outfile_prefix = Path(hyperparams['experiments_json']).stem
    elif hyperparams['experiments_csv'] is not None:
        experiments = pd.read_csv(hyperparams['experiments_csv'])
        iterations = experiments.shape[0]
        outfile_prefix = Path(hyperparams['experiments_csv']).stem
    else:
        experiments = None
        iterations = hyperparams['iterations']
        outfile_prefix = 'experiment'

    # Get model name of CPU
    cpu_name = cpuinfo.get_cpu_info()['brand_raw']

    # Get model names of all GPUs on system
    gpu_names = []
    for gpu in tf.config.list_physical_devices(device_type = 'GPU'):
        gpu_names.append(tf.config.experimental.get_device_details(gpu)['device_name'])

    # Initialize data list variables for CSV output at end of program
    experiment_data_list = []
    per_class_data_lists = {}

    print('-------------------------------------------------------------------')
    print('-------------------------------------------------------------------')
    print('-------------------------------------------------------------------')
    print('BEGINNING EXPERIMENTS...')
    print('vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv')
    print()

    # Make a generater to create prime number random seeds
    primes = prime_generator()

    # Set variables that carry state over experiments to None
    dataset_choice = None
    data = None
    train_gt = None
    test_gt = None
    dataset_info = {
        'name': None,
        'num_classes': None,
        'ignored_labels': None,
        'class_labels': None,
        'label_mapping': None,
    }
    train_dataset = None
    val_dataset = None
    test_dataset = None
    target_test = None

    # Go through experiment iterations
    for iteration in range(iterations):

        print('*******************************************************')
        print(f'<<< EXPERIMENT #{iteration+1}  STARTING >>>')
        print('*******************************************************')
        print()

        experiment_data = {
            'experiment_number': iteration + 1,
            'success': False,
            'random_seed': None,
            'dataset': None,
            'channels': None,
            'model': None,
            'device': None,
            'epochs': None,
            'batch_size': None,
            'patch_size': None,
            'train_split': None,
            'optimizer': None,
            'learning_rate': None,
            'loss': None,
            'train_time': 0.0,
            'test_time': 0.0,
            'test_score': 0.0,
            'test_accuracy': 0.0,
            'overall_accuracy': 0.0,
            'average_accuracy': 0.0,
            'precision_score': 0.0,
            'recall_score': 0.0,
            'cohen_kappa_score': 0.0,
        }

        per_class_data = {
            'experiment_number': iteration + 1, 
            'random_seed': None,
            'model': None, 
            'overall_accuracy': 0.0, 
            'average_accuracy': 0.0,
        }

        experiments_results_file = f'{outfile_prefix}_results.csv'
        class_results_file = f'{outfile_prefix}__{dataset_choice}__class_results.csv'

        # Experiment has begun, so make sure to catch any failures that
        # may occur
        try:
            # If loading experiments from a file, get new set of hyperparams
            if experiments is not None:
                hyperparams = experiments.iloc[iteration].to_dict()

                # Ignore the output path in the experiments, use the path
                # from command line arguments
                hyperparams['output_path'] = output_path

                # Ignore the workers argument in experiments, use the
                # value from the command line
                hyperparams['workers'] = workers

                print('<~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>')
                print(f'EXPERIMENT NAME: {experiments.index[iteration]}')
                print('<~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>')
                print()


            # Print out parameters for experiment
            print('.......................................................')
            print('EXPERIMENT PARAMETERS')
            print('.......................................................')
            header = '{:<40} | {:<40}'.format('PARAMETER', 'VALUE')
            print(header)
            print('=' * len(header))
            for key in hyperparams:
                print('{:<40} | {:<40}'.format(key, str(hyperparams[key])))
                print('-' * len(header))
            print('-' * len(header))
            print('.......................................................')


            # Model checks
            if (hyperparams['model_id'] == 'fusion-fcn' 
                and hyperparams['dataset'] != 'grss_dfc_2018'):
                print('Cannot use fusion-fcn model without the grss_dfc_2018 dataset!')
                exit(1)
            elif (hyperparams['model_id'] == 'fusion-fcn-v2' 
                and hyperparams['dataset'] != 'grss_dfc_2018'):
                print('Cannot use fusion-fcn-v2 model without the grss_dfc_2018 dataset!')
                exit(1)

            # Initialize random seed for sampling function
            # Each random seed is a prime number, in order
            if hyperparams['random_seed'] is not None:
                seed = hyperparams['random_seed']
            else:
                seed = next(primes)
            print(f'< Iteration #{iteration} random seed: {seed} >')
            print()
            np.random.seed(seed)

            # Choose the appropriate device from the hyperparameters
            device = get_device(hyperparams['cuda'])

            if 'CPU' in device:
                device_name = cpu_name
            else:
                gpu_num = int(device.split(':')[-1])
                device_name = gpu_names[gpu_num]
                gpu = tf.config.list_physical_devices('GPU')[gpu_num]
                tf.config.experimental.set_memory_growth(gpu, True)
        
            # Device has been selected, so do all possible computation with device
            with tf.device(device):
                reuse_last_dataset = hyperparams['reuse_last_dataset']
                if reuse_last_dataset and dataset_choice is not None:
                    print()
                    print(f'< Reusing last dataset: {dataset_choice} >')
                else:

                    reuse_last_dataset = False

                    print()
                    print('-------------------------------------------------------------------')
                    print('LOADING DATASET...')
                    print('vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv')

                    # Get dataset choice parameter
                    dataset_choice = hyperparams['dataset']
                    class_results_file = f'{outfile_prefix}__{dataset_choice}__class_results.csv'
                    print()
                    print(f' < Dataset Chosen: {dataset_choice} >')
                    print()

                    # Make sure dataset is in per-class data list dictionary
                    if dataset_choice not in per_class_data_lists:
                        per_class_data_lists[dataset_choice] = []

                    # Get selected dataset
                    if not reuse_last_dataset:
                        if dataset_choice == 'grss_dfc_2018':
                            # Determine what parts of dataset to use
                            if (not hyperparams['use_hs_data']
                                and not hyperparams['use_lidar_ms_data']
                                and not hyperparams['use_lidar_ndsm_data']
                                and not hyperparams['use_vhr_data']
                                and not hyperparams['use_all_data']):

                                print('<!> No specific data selected, defaulting to using only hyperspectral data... <!>')
                                hyperparams['use_hs_data'] = True
                            
                            data, train_gt, test_gt, dataset_info = load_grss_dfc_2018_uh_dataset(**hyperparams)
                        elif dataset_choice == 'indian_pines':
                            data, train_gt, test_gt, dataset_info = load_indian_pines_dataset(**hyperparams)
                        elif dataset_choice == 'pavia_center':
                            data, train_gt, test_gt, dataset_info = load_pavia_center_dataset(**hyperparams)
                        elif dataset_choice == 'university_of_pavia':
                            data, train_gt, test_gt, dataset_info = load_university_of_pavia_dataset(**hyperparams)
                        else:
                            print('No dataset chosen! Defaulting to only hyperspectral bands of grss_dfc_2018...')
                            dataset_choice = 'grss_dfc_2018'
                            hyperparams['use_hs_data'] = True
                            data, train_gt, test_gt, dataset_info = load_grss_dfc_2018_uh_dataset(**hyperparams)

                    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                    print('DATASET LOADED!')
                    print('-------------------------------------------------------------------')
                    print()

                    if not hyperparams['skip_data_preprocessing']:
                        print('-------------------------------------------------------------------')
                        print('PREPROCESS THE DATA')
                        print('-------------------------------------------------------------------')
                        data = preprocess_data(data, **hyperparams)
                        print('-------------------------------------------------------------------')
                        print()
                    
                    if not hyperparams['skip_band_selection']:
                        print('-------------------------------------------------------------------')
                        print('RUN BAND SELECTION ALGORITHM')
                        print('-------------------------------------------------------------------')
                        data = band_selection(data, dataset_info['class_labels'], **hyperparams)
                        print('-------------------------------------------------------------------')
                        print()

                # Set dataset variables
                dataset_name = dataset_info['name']
                num_classes = dataset_info['num_classes']
                hs_channels = dataset_info['hs_channels']
                lidar_ms_channels = dataset_info['lidar_ms_channels']
                lidar_ndsm_channels = dataset_info['lidar_ndsm_channels']
                vhr_rgb_channels = dataset_info['vhr_rgb_channels']
                ignored_labels = dataset_info['ignored_labels']
                all_class_labels = dataset_info['class_labels']
                valid_class_labels = [label for index, label in enumerate(all_class_labels) 
                                        if index not in ignored_labels]


                epochs = hyperparams['epochs']
                supervision = 'full'
                batch_size = hyperparams['batch_size']
                patch_size = hyperparams['patch_size']
                train_split = hyperparams['train_split']
                optimizer = hyperparams['optimizer']
                learning_rate = hyperparams['lr']
                loss = 'sparse_categorical_crossentropy'
                img_channels = data.shape[-1]
                img_rows = patch_size
                img_cols = patch_size

                if (hyperparams['model_id'] == 'fusion-fcn'
                    or hyperparams['model_id'] == 'fusion-fcn-v2'):
                    branch_1_channels = (*vhr_rgb_channels, *lidar_ms_channels)
                    branch_2_channels = (*lidar_ndsm_channels,)
                    branch_3_channels = (*hs_channels,)
                    input_channels = (branch_1_channels, branch_2_channels, branch_3_channels)
                else:
                    input_channels = None

                # Check to see if model uses 3d convolutions - if so
                # then the input dimensions will need to be expanded
                # to include the 'planes' dimension
                if (hyperparams['model_id'] == '3d-densenet'
                    or hyperparams['model_id'] == '3d-cnn'):
                    expand_dims = True
                else:
                    expand_dims = False

                # Add and update hyperparameters for model training
                hyperparams.update(
                    {
                        'random_seed': seed,
                        'n_classes': num_classes,
                        'n_bands': img_channels,
                        'ignored_labels': ignored_labels,
                        'device': device,
                        'supervision': supervision,
                        'center_pixel': True,
                        'one_hot_encoding': True,
                        'metrics': ['sparse_categorical_accuracy'],
                        'loss': loss,
                        'input_channels': input_channels,
                        'expand_dims': expand_dims,
                    }
                )

                # Update experiment data
                experiment_data.update({
                    'random_seed': seed,
                    'dataset': dataset_name,
                    'channels': img_channels,
                    'device': device_name,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'patch_size': patch_size,
                    'train_split': train_split,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'loss': loss,
                })

                # Update per-class data for experiment
                per_class_data.update({
                    'random_seed': seed,
                })
                for label in all_class_labels:
                    per_class_data[label] = 0.0

                if not reuse_last_dataset:
                    print('-------------------------------------------------------------------')
                    print('SPLIT DATA FOR TRAINING, VALIDATION, AND TESTING')
                    print('-------------------------------------------------------------------')

                    print('Breaking down image into data patches and splitting data into train, validation, and test sets...')
                    train_dataset, val_dataset, test_dataset, target_test = create_datasets(data, train_gt, test_gt, **hyperparams)

                    print('-------------------------------------------------------------------')
                    print()


                print('-------------------------------------------------------------------')
                print('CREATE MODEL')
                print('-------------------------------------------------------------------')

                # Create specified model
                if hyperparams['model_id'] == '2d-densenet':
                    model = densenet_2d_model(img_rows=img_rows, 
                                    img_cols=img_cols, 
                                    img_channels=img_channels, 
                                    nb_classes=num_classes)
                elif hyperparams['model_id'] == '2d-densenet-multi':
                    pass    #TODO
                elif hyperparams['model_id'] == '3d-densenet':
                    model = densenet_3d_model(img_rows=img_rows, 
                                    img_cols=img_cols, 
                                    img_channels=img_channels, 
                                    nb_classes=num_classes)
                elif hyperparams['model_id'] == '2d-cnn':
                    model = cnn_2d_model(img_rows=img_rows, 
                                    img_cols=img_cols, 
                                    img_channels=img_channels, 
                                    nb_classes=num_classes)
                elif hyperparams['model_id'] == '3d-cnn':
                    model = cnn_3d_model(img_rows=img_rows, 
                                    img_cols=img_cols, 
                                    img_channels=img_channels, 
                                    nb_classes=num_classes)
                elif hyperparams['model_id'] == 'cnn-baseline':
                    filter_size = patch_size // 2 + 1
                    model = baseline_cnn_model(img_rows=img_rows, 
                                            img_cols=img_cols, 
                                            img_channels=img_channels, 
                                            patch_size=filter_size, 
                                            nb_filters=num_classes * 2, 
                                            nb_classes=num_classes)
                elif hyperparams['model_id'] == 'fusion-fcn':
                    branch_1_shape = (img_rows, img_cols, 
                                len(lidar_ms_channels) + len(vhr_rgb_channels))
                    branch_2_shape = (img_rows, img_cols, 
                                len(lidar_ndsm_channels))
                    branch_3_shape = (img_rows, img_cols, len(hs_channels))
                    model = fusion_fcn_model(
                                    branch_1_shape=branch_1_shape,
                                    branch_2_shape=branch_2_shape,
                                    branch_3_shape=branch_3_shape, 
                                    nb_classes=num_classes)
                elif hyperparams['model_id'] == 'fusion-fcn-v2':
                    branch_1_shape = (img_rows, img_cols, 
                                len(lidar_ms_channels) + len(vhr_rgb_channels))
                    branch_2_shape = (img_rows, img_cols, 
                                len(lidar_ndsm_channels))
                    branch_3_shape = (img_rows, img_cols, len(hs_channels))
                    model = fusion_fcn_v2_model(
                                    branch_1_shape=branch_1_shape,
                                    branch_2_shape=branch_2_shape,
                                    branch_3_shape=branch_3_shape, 
                                    nb_classes=num_classes)
                else:
                    print('<!> No model specified, defaulting to 3d-densenet <!>')
                    model = densenet_3d_model(img_rows=img_rows, 
                                    img_cols=img_cols, 
                                    img_channels=img_channels, 
                                    nb_classes=num_classes)
                
                # Record model name for output
                experiment_data['model'] = model.name
                per_class_data['model'] = model.name

                print('-------------------------------------------------------------------')
                print()
                
                print('-------------------------------------------------------------------')
                print('RUN MODEL')
                print('-------------------------------------------------------------------')

                # Run experiment on model
                results = run_model(model=model, 
                                    train_dataset=train_dataset, 
                                    val_dataset=val_dataset, 
                                    test_dataset=test_dataset,
                                    target_test=target_test, 
                                    labels=all_class_labels, 
                                    iteration=iteration,
                                    **hyperparams)
                
                # Copy results to output data
                experiment_data['train_time'] = results['train_time']
                experiment_data['test_time'] = results['test_time']
                experiment_data['test_score'] = results['test_score']
                experiment_data['test_accuracy'] = results['test_accuracy']
                experiment_data['overall_accuracy'] = results['overall_accuracy']
                experiment_data['average_accuracy'] = results['average_accuracy']
                experiment_data['precision_score'] = results['precision_score']
                experiment_data['recall_score'] = results['recall_score']
                experiment_data['cohen_kappa_score'] = results['cohen_kappa_score']

                per_class_data['overall_accuracy'] = results['overall_accuracy']
                per_class_data['average_accuracy'] = results['average_accuracy']

                # per_class_accuracies = results['per_class_accuracies']
                # if len(per_class_accuracies) != all_class_labels:
                #     for index in ignored_labels:
                #         np.insert(per_class_accuracies, index, 0.0)

                for index, acc in enumerate(results['per_class_accuracies']):
                    per_class_data[results['labels'][index]] = acc

                print('-------------------------------------------------------------------')
                print()


                if not hyperparams['skip_data_postprocessing']:
                    print('-------------------------------------------------------------------')
                    print('PREPROCESS THE DATA')
                    print('-------------------------------------------------------------------')
                    # data = postprocess_data(data, **hyperparams)
                    print('-------------------------------------------------------------------')
                    print()

                experiment_data['success'] = True

        except Exception as e:
            print()
            print('###################################################')
            print('!!! EXCEPTION OCCURRED !!!')
            print('###################################################')
            print(f'Exception Type: {type(e)}')
            print(f'Exception Line: {e.__traceback__.tb_lineno}')
            print(f'Exception Desc: {e}')
            print()
            print('---------------------------------------------------')
            print('** Full Traceback **')
            print()
            # Print full exception
            traceback.print_exc()
            print('###################################################')
            print()

            # Write exception to file
            with open(os.path.join(output_path, f'experiment_{iteration+1}_exception.log'),'w') as ef:
                ef.write('\n')
                ef.write('###################################################\n')
                ef.write('!!! EXCEPTION OCCURRED !!!\n')
                ef.write('###################################################\n')
                ef.write(f'Exception Type: {type(e)}\n')
                ef.write(f'Exception Line: {e.__traceback__.tb_lineno}\n')
                ef.write(f'Exception Desc: {e}\n')
                ef.write('\n')
                ef.write('---------------------------------------------------\n')
                ef.write('** Full Traceback **\n')
                ef.write('\n')
                # Print full exception
                ef.write(f'{traceback.format_exc()}\n')
                ef.write('###################################################\n')
                ef.write('\n')

            print(f'Experiment #{iteration+1} crashed and thus failed!')
        
        experiment_data_list.append(experiment_data)
        per_class_data_lists[dataset_choice].append(per_class_data)

        print()
        print('-------------------------------------------------------------------')
        print('SAVING RESULTS...')

        experiment_results = pd.DataFrame(experiment_data_list)
        experiment_results.to_csv(os.path.join(output_path, experiments_results_file))
        
        print('  >>> Experiment results saved!')

        for dataset_choice in per_class_data_lists:
            per_class_data_results = pd.DataFrame(per_class_data_lists[dataset_choice])
            per_class_data_results.to_csv(os.path.join(output_path, class_results_file))
            print(f'  >>> {dataset_choice} per-class results saved!')

        print('RESULTS SAVED!')
        print('-------------------------------------------------------------------')

        print()
        print('*******************************************************')
        print(f'<<< EXPERIMENT #{iteration+1}  COMPLETE! >>>')
        print('*******************************************************')
        print()
    
    print()
    print()
    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    print('EXPERIMENTS COMPLETE!')
    print('-------------------------------------------------------------------')
    print('-------------------------------------------------------------------')
    print('-------------------------------------------------------------------')
    print()

    test_harness_end = time.time()
    test_harness_runtime = datetime.timedelta(seconds=(test_harness_end - test_harness_start))

    print(f' < Total Test Harness Runtime: {test_harness_runtime} >')

    print()
    print('   ~~~ EXITTING TEST HARNESS PROGRAM ~~~')
    print()
