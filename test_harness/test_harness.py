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
import datetime
import gc
import os
from pathlib import Path
import time
import traceback


### Other Library Imports ###
import cpuinfo
import numpy as np
import pandas as pd
import tensorflow as tf

### Local Imports ###
from test_harness.band_selection import (
    band_selection
)
from data.datasets import (
    create_datasets,
    load_grss_dfc_2018_uh_dataset,
    load_indian_pines_dataset,
    load_pavia_center_dataset,
    load_university_of_pavia_dataset,
)
from test_harness.command_line_parser import PARAMETER_LIST
from test_harness.evaluation import (
    calculate_model_statistics,
    create_confusion_matrix_plot,
    output_experiment_results,
    test_model,
)
from models.models import (
    fusion_fcn_model,
    fusion_fcn_v2_model,
    get_optimizer,
    densenet_2d_model,
    densenet_2d_multi_model,
    densenet_3d_model,
    cnn_2d_model,
    cnn_3d_model,
    baseline_cnn_model,
    nin_model,
)
from models.densenet_3d_fusion_model import (
    densenet_3d_fusion_model,
    densenet_3d_fusion_model2,
    densenet_3d_fusion_model3,
    # densenet_3d_fusion_model4,
)
from models.densenet_3d_modified import densenet_3d_modified_model
from test_harness.training import (
    train_model,
)
from test_harness.utils import (
    filter_pred_results,
    get_device,
    prime_generator,
    preprocess_data,
    postprocess_data,
)

def run_test_harness(**hyperparams):
    """
    """

    experiment_name = hyperparams['experiment_name']

    # Get output path
    if hyperparams['experiment_number'] < 1:
        experiment_number = 1
    else:
        experiment_number = hyperparams['experiment_number']

    # Get output path
    if hyperparams['output_path'] is not None:
        output_path = hyperparams['output_path']
    else:
        output_path = './'

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
        if hyperparams['experiment_name'] is None:
            outfile_prefix = 'experiment'
        else:
            outfile_prefix = hyperparams['experiment_name']

    # Get model name of CPU
    cpu_name = cpuinfo.get_cpu_info()['brand_raw']

    # Get model names of all GPUs on system
    gpu_names = []
    for gpu in tf.config.list_physical_devices(device_type = 'GPU'):
        gpu_names.append(tf.config.experimental.get_device_details(gpu)['device_name'])

    # Initialize data list variables for CSV output at end of program
    experiment_data_list = []
    per_class_data_lists = {}
    per_class_selected_band_lists = {}

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
    band_selection_time = None
    bands_selected = None

    # Go through experiment iterations
    for iteration in range(experiment_number - 1, experiment_number - 1 + iterations):

        # Clean memory in each iteration (otherwise the machine may
        # randomly run out of memory if it is being pushed to its
        # limit)
        gc.collect()

        print('*******************************************************')
        print(f'<<< EXPERIMENT #{iteration+1}  STARTING >>>')
        print('*******************************************************')
        print()

        experiment_data = {
            'experiment_number': iteration + 1,
            'experiment_name': experiment_name,
            'success': False,
            'random_seed': None,
            'dataset': None,
            'band_reduction_method': None,
            'band_selection_time': None,
            'bands_selected': None,
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
            'overall_accuracy': 0.0,
            'average_accuracy': 0.0,
            'precision_score': 0.0,
            'recall_score': 0.0,
            'cohen_kappa_score': 0.0,
        }

        per_class_data = {
            'experiment_number': iteration + 1, 
            'experiment_name': experiment_name,
            'random_seed': None,
            'band_reduction_method': None,
            'band_selection_time': None,
            'bands_selected': None,
            'model': None, 
            'train_time': 0.0,
            'test_time': 0.0,
            'overall_accuracy': 0.0, 
            'average_accuracy': 0.0,
            'precision_score': 0.0,
            'recall_score': 0.0,
            'cohen_kappa_score': 0.0,
        }

        per_class_selected_bands = None

        experiments_results_file = f'{outfile_prefix}_results.csv'
        class_results_file = f'{outfile_prefix}__{dataset_choice}__class_results.csv'
        selected_bands_file = f'{outfile_prefix}__{dataset_choice}__selected_bands.csv'

        # Experiment has begun, so make sure to catch any failures that
        # may occur
        try:
            
            # If loading experiments from a file, get new set of hyperparams
            if experiments is not None:
                # Get hyperparameters from dictionary
                hyperparams = experiments.iloc[iteration].to_dict()

                experiment_name = experiments.index[iteration]

                hyperparams['experiment_name'] = experiment_name
                experiment_data['experiment_name'] = experiment_name
                per_class_data['experiment_name'] = experiment_name

                # Fill in any missing parameters with None
                for param in PARAMETER_LIST:
                    if param not in hyperparams:
                        hyperparams[param] = None

                if hyperparams['experiment_number'] > 0:
                    iteration = hyperparams['experiment_number'] - 1
                    experiment_data['experiment_number'] = hyperparams['experiment_number']
                    per_class_data['experiment_number'] = hyperparams['experiment_number']

                # Ignore the output path in the experiments, use the path
                # from command line arguments
                hyperparams['output_path'] = output_path

                
                print('<~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>')
                print(f'EXPERIMENT NAME: {experiments.index[iteration]}')
                print('<~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>')
                print()
            else:
                # Fill in any missing parameters with None
                for param in PARAMETER_LIST:
                    if param not in hyperparams:
                        hyperparams[param] = None
                
                experiment_name = hyperparams['experiment_name']
            
            # Save hyperparameters to experiment file if argument
            # has been given
            if hyperparams['save_experiment_path'] is not None:
                extension = Path(hyperparams['save_experiment_path']).suffix
                if extension == '.json':
                    experiments_params_df = pd.DataFrame.from_dict({experiment_name:hyperparams,}, orient='index')
                    experiments_params_df.to_json(hyperparams['save_experiment_path'], orient='index', indent=4)
                elif extension == '.csv':
                    # experiments_params_df = pd.DataFrame.from_dict(hyperparams)
                    experiments_params_df = pd.DataFrame.from_dict({experiment_name:hyperparams,}, orient='index')
                    # experiments_params_df.to_csv(hyperparams['save_experiment_path'])
                    experiments_params_df.to_csv(hyperparams['save_experiment_path'])
                else:
                    #TODO
                    pass


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
            elif (hyperparams['model_id'] == '3d-densenet-fusion' 
                and hyperparams['dataset'] != 'grss_dfc_2018'):
                print('Cannot use 3d-densenet-fusion model without the grss_dfc_2018 dataset!')
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
                # tf.config.experimental.set_virtual_device_configuration(
                #     gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
                # )
        
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
                    print()
                    print(f' < Dataset Chosen: {dataset_choice} >')
                    print()

                    # Make sure dataset is in per-class data list dictionary
                    if dataset_choice not in per_class_data_lists:
                        per_class_data_lists[dataset_choice] = []
                    
                    # Make sure dataset is in per-class data list dictionary
                    if dataset_choice not in per_class_selected_band_lists:
                        per_class_selected_band_lists[dataset_choice] = []

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
                        #TODO - allow per-modality band selection

                        if hyperparams['select_only_hs_bands']:
                            hs_channels = dataset_info['hs_channels']
                            if hs_channels is not None:
                                # Get the non-hyperspectral data so it
                                # can be appended to the reduced
                                # hyperspectral data later
                                non_hs_channels = [channel for channel in range(data.shape[-1]) if channel not in hs_channels]
                                non_hs_data = data[..., non_hs_channels]
                                data = data[..., hs_channels]

                                hs_channel_labels = [label for channel, label in enumerate(dataset_info['channel_labels']) if channel in hs_channels]
                                non_hs_channel_labels = [label for channel, label in enumerate(dataset_info['channel_labels']) if channel not in hs_channels]

                                data, band_selection_time, bands_selected = band_selection(data, train_gt, **hyperparams)

                                # Update the channel indices to reflect 
                                # reduced HS data
                                num_hs_channels = data.shape[-1]
                                dataset_info['hs_channels'] = range(num_hs_channels)
                                dataset_info['lidar_ms_channels'] = [new_channel + num_hs_channels for new_channel, channel in enumerate(non_hs_channels) if channel in dataset_info['lidar_ms_channels']]
                                dataset_info['lidar_ndsm_channels'] = [new_channel + num_hs_channels for new_channel, channel in enumerate(non_hs_channels) if channel in dataset_info['lidar_ndsm_channels']]
                                dataset_info['vhr_rgb_channels'] = [new_channel + num_hs_channels for new_channel, channel in enumerate(non_hs_channels) if channel in dataset_info['vhr_rgb_channels']]

                                # Stack the non hyperspectral data onto
                                # the data array
                                data = np.dstack((data, non_hs_data))

                                if bands_selected is not None:
                                    # Initialize dictionary of band selection
                                    # data to output to file
                                    per_class_selected_bands = {
                                        'experiment_number': iteration + 1, 
                                        'experiment_name': experiment_name,
                                        'random_seed': seed,
                                        'band_reduction_method': hyperparams['band_reduction_method'],
                                        'band_selection_time': band_selection_time,
                                        'model': None, 
                                        'overall_accuracy': 0.0, 
                                        'average_accuracy': 0.0,
                                        'precision_score': 0.0,
                                        'recall_score': 0.0,
                                        'cohen_kappa_score': 0.0,
                                    }

                                    
                                    # Write selected bands to band selection
                                    # dictionary
                                    for channel, label in enumerate(hs_channel_labels):
                                        if bands_selected is not None and channel in bands_selected:
                                            per_class_selected_bands[f'{label} (channel {channel})'] = True
                                        else:
                                            per_class_selected_bands[f'{label} (channel {channel})'] = False
                                    
                                    for channel, label in enumerate(non_hs_channel_labels):
                                        per_class_selected_bands[f'{label} (channel {channel + len(hs_channels)})'] = True

                            else:
                                print('There are no hyperspectral channels in this experiment! Skipping band selection...')
                                
                        else:
                            data, band_selection_time, bands_selected = band_selection(data, train_gt, **hyperparams)

                            if bands_selected is not None:
                                per_class_selected_bands = {
                                    'experiment_number': iteration + 1, 
                                    'experiment_name': experiment_name,
                                    'random_seed': seed,
                                    'band_reduction_method': hyperparams['band_reduction_method'],
                                    'band_selection_time': band_selection_time,
                                    'model': None, 
                                    'overall_accuracy': 0.0, 
                                    'average_accuracy': 0.0,
                                    'precision_score': 0.0,
                                    'recall_score': 0.0,
                                    'cohen_kappa_score': 0.0,
                                }

                                if dataset_info['channel_labels'] is not None:
                                    for channel, label in enumerate(dataset_info['channel_labels']):
                                        if bands_selected is not None and channel in bands_selected:
                                            per_class_selected_bands[f'{label} (channel {channel})'] = True
                                        else:
                                            per_class_selected_bands[f'{label} (channel {channel})'] = False

                                            # Remove any channels from
                                            # modality channel lists if they
                                            # are not selected
                                            if channel in dataset_info['hs_channels']:
                                                dataset_info['hs_channels'].remove(channel)
                                            if channel in dataset_info['lidar_ms_channels']:
                                                dataset_info['lidar_ms_channels'].remove(channel)
                                            if channel in dataset_info['lidar_ndsm_channels']:
                                                dataset_info['lidar_ndsm_channels'].remove(channel)
                                            if channel in dataset_info['vhr_rgb_channels']:
                                                dataset_info['vhr_rgb_channels'].remove(channel)

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
                    input_sizes = [len(input_channel) for input_channel in input_channels]
                elif (hyperparams['model_id'] == '3d-densenet-fusion'
                      or hyperparams['model_id'] == '3d-densenet-fusion2'
                      or hyperparams['model_id'] == '3d-densenet-fusion3'
                      or hyperparams['model_id'] == '3d-densenet-fusion4'):
                    # branch_1_channels = (*vhr_rgb_channels, )
                    # branch_2_channels = (*lidar_ms_channels, *lidar_ndsm_channels, )
                    # branch_3_channels = (*hs_channels, )
                    # input_channels = (branch_1_channels, branch_2_channels, branch_3_channels)
                    branch_1_channels = (*hs_channels, )
                    branch_2_channels = (*lidar_ms_channels, )
                    branch_3_channels = (*lidar_ndsm_channels, )
                    branch_4_channels = (*vhr_rgb_channels, )
                    input_channels = (branch_1_channels, branch_2_channels, branch_3_channels, branch_4_channels)
                    input_sizes = [len(input_channel) for input_channel in input_channels]
                elif hyperparams['model_id'] == '3d-densenet-modified':
                    if hyperparams['add_branch'] is not None:
                        input_channels = []
                        branch_list = []
                        for branch in hyperparams['add_branch']:
                            branch_channels = []
                            branch_modalities = []
                            for modality in str(branch).split(','):
                                if modality == 'hs' and (hyperparams['use_hs_data'] or hyperparams['use_all_data']):
                                    branch_channels += [*hs_channels, ]
                                elif modality == 'lidar_ms' and (hyperparams['use_lidar_ms_data'] or hyperparams['use_all_data']):
                                    branch_channels += [*lidar_ms_channels, ]
                                elif modality == 'lidar_ndsm' and (hyperparams['use_lidar_ndsm_data'] or hyperparams['use_all_data']):
                                    branch_channels += [*lidar_ndsm_channels, ]
                                elif modality == 'vhr_rgb' and (hyperparams['use_vhr_data'] or hyperparams['use_all_data']):
                                    branch_channels += [*vhr_rgb_channels, ]
                                
                                branch_modalities.append(modality)
                            branch_list.append(branch_modalities)
                            if len(branch_channels) > 0:
                                input_channels.append(branch_channels)
                        
                        for index, modalities in enumerate(branch_list):
                            print(f'Branch {index} modalities: {modalities}')

                        if len(input_channels) > 0:
                            input_sizes = [len(input_channel) for input_channel in input_channels]
                        else:
                            input_channels = None
                            input_sizes = [img_channels]

                    else:
                        input_channels = None
                        input_sizes = [img_channels]
                    
                else:
                    input_channels = None
                    input_sizes = None

                # Check to see if model uses 3d convolutions - if so
                # then the input dimensions will need to be expanded
                # to include the 'planes' dimension
                if (hyperparams['model_id'] == '3d-densenet'
                    # or hyperparams['model_id'] == '3d-densenet-fusion'
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
                        'all_class_labels': all_class_labels,
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
                    'band_reduction_method': hyperparams['band_reduction_method'],
                    'band_selection_time': band_selection_time,
                    'bands_selected': bands_selected,
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
                    'band_reduction_method': hyperparams['band_reduction_method'],
                    'band_selection_time': band_selection_time,
                    'bands_selected': bands_selected,
                })
                for label in all_class_labels:
                    per_class_data[label] = 0.0

                if not reuse_last_dataset:
                    print('-------------------------------------------------------------------')
                    print('SPLIT DATA FOR TRAINING, VALIDATION, AND TESTING')
                    print('-------------------------------------------------------------------')

                    print('Breaking down image into data patches and splitting data into train, validation, and test sets...')
                    train_dataset, val_dataset, test_dataset = create_datasets(data, train_gt, test_gt, **hyperparams)
                    target_test = np.array(test_dataset.labels)
                    # datasets = create_datasets_v2(data, train_gt, test_gt, **hyperparams)

                    # train_dataset = (datasets['train_dataset'], datasets['train_steps'])
                    # val_dataset = (datasets['val_dataset'], datasets['val_steps'])
                    # test_dataset = (datasets['test_dataset'], datasets['test_steps'])
                    # target_test = datasets['target_test']

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
                elif hyperparams['model_id'] == '3d-densenet-modified':
                    model = densenet_3d_modified_model(img_rows=img_rows, 
                                                       img_cols=img_cols, 
                                                       img_channels_list=input_sizes, 
                                                       nb_classes=num_classes, 
                                                       num_dense_blocks=3,
                                                       growth_rate=32, 
                                                       num_1x1_convs=0,
                                                       first_conv_filters=64,
                                                       first_conv_kernel=(5,5,5),
                                                       dropout_1=0.5,
                                                       dropout_2=0.5,
                                                       activation='leaky_relu')
                elif hyperparams['model_id'] == '3d-densenet-fusion':
                    model = densenet_3d_modified_model(img_rows=img_rows, 
                                                       img_cols=img_cols, 
                                                       img_channels_list=input_sizes, 
                                                       nb_classes=num_classes, 
                                                       num_dense_blocks=3,
                                                       growth_rate=32, 
                                                       num_1x1_convs=0,
                                                       first_conv_filters=64,
                                                       first_conv_kernel=(5,5,5),
                                                       dropout_1=0.5,
                                                       dropout_2=0.5,
                                                       activation='leaky_relu')
                elif hyperparams['model_id'] == '3d-densenet-fusion2':
                    model = densenet_3d_fusion_model2(img_rows=img_rows, 
                                                     img_cols=img_cols, 
                                                     img_channels_list=[
                                                            len(hs_channels), 
                                                            len(lidar_ms_channels),
                                                            len(lidar_ndsm_channels), 
                                                            len(vhr_rgb_channels),
                                                     ], 
                                                     nb_classes=num_classes, 
                                                     num_dense_blocks=3)
                elif hyperparams['model_id'] == '3d-densenet-fusion3':
                    model = densenet_3d_fusion_model3(img_rows=img_rows, 
                                                     img_cols=img_cols, 
                                                     img_channels_list=[
                                                            len(hs_channels), 
                                                            len(lidar_ms_channels),
                                                            len(lidar_ndsm_channels), 
                                                            len(vhr_rgb_channels),
                                                     ], 
                                                     nb_classes=num_classes, 
                                                     num_dense_blocks=3)
                # elif hyperparams['model_id'] == '3d-densenet-fusion':
                #     model = densenet_3d_fusion_model(img_rows=img_rows, 
                #                                      img_cols=img_cols, 
                #                                      img_channels_1=len(vhr_rgb_channels), 
                #                                      img_channels_2=len(lidar_ms_channels) + len(lidar_ndsm_channels), 
                #                                      img_channels_3=len(hs_channels), 
                #                                      nb_classes=num_classes, 
                #                                      num_dense_blocks=3)
                # elif hyperparams['model_id'] == '3d-densenet-fusion2':
                #     model = densenet_3d_fusion_model2(img_rows=img_rows, 
                #                                      img_cols=img_cols, 
                #                                      img_channels_1=len(vhr_rgb_channels), 
                #                                      img_channels_2=len(lidar_ms_channels) + len(lidar_ndsm_channels), 
                #                                      img_channels_3=len(hs_channels), 
                #                                      nb_classes=num_classes, 
                #                                      num_dense_blocks=3)
                # elif hyperparams['model_id'] == '3d-densenet-fusion3':
                #     model = densenet_3d_fusion_model3(img_rows=img_rows, 
                #                                      img_cols=img_cols, 
                #                                      img_channels_1=len(vhr_rgb_channels), 
                #                                      img_channels_2=len(lidar_ms_channels) + len(lidar_ndsm_channels), 
                #                                      img_channels_3=len(hs_channels), 
                #                                      nb_classes=num_classes, 
                #                                      num_dense_blocks=3)
                # elif hyperparams['model_id'] == '3d-densenet-fusion4':
                #     model = densenet_3d_fusion_model4(img_rows=img_rows, 
                #                                      img_cols=img_cols, 
                #                                      img_channels_1=len(vhr_rgb_channels), 
                #                                      img_channels_2=len(lidar_ms_channels) + len(lidar_ndsm_channels), 
                #                                      img_channels_3=len(hs_channels), 
                #                                      nb_classes=num_classes, 
                #                                      num_dense_blocks=3)
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
                    num_filters = img_channels * 2
                    model = baseline_cnn_model(img_rows=img_rows, 
                                            img_cols=img_cols, 
                                            img_channels=img_channels, 
                                            patch_size=filter_size, 
                                            nb_filters=num_filters, 
                                            nb_classes=num_classes)
                elif hyperparams['model_id'] == 'nin':
                    model = nin_model(img_rows=img_rows, 
                                    img_cols=img_cols, 
                                    img_channels=img_channels, 
                                    num_classes=num_classes)              
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
                if per_class_selected_bands is not None:
                    per_class_selected_bands['model'] = model.name

                if hyperparams['restore'] is not None:
                    print(f'Restoring {model.name} weights from {hyperparams["restore"]}')
                    model.load_weights(hyperparams['restore'])


                print('-------------------------------------------------------------------')
                print()
                
                if not hyperparams['predict_only'] or hyperparams['predict_only'] is None:
                    print('-------------------------------------------------------------------')
                    print('TRAIN MODEL')
                    print('-------------------------------------------------------------------')

                    # Run experiment on model
                    model, model_train_time = train_model(model=model, 
                                                        train_dataset=train_dataset, 
                                                        val_dataset=val_dataset,  
                                                        iteration=iteration,
                                                        **hyperparams)
                else:
                    model_train_time = None

                print('-------------------------------------------------------------------')
                print('TEST MODEL')
                print('-------------------------------------------------------------------')

                pred_test, model_test_time = test_model(model=model,
                                                        test_dataset=test_dataset,
                                                        **hyperparams)

                if not hyperparams['skip_data_postprocessing']:
                    print('-------------------------------------------------------------------')
                    print('POSTPROCESS THE TEST RESULTS')
                    print('-------------------------------------------------------------------')

                    # Check whether pred_test is the right size
                    print(f'pred_test shape: {pred_test.shape}')
                    print(f'pred_test size:  {pred_test.size}')
                    print(f'test_gt size:    {test_gt.size}')
                    if pred_test.size != test_gt.size:
                        print('Error! pred_test and test_gt do not have same number of elements!')
                        print(f'       pred_test delta: {pred_test.size - test_gt.size} more elements')
                    
                    # Reshape pred_test to original gt image size so that
                    # postprocessing can occur
                    pred_test = np.reshape(pred_test, test_gt.shape)
                    print(f'reshaped pred_test shape: {pred_test.shape}')
                    print(f'test_gt shape:            {test_gt.shape}')

                    pred_test = postprocess_data(pred_test, **hyperparams)
                    print('-------------------------------------------------------------------')
                    print()
                
                    # Remove ignored labels from target and predicted data
                    target_test, pred_test = filter_pred_results(test_gt, pred_test, ignored_labels)
                    
                    

                # Calculate the model performance statistics
                experiment_results = calculate_model_statistics(pred_test, target_test, all_class_labels, **hyperparams)
                experiment_results.update({
                    'experiment_name': experiment_name,
                    'model_name': model.name,
                    'model_train_time': model_train_time,
                    'model_test_time': model_test_time,
                })

                # Copy results to output data
                experiment_data['train_time'] = experiment_results['model_train_time']
                experiment_data['test_time'] = experiment_results['model_test_time']
                experiment_data['overall_accuracy'] = experiment_results['overall_accuracy']
                experiment_data['average_accuracy'] = experiment_results['average_accuracy']
                experiment_data['precision_score'] = experiment_results['precision_score']
                experiment_data['recall_score'] = experiment_results['recall_score']
                experiment_data['cohen_kappa_score'] = experiment_results['cohen_kappa_score']

                per_class_data['train_time'] = model_train_time
                per_class_data['test_time'] = model_test_time
                per_class_data['overall_accuracy'] = experiment_results['overall_accuracy']
                per_class_data['average_accuracy'] = experiment_results['average_accuracy']
                per_class_data['precision_score'] = experiment_results['precision_score']
                per_class_data['recall_score'] = experiment_results['recall_score']
                per_class_data['cohen_kappa_score'] = experiment_results['cohen_kappa_score']

                for index, acc in enumerate(experiment_results['per_class_accuracies']):
                    per_class_data[experiment_results['labels'][index]] = acc
                
                if per_class_selected_bands is not None:
                    per_class_selected_bands['overall_accuracy'] = experiment_results['overall_accuracy']
                    per_class_selected_bands['average_accuracy'] = experiment_results['average_accuracy']
                    per_class_selected_bands['precision_score'] = experiment_results['precision_score']
                    per_class_selected_bands['recall_score'] = experiment_results['recall_score']
                    per_class_selected_bands['cohen_kappa_score'] = experiment_results['cohen_kappa_score']

                # Output experimental results
                output_experiment_results(experiment_results)

                # Save image of confusion matrix
                create_confusion_matrix_plot(experiment_results['confusion_matrix'], 
                                             all_class_labels, 
                                             model.name, 
                                             output_path = output_path,
                                             iteration=iteration)

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
        if per_class_selected_bands is not None:
            per_class_selected_band_lists[dataset_choice].append(per_class_selected_bands)

        print()
        print('-------------------------------------------------------------------')
        print('SAVING RESULTS...')

        experiment_results = pd.DataFrame(experiment_data_list)
        experiment_results.set_index('experiment_number', inplace=True)
        experiment_results.to_csv(os.path.join(output_path, experiments_results_file))
        
        print('  >>> Experiment results saved!')

        for dataset_choice in per_class_data_lists:
            if len(per_class_data_lists[dataset_choice]) > 0:
                file_name = f'{outfile_prefix}__{dataset_choice}__class_results.csv'
                per_class_data_results = pd.DataFrame(per_class_data_lists[dataset_choice])
                per_class_data_results.set_index('experiment_number', inplace=True)
                per_class_data_results.to_csv(os.path.join(output_path, file_name))
                print(f'  >>> {dataset_choice} per-class results saved!')

        for dataset_choice in per_class_selected_band_lists:
            if len(per_class_selected_band_lists[dataset_choice]) > 0:
                file_name = f'{outfile_prefix}__{dataset_choice}__selected_band_results.csv'
                per_class_selected_band_results = pd.DataFrame(per_class_selected_band_lists[dataset_choice])
                per_class_selected_band_results.set_index('experiment_number', inplace=True)
                per_class_selected_band_results.to_csv(os.path.join(output_path, file_name))
                print(f'  >>> {dataset_choice} per-class selected band results saved!')

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