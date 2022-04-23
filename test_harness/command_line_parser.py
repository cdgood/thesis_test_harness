#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Command line parsing functions module of thesis testing harness.

Author:  Christopher Good
Version: 1.0.0

Usage: command_line_parser.py

"""
# See following link for proper docstring documentation
# https://pandas.pydata.org/docs/development/contributing_docstring.html 

### Futures ###
#TODO

### Built-in Imports ###
import argparse

### Other Library Imports ###
#TODO

### Local Imports ###
#TODO

### Constants ###
PARAMETER_LIST = (
    "cuda",
    "restore",
    "output_path",
    "dataset",
    "reuse_last_dataset",
    "skip_data_preprocessing",
    "skip_band_selection",
    "skip_data_postprocessing",
    "model_id",
    "random_seed",
    "epochs",
    "epochs_before_decay",
    "batch_size",
    "patch_size",
    "train_split",
    "split_mode",
    "class_balancing",
    "test_stride",
    "iterations",
    "patience",
    "optimizer",
    "lr",
    "lr_decay_rate",
    "momentum",
    "epsilon",
    "initial_accumulator_value",
    "beta",
    "beta_1",
    "beta_2",
    "amsgrad",
    "rho",
    "centered",
    "nesterov",
    "learning_rate_power",
    "l1_regularization_strength",
    "l2_regularization_strength",
    "l2_shrinkage_regularization_strength",
    "center_pixel",
    "flip_augmentation",
    "radiation_augmentation",
    "mixture_augmentation",
    "use_hs_data",
    "use_lidar_ms_data",
    "use_lidar_ndsm_data",
    "use_vhr_data",
    "use_all_data",
    "normalize_hs_data",
    "normalize_lidar_ms_data",
    "normalize_lidar_ndsm_data",
    "normalize_vhr_data",
    "hs_resampling",
    "lidar_ms_resampling",
    "lidar_ndsm_resampling",
    "vhr_resampling",
    "hs_histogram_equalization",
    "lidar_ms_histogram_equalization",
    "lidar_dsm_histogram_equalization",
    "lidar_dem_histogram_equalization",
    "lidar_ndsm_histogram_equalization",
    "vhr_histogram_equalization",
    "hs_data_filter",
    "lidar_ms_data_filter",
    "lidar_dsm_data_filter",
    "lidar_dem_data_filter",
    "vhr_data_filter",
    "band_reduction_method",
    "n_components",
)

### Function Definitions ###

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