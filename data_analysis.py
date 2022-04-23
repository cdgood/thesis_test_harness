#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""GRSS DFC 2018 data analysis script for thesis test harness.

Author:  Christopher Good
Version: 1.0.0

Usage: data_analysis.py

"""
# See following link for proper docstring documentation
# https://pandas.pydata.org/docs/development/contributing_docstring.html 

### Futures ###
#TODO

### Built-in Imports ###
import argparse
import datetime
import os
import time

### Other Library Imports ###
from chart_studio import tools
import chart_studio.plotly as py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from plotly import tools
from plotly.offline import init_notebook_mode, iplot
import plotly.figure_factory as ff
import plotly.graph_objects as go
# import plotly.plotly as py
import seaborn as sns

### Local Imports ###
from data.grss_dfc_2018_uh import UH_2018_Dataset

### Environment ###
init_notebook_mode(connected=True)
pd.set_option('display.max_columns', 100)
sns.set(rc={'figure.figsize':(12, 16)})

### Constants ###
#TODO

### Definitions ###
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def progressBar(iterable, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)
    # Progress Bar Printing Function
    def printProgressBar (iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()

def load_data():
    """
    """
    dataset = UH_2018_Dataset()
    gt_train = dataset.load_full_gt_image(train_only=True)
    gt_test = dataset.load_full_gt_image(test_only=True)
    hs_data = dataset.load_full_hs_image(thres=False, 
                                         normalize=False, 
                                         resampling='average',   # Also need to try 'nearest'
                                        )
    lidar_ms_data = dataset.load_full_lidar_ms_image(thres=False,
                                                     normalize=False,
                                                     resampling=None,
                                                    )
    lidar_dsm_data = dataset.load_full_lidar_dsm_image(thres=False,
                                                       normalize=False,
                                                       resampling=None,
                                                      )
    lidar_dem_data = dataset.load_full_lidar_dem_image(thres=False,
                                                       normalize=False,
                                                       resampling=None,
                                                      )
    vhr_rgb_data = dataset.load_full_vhr_image(thres=False,
                                               normalize=False,
                                               resampling='cubic_spline',
                                              )

def load_hs_data():
    """
    """
    dataset = UH_2018_Dataset()
    gt_train = dataset.load_full_gt_image(train_only=True)
    hs_data = dataset.load_full_hs_image(thres=False, 
                                         normalize=False, 
                                         resampling='average',   # Also need to try 'nearest'
                                        )

    return hs_data, gt_train

def load_lidar_ms_data():
    """
    """
    dataset = UH_2018_Dataset()
    gt_train = dataset.load_full_gt_image(train_only=True)
    lidar_ms_data = dataset.load_full_lidar_ms_image(thres=False,
                                                     normalize=False,
                                                     resampling=None,
                                                    )

    return lidar_ms_data, gt_train

def load_lidar_dsm_data():
    """
    """
    dataset = UH_2018_Dataset()
    gt_train = dataset.load_full_gt_image(train_only=True)
    lidar_dsm_data = dataset.load_full_lidar_dsm_image(thres=False,
                                                       normalize=False,
                                                       resampling=None,
                                                      )

    return lidar_dsm_data, gt_train

def load_lidar_dem_data():
    """
    """
    dataset = UH_2018_Dataset()
    gt_train = dataset.load_full_gt_image(train_only=True)
    lidar_dem_data = dataset.load_full_lidar_dem_image(thres=False,
                                                       normalize=False,
                                                       resampling=None,
                                                      )

    return lidar_dem_data, gt_train

def load_vhr_rgb_data():
    """
    """
    dataset = UH_2018_Dataset()
    gt_train = dataset.load_full_gt_image(train_only=True)
    vhr_rgb_data = dataset.load_full_vhr_image(thres=False,
                                               normalize=False,
                                               resampling='cubic_spline',
                                              )
    return vhr_rgb_data, gt_train

def get_sample_pixels(data, gt, ignored_labels, num_samples):
    """
    """
    pixel_class_map = {val: [] for val in np.unique(gt)}
    
    height, width = gt.shape

    for row in range(height):
        for col in range(width):
            pixel_class_map[gt[row, col]].append({'pixel':(row, col), 'channels':data[row, col]})
    
    sample_pixels = {}

    for label in pixel_class_map:
        if label in ignored_labels: continue

        pixels = np.array(pixel_class_map[label], dtype=object)
        num_pixels = len(pixel_class_map[label])
        sample_indices = list(np.random.choice(num_pixels, size=num_samples))
        sample_pixels[label] = list(pixels[sample_indices])

    return sample_pixels

def save_sample_pixel_profiles(sample_pixels, class_labels, channel_labels, channel_colors=None, output_path='./'):
    """
    """
    for lval, samples in sample_pixels.items():
        label = class_labels[lval]
        sample_output_dir = os.path.join(output_path, label)

        for index, sample in enumerate(samples):
            
            # Get pixel values
            pixel_loc = sample['pixel']
            r, c = pixel_loc
            channels = sample['channels']
            channel_list = [x for x in range(len(channels))]

            # Create vertical plot file name
            file_name = f'vertical_plots/{label}_pixel_r{r}_c{c}_profile_vertical.png'
            file_path = os.path.join(sample_output_dir, file_name)

            if not os.path.exists(file_path):
                # Set up vertical bar graph plot
                plt.figure(figsize=(12,9))
                bar = plt.bar(channel_list, channels, color=channel_colors)
                plt.bar_label(bar, channels, rotation='vertical')
                plt.grid(color='grey', linestyle='-', linewidth=1, axis='y')
                plt.xticks(channel_list, channel_labels, rotation='vertical')
                plt.xlabel('Channel')
                plt.ylim(0, 16000)
                plt.ylabel('Intensity')
                plt.title(f'Channel profile for point {pixel_loc} of class "{label}"')
                plt.tight_layout()

                # Save bar plot image
                print(f'>>> Saving sample plot {file_name}...')
                plt.savefig(file_path, bbox_inches='tight')

                # Clear plot data for next plot
                plt.clf()
                plt.close()

            # Create horizontal plot file name
            file_name = f'horizontal_plots/{label}_pixel_r{r}_c{c}_profile_horizontal.png'
            file_path = os.path.join(sample_output_dir, file_name)

            if not os.path.exists(file_path):
                # Set up horizontal bar graph plot
                plt.figure(figsize=(12,9))
                bar = plt.barh(channel_list, channels, color=channel_colors)
                plt.bar_label(bar, channels)
                plt.grid(color='grey', linestyle='-', linewidth=1, axis='x')
                plt.yticks(channel_list, channel_labels)
                plt.ylabel('Channel')
                plt.xlim(0, 16000)
                plt.xlabel('Intensity')
                plt.title(f'Channel profile for point {pixel_loc} of class "{label}"')
                plt.tight_layout()

                # Save bar plot image
                print(f'>>> Saving sample plot {file_name}...')
                plt.savefig(file_path, bbox_inches='tight')

                # Clear plot data for next plot
                plt.clf()
                plt.close()


def get_class_dataframes(data, gt, ignored_labels, class_labels, channel_labels=['data']):
    """
    """
    if data.ndim == 3:
        height, width, channels = data.shape
    elif data.ndim == 2:
        height, width = data.shape
        channels = 1
    else:
        raise Exception('Data does not have 2 or 3 dimensions!')
    
    class_info = {label: [] for lval, label in enumerate(class_labels) if lval not in ignored_labels}

    for row in range(height):
        for col in range(width):
            lval = gt[row, col]
            label = class_labels[lval]
            if lval in ignored_labels: continue

            class_info[label].append(data[row, col])
    
    for label in class_info:
        class_info[label] = pd.DataFrame(np.array(class_info[label]),
                                         columns=channel_labels)
    
    return class_info

def save_dataframe_descriptions(dataframes, output_path='./'):
    """
    """
    for label in dataframes:
        file_name = f'{label}_description.csv'
        file_path = os.path.join(output_path, file_name)
        if not os.path.exists(file_path):
            print(f'Saving "{label}" data description to "{file_name}"...')
            dataframes[label].describe().to_csv(file_path)

def get_box_plots(dataframes, output_path='./', show_figures=True):
    """
    """
    
    for label in dataframes:
        # Make dataframe variable
        df = dataframes[label]

        # Generate file path for saving image
        file_name = f'{label}__box_plots.png'
        file_path = os.path.join(output_path, file_name)

        # Get random color profile for box plot
        red, green, blue = list(np.random.choice(256, size=3))

        data = [
            go.Box(
                y=df[column],
                name=column,
                marker=dict(
                    color = f'rgb({red},{green},{blue})',
                ),
            )
            for column in df.columns
        ]

        layout = go.Layout(
            title = f'Plots of channel intensity distribution for "{label}"'
        )

        # Create figure and save image
        fig = go.Figure(data=data, layout=layout)
        # fig.write_image(file_path)

        # Show figure
        if show_figures:
            py.iplot(fig)

def get_sns_box_plots(dataframes, output_path='./', show_figures=True):
    """
    """
    
    for label in dataframes:
        # Make dataframe variable
        df = dataframes[label]

        # Generate file path for saving image
        file_name = f'{label}__box_plots.png'
        file_path = os.path.join(output_path, file_name)

        if not os.path.exists(file_path) or show_figures:
            sns.boxplot(data=df)
            plt.xlabel('Intensity Values')
            plt.xticks(rotation=90)
            plt.ylabel('Channels')
            plt.title(f'{label} Channel Intensities Distribution')

            # Create figure and save image
            print(f'Saving "{label}" box plot to "{file_name}"...')
            plt.savefig(file_path)

            if show_figures:
                # Show figure
                plt.show()

            # Clear figure for next iteration
            plt.clf()
            plt.close()

def get_sns_violin_plots(dataframes, output_path='./', show_figures=True):
    """
    """
    
    for label in dataframes:
        # Make dataframe variable
        df = dataframes[label]

        # Generate file path for saving image
        file_name = f'{label}__violin_plots.png'
        file_path = os.path.join(output_path, file_name)

        if not os.path.exists(file_path) or show_figures:
            sns.violinplot(data=df)
            plt.xlabel('Intensity Values')
            plt.xticks(rotation=90)
            plt.ylabel('Channels')
            plt.title(f'{label} Channel Intensities Distribution')

            # Create figure and save image
            print(f'Saving "{label}" violin plot to "{file_name}"...')
            plt.savefig(file_path)

            if show_figures:
                # Show figure
                plt.show()

            # Clear figure for next iteration
            plt.clf()
            plt.close()

def create_data_df(data, channel_labels):
    """
    """
    
    df = {'row':[], 'column':[]}
    for channel in channel_labels:
        df[channel] = []

    if data.ndim == 3:
        height, width, channels = data.shape
    elif data.ndim == 2:
        height, width = data.shape
        channels = 1
    else:
        return None

    total_samples = height * width
    printProgressBar(0, total_samples, 
                     prefix='Progress:', 
                     suffix='Complete', 
                     length=50)

    for row in range(height):
        for col in range(width):
            df['row'].append(row)
            df['column'].append(col)
            for ch_idx, intensity in enumerate(data[row, col]):
                df[channel_labels[ch_idx]].append(intensity)
            
            printProgressBar(col+row*width, total_samples, 
                             prefix='Progress:', 
                             suffix='Complete', 
                             length=50)
    
    print() # New line so text doesn't overlap progress bar
    df = pd.DataFrame.from_dict(df)

    return df

def create_hs_data_csv(output_path='./'):
    """
    """
    dataset = UH_2018_Dataset()
    channel_labels = dataset.hs_band_wavelength_labels

    print('Loading resampled hyperspectral image...')
    hs_data_resample = dataset.load_full_hs_image(thres=False, 
                                                  normalize=False, 
                                                  resampling='average',   # Also need to try 'nearest'
                                                  gsd=dataset.gsd_gt
                                                 )

    print('Creating dataframe from resampled hyperspectral image...')
    df_hs = create_data_df(hs_data_resample, channel_labels)
    file_path = os.path.join(output_path, 'hyperspectral_resample.csv')
    
    print('Creating CSV file of resampled hyperspectral data cube')
    df_hs.to_csv(file_path)

    # Delete to save memory
    del hs_data_resample
    del df_hs

    print('Loading original size hyperspectral image...')
    hs_data_orig = dataset.load_full_hs_image(thres=False, 
                                              normalize=False, 
                                              resampling=None,   # Also need to try 'nearest'
                                              gsd=dataset.gsd_hs
                                             )

    print('Creating dataframe from original hyperspectral image...')
    df_hs = create_data_df(hs_data_orig, channel_labels)
    file_path = os.path.join(output_path, 'hyperspectral_original.csv')
    
    print('Creating CSV file of original hyperspectral data cube')
    df_hs.to_csv(file_path)

def create_lidar_ms_data_csv(output_path='./'):
    """
    """
    dataset = UH_2018_Dataset()
    channel_labels = dataset.lidar_ms_band_wavelength_labels

    print('Loading original size LiDAR multispectral image...')
    lidar_ms_data_orig = dataset.load_full_lidar_ms_image(thres=False, 
                                                          normalize=False, 
                                                          resampling=None,
                                                          gsd=dataset.gsd_lidar
                                                         )

    print('Creating dataframe from original LiDAR multispectral image...')
    df_lidar_ms = create_data_df(lidar_ms_data_orig, channel_labels)
    file_path = os.path.join(output_path, 'lidar_ms_original.csv')
    
    print('Creating CSV file of original LiDAR multispectral data cube')
    df_lidar_ms.to_csv(file_path)

def create_vhr_rgb_data_csv(output_path='./'):
    """
    """
    dataset = UH_2018_Dataset()
    channel_labels = dataset.vhr_channel_labels

    print('Loading original size VHR RGB image...')
    vhr_data_orig = dataset.load_full_vhr_image(thres=False, 
                                                normalize=False, 
                                                resampling=None,
                                                gsd=dataset.gsd_vhr
                                               )

    print('Creating dataframe from original VHR RGB image...')
    df_vhr = create_data_df(vhr_data_orig, channel_labels)
    file_path = os.path.join(output_path, 'vhr_rgb_original.csv')
    
    print('Creating CSV file of original VHR RGB data cube')
    df_vhr.to_csv(file_path)

    # Delete to save memory
    del vhr_data_orig
    del df_vhr

    print('Loading resampled size VHR RGB image...')
    vhr_data_resampled = dataset.load_full_vhr_image(thres=False, 
                                                     normalize=False, 
                                                     resampling='cubic_spline',
                                                     gsd=dataset.gsd_gt
                                                    )

    print('Creating dataframe from resampled VHR RGB image...')
    df_vhr = create_data_df(vhr_data_resampled, channel_labels)
    file_path = os.path.join(output_path, 'vhr_rgb_resampled.csv')
    
    print('Creating CSV file of original VHR RGB data cube')
    df_vhr.to_csv(file_path)

def get_list_of_analysis_subdirectories(dataset_name=None, class_labels=[]):
    """
    """

    if len(class_labels) == 0:
        subdirectories = [
            'box_plots',
            'dataset_csvs',
            'descriptions',
            'violin_plots',
            'sample_profiles',
            'sample_profiles/vertical_plots',
            'sample_profiles/horizontal_plots',
        ]
    else:
        subdirectories = [
            'box_plots',
            'dataset_csvs',
            'descriptions',
            'violin_plots',
            'sample_profiles',
        ]
        for label in class_labels:
            subdirectories.append(f'sample_profiles/{label}')
            subdirectories.append(f'sample_profiles/{label}/vertical_plots')
            subdirectories.append(f'sample_profiles/{label}/horizontal_plots')
    
    if dataset_name is not None:
        subdirectories = [dataset_name] + [f'{dataset_name}/{subdirectory}' for subdirectory in subdirectories]
    
    return subdirectories

def make_analysis_dirs(location='./', subdirectories = []):
    """
    """
    print(' --- Analysis Directory Creation ---')

    base_analysis_dir = os.path.join(location, 'analysis')
    if not os.path.exists(base_analysis_dir):
        print(f'Creating directory "{base_analysis_dir}"...')
        os.mkdir(base_analysis_dir)
    
    if len(subdirectories) > 0:
        for subdirectory in subdirectories:
            dir_path = os.path.join(base_analysis_dir, subdirectory)
            if not os.path.exists(dir_path):
                print(f'Creating directory "{dir_path}"...')
                os.mkdir(dir_path)
    
    print(' --- Analysis Directory Creation Complete ! ---')

def clean_analysis_dirs(location='./', subdirectories = []):
    """
    """
    print(' --- Analysis Directory Clean ---')

    base_analysis_dir = os.path.join(location, 'analysis')
    if os.path.exists(base_analysis_dir):
        for subdirectory in subdirectories:
            dir_path = os.path.join(base_analysis_dir, subdirectory)
            if os.path.exists(dir_path):
                print(f'Cleaning directory "{dir_path}"...')
                for file in os.scandir(dir_path):
                    if not os.path.isdir(file.path):
                        os.remove(file.path)
    
    print(' --- Analysis Directory Clean Complete ! ---')

### Main ###
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('Dataset analysis script')
    parser.add_argument(
        "--no-show-figures",
        action='store_true',
        help="Don't show figures when analyzing dataset",
    )
    parser.add_argument(
        "--analysis-dir-path",
        type=str,
        default='./',
        help="Path to create analysis directory at",
    )
    parser.add_argument(
        "--clean-analysis-dir",
        action='store_true',
        help="Clean files from the analysis directories",
    )

    args = parser.parse_args()

    show_figures = not args.no_show_figures
    analysis_dir_path = args.analysis_dir_path
    clean = args.clean_analysis_dir

    dataset = UH_2018_Dataset()
    ignored_labels = dataset.gt_ignored_labels
    class_labels = dataset.gt_class_label_list
    channel_labels = dataset.hs_band_wavelength_labels
    channel_colors = dataset.hs_band_rgb_list

    subdirectories = get_list_of_analysis_subdirectories(dataset_name='grss_dfc_2018',
                                                         class_labels=class_labels)

    if clean:
        clean_analysis_dirs(location=analysis_dir_path,
                            subdirectories=subdirectories)
    
    # Make sure all analysis directories are created
    make_analysis_dirs(location=analysis_dir_path,
                       subdirectories=subdirectories)

    data, gt = load_hs_data()

    sample_pixels = get_sample_pixels(data, gt, ignored_labels, 10)
    save_sample_pixel_profiles(sample_pixels, 
                               class_labels,
                               channel_labels,
                               channel_colors=channel_colors,
                               output_path='./analysis/grss_dfc_2018/sample_profiles')

    dataframes = get_class_dataframes(data, gt, 
                                      ignored_labels, 
                                      class_labels, 
                                      channel_labels=channel_labels)
    save_dataframe_descriptions(dataframes, 
                                output_path='./analysis/grss_dfc_2018/descriptions/')
    get_sns_violin_plots(dataframes, 
                         output_path='./analysis/grss_dfc_2018/violin_plots/', 
                         show_figures=show_figures)
    get_sns_box_plots(dataframes, 
                      output_path='./analysis/grss_dfc_2018/box_plots/', 
                      show_figures=show_figures)

    create_hs_data_csv(output_path='./analysis/grss_dfc_2018/dataset_csvs')
    create_lidar_ms_data_csv(output_path='./analysis/grss_dfc_2018/dataset_csvs')
    create_vhr_rgb_data_csv(output_path='./analysis/grss_dfc_2018/dataset_csvs')