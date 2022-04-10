#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Dataset exploratory data analysis module

This script conducts exploratory data analysis on hyperspectral and 
datafusion data sets.

Author:  Christopher Good
Version: 1.0.0

Usage: data_analysis.py

"""

### Built-in Imports ###
import os

### Other Library Imports ###
import matplotlib.pyplot as plt
import numpy as np
import spectral

### Local Imports ###
from grss_dfc_2018_uh import UH_2018_Dataset

### Function Definitions ###

def normalize_img(img):
    img = img.astype(float, copy=False)
    img -= img.min()
    img /= img.max()

    return img

def threshold_img(img, threshold):
    img = img[img > threshold] = img.min()
    return img

def get_indices(gt, ignored_classes=None):
    """
    """

    indices = []
    if ignored_classes is None:
        ignored_classes = []
    for row in range(gt.shape[0]):
        for col in range(gt.shape[1]):
            index = (row, col)
            if gt[index] not in ignored_classes:
                indices.append(index)
    
    return indices

def get_spectrum_distribution(img, gt, 
                              indices=None, 
                              ignored_classes=None,
                              max_intensity=None):
    """
    """


    if indices is None:
        indices = get_indices(gt, ignored_classes=ignored_classes)

    if max_intensity is None:
        max_intensity = np.max(img)

    bins = [[] for _ in range(np.max(gt)+1)]

    for index in indices:
        bins[gt[index]].append(img[index])

    return np.array(bins)


def create_spectrum_graphs_and_images(img, gt, classes, 
                                        ignored_classes=None,
                                        band_labels=None,
                                        band_rgb=None,
                                        output_dir='./'):

    # Normalize image
    img = normalize_img(img)

    # Get number of bands
    bands = img.shape[-1]

    # Get valid indices
    print('Getting valid indices...')
    indices = get_indices(gt, ignored_classes=ignored_classes)

    # Initialize lists and dictionaries
    print('Initializing lists and dictionaries...')
    max_intensity = np.max(img)
    intensities_list = [x for x in range(max_intensity)]
    band_list = [x for x in range(bands)]
    class_averages = {class_label: np.zeros(bands) for class_label in classes}
    class_num_samples= {class_label: 0 for class_label in classes}

    # Create grayscale images directory
    grayscale_image_dir = os.path.join(output_dir, 'grayscale_images')
    print(f'Creating directory "{grayscale_image_dir}"')
    os.makedirs(grayscale_image_dir, exist_ok=True)

    # Create class averages directory
    class_averages_dir = os.path.join(output_dir, 'class_averages')
    print(f'Creating directory "{class_averages_dir}"')
    os.makedirs(class_averages_dir, exist_ok=True)

    # Create class directories
    for class_label in classes:
        dir_name = os.path.join(output_dir, f'classes/{class_label}')
        print(f'Creating directory "{dir_name}"')
        os.makedirs(dir_name, exist_ok=True)
    

    # Loop through bands to get distributions
    for band in range(bands):
        # if band_labels is not None:
        #     band_label = f'Band_#{band}_{band_labels[band]}'
        # else:
        #     band_label = f'Band_#{band}'
        
        band_label = f'Band_#{band}'

        # Create grayscale image of band
        grayscale_image_path = os.path.join(grayscale_image_dir,
            f'{band_label}.png')
        print(f'Saving {grayscale_image_path}...')
        spectral.save_rgb(grayscale_image_path, img, [band], format='png')

        # Get distribution of intensities
        print(f'Getting {band_label} spectrum distribution...')
        bins = get_spectrum_distribution(img[:,:,band],gt,
                                         indices=indices,
                                         ignored_classes=ignored_classes,
                                         max_intensity=max_intensity)

        # Create intensity distribution graphs
        for index, class_label in enumerate(classes):

            class_dir = os.path.join(output_dir, f'classes/{class_label}')

            # Add amount of samples to class
            class_num_samples[class_label] += sum(bins[index])

            # Add values to averages
            for intensity in range(bins.shape[-1]):
                class_averages[class_label][band] += bins[index][intensity] * intensity

            # Create distribution plot name
            plot_file_name = os.path.join(class_dir, 
                f'{class_label}_{band_label}_intensities_distribution.png')

            # Set up bar graph plot
            print(f'bins[index].shape: {bins[index].shape}')
            print(f'intensities_list len: {len(intensities_list)}')
            plt.figure(figsize=(12,9))
            plt.bar(intensities_list, bins[index])
            plt.grid(color='grey', linestyle='-', linewidth=1, axis='y')
            plt.xticks(intensities_list, intensities_list, rotation='vertical')
            plt.xlabel('Intensity value')
            plt.ylim(0, 8000)
            plt.ylabel('Number of samples')
            plt.title(f'{class_label} {band_label} {band_labels[band]} intensities distribution')
            plt.tight_layout()

            print(f'Saving {plot_file_name}...')

            # Save plot
            plt.savefig(plot_file_name, bbox_inches='tight')

            # Clear plot data for next plot
            plt.clf()
    
    # Generate averages graphs
    for class_label in classes:
        class_averages[class_label] /= class_num_samples[class_label]


        # Create average distribution plot name
        plot_file_name = os.path.join(class_averages_dir, 
            f'{class_label}_average_wavelengths.png')


        # Set up average wavelength bar graph plot
        plt.figure(figsize=(12,9))
        plt.bar(band_list, class_averages[class_label], color=band_rgb)
        plt.grid(color='grey', linestyle='-', linewidth=1, axis='y')
        plt.xticks(band_list, band_labels, rotation='vertical')
        plt.xlabel('Wavelength')
        plt.ylim(0, 8000)
        plt.ylabel('Intensity')
        plt.title(f'Average wavelength for {class_label}')
        plt.tight_layout()

        # Save plot
        print(f'Saving {plot_file_name}...')
        plt.savefig(plot_file_name, bbox_inches='tight')

        # Clear plot data for next plot
        plt.clf()

### Main ###

if __name__ == "__main__":
    dataset = UH_2018_Dataset()

    gt = dataset.load_full_gt_image()
    hs_img = dataset.load_full_hs_image(thres=False, normalize=False)
    classes = dataset.gt_class_label_list
    ignored_classes = dataset.gt_ignored_labels
    band_labels = dataset.hs_band_wavelength_labels
    band_rgb = dataset.hs_band_rgb_list
    output_dir = './analysis/grss_dfc_2018/'

    create_spectrum_graphs_and_images(hs_img, gt, classes, 
                                      ignored_classes=ignored_classes,
                                      band_labels=band_labels,
                                      band_rgb=band_rgb,
                                      output_dir=output_dir)