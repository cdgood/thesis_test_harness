#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Dataset loading and setup module

This script sets up hyperspectral and datafusion data sets.

Author:  Christopher Good
Version: 1.0.0

Usage: datasets.py

"""
# See following link for proper docstring documentation
# https://pandas.pydata.org/docs/development/contributing_docstring.html 

### Futures ###
#TODO

### Built-in Imports ###
import math

### Other Library Imports ###
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import (
    Sequence,
    to_categorical, 
) 

### Local Imports ###
from data.grss_dfc_2018_uh import UH_2018_Dataset

### Class Definitions ###
class HyperspectralDataset(Sequence):
    def __init__(self, data, gt, shuffle=True, **params):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            shuffle: bool, set to True to shuffle data at each epoch
            hyperparams: extra hyperparameters for setting up dataset
        """
        # super(HyperspectralDataset, self).__init__()
        self.data = data
        self.input_channels = params['input_channels']
        self.gt = gt
        self.shuffle = shuffle
        self.batch_size = params['batch_size']
        self.patch_size = params['patch_size']
        self.supervision = params['supervision']
        self.ignored_labels = set(params['ignored_labels'])
        self.num_classes = params['n_classes']
        self.loss = params['loss']
        self.expand_dims = params['expand_dims']
        self.flip_augmentation = params["flip_augmentation"]
        self.radiation_augmentation = params["radiation_augmentation"]
        self.mixture_augmentation = params["mixture_augmentation"]
        self.center_pixel = params["center_pixel"]

        if self.input_channels is not None:
            self.multi_input = True
        else:
            self.multi_input = False
        
        if self.supervision == "full":
            mask = np.ones_like(gt)
            for label in self.ignored_labels:
                mask[gt == label] = 0
        # Semi-supervised : use all pixels, except padding
        elif self.supervision == "semi":
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        num_neighbors = self.patch_size // 2
        self.indices = np.array(
            [
                (x, y)
                for x, y in zip(x_pos, y_pos)
                if x > num_neighbors 
                    and x < data.shape[0] - num_neighbors 
                    and y > num_neighbors 
                    and y < data.shape[1] - num_neighbors
            ]
        )

        self.labels = np.array([gt[x, y] for x, y in self.indices])

        # Run epoch end function to initialize dataset
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, i):
        if self.multi_input:
            batch_data = [[] for _ in self.input_channels]
        else:
            batch_data = []
        batch_labels = []

        # Get all items in batch
        for item in range(i*self.batch_size,(i+1)*self.batch_size):

            # Make sure not to look for item id greater than number of
            # indices
            if item >= len(self.indices): break

            # Get index tuple from indices
            index = tuple(self.indices[item])

            # Get data patch for the index
            data, label= self.__get_patch(self.data, 
                                          self.gt,
                                          index, 
                                          self.patch_size)

            if self.flip_augmentation and self.patch_size > 1:
                # Perform data augmentation (only on 2D patches)
                data, label = self.flip(data, label)
            if self.radiation_augmentation and np.random.random() < 0.1:
                data = self.radiation_noise(data)
            if self.mixture_augmentation and np.random.random() < 0.2:
                data = self.mixture_noise(data, label)

            # Extract the center label if needed
            if self.center_pixel and self.patch_size > 1:
                label = label[self.patch_size // 2, self.patch_size // 2]
            # Remove unused dimensions when we work with invidual spectrums
            elif self.patch_size == 1:
                data = data[:, 0, 0]
                label = label[0, 0]

            # Add a fourth dimension for 3D CNN
            if self.expand_dims and self.patch_size > 1:
                # Make 4D data ((Batch x) Planes x Channels x Width x Height)
                # E.g. adding a dimension for 'planes'
                axis = len(data.shape) if K.image_data_format() == 'channels_last' else 0
                data = np.expand_dims(data, axis)
                # patch = tf.expand_dims(patch, 0)

            # Break the data into inputs if needed
            if self.multi_input:
                data = [data.take(channels, axis=data.ndim-1) for channels in self.input_channels]

            # If categorical cross-entropy, make sure labels are one-hot
            # encoded
            if self.loss == 'categorical_crossentropy':
                label = to_categorical(label, num_classes = self.num_classes)

            # Add data to lists
            if self.multi_input:
                for idx in range(len(batch_data)):
                    batch_data[idx].append(tf.convert_to_tensor(data[idx], dtype='float32'))
            else:
                batch_data.append(tf.convert_to_tensor(data, dtype='float32'))
            batch_labels.append(label)

        if self.multi_input:
            for idx in range(len(batch_data)):
                batch_data[idx] = tf.convert_to_tensor(batch_data[idx])
            batch_data = (*batch_data,)
        else:
            batch_data = tf.convert_to_tensor(batch_data)

        batch_labels = tf.convert_to_tensor(batch_labels)

        return batch_data, batch_labels

    @staticmethod
    def __get_patch(data, gt, index, patch_size):
        x, y = index
        x1 = x - patch_size // 2    # Leftmost edge of patch
        y1 = y - patch_size // 2    # Topmost edge of patch
        x2 = x1 + patch_size        # Rightmost edge of patch
        y2 = y1 + patch_size        # Bottommost edge of patch

        patch = data[x1:x2, y1:y2]
        label = gt[x1:x2, y1:y2]

        # Copy the data into numpy arrays
        patch = np.asarray(np.copy(patch), dtype="float32")
        label = np.asarray(np.copy(label), dtype='uint8')
        
        return patch, label
    
    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1 / 25):
        alpha1, alpha2 = np.random.uniform(0.01, 1.0, size=2)
        noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert self.labels[l_indice] == value
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

### Function Definitions ###

def get_hyperspectral_dataset(data, gt, shuffle=True, **params):
    """
    """

    random_seed = params['random_seed']
    input_channels = params['input_channels']
    batch_size = params['batch_size']
    patch_size = params['patch_size']
    supervision = params['supervision']
    ignored_labels = set(params['ignored_labels'])
    num_classes = params['n_classes']
    loss = params['loss']
    expand_dims = params['expand_dims']

    # Determine if this dataset is feeding a multi-input model
    multi_input = True if input_channels is not None else False

    # Get expected data shapes to make sure data is properly formatted
    # if the Shape needs to be fixed
    if multi_input:
        x_shape = [[None, None, None, None, len(channels)] if expand_dims 
                        else [None, None, None, len(channels)]
                        for channels in input_channels]
    else:
        x_shape = [None, None, None, None, data.shape[-1]] if expand_dims \
                    else [None, None, None, data.shape[-1]]
    
    if loss == 'categorical_crossentropy':
        y_shape = [None, num_classes]
    else:
        y_shape = [None, 1]


    # Fully supervised : use all pixels with label not ignored
    if supervision == "full":
        mask = np.ones_like(gt)
        for label in ignored_labels:
            mask[gt == label] = 0
    # Semi-supervised : use all pixels, except padding
    elif supervision == "semi":
        mask = np.ones_like(gt)
    x_pos, y_pos = np.nonzero(mask)
    num_neighbors = patch_size // 2
    indices = np.array(
        [
            (x, y)
            for x, y in zip(x_pos, y_pos)
            if x > num_neighbors 
                and x < data.shape[0] - num_neighbors 
                and y > num_neighbors 
                and y < data.shape[1] - num_neighbors
        ]
    )

    labels = np.array([gt[x, y] for x, y in indices])

    class HSDataset:
        def __init__(self, data, gt, **params):
            
            # Save parameters
            self.data = data
            self.gt = gt

            self.input_channels = params['input_channels']
            self.patch_size = params['patch_size']
            self.num_classes = params['n_classes']
            self.loss = params['loss']
            self.expand_dims = params['expand_dims']

            # Determine if this dataset is feeding a multi-input model
            self.multi_input = True if self.input_channels is not None else False
        
        def __call__(self, i):
            i = tuple(i.numpy())

            x, y = i
            x1 = x - self.patch_size // 2    # Leftmost edge of patch
            y1 = y - self.patch_size // 2    # Topmost edge of patch
            x2 = x1 + self.patch_size        # Rightmost edge of patch
            y2 = y1 + self.patch_size        # Bottommost edge of patch

            patch = self.data[x1:x2, y1:y2]

            # Copy the data into numpy arrays
            patch = np.asarray(np.copy(patch), dtype="float32")
            # patch = tf.convert_to_tensor(patch, dtype="float32")

            if self.patch_size == 1:
                patch = patch[:, 0, 0]

            # Add a fourth dimension for 3D CNN
            if self.expand_dims and self.patch_size > 1:
                # Make 4D data ((Batch x) Planes x Channels x Width x Height)
                # E.g. adding a dimension for 'planes'
                axis = len(patch.shape) if K.image_data_format() == 'channels_last' else 0
                patch = np.expand_dims(patch, axis)
            
            if self.multi_input:
                # Break the data into inputs
                patch = [patch.take(channels, axis=data.ndim-1) for channels in self.input_channels]

            sample = patch

            # Get label for the patch
            label = self.gt[i]

            if self.loss == 'categorical_crossentropy':
                label = to_categorical(label, num_classes = self.num_classes)

            
            return sample, label

    class FixShape:
        def __init__(self, x_shape, y_shape, multi_input):
            self.x_shape = x_shape
            self.y_shape = y_shape
            self.multi_input = multi_input
        
        def __call__(self, x, y):
            if self.multi_input:
                _x = []
                for index, data in enumerate(x):
                    _x.append(data.set_shape(self.x_shape[index]))
                
                x = tuple(_x)
            else:
                x.set_shape(self.x_shape)
            
            y.set_shape(self.y_shape)

            return x, y
    
    hs_dataset = HSDataset(data, gt, **params)

    output_signature = (tf.TensorSpec(shape=(2), dtype=tf.uint32))
    dataset = tf.data.Dataset.from_generator(lambda: indices, 
                                            output_signature=output_signature)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(indices), seed=random_seed,
                                reshuffle_each_iteration=True)

    if multi_input:
        Tout = [tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.uint8)]
    else:
        Tout = [tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
            tf.TensorSpec(shape=None, dtype=tf.uint8)]
    dataset = dataset.map(lambda i: tf.py_function(func=hs_dataset,
                                                   inp=[i],
                                                   Tout=Tout
                                                   ),
                          num_parallel_calls=tf.data.AUTOTUNE)

    # fixup_shape = FixShape(x_shape, y_shape, multi_input)
    # dataset = dataset.batch(batch_size).map(fixup_shape)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, labels

def get_valid_gt_indices(gt, ignored_labels=[]):

    mask = np.ones_like(gt)
    for label in ignored_labels:
        mask[gt == label] = 0

    x_pos, y_pos = np.nonzero(mask)
    indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])

    return indices

def sample_gt(gt, train_size, mode='random'):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    indices = np.nonzero(gt)
    X = list(zip(*indices)) # x,y features
    y = gt[indices].ravel() # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
       train_size = int(train_size)

    if mode == 'random':
       train_indices, test_indices = train_test_split(X, train_size=train_size, stratify=y)
       train_indices = [list(t) for t in zip(*train_indices)]
       test_indices = [list(t) for t in zip(*test_indices)]
       train_gt[train_indices] = gt[train_indices]
       test_gt[test_indices] = gt[test_indices]
    elif mode == 'fixed':
       print(f'Sampling {mode} with train size = {train_size}')
       train_indices, test_indices = [], []
       for c in np.unique(gt):
           if c == 0:
              continue
           indices = np.nonzero(gt == c)
           X = list(zip(*indices)) # x,y features

           train, test = train_test_split(X, train_size=train_size)
           train_indices += train
           test_indices += test
       train_indices = tuple([list(t) for t in zip(*train_indices)])
       test_indices = tuple([list(t) for t in zip(*test_indices)])
       train_gt[train_indices] = gt[train_indices]
       test_gt[test_indices] = gt[test_indices]

    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / (first_half_count + second_half_count)
                    if ratio > 0.9 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError(f'{mode} sampling is not implemented yet.')
    return train_gt, test_gt

def normalize_image(img):
    """
    """
    img = img.astype(float, copy=False)
    img -= img.min()
    img /= img.max()

    return img

def threshold_image(img, threshold):
    """
    """
    img = img[img > threshold] = img.min()
    return img

def filter_image(img, filter_type, filter_size=None, normalize=False):
    """
    """

    failure = False

    if img.ndim == 2:
        height, width = img.shape
        channels = 1
    else:
        height, width, channels = img.shape

    if filter_type == 'median':
        if filter_size is None: filter_size = (3, 3, channels)
        img = median_filter(img, size=filter_size)
    elif filter_type == 'gaussian':
        img = gaussian_filter(img, 1)
    else:
        print(f'Bad filter argument {filter_type}! Image left alone...')
        failure = True
    
    if not failure and normalize:
        img=normalize_image(img)
    
    return img

def histogram_equalization(img):
    """
    """

    # def _equalize_layer(layer):
    #     """
    #     Internal function to equalize a single image channel slice.
    #     """
    #     height, width = layer.shape
    #     h, bin = np.histogram(layer.flatten(), 256, [0, 256])

    #     cdf = np.cumsum(h)

    #     cdf_m = np.ma.masked_equal(cdf,0)
    #     cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    #     cdf_final = np.ma.filled(cdf_m,0).astype('uint8')

    # if img.ndim == 2:
    #     return cv2.equalizeHist(img)
    # else:
    #     for channel in range(img.shape[-1]):
    #         img[:,:, channel] = cv2.equalizeHist(img[:,:,channel])
    #     return img
        #return cv2.equalizeHist(img)
    
    return img
       

def pad_img(img, pad_width, ignore_dims=[]):
    """
    """

    padding = [(pad_width,) if dim not in ignore_dims else (0,) for dim in range(img.ndim)]
    img = np.pad(img, padding, mode='constant')
    
    return img

def load_grss_dfc_2018_uh_dataset(**hyperparams):
    """
    """

    skip_data_preprocessing = hyperparams['skip_data_preprocessing']
    
    hs_resampling = hyperparams['hs_resampling']
    lidar_ms_resampling = hyperparams['lidar_ms_resampling']
    lidar_ndsm_resampling = hyperparams['lidar_ndsm_resampling']
    vhr_resampling = hyperparams['vhr_resampling']

    normalize_hs_data = hyperparams['normalize_hs_data']
    normalize_lidar_ms_data = hyperparams['normalize_lidar_ms_data']
    normalize_lidar_ndsm_data = hyperparams['normalize_lidar_ndsm_data']
    normalize_vhr_data = hyperparams['normalize_vhr_data']

    hs_histogram_equalization = hyperparams['hs_histogram_equalization']
    lidar_ms_histogram_equalization = hyperparams['lidar_ms_histogram_equalization']
    lidar_dsm_histogram_equalization = hyperparams['lidar_dsm_histogram_equalization']
    lidar_dem_histogram_equalization = hyperparams['lidar_dem_histogram_equalization']
    lidar_ndsm_histogram_equalization = hyperparams['lidar_ndsm_histogram_equalization']
    vhr_histogram_equalization = hyperparams['vhr_histogram_equalization']

    hs_data_filter = hyperparams['hs_data_filter']
    lidar_ms_data_filter = hyperparams['lidar_ms_data_filter']
    lidar_dsm_data_filter = hyperparams['lidar_dsm_data_filter']
    lidar_dem_data_filter = hyperparams['lidar_dem_data_filter']
    vhr_data_filter = hyperparams['vhr_data_filter']


    use_all_data = hyperparams['use_all_data']
    if use_all_data:
        use_hs_data = True
        use_lidar_ms_data = True
        use_lidar_ndsm_data = True
        use_vhr_data = True
    else:
        use_hs_data = hyperparams['use_hs_data']
        use_lidar_ms_data = hyperparams['use_lidar_ms_data']
        use_lidar_ndsm_data = hyperparams['use_lidar_ndsm_data']
        use_vhr_data = hyperparams['use_vhr_data']

    hs_channels = []
    lidar_ms_channels = []
    lidar_ndsm_channels = []
    vhr_rgb_channels = []


    dataset = UH_2018_Dataset()
    train_gt = dataset.load_full_gt_image(train_only=True)
    test_gt = dataset.load_full_gt_image(test_only=True)

    data = None

    # Check to see if hyperspectral data is being used
    if use_hs_data:
        # Load hyperspectral data
        if dataset.hs_image is None:
            hs_data = dataset.load_full_hs_image(resampling=hs_resampling)
        else:
            hs_data = dataset.hs_image
        print(f'{dataset.name} hs_data shape: {hs_data.shape}')

        # Check for data equalization, filtering and normalization
        if hs_histogram_equalization and not skip_data_preprocessing:
            hs_data = histogram_equalization(hs_data)
        if hs_data_filter is not None and not skip_data_preprocessing:
            print(f'Filtering hyperspectral data with {hs_data_filter} filter...')
            hs_data = filter_image(hs_data, hs_data_filter)
        if normalize_hs_data:
            print('Normalizing hyperspectral data...')
            hs_data = normalize_image(hs_data)

        # Add hyperspectral data to data cube and save channel indices
        # for hyperspectral data
        if data is None:
            hs_channels = range(hs_data.shape[-1])
            data = np.copy(hs_data)
        else:
            hs_channels = [x + data.shape[-1] for x in range(hs_data.shape[-1])]
            data = np.dstack((data, hs_data))

    # Check to see if lidar multispectral intensity data is being used
    if use_lidar_ms_data:
        # Load LiDAR multispectral data
        if dataset.lidar_ms_image is None:
            lidar_ms_data = dataset.load_full_lidar_ms_image(normalize=normalize_lidar_ms_data,
                                                             resampling=lidar_ms_resampling)
        else:
            lidar_ms_data = dataset.lidar_ms_image
        print(f'{dataset.name} lidar_ms_data shape: {lidar_ms_data.shape}')

        # Check for data equalization, filtering and normalization
        if lidar_ms_histogram_equalization and not skip_data_preprocessing:
            lidar_ms_data = histogram_equalization(lidar_ms_data)
        if lidar_ms_data_filter is not None and not skip_data_preprocessing:
            print(f'Filtering LiDAR multispectral data with {lidar_ms_data_filter} filter...')
            lidar_ms_data = filter_image(lidar_ms_data, lidar_ms_data_filter)
        if normalize_lidar_ms_data:
            print('Normalizing LiDAR multispectral data...')
            lidar_ms_data = normalize_image(lidar_ms_data)

        # Add lidar multispectral data to data cube and save channel
        # indices for lidar multispectral data
        if data is None:
            lidar_ms_channels = range(lidar_ms_data.shape[-1])
            data = np.copy(lidar_ms_data)
        else:
            lidar_ms_channels = [x + data.shape[-1] for x in range(lidar_ms_data.shape[-1])]
            data = np.dstack((data, lidar_ms_data))

    # Check to see if lidar normalized digital surface model data is
    # being used
    if use_lidar_ndsm_data:
        if dataset.lidar_dsm_image is None or dataset.lidar_dem_image is None:
            lidar_dsm_data = dataset.load_full_lidar_dsm_image(resampling=lidar_ndsm_resampling)
            lidar_dem_data = dataset.load_full_lidar_dem_image(resampling=lidar_ndsm_resampling)
        else:
            lidar_dsm_data = dataset.lidar_dsm_image
            lidar_dem_data = dataset.lidar_dem_image
        print(f'{dataset.name} lidar_dsm_data shape: {lidar_dsm_data.shape}')
        print(f'{dataset.name} lidar_dem_data shape: {lidar_dem_data.shape}')

        # Check for data equalization, filtering and normalization
        if lidar_dsm_histogram_equalization and not skip_data_preprocessing:
            lidar_dem_data = histogram_equalization(lidar_dsm_data)
        # Check for data equalization, filtering and normalization
        if lidar_dem_histogram_equalization and not skip_data_preprocessing:
            lidar_dem_data = histogram_equalization(lidar_dem_data)

        # Check for data filtering
        if lidar_dsm_data_filter is not None and not skip_data_preprocessing:
            print(f'Filtering LiDAR DSM data with {lidar_dsm_data_filter} filter...')
            lidar_dsm_data = filter_image(lidar_dsm_data, lidar_dsm_data_filter)
        if lidar_dem_data_filter is not None:
            print(f'Filtering LiDAR DEM data with {lidar_dem_data_filter} filter...')
            lidar_dem_data = filter_image(lidar_dem_data, lidar_dem_data_filter)

        # Create NDSM image
        print('Creating NDSM image from DSM and DEM (NDSM = DSM - DEM)...')
        lidar_ndsm_data = lidar_dsm_data - lidar_dem_data

        # Check for data equalization, filtering and normalization
        if lidar_ndsm_histogram_equalization and not skip_data_preprocessing:
            lidar_ndsm_data = histogram_equalization(lidar_ndsm_data)

        # Check for data normalization
        if normalize_lidar_ndsm_data:
            print('Normalizing LiDAR NDSM data...')
            lidar_ndsm_data = normalize_image(lidar_ndsm_data)

        # Add lidar NDSM data to data cube and save channel
        # index for lidar NDSM data
        if data is None:
            lidar_ndsm_channels = [0]
            data = np.copy(lidar_ndsm_data)
        else:
            lidar_ndsm_channels = [data.shape[-1]]
            data = np.dstack((data, lidar_ndsm_data))

    # Check to see if very high resolution RGB image data is being used
    if use_vhr_data:
        # Load Very High Resolution RGB image
        if dataset.vhr_image is None:
            vhr_data = dataset.load_full_vhr_image(normalize=normalize_vhr_data,
                                                   resampling=vhr_resampling)
        else:
            vhr_data = dataset.vhr_image
        print(f'{dataset.name} vhr_data shape: {vhr_data.shape}')

        # Check for data equalization, filtering and normalization
        if vhr_histogram_equalization and not skip_data_preprocessing:
            vhr_data = histogram_equalization(vhr_data)
        if vhr_data_filter is not None and not skip_data_preprocessing:
            print(f'Filtering VHR RGB data with {vhr_data_filter} filter...')
            vhr_data = filter_image(vhr_data, vhr_data_filter)
        if normalize_vhr_data:
            print('Normalizing VHR RGB data...')
            vhr_data = normalize_image(vhr_data)

        # Add VHR data to data cube and save channel indices for VHR
        # RGB data
        if data is None:
            vhr_rgb_channels = range(vhr_data.shape[-1])
            data = np.copy(vhr_data)
        else:
            vhr_rgb_channels = [x + data.shape[-1] for x in range(vhr_data.shape[-1])]
            data = np.dstack((data, vhr_data))
    
    # Verify that some data was loaded
    if data is not None:
        print(f'{dataset.name} full dataset shape: {data.shape}')
    else:
        print('No data was loaded! Training cancelled...')
        return


    print(f'{dataset.name} train_gt shape: {train_gt.shape}')
    print(f'{dataset.name} test_gt shape: {test_gt.shape}')

    dataset_info = {
        'name': dataset.name,
        'num_classes': dataset.gt_num_classes,
        'ignored_labels': dataset.gt_ignored_labels,
        'class_labels': dataset.gt_class_label_list,
        'label_mapping': dataset.gt_class_value_mapping,
        'hs_channels': hs_channels,
        'lidar_ms_channels': lidar_ms_channels,
        'lidar_ndsm_channels': lidar_ndsm_channels,
        'vhr_rgb_channels': vhr_rgb_channels,
    }

    return data, train_gt, test_gt, dataset_info

def load_indian_pines_dataset(**hyperparams):
    """
    """

    data = None
    train_gt = None
    test_gt = None

    labels = [
            'Undefined',
            'Alfalfa',
            'Corn-notill',
            'Corn-mintill',
            'Corn',
            'Grass-pasture',
            'Grass-trees',
            'Grass-pasture-mowed',
            'Hay-windrowed',
            'Oats',
            'Soybean-notill',
            'Soybean-mintill',
            'Soybean-clean',
            'Wheat',
            'Woods',
            'Buildings-Grass-Trees-Drives',
            'Stone-Steel-Towers',
        ]

    dataset_info = {
        'name': 'Indian Pines',
        'num_classes': len(labels),
        'ignored_labels': [0],
        'class_labels': labels,
        'label_mapping': {index: label for index, label in enumerate(labels)},
    }

    return data, train_gt, test_gt, dataset_info

def load_pavia_center_dataset(**hyperparams):
    """
    """
    
    data = None
    train_gt = None
    test_gt = None

    labels = [
            'Undefined',
            'Water',
            'Trees',
            'Asphalt',
            'Self-Blocking Bricks',
            'Bitumen',
            'Tiles',
            'Shadows',
            'Meadows',
            'Bare Soil',
        ]

    dataset_info = {
        'name': 'University of Pavia',
        'num_classes': len(labels),
        'ignored_labels': [0],
        'class_labels': labels,
        'label_mapping': {index: label for index, label in enumerate(labels)},
    }

    return data, train_gt, test_gt, dataset_info

def load_university_of_pavia_dataset(**hyperparams):
    """
    """
    
    data = None
    train_gt = None
    test_gt = None

    labels = [
            'Undefined',
            'Asphalt',
            'Meadows',
            'Gravel',
            'Trees',
            'Painted metal sheets',
            'Bare Soil',
            'Bitumen',
            'Self-Blocking Bricks',
            'Shadows',
        ]

    dataset_info = {
        'name': 'University of Pavia',
        'num_classes': len(labels),
        'ignored_labels': [0],
        'class_labels': labels,
        'label_mapping': {index: label for index, label in enumerate(labels)},
    }

    return data, train_gt, test_gt, dataset_info


def create_datasets(data, train_gt, test_gt, **hyperparams):
    """
    """

    # Get data from hyperparameters
    patch_size = hyperparams['patch_size']  # N in NxN patch per sample
    train_split = hyperparams['train_split']    # training percent in val/train split
    split_mode = hyperparams['split_mode']

    # Set pad length per dimension
    pad = patch_size // 2

    # Pad only first two dimensions
    ignore_dims = [x for x in range(data.ndim) if x >= 2]

    # Pad all images
    data = pad_img(data, pad, ignore_dims=ignore_dims)
    train_gt = pad_img(train_gt, pad)
    test_gt = pad_img(test_gt, pad)

    # Show updated padded dataset shapes
    print(f'padded data shape: {data.shape}')
    print(f'padded train_gt shape: {train_gt.shape}')
    print(f'padded test_gt shape: {test_gt.shape}')

    # Create validation dataset from training set
    train_gt, val_gt = sample_gt(train_gt, train_split, mode=split_mode)

    dataset_params = (
        'input_channels', 
        'batch_size', 
        'patch_size', 
        'supervision', 
        'ignored_labels', 
        'n_classes', 
        'loss', 
        'expand_dims', 
        'flip_augmentation',
        'radiation_augmentation',
        'mixture_augmentation',
        'center_pixel',
    )

    # Create dataset parameter subset from hyperparameters
    params = {param: hyperparams[param] for param in dataset_params}

    train_dataset = HyperspectralDataset(data, train_gt, **params)

    # Don't use augmentation for validation and test sets
    # params['flip_augmentation'] = False
    # params['radiation_augmentation'] = False
    # params['mixture_augmentation'] = False
    val_dataset = HyperspectralDataset(data, val_gt, **params)

    # If postprocessing is going to occur, change supervision parameter 
    # to 'semi' so all pixels are used (so we can predict the full 
    # image, the prediction then being used for postprocessing)
    if not hyperparams['skip_data_postprocessing']:
        params['supervision'] = 'semi'

    # Don't use augmentation for test set
    params['flip_augmentation'] = False
    params['radiation_augmentation'] = False
    params['mixture_augmentation'] = False
    test_dataset = HyperspectralDataset(data, test_gt, shuffle=False, **params)

    return train_dataset, val_dataset, test_dataset

def create_datasets_v2(data, train_gt, test_gt, **hyperparams):
    """
    """

    # Get data from hyperparameters
    patch_size = hyperparams['patch_size']  # N in NxN patch per sample
    train_split = hyperparams['train_split']    # training percent in val/train split
    split_mode = hyperparams['split_mode']
    batch_size = hyperparams['batch_size']

    # Set pad length per dimension
    pad = patch_size // 2

    # Pad only first two dimensions
    ignore_dims = [x for x in range(data.ndim) if x >= 2]

    # Pad all images
    data = pad_img(data, pad, ignore_dims=ignore_dims)
    train_gt = pad_img(train_gt, pad)
    test_gt = pad_img(test_gt, pad)

    # Show updated padded dataset shapes
    print(f'padded data shape: {data.shape}')
    print(f'padded train_gt shape: {train_gt.shape}')
    print(f'padded test_gt shape: {test_gt.shape}')

    # Create validation dataset from training set
    train_gt, val_gt = sample_gt(train_gt, train_split, mode=split_mode)

    dataset_params = (
        'random_seed',
        'input_channels', 
        'batch_size', 
        'patch_size', 
        'supervision', 
        'ignored_labels', 
        'n_classes', 
        'loss', 
        'expand_dims', 
        'flip_augmentation',
        'radiation_augmentation',
        'mixture_augmentation',
        'center_pixel',
    )

    # Create dataset parameter subset from hyperparameters
    params = {param: hyperparams[param] for param in dataset_params}

    train_dataset, train_labels = get_hyperspectral_dataset(data, train_gt, **params)
    val_dataset, val_labels = get_hyperspectral_dataset(data, val_gt, **params)

    # If postprocessing is going to occur, change supervision parameter 
    # to 'semi' so all pixels are used (so we can predict the full 
    # image, the prediction then being used for postprocessing)
    if not hyperparams['skip_data_postprocessing']:
        params['supervision'] = 'semi'

    test_dataset, target_test = get_hyperspectral_dataset(data, test_gt, shuffle=False, **params)

    datasets = {
        'train_dataset': train_dataset,
        'train_steps': math.ceil(len(train_labels) / batch_size),
        'val_dataset': val_dataset,
        'val_steps': math.ceil(len(val_labels) / batch_size),
        'test_dataset': test_dataset,
        'test_steps': math.ceil(len(target_test) / batch_size),
        'target_test': target_test,
    }

    return datasets