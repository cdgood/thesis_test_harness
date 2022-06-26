#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Machine Learning Model module

This script defines machine learning model creation functions.

Author:  Christopher Good
Version: 1.0.0

Usage: 3d_densenet_fusion.py

"""
# See following link for proper docstring documentation
# https://pandas.pydata.org/docs/development/contributing_docstring.html 

### Futures ###
#TODO

### Built-in Imports ###

### Other Library Imports ###
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Activation,
    Add,
    Average,
    AveragePooling2D,
    AveragePooling3D,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Conv2D,
    Conv3D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    GlobalAveragePooling3D,
    Input,
    MaxPooling2D,
    MaxPooling3D,
    Reshape,
    Resizing,
)
from tensorflow.keras.models import (
    Model, 
    Sequential,
) 
from tensorflow.keras.optimizers import (
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    Ftrl,
    Nadam,
    RMSprop,
    SGD,
)

### Local Imports ###
#TODO

### Function definitions ###

def dense_block_3d(x, blocks, name, growth_rate=32, activation='relu'):
    """A dense block.

    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block_3d(x, growth_rate, activation=activation, name=name + '_block' + str(i + 1))
    return x


def conv_block_3d(x, growth_rate, name, activation='relu'):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 4 if K.image_data_format() == 'channels_last' else 1
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_0_bn')(x)
    x1 = Activation(activation, name=name + f'_0_{activation}')(x1)
    x1 = Conv3D(4 * growth_rate, 1, use_bias=False,
                name=name + '_1_conv', padding='same')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_1_bn')(x1)
    x1 = Activation(activation, name=name + f'_1_{activation}')(x1)
    x1 = Conv3D(growth_rate, 3, padding='same', use_bias=False,
                name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def transition_block_3d(x, reduction, name, activation='relu'):
    """A transition block.

    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 4 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name=name + '_bn')(x)
    x = Activation(activation, name=name + f'_{activation}')(x)
    x = Conv3D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias=False,
               name=name + '_conv', padding='same')(x)
    x = AveragePooling3D(1, strides=(2, 2, 2), name=name + '_pool', padding='same')(x)
    return x

def densenet_3d_modified_model(img_rows, img_cols, img_channels_list, nb_classes, 
                               num_dense_blocks=3,
                               growth_rate=32, 
                               num_1x1_convs=0,
                               first_conv_filters=64,
                               first_conv_kernel=(3,3,3),
                               dropout_1=0.5,
                               dropout_2=0.5,
                               activation='relu'):

    branch_shapes = []

    # Initialize shapes
    for img_channels in img_channels_list:
        if K.image_data_format() == 'channels_last':
            branch_shapes.append((img_rows, img_cols, img_channels))
        else:
            branch_shapes.append((img_channels, img_rows, img_cols))

    # Initialize inputs
    branch_inputs = [Input(shape=shape) for shape in branch_shapes]

    # Print input shapes
    for index, branch_input in enumerate(branch_inputs):
        print(f'Branch {index+1} input shape: {branch_input.shape}')


    # Set up branches
    branches = []

    for index, branch_input in enumerate(branch_inputs):
        num_channels = img_channels_list[index]
        branch_num = index + 1
        x = branch_input

        for conv_1x1_num in range(num_1x1_convs):
            x = Conv2D(num_channels, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                        kernel_initializer='he_normal', name=f'Branch_{branch_num}_Conv1x1_{conv_1x1_num}')(x)
            x = Activation(activation, name=f'Branch_{branch_num}_Conv1x1_{activation}_{conv_1x1_num}')(x)

        if K.image_data_format() == 'channels_last':
            x = Reshape((*branch_input.shape[1:], 1), name=f'Branch_{branch_num}_Reshape')(x)
        else:
            x = Reshape((1, *branch_input.shape[1:]), name=f'Branch_{branch_num}_Reshape')(x)


        x = Conv3D(first_conv_filters, kernel_size=first_conv_kernel, strides=(1, 1, 1), padding='same',
                    kernel_initializer='he_normal', name=f'Branch_{branch_num}_3DConv')(x)
        x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', 
                    name=f'Branch_{branch_num}_MaxPooling3D')(x)

        # Dense Blocks
        x = dense_block_3d(x, num_dense_blocks, growth_rate=growth_rate, activation=activation, name=f'Branch_{branch_num}__conv1')
        x = transition_block_3d(x, dropout_1, activation=activation, name=f'Branch_{branch_num}__pool1')
        x = dense_block_3d(x, num_dense_blocks, growth_rate=growth_rate, activation=activation, name=f'Branch_{branch_num}__conv2')
        x = transition_block_3d(x, dropout_2, activation=activation, name=f'Branch_{branch_num}__pool2')
        x = dense_block_3d(x, num_dense_blocks, growth_rate=growth_rate, activation=activation, name=f'Branch_{branch_num}__conv3')

        x = GlobalAveragePooling3D(name=f'Branch_{branch_num}__avg_pool')(x)

        branches.append(x)

    # Print the output shape of the branches
    for index, branch in enumerate(branches):
        print(f'Branch {index+1} output shape: {branch.shape}')


    if len(img_channels_list) > 1:
    # Branch fusion
        fusion = Concatenate(axis=1, name='Fusion_Concatenate')(branches)
        print(f'Shape after concatenation: {fusion.shape}')
        fusion = Dense(units=fusion.shape[-1], kernel_initializer='he_normal', name='Fusion_Dense_1')(fusion)
        fusion = Activation(activation, name=f'Fusion_{activation}')(fusion)
        fusion = Dense(units=nb_classes, kernel_initializer='he_normal', name='Fusion_Dense_2')(fusion)
        out = Activation('softmax', name='Fusion_Softmax')(fusion)

        print(f'Fusion output shape: {out.shape}')

        model_name = '3D-Densenet-Fusion'
    else:
        # x = Dense(units=x.shape[-1], kernel_initializer='he_normal', name='Dense_1')(x)
        # x = Activation(activation, name=f'Dense_1_{activation}')(x)
        x = Dense(units=nb_classes, kernel_initializer='he_normal', name='Dense_2')(x)
        out = Activation('softmax', name='Softmax')(x)

        print(f'Output shape: {out.shape}')

        model_name = '3D-Densenet'

    model = Model(inputs=branch_inputs, 
                  outputs=out,
                  name=model_name)

    return model