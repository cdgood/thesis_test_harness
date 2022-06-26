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
from tkinter import W
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

def dense_block_3d(x, blocks, name):
    """A dense block.

    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block_3d(x, 32, name=name + '_block' + str(i + 1))
    return x


def conv_block_3d(x, growth_rate, name):
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
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv3D(4 * growth_rate, 1, use_bias=False,
                name=name + '_1_conv', padding='same')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv3D(growth_rate, 3, padding='same', use_bias=False,
                name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def transition_block_3d(x, reduction, name):
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
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv3D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias=False,
               name=name + '_conv', padding='same')(x)
    x = AveragePooling3D(1, strides=(2, 2, 2), name=name + '_pool', padding='same')(x)
    return x

def densenet_3d_fusion_model(img_rows, img_cols, 
                             img_channels_1, img_channels_2, img_channels_3, 
                             nb_classes, num_dense_blocks=3):

    # Note - normal num_dense_blocks is 6

    # Initialize Inputs
    if K.image_data_format() == 'channels_last':
        branch_1_shape = (img_rows, img_cols, img_channels_1)
        branch_2_shape = (img_rows, img_cols, img_channels_2)
        branch_3_shape = (img_rows, img_cols, img_channels_3)
    else:
        branch_1_shape = (img_channels_1, img_rows, img_cols)
        branch_2_shape = (img_channels_2, img_rows, img_cols)
        branch_3_shape = (img_channels_3, img_rows, img_cols)

    branch_1_input = Input(shape=branch_1_shape)
    branch_2_input = Input(shape=branch_2_shape)
    branch_3_input = Input(shape=branch_3_shape)

    print(f'Branch 1 input shape: {branch_1_input.shape}')
    print(f'Branch 2 input shape: {branch_2_input.shape}')
    print(f'Branch 3 input shape: {branch_3_input.shape}')

    ### Branch 1

    # 3D Convolution and pooling
    x = Conv2D(img_channels_1, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                kernel_initializer='he_normal', name='Branch_1_Conv1x1')(branch_1_input)
    if K.image_data_format() == 'channels_last':
        x = Reshape((*branch_1_input.shape[1:], 1), name='Branch_1_Reshape')(x)
    else:
        x = Reshape((1, *branch_1_input.shape[1:]), name='Branch_1_Reshape')(x)
    x = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME',
                kernel_initializer='he_normal', name='Branch_1_3DConv')(x)
    x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', 
                name='Branch_1_MaxPooling3D')(x)

    # Dense Blocks
    x = dense_block_3d(x, num_dense_blocks, name='Branch_1__conv1')
    x = transition_block_3d(x, 0.5, name='Branch_1__pool1')
    x = dense_block_3d(x, num_dense_blocks, name='Branch_1__conv2')
    x = transition_block_3d(x, 0.5, name='Branch_1__pool2')
    x = dense_block_3d(x, num_dense_blocks, name='Branch_1__conv3')
    branch_1 = GlobalAveragePooling3D(name='Branch_1__avg_pool')(x)

    ### Branch 2

    # 3D Convolution and pooling
    y = Conv2D(img_channels_2, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                kernel_initializer='he_normal', name='Branch_2_Conv1x1')(branch_2_input)
    if K.image_data_format() == 'channels_last':
        y = Reshape((*branch_2_input.shape[1:], 1), name='Branch_2_Reshape')(y)
    else:
        y = Reshape((1, *branch_2_input.shape[1:]), name='Branch_2_Reshape')(y)
    y = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME', 
                kernel_initializer='he_normal', name='Branch_2_3DConv')(y)
    y = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', 
                name='Branch_2_MaxPooling3D')(y)

    # Dense Blocks
    y = dense_block_3d(y, num_dense_blocks, name='Branch_2__conv1')
    y = transition_block_3d(y, 0.5, name='Branch_2__pool1')
    y = dense_block_3d(y, num_dense_blocks, name='Branch_2__conv2')
    y = transition_block_3d(y, 0.5, name='Branch_2__pool2')
    y = dense_block_3d(y, num_dense_blocks, name='Branch_2__conv3')
    branch_2 = GlobalAveragePooling3D(name='Branch_2__avg_pool')(y)


    ### Branch 3

    # 3D Convolution and pooling
    z = Conv2D(img_channels_3, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                kernel_initializer='he_normal', name='Branch_3_Conv1x1')(branch_3_input)
    if K.image_data_format() == 'channels_last':
        z = Reshape((*branch_3_input.shape[1:], 1), name='Branch_3_Reshape')(z)
    else:
        z = Reshape((1, *branch_3_input.shape[1:]), name='Branch_3_Reshape')(z)
    z = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME', 
                kernel_initializer='he_normal', name='Branch_3_3DConv')(z)
    z = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', 
                name='Branch_3_MaxPooling3D')(z)

    # Dense Blocks
    z = dense_block_3d(z, num_dense_blocks, name='Branch_3__conv1')
    z = transition_block_3d(z, 0.5, name='Branch_3__pool1')
    z = dense_block_3d(z, num_dense_blocks, name='Branch_3__conv2')
    z = transition_block_3d(z, 0.5, name='Branch_3__pool2')
    z = dense_block_3d(z, num_dense_blocks, name='Branch_3__conv3')
    branch_3 = GlobalAveragePooling3D(name='Branch_3__avg_pool')(z)


    print(f'Branch 1 output shape: {branch_1.shape}')
    print(f'Branch 2 output shape: {branch_2.shape}')
    print(f'Branch 3 output shape: {branch_3.shape}')

    # Branch fusion
    fusion = Concatenate(axis=1, name='Fusion_Concatenate')([branch_1, branch_2, branch_3])
    print(f'Shape after concatenation: {fusion.shape}')
    fusion = Dense(units=fusion.shape[-1], kernel_initializer='he_normal', name='Fusion_Dense_1')(fusion)
    fusion = Activation('relu', name='Fusion_ReLU')(fusion)
    fusion = Dense(units=nb_classes, kernel_initializer='he_normal', name='Fusion_Dense_2')(fusion)
    out = Activation('softmax', name='Fusion_Softmax')(fusion)

    print(f'Output shape: {out.shape}')

    model = Model(inputs=[branch_1_input, branch_2_input, branch_3_input], 
                  outputs=out,
                  name='3D-Densenet-Fusion')

    return model

def densenet_3d_fusion_model2(img_rows, img_cols, 
                             img_channels_1, img_channels_2, img_channels_3, 
                             nb_classes, num_dense_blocks=3):

    # Note - normal num_dense_blocks is 6

    # Initialize Inputs
    if K.image_data_format() == 'channels_last':
        branch_1_shape = (img_rows, img_cols, img_channels_1)
        branch_2_shape = (img_rows, img_cols, img_channels_2)
        branch_3_shape = (img_rows, img_cols, img_channels_3)
    else:
        branch_1_shape = (img_channels_1, img_rows, img_cols)
        branch_2_shape = (img_channels_2, img_rows, img_cols)
        branch_3_shape = (img_channels_3, img_rows, img_cols)

    branch_1_input = Input(shape=branch_1_shape)
    branch_2_input = Input(shape=branch_2_shape)
    branch_3_input = Input(shape=branch_3_shape)

    print(f'Branch 1 input shape: {branch_1_input.shape}')
    print(f'Branch 2 input shape: {branch_2_input.shape}')
    print(f'Branch 3 input shape: {branch_3_input.shape}')

    ### Branch 1

    # 3D Convolution and pooling
    x = Conv2D(img_channels_1, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                kernel_initializer='he_normal', name='Branch_1_Conv1x1_1')(branch_1_input)
    x = Conv2D(img_channels_1, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                kernel_initializer='he_normal', name='Branch_1_Conv1x1_2')(x)
    if K.image_data_format() == 'channels_last':
        x = Reshape((*branch_1_input.shape[1:], 1), name='Branch_1_Reshape')(x)
    else:
        x = Reshape((1, *branch_1_input.shape[1:]), name='Branch_1_Reshape')(x)
    x = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME',
                kernel_initializer='he_normal', name='Branch_1_3DConv')(x)
    x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', 
                name='Branch_1_MaxPooling3D')(x)

    # Dense Blocks
    x = dense_block_3d(x, num_dense_blocks, name='Branch_1__conv1')
    x = transition_block_3d(x, 0.5, name='Branch_1__pool1')
    x = dense_block_3d(x, num_dense_blocks, name='Branch_1__conv2')
    x = transition_block_3d(x, 0.5, name='Branch_1__pool2')
    branch_1 = dense_block_3d(x, num_dense_blocks, name='Branch_1__conv3')

    ### Branch 2

    # 3D Convolution and pooling
    y = Conv2D(img_channels_2, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                kernel_initializer='he_normal', name='Branch_2_Conv1x1_1')(branch_2_input)
    y = Conv2D(img_channels_2, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                kernel_initializer='he_normal', name='Branch_2_Conv1x1_2')(y)
    if K.image_data_format() == 'channels_last':
        y = Reshape((*branch_2_input.shape[1:], 1), name='Branch_2_Reshape')(y)
    else:
        y = Reshape((1, *branch_2_input.shape[1:]), name='Branch_2_Reshape')(y)
    y = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME', 
                kernel_initializer='he_normal', name='Branch_2_3DConv')(y)
    y = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', 
                name='Branch_2_MaxPooling3D')(y)

    # Dense Blocks
    y = dense_block_3d(y, num_dense_blocks, name='Branch_2__conv1')
    y = transition_block_3d(y, 0.5, name='Branch_2__pool1')
    y = dense_block_3d(y, num_dense_blocks, name='Branch_2__conv2')
    y = transition_block_3d(y, 0.5, name='Branch_2__pool2')
    branch_2 = dense_block_3d(y, num_dense_blocks, name='Branch_2__conv3')


    ### Branch 3

    # 3D Convolution and pooling
    z = Conv2D(img_channels_3, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                kernel_initializer='he_normal', name='Branch_3_Conv1x1_1')(branch_3_input)
    z = Conv2D(img_channels_3, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                kernel_initializer='he_normal', name='Branch_3_Conv1x1_2')(z)
    if K.image_data_format() == 'channels_last':
        z = Reshape((*branch_3_input.shape[1:], 1), name='Branch_3_Reshape')(z)
    else:
        z = Reshape((1, *branch_3_input.shape[1:]), name='Branch_3_Reshape')(z)
    z = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME', 
                kernel_initializer='he_normal', name='Branch_3_3DConv')(z)
    z = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', 
                name='Branch_3_MaxPooling3D')(z)

    # Dense Blocks
    z = dense_block_3d(z, num_dense_blocks, name='Branch_3__conv1')
    z = transition_block_3d(z, 0.5, name='Branch_3__pool1')
    z = dense_block_3d(z, num_dense_blocks, name='Branch_3__conv2')
    z = transition_block_3d(z, 0.5, name='Branch_3__pool2')
    branch_3 = dense_block_3d(z, num_dense_blocks, name='Branch_3__conv3')


    print(f'Branch 1 output shape: {branch_1.shape}')
    print(f'Branch 2 output shape: {branch_2.shape}')
    print(f'Branch 3 output shape: {branch_3.shape}')

    # If channels_last, then (batch, row, col, channels, planes)
    # else if channels_first, then (batch, planes, channels, row, col)
    channel_axis = 3 if K.image_data_format() == 'channels_last' else 2
    plane_axis = 4 if K.image_data_format() == 'channels_last' else 1

    # Branch fusion
    fusion = Concatenate(axis=channel_axis, name='Fusion_Concatenate')([branch_1, branch_2, branch_3])
    print(f'Shape after concatenation: {fusion.shape}')
    # if K.image_data_format() == 'channels_last':
    #     fusion = Reshape(fusion.shape[1:-1], name='Fusion_Reshape')(fusion)
    # else:
    #     fusion = Reshape(fusion.shape[2:], name='Fusion_Reshape')(fusion)
    # print(f'Shape after Reshaping: {fusion.shape}')
    # fusion = Conv2D(fusion.shape[channel_axis], (1,1), strides=(1,1), padding='same',
    #                     name='Fusion_Conv1x1')(fusion)
    fusion = Conv3D(fusion.shape[plane_axis], (1,1,1), strides=(1,1,1), padding='same',
                        name='Fusion_Conv1x1x1_1')(fusion)
    fusion = Activation('relu', name='Fusion_ReLU_1')(fusion)
    fusion = Conv3D(fusion.shape[plane_axis], (1,1,1), strides=(1,1,1), padding='same',
                        name='Fusion_Conv1x1x1_2')(fusion)
    fusion = Activation('relu', name='Fusion_ReLU_2')(fusion)
    #fusion = Flatten(name='Fusion_Flatten')(fusion)
    fusion = GlobalAveragePooling3D(name='Fusion_Average_Pooling')(fusion)
    fusion = Dense(units=fusion.shape[-1], kernel_initializer='he_normal', name='Fusion_Dense_1')(fusion)
    fusion = Activation('relu', name='Fusion_ReLU_3')(fusion)
    fusion = Dense(units=nb_classes, kernel_initializer='he_normal', name='Fusion_Dense_2')(fusion)
    out = Activation('softmax', name='Fusion_Softmax')(fusion)

    print(f'Output shape: {out.shape}')

    model = Model(inputs=[branch_1_input, branch_2_input, branch_3_input], 
                  outputs=out,
                  name='3D-Densenet-Fusion')

    return model

def densenet_3d_fusion_model3(img_rows, img_cols, 
                             img_channels_1, img_channels_2, img_channels_3, 
                             nb_classes, num_dense_blocks=3):

    # Note - normal num_dense_blocks is 6

    # Initialize Inputs
    if K.image_data_format() == 'channels_last':
        branch_1_shape = (img_rows, img_cols, img_channels_1)
        branch_2_shape = (img_rows, img_cols, img_channels_2)
        branch_3_shape = (img_rows, img_cols, img_channels_3)
    else:
        branch_1_shape = (img_channels_1, img_rows, img_cols)
        branch_2_shape = (img_channels_2, img_rows, img_cols)
        branch_3_shape = (img_channels_3, img_rows, img_cols)

    branch_1_input = Input(shape=branch_1_shape)
    branch_2_input = Input(shape=branch_2_shape)
    branch_3_input = Input(shape=branch_3_shape)

    print(f'Branch 1 input shape: {branch_1_input.shape}')
    print(f'Branch 2 input shape: {branch_2_input.shape}')
    print(f'Branch 3 input shape: {branch_3_input.shape}')

    ### Branch 1

    # 3D Convolution and pooling
    x = Conv2D(img_channels_1, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                kernel_initializer='he_normal', name='Branch_1_Conv1x1_1')(branch_1_input)
    x = Conv2D(img_channels_1, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                kernel_initializer='he_normal', name='Branch_1_Conv1x1_2')(x)
    if K.image_data_format() == 'channels_last':
        x = Reshape((*branch_1_input.shape[1:], 1), name='Branch_1_Reshape')(x)
    else:
        x = Reshape((1, *branch_1_input.shape[1:]), name='Branch_1_Reshape')(x)
    x = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME',
                kernel_initializer='he_normal', name='Branch_1_3DConv')(x)
    x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', 
                name='Branch_1_MaxPooling3D')(x)

    # Dense Blocks
    x = dense_block_3d(x, num_dense_blocks, name='Branch_1__conv1')
    x = transition_block_3d(x, 0.5, name='Branch_1__pool1')
    x = dense_block_3d(x, num_dense_blocks, name='Branch_1__conv2')
    x = transition_block_3d(x, 0.5, name='Branch_1__pool2')
    x = dense_block_3d(x, num_dense_blocks, name='Branch_1__conv3')
    x = GlobalAveragePooling3D(name='Branch_1__avg_pool')(x)
    branch_1 = Dense(units=nb_classes, kernel_initializer='he_normal', name='Branch_1_Dense')(x)

    ### Branch 2

    # 3D Convolution and pooling
    y = Conv2D(img_channels_2, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                kernel_initializer='he_normal', name='Branch_2_Conv1x1_1')(branch_2_input)
    y = Conv2D(img_channels_2, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                kernel_initializer='he_normal', name='Branch_2_Conv1x1_2')(y)
    if K.image_data_format() == 'channels_last':
        y = Reshape((*branch_2_input.shape[1:], 1), name='Branch_2_Reshape')(y)
    else:
        y = Reshape((1, *branch_2_input.shape[1:]), name='Branch_2_Reshape')(y)
    y = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME', 
                kernel_initializer='he_normal', name='Branch_2_3DConv')(y)
    y = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', 
                name='Branch_2_MaxPooling3D')(y)

    # Dense Blocks
    y = dense_block_3d(y, num_dense_blocks, name='Branch_2__conv1')
    y = transition_block_3d(y, 0.5, name='Branch_2__pool1')
    y = dense_block_3d(y, num_dense_blocks, name='Branch_2__conv2')
    y = transition_block_3d(y, 0.5, name='Branch_2__pool2')
    y = dense_block_3d(y, num_dense_blocks, name='Branch_2__conv3')
    y = GlobalAveragePooling3D(name='Branch_2__avg_pool')(y)
    branch_2 = Dense(units=nb_classes, kernel_initializer='he_normal', name='Branch_2_Dense')(y)


    ### Branch 3

    # 3D Convolution and pooling
    z = Conv2D(img_channels_3, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                kernel_initializer='he_normal', name='Branch_3_Conv1x1_1')(branch_3_input)
    z = Conv2D(img_channels_3, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                kernel_initializer='he_normal', name='Branch_3_Conv1x1_2')(z)
    if K.image_data_format() == 'channels_last':
        z = Reshape((*branch_3_input.shape[1:], 1), name='Branch_3_Reshape')(z)
    else:
        z = Reshape((1, *branch_3_input.shape[1:]), name='Branch_3_Reshape')(z)
    z = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME', 
                kernel_initializer='he_normal', name='Branch_3_3DConv')(z)
    z = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', 
                name='Branch_3_MaxPooling3D')(z)

    # Dense Blocks
    z = dense_block_3d(z, num_dense_blocks, name='Branch_3__conv1')
    z = transition_block_3d(z, 0.5, name='Branch_3__pool1')
    z = dense_block_3d(z, num_dense_blocks, name='Branch_3__conv2')
    z = transition_block_3d(z, 0.5, name='Branch_3__pool2')
    z = dense_block_3d(z, num_dense_blocks, name='Branch_3__conv3')
    z = GlobalAveragePooling3D(name='Branch_3__avg_pool')(z)
    branch_3 = Dense(units=nb_classes, kernel_initializer='he_normal', name='Branch_3_Dense')(z)


    print(f'Branch 1 output shape: {branch_1.shape}')
    print(f'Branch 2 output shape: {branch_2.shape}')
    print(f'Branch 3 output shape: {branch_3.shape}')

    # Branch fusion
    fusion = Average(name='Fusion_Average')([branch_1, branch_2, branch_3])
    print(f'Shape after average: {fusion.shape}')
    out = Activation('softmax', name='Fusion_Softmax')(fusion)

    print(f'Output shape: {out.shape}')

    model = Model(inputs=[branch_1_input, branch_2_input, branch_3_input], 
                  outputs=out,
                  name='3D-Densenet-Fusion')

    return model

def densenet_3d_fusion_model4(img_rows, img_cols, 
                             img_channels_1, img_channels_2, img_channels_3, 
                             nb_classes, num_dense_blocks=3):

    # Note - normal num_dense_blocks is 6

    # Initialize Inputs
    if K.image_data_format() == 'channels_last':
        branch_1_shape = (img_rows, img_cols, img_channels_1)
        branch_2_shape = (img_rows, img_cols, img_channels_2)
        branch_3_shape = (img_rows, img_cols, img_channels_3)
    else:
        branch_1_shape = (img_channels_1, img_rows, img_cols)
        branch_2_shape = (img_channels_2, img_rows, img_cols)
        branch_3_shape = (img_channels_3, img_rows, img_cols)

    branch_1_input = Input(shape=branch_1_shape)
    branch_2_input = Input(shape=branch_2_shape)
    branch_3_input = Input(shape=branch_3_shape)

    print(f'Branch 1 input shape: {branch_1_input.shape}')
    print(f'Branch 2 input shape: {branch_2_input.shape}')
    print(f'Branch 3 input shape: {branch_3_input.shape}')

    ### Branch 1

    # 3D Convolution and pooling
    x = Conv2D(img_channels_1, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                kernel_initializer='he_normal', name='Branch_1_Conv1x1_1')(branch_1_input)
    x = Conv2D(img_channels_1, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                kernel_initializer='he_normal', name='Branch_1_Conv1x1_2')(x)
    if K.image_data_format() == 'channels_last':
        x = Reshape((*branch_1_input.shape[1:], 1), name='Branch_1_Reshape')(x)
    else:
        x = Reshape((1, *branch_1_input.shape[1:]), name='Branch_1_Reshape')(x)
    x = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME',
                kernel_initializer='he_normal', name='Branch_1_3DConv')(x)
    x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', 
                name='Branch_1_MaxPooling3D')(x)

    # Dense Blocks
    x = dense_block_3d(x, num_dense_blocks, name='Branch_1__conv1')
    x = transition_block_3d(x, 0.5, name='Branch_1__pool1')
    x = dense_block_3d(x, num_dense_blocks, name='Branch_1__conv2')
    x = transition_block_3d(x, 0.5, name='Branch_1__pool2')
    x = dense_block_3d(x, num_dense_blocks, name='Branch_1__conv3')
    branch_1 = GlobalAveragePooling3D(name='Branch_1__avg_pool')(x)

    ### Branch 2

    # 3D Convolution and pooling
    y = Conv2D(img_channels_2, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                kernel_initializer='he_normal', name='Branch_2_Conv1x1_1')(branch_2_input)
    y = Conv2D(img_channels_2, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                kernel_initializer='he_normal', name='Branch_2_Conv1x1_2')(y)
    if K.image_data_format() == 'channels_last':
        y = Reshape((*branch_2_input.shape[1:], 1), name='Branch_2_Reshape')(y)
    else:
        y = Reshape((1, *branch_2_input.shape[1:]), name='Branch_2_Reshape')(y)
    y = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME', 
                kernel_initializer='he_normal', name='Branch_2_3DConv')(y)
    y = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', 
                name='Branch_2_MaxPooling3D')(y)

    # Dense Blocks
    y = dense_block_3d(y, num_dense_blocks, name='Branch_2__conv1')
    y = transition_block_3d(y, 0.5, name='Branch_2__pool1')
    y = dense_block_3d(y, num_dense_blocks, name='Branch_2__conv2')
    y = transition_block_3d(y, 0.5, name='Branch_2__pool2')
    y = dense_block_3d(y, num_dense_blocks, name='Branch_2__conv3')
    branch_2 = GlobalAveragePooling3D(name='Branch_2__avg_pool')(y)

    ### Branch 3

    # 3D Convolution and pooling
    z = Conv2D(img_channels_3, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                kernel_initializer='he_normal', name='Branch_3_Conv1x1_1')(branch_3_input)
    z = Conv2D(img_channels_3, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                kernel_initializer='he_normal', name='Branch_3_Conv1x1_2')(z)
    if K.image_data_format() == 'channels_last':
        z = Reshape((*branch_3_input.shape[1:], 1), name='Branch_3_Reshape')(z)
    else:
        z = Reshape((1, *branch_3_input.shape[1:]), name='Branch_3_Reshape')(z)
    z = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME', 
                kernel_initializer='he_normal', name='Branch_3_3DConv')(z)
    z = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', 
                name='Branch_3_MaxPooling3D')(z)

    # Dense Blocks
    z = dense_block_3d(z, num_dense_blocks, name='Branch_3__conv1')
    z = transition_block_3d(z, 0.5, name='Branch_3__pool1')
    z = dense_block_3d(z, num_dense_blocks, name='Branch_3__conv2')
    z = transition_block_3d(z, 0.5, name='Branch_3__pool2')
    z = dense_block_3d(z, num_dense_blocks, name='Branch_3__conv3')
    branch_3 = GlobalAveragePooling3D(name='Branch_3__avg_pool')(z)


    print(f'Branch 1 output shape: {branch_1.shape}')
    print(f'Branch 2 output shape: {branch_2.shape}')
    print(f'Branch 3 output shape: {branch_3.shape}')

    # Branch fusion
    fusion = Add(name='Fusion_Addition')([branch_1, branch_2, branch_3])
    print(f'Shape after addition: {fusion.shape}')
    # fusion = Dense(units=fusion.shape[-1], kernel_initializer='he_normal', name='Fusion_Dense_1')(fusion)
    # fusion = Activation('relu', name='Fusion_ReLU')(fusion)
    fusion = Dense(units=nb_classes, kernel_initializer='he_normal', name='Fusion_Dense')(fusion)
    out = Activation('softmax', name='Fusion_Softmax')(fusion)

    print(f'Output shape: {out.shape}')

    model = Model(inputs=[branch_1_input, branch_2_input, branch_3_input], 
                  outputs=out,
                  name='3D-Densenet-Fusion')

    return model

def densenet_3d_fusion_model5(img_rows, img_cols, 
                             img_channels_1, img_channels_2, img_channels_3, img_channels_4,
                             nb_classes, num_dense_blocks=3):

    # Note - normal num_dense_blocks is 6

    # Initialize Inputs
    if K.image_data_format() == 'channels_last':
        branch_1_shape = (img_rows, img_cols, img_channels_1)
        branch_2_shape = (img_rows, img_cols, img_channels_2)
        branch_3_shape = (img_rows, img_cols, img_channels_3)
        branch_4_shape = (img_rows, img_cols, img_channels_4)
    else:
        branch_1_shape = (img_channels_1, img_rows, img_cols)
        branch_2_shape = (img_channels_2, img_rows, img_cols)
        branch_3_shape = (img_channels_3, img_rows, img_cols)
        branch_4_shape = (img_channels_4, img_rows, img_cols)

    branch_1_input = Input(shape=branch_1_shape)
    branch_2_input = Input(shape=branch_2_shape)
    branch_3_input = Input(shape=branch_3_shape)
    branch_4_input = Input(shape=branch_4_shape)

    print(f'Branch 1 input shape: {branch_1_input.shape}')
    print(f'Branch 2 input shape: {branch_2_input.shape}')
    print(f'Branch 3 input shape: {branch_3_input.shape}')
    print(f'Branch 4 input shape: {branch_4_input.shape}')

    ### Branch 1

    # 3D Convolution and pooling
    if img_channels_1 > 2:
        w = Conv2D(img_channels_1, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                    kernel_initializer='he_normal', name='Branch_1_Conv1x1_1')(branch_1_input)
        w = Conv2D(img_channels_1, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                    kernel_initializer='he_normal', name='Branch_1_Conv1x1_2')(w)
        if K.image_data_format() == 'channels_last':
            w = Reshape((*branch_1_input.shape[1:], 1), name='Branch_1_Reshape')(w)
        else:
            w = Reshape((1, *branch_1_input.shape[1:]), name='Branch_1_Reshape')(w)
        w = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME',
                    kernel_initializer='he_normal', name='Branch_1_3DConv')(w)
        w = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', 
                    name='Branch_1_MaxPooling3D')(w)
    else:
        if K.image_data_format() == 'channels_last':
            w = Reshape((*branch_1_input.shape[1:], 1), name='Branch_1_Reshape')(branch_1_input)
        else:
            w = Reshape((1, *branch_1_input.shape[1:]), name='Branch_1_Reshape')(w)
        w = Conv3D(64, kernel_size=(3, 3, img_channels_1), strides=(1, 1, 1), padding='SAME',
                    kernel_initializer='he_normal', name='Branch_1_3DConv')(w)
        w = MaxPooling3D(pool_size=(3, 3, img_channels_1), strides=(2, 2, img_channels_1), padding='same', 
                    name='Branch_1_MaxPooling3D')(w)

    # Dense Blocks
    w = dense_block_3d(w, num_dense_blocks, name='Branch_1__conv1')
    w = transition_block_3d(w, 0.5, name='Branch_1__pool1')
    w = dense_block_3d(w, num_dense_blocks, name='Branch_1__conv2')
    w = transition_block_3d(w, 0.5, name='Branch_1__pool2')
    branch_1 = dense_block_3d(w, num_dense_blocks, name='Branch_1__conv3')

    ### Branch 2

    # 3D Convolution and pooling
    if img_channels_2 > 2:
        x = Conv2D(img_channels_2, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                    kernel_initializer='he_normal', name='Branch_2_Conv1x1_1')(branch_2_input)
        x = Conv2D(img_channels_2, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                    kernel_initializer='he_normal', name='Branch_2_Conv1x1_2')(x)
        if K.image_data_format() == 'channels_last':
            x = Reshape((*branch_2_input.shape[1:], 1), name='Branch_2_Reshape')(x)
        else:
            x = Reshape((1, *branch_2_input.shape[1:]), name='Branch_2_Reshape')(x)
        x = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME',
                    kernel_initializer='he_normal', name='Branch_2_3DConv')(x)
        x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', 
                    name='Branch_1_MaxPooling3D')(x)
    else:
        if K.image_data_format() == 'channels_last':
            x = Reshape((*branch_2_input.shape[1:], 1), name='Branch_2_Reshape')(branch_2_input)
        else:
            x = Reshape((1, *branch_2_input.shape[1:]), name='Branch_2_Reshape')(x)
        x = Conv3D(64, kernel_size=(3, 3, img_channels_2), strides=(1, 1, 1), padding='SAME',
                    kernel_initializer='he_normal', name='Branch_2_3DConv')(x)
        x = MaxPooling3D(pool_size=(3, 3, img_channels_2), strides=(2, 2, img_channels_2), padding='same', 
                    name='Branch_2_MaxPooling3D')(x)

    # Dense Blocks
    x = dense_block_3d(x, num_dense_blocks, name='Branch_2__conv1')
    x = transition_block_3d(x, 0.5, name='Branch_2__pool1')
    x = dense_block_3d(x, num_dense_blocks, name='Branch_2__conv2')
    x = transition_block_3d(x, 0.5, name='Branch_2__pool2')
    branch_2 = dense_block_3d(x, num_dense_blocks, name='Branch_2__conv3')



    ### Branch 3

    # 3D Convolution and pooling
    y = Conv2D(img_channels_2, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                kernel_initializer='he_normal', name='Branch_2_Conv1x1_1')(branch_2_input)
    y = Conv2D(img_channels_2, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                kernel_initializer='he_normal', name='Branch_2_Conv1x1_2')(y)
    if K.image_data_format() == 'channels_last':
        y = Reshape((*branch_2_input.shape[1:], 1), name='Branch_2_Reshape')(y)
    else:
        y = Reshape((1, *branch_2_input.shape[1:]), name='Branch_2_Reshape')(y)
    y = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME', 
                kernel_initializer='he_normal', name='Branch_2_3DConv')(y)
    y = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', 
                name='Branch_2_MaxPooling3D')(y)

    # Dense Blocks
    y = dense_block_3d(y, num_dense_blocks, name='Branch_2__conv1')
    y = transition_block_3d(y, 0.5, name='Branch_2__pool1')
    y = dense_block_3d(y, num_dense_blocks, name='Branch_2__conv2')
    y = transition_block_3d(y, 0.5, name='Branch_2__pool2')
    branch_2 = dense_block_3d(y, num_dense_blocks, name='Branch_2__conv3')


    ### Branch 3

    # 3D Convolution and pooling
    z = Conv2D(img_channels_3, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                kernel_initializer='he_normal', name='Branch_3_Conv1x1_1')(branch_3_input)
    z = Conv2D(img_channels_3, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                kernel_initializer='he_normal', name='Branch_3_Conv1x1_2')(z)
    if K.image_data_format() == 'channels_last':
        z = Reshape((*branch_3_input.shape[1:], 1), name='Branch_3_Reshape')(z)
    else:
        z = Reshape((1, *branch_3_input.shape[1:]), name='Branch_3_Reshape')(z)
    z = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME', 
                kernel_initializer='he_normal', name='Branch_3_3DConv')(z)
    z = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', 
                name='Branch_3_MaxPooling3D')(z)

    # Dense Blocks
    z = dense_block_3d(z, num_dense_blocks, name='Branch_3__conv1')
    z = transition_block_3d(z, 0.5, name='Branch_3__pool1')
    z = dense_block_3d(z, num_dense_blocks, name='Branch_3__conv2')
    z = transition_block_3d(z, 0.5, name='Branch_3__pool2')
    branch_3 = dense_block_3d(z, num_dense_blocks, name='Branch_3__conv3')


    print(f'Branch 1 output shape: {branch_1.shape}')
    print(f'Branch 2 output shape: {branch_2.shape}')
    print(f'Branch 3 output shape: {branch_3.shape}')

    # If channels_last, then (batch, row, col, channels, planes)
    # else if channels_first, then (batch, planes, channels, row, col)
    channel_axis = 3 if K.image_data_format() == 'channels_last' else 2
    plane_axis = 4 if K.image_data_format() == 'channels_last' else 1

    # Branch fusion
    fusion = Concatenate(axis=channel_axis, name='Fusion_Concatenate')([branch_1, branch_2, branch_3])
    print(f'Shape after concatenation: {fusion.shape}')
    # if K.image_data_format() == 'channels_last':
    #     fusion = Reshape(fusion.shape[1:-1], name='Fusion_Reshape')(fusion)
    # else:
    #     fusion = Reshape(fusion.shape[2:], name='Fusion_Reshape')(fusion)
    # print(f'Shape after Reshaping: {fusion.shape}')
    # fusion = Conv2D(fusion.shape[channel_axis], (1,1), strides=(1,1), padding='same',
    #                     name='Fusion_Conv1x1')(fusion)
    fusion = Conv3D(fusion.shape[plane_axis], (1,1,1), strides=(1,1,1), padding='same',
                        name='Fusion_Conv1x1x1')(fusion)
    fusion = Activation('relu', name='Fusion_ReLU')(fusion)
    #fusion = Flatten(name='Fusion_Flatten')(fusion)
    fusion = GlobalAveragePooling3D(name='Fusion_Average_Pooling')(fusion)
    fusion = Dense(units=nb_classes, kernel_initializer='he_normal', name='Fusion_Dense')(fusion)
    out = Activation('softmax', name='Fusion_Softmax')(fusion)

    print(f'Output shape: {out.shape}')

    model = Model(inputs=[branch_1_input, branch_2_input, branch_3_input], 
                  outputs=out,
                  name='3D-Densenet-Fusion')

    return model