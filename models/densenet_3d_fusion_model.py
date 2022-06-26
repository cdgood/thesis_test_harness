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

# def densenet_3d_fusion_model(img_rows, img_cols, 
#                              img_channels_1, img_channels_2, img_channels_3, 
#                              nb_classes, num_dense_blocks=3):

#     # Note - normal num_dense_blocks is 6

#     # Initialize Inputs
#     if K.image_data_format() == 'channels_last':
#         branch_1_shape = (img_rows, img_cols, img_channels_1)
#         branch_2_shape = (img_rows, img_cols, img_channels_2)
#         branch_3_shape = (img_rows, img_cols, img_channels_3)
#     else:
#         branch_1_shape = (img_channels_1, img_rows, img_cols)
#         branch_2_shape = (img_channels_2, img_rows, img_cols)
#         branch_3_shape = (img_channels_3, img_rows, img_cols)

#     branch_1_input = Input(shape=branch_1_shape)
#     branch_2_input = Input(shape=branch_2_shape)
#     branch_3_input = Input(shape=branch_3_shape)

#     print(f'Branch 1 input shape: {branch_1_input.shape}')
#     print(f'Branch 2 input shape: {branch_2_input.shape}')
#     print(f'Branch 3 input shape: {branch_3_input.shape}')

#     ### Branch 1

#     # 3D Convolution and pooling
#     # x = Conv2D(img_channels_1, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
#     #             kernel_initializer='he_normal', name='Branch_1_Conv1x1')(branch_1_input)
#     x = branch_1_input
#     if K.image_data_format() == 'channels_last':
#         x = Reshape((*branch_1_input.shape[1:], 1), name='Branch_1_Reshape')(x)
#     else:
#         x = Reshape((1, *branch_1_input.shape[1:]), name='Branch_1_Reshape')(x)
#     x = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME',
#                 kernel_initializer='he_normal', name='Branch_1_3DConv')(x)
#     x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', 
#                 name='Branch_1_MaxPooling3D')(x)

#     # Dense Blocks
#     x = dense_block_3d(x, num_dense_blocks, name='Branch_1__conv1')
#     x = transition_block_3d(x, 0.5, name='Branch_1__pool1')
#     x = dense_block_3d(x, num_dense_blocks, name='Branch_1__conv2')
#     x = transition_block_3d(x, 0.5, name='Branch_1__pool2')
#     x = dense_block_3d(x, num_dense_blocks, name='Branch_1__conv3')
#     branch_1 = GlobalAveragePooling3D(name='Branch_1__avg_pool')(x)

#     ### Branch 2

#     # 3D Convolution and pooling
#     # y = Conv2D(img_channels_2, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
#     #             kernel_initializer='he_normal', name='Branch_2_Conv1x1')(branch_2_input)
#     y = branch_2_input
#     if K.image_data_format() == 'channels_last':
#         y = Reshape((*branch_2_input.shape[1:], 1), name='Branch_2_Reshape')(y)
#     else:
#         y = Reshape((1, *branch_2_input.shape[1:]), name='Branch_2_Reshape')(y)
#     y = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME', 
#                 kernel_initializer='he_normal', name='Branch_2_3DConv')(y)
#     y = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', 
#                 name='Branch_2_MaxPooling3D')(y)

#     # Dense Blocks
#     y = dense_block_3d(y, num_dense_blocks, name='Branch_2__conv1')
#     y = transition_block_3d(y, 0.5, name='Branch_2__pool1')
#     y = dense_block_3d(y, num_dense_blocks, name='Branch_2__conv2')
#     y = transition_block_3d(y, 0.5, name='Branch_2__pool2')
#     y = dense_block_3d(y, num_dense_blocks, name='Branch_2__conv3')
#     branch_2 = GlobalAveragePooling3D(name='Branch_2__avg_pool')(y)


#     ### Branch 3

#     # 3D Convolution and pooling
#     # z = Conv2D(img_channels_3, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
#     #             kernel_initializer='he_normal', name='Branch_3_Conv1x1')(branch_3_input)
#     z = branch_3_input
#     if K.image_data_format() == 'channels_last':
#         z = Reshape((*branch_3_input.shape[1:], 1), name='Branch_3_Reshape')(z)
#     else:
#         z = Reshape((1, *branch_3_input.shape[1:]), name='Branch_3_Reshape')(z)
#     z = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME', 
#                 kernel_initializer='he_normal', name='Branch_3_3DConv')(z)
#     z = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', 
#                 name='Branch_3_MaxPooling3D')(z)

#     # Dense Blocks
#     z = dense_block_3d(z, num_dense_blocks, name='Branch_3__conv1')
#     z = transition_block_3d(z, 0.5, name='Branch_3__pool1')
#     z = dense_block_3d(z, num_dense_blocks, name='Branch_3__conv2')
#     z = transition_block_3d(z, 0.5, name='Branch_3__pool2')
#     z = dense_block_3d(z, num_dense_blocks, name='Branch_3__conv3')
#     branch_3 = GlobalAveragePooling3D(name='Branch_3__avg_pool')(z)


#     print(f'Branch 1 output shape: {branch_1.shape}')
#     print(f'Branch 2 output shape: {branch_2.shape}')
#     print(f'Branch 3 output shape: {branch_3.shape}')

#     # Branch fusion
#     fusion = Concatenate(axis=1, name='Fusion_Concatenate')([branch_1, branch_2, branch_3])
#     print(f'Shape after concatenation: {fusion.shape}')
#     fusion = Dense(units=fusion.shape[-1], kernel_initializer='he_normal', name='Fusion_Dense_1')(fusion)
#     fusion = Activation('relu', name='Fusion_ReLU')(fusion)
#     fusion = Dense(units=nb_classes, kernel_initializer='he_normal', name='Fusion_Dense_2')(fusion)
#     out = Activation('softmax', name='Fusion_Softmax')(fusion)

#     print(f'Output shape: {out.shape}')

#     model = Model(inputs=[branch_1_input, branch_2_input, branch_3_input], 
#                   outputs=out,
#                   name='3D-Densenet-Fusion')

#     return model

def densenet_3d_fusion_model(img_rows, img_cols, img_channels_list,
                             nb_classes, num_dense_blocks=3):

    # Note - normal num_dense_blocks is 6

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

        # x = Conv2D(num_channels, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
        #             kernel_initializer='he_normal', name=f'Branch_{branch_num}_Conv1x1_1')(x)
        # x = Conv2D(num_channels, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
        #             kernel_initializer='he_normal', name=f'Branch_{branch_num}_Conv1x1_2')(x)
        if K.image_data_format() == 'channels_last':
            x = Reshape((*branch_input.shape[1:], 1), name=f'Branch_{branch_num}_Reshape')(x)
        else:
            x = Reshape((1, *branch_input.shape[1:]), name=f'Branch_{branch_num}_Reshape')(x)
        x = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                    kernel_initializer='he_normal', name=f'Branch_{branch_num}_3DConv')(x)
        x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', 
                    name=f'Branch_{branch_num}_MaxPooling3D')(x)

        # Dense Blocks
        x = dense_block_3d(x, num_dense_blocks, name=f'Branch_{branch_num}__conv1')
        x = transition_block_3d(x, 0.5, name=f'Branch_{branch_num}__pool1')
        x = dense_block_3d(x, num_dense_blocks, name=f'Branch_{branch_num}__conv2')
        x = transition_block_3d(x, 0.5, name=f'Branch_{branch_num}__pool2')
        x = dense_block_3d(x, num_dense_blocks, name=f'Branch_{branch_num}__conv3')

        x = GlobalAveragePooling3D(name=f'Branch_{branch_num}__avg_pool')(x)

        branches.append(x)

    # Print the output shape of the branches
    for index, branch in enumerate(branches):
        print(f'Branch {index+1} output shape: {branch.shape}')


    # Branch fusion
    fusion = Concatenate(axis=1, name='Fusion_Concatenate')(branches)
    print(f'Shape after concatenation: {fusion.shape}')
    fusion = Dense(units=fusion.shape[-1], kernel_initializer='he_normal', name='Fusion_Dense_1')(fusion)
    fusion = Activation('relu', name='Fusion_ReLU')(fusion)
    fusion = Dense(units=nb_classes, kernel_initializer='he_normal', name='Fusion_Dense_2')(fusion)
    out = Activation('softmax', name='Fusion_Softmax')(fusion)

    print(f'Fusion output shape: {out.shape}')

    model = Model(inputs=branch_inputs, 
                  outputs=out,
                  name='3D-Densenet-Fusion')

    return model

def densenet_3d_fusion_model2(img_rows, img_cols, img_channels_list,
                             nb_classes, num_dense_blocks=3):

    # Note - normal num_dense_blocks is 6

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

        x = Conv2D(num_channels, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                    kernel_initializer='he_normal', name=f'Branch_{branch_num}_Conv1x1_1')(x)
        if K.image_data_format() == 'channels_last':
            x = Reshape((*branch_input.shape[1:], 1), name=f'Branch_{branch_num}_Reshape')(x)
        else:
            x = Reshape((1, *branch_input.shape[1:]), name=f'Branch_{branch_num}_Reshape')(x)
        x = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                    kernel_initializer='he_normal', name=f'Branch_{branch_num}_3DConv')(x)
        x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', 
                    name=f'Branch_{branch_num}_MaxPooling3D')(x)

        # Dense Blocks
        x = dense_block_3d(x, num_dense_blocks, name=f'Branch_{branch_num}__conv1')
        x = transition_block_3d(x, 0.5, name=f'Branch_{branch_num}__pool1')
        x = dense_block_3d(x, num_dense_blocks, name=f'Branch_{branch_num}__conv2')
        x = transition_block_3d(x, 0.5, name=f'Branch_{branch_num}__pool2')
        x = dense_block_3d(x, num_dense_blocks, name=f'Branch_{branch_num}__conv3')

        x = GlobalAveragePooling3D(name=f'Branch_{branch_num}__avg_pool')(x)

        branches.append(x)

    # Print the output shape of the branches
    for index, branch in enumerate(branches):
        print(f'Branch {index+1} output shape: {branch.shape}')


    # Branch fusion
    fusion = Concatenate(axis=1, name='Fusion_Concatenate')(branches)
    print(f'Shape after concatenation: {fusion.shape}')
    fusion = Dense(units=fusion.shape[-1], kernel_initializer='he_normal', name='Fusion_Dense_1')(fusion)
    fusion = Activation('relu', name='Fusion_ReLU')(fusion)
    fusion = Dense(units=nb_classes, kernel_initializer='he_normal', name='Fusion_Dense_2')(fusion)
    out = Activation('softmax', name='Fusion_Softmax')(fusion)

    print(f'Fusion output shape: {out.shape}')

    model = Model(inputs=branch_inputs, 
                  outputs=out,
                  name='3D-Densenet-Fusion')

    return model

def densenet_3d_fusion_model3(img_rows, img_cols, img_channels_list,
                             nb_classes, num_dense_blocks=3):

    # Note - normal num_dense_blocks is 6

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

        x = Conv2D(num_channels, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                    kernel_initializer='he_normal', name=f'Branch_{branch_num}_Conv1x1_1')(x)
        x = Conv2D(num_channels, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
                    kernel_initializer='he_normal', name=f'Branch_{branch_num}_Conv1x1_2')(x)
        if K.image_data_format() == 'channels_last':
            x = Reshape((*branch_input.shape[1:], 1), name=f'Branch_{branch_num}_Reshape')(x)
        else:
            x = Reshape((1, *branch_input.shape[1:]), name=f'Branch_{branch_num}_Reshape')(x)
        x = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                    kernel_initializer='he_normal', name=f'Branch_{branch_num}_3DConv')(x)
        x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same', 
                    name=f'Branch_{branch_num}_MaxPooling3D')(x)

        # Dense Blocks
        x = dense_block_3d(x, num_dense_blocks, name=f'Branch_{branch_num}__conv1')
        x = transition_block_3d(x, 0.5, name=f'Branch_{branch_num}__pool1')
        x = dense_block_3d(x, num_dense_blocks, name=f'Branch_{branch_num}__conv2')
        x = transition_block_3d(x, 0.5, name=f'Branch_{branch_num}__pool2')
        x = dense_block_3d(x, num_dense_blocks, name=f'Branch_{branch_num}__conv3')

        x = GlobalAveragePooling3D(name=f'Branch_{branch_num}__avg_pool')(x)

        branches.append(x)

    # Print the output shape of the branches
    for index, branch in enumerate(branches):
        print(f'Branch {index+1} output shape: {branch.shape}')


    # Branch fusion
    fusion = Concatenate(axis=1, name='Fusion_Concatenate')(branches)
    print(f'Shape after concatenation: {fusion.shape}')
    fusion = Dense(units=fusion.shape[-1], kernel_initializer='he_normal', name='Fusion_Dense_1')(fusion)
    fusion = Activation('relu', name='Fusion_ReLU')(fusion)
    fusion = Dense(units=nb_classes, kernel_initializer='he_normal', name='Fusion_Dense_2')(fusion)
    out = Activation('softmax', name='Fusion_Softmax')(fusion)

    print(f'Fusion output shape: {out.shape}')

    model = Model(inputs=branch_inputs, 
                  outputs=out,
                  name='3D-Densenet-Fusion')

    return model