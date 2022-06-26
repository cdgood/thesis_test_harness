#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Machine Learning Model module

This script defines machine learning model creation functions.

Author:  Christopher Good
Version: 1.0.0

Usage: models.py

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

def get_optimizer(**hyperparams):
    """
    Returns appropriately constructed optimizer from hyperparameter
    inputs.

    Parameters
    ----------
    **hyperparams : dict
        dictionary of hyperparameter values to use to construct the
        optimizer

    Returns
    -------
    optimizer : tensorflow.keras.optimizer.Optimizer
        A keras Optimizer object constructed to hyperparam specification
    """
    
    # Get requisite hyperparameter values
    optimizer_name = hyperparams['optimizer']
    learning_rate = hyperparams['lr']
    momentum = hyperparams['momentum']
    epsilon = hyperparams['epsilon']
    initial_accumulator_value = hyperparams['initial_accumulator_value']
    beta_1 = hyperparams['beta_1']
    beta_2 = hyperparams['beta_2']
    amsgrad = hyperparams['amsgrad']
    rho = hyperparams['rho']
    centered = hyperparams['centered']
    nesterov = hyperparams['nesterov']
    learning_rate_power = hyperparams['learning_rate_power']
    l1_regularization_strength = hyperparams['l1_regularization_strength']
    l2_regularization_strength = hyperparams['l2_regularization_strength']
    l2_shrinkage_regularization_strength = hyperparams['l2_shrinkage_regularization_strength']
    beta = hyperparams['beta']

    # Set up the optimizers according to the input hyperparameters
    if optimizer_name == 'adadelta':
        if learning_rate is None: learning_rate = 0.001
        if rho is None: rho = 0.95
        if epsilon is None: epsilon = 1e-7
        optimizer = Adadelta(learning_rate=learning_rate,
                             rho=rho,
                             epsilon=epsilon)
    elif optimizer_name == 'adagrad':
        if learning_rate is None: learning_rate = 0.001
        if initial_accumulator_value is None: initial_accumulator_value = 0.1
        if epsilon is None: epsilon = 1e-7
        optimizer = Adagrad(learning_rate=learning_rate,
                            initial_accumulator_value=initial_accumulator_value,
                            epsilon=epsilon)
    elif optimizer_name == 'adam':
        if learning_rate is not None: learning_rate = 0.001
        if beta_1 is None: beta_1 = 0.9
        if beta_2 is None: beta_2 = 0.999
        if epsilon is None: epsilon = 1e-7
        if amsgrad is None: amsgrad = False
        optimizer = Adam(learning_rate=learning_rate,
                         beta_1=beta_1,
                         beta_2=beta_2,
                         epsilon=epsilon,
                         amsgrad=amsgrad)
    elif optimizer_name == 'adamax':
        if learning_rate is not None: learning_rate = 0.001
        if beta_1 is None: beta_1 = 0.9
        if beta_2 is None: beta_2 = 0.999
        if epsilon is None: epsilon = 1e-7
        optimizer = Adamax(learning_rate=learning_rate,
                           beta_1=beta_1,
                           beta_2=beta_2,
                           epsilon=epsilon)
    elif optimizer_name == 'ftrl':
        if learning_rate is not None: learning_rate = 0.001
        if learning_rate_power is None: learning_rate_power = -0.5
        if initial_accumulator_value is None: initial_accumulator_value = 0.1
        if l1_regularization_strength is None: l1_regularization_strength = 0.0
        if l2_regularization_strength is None: l2_regularization_strength = 0.0
        if l2_shrinkage_regularization_strength is None: l2_shrinkage_regularization_strength = 0.0
        if beta is None: beta = 0.0
        optimizer = Ftrl(learning_rate=learning_rate,
                         learning_rate_power=learning_rate_power,
                         initial_accumulator_value=initial_accumulator_value,
                         l1_regularization_strength=l1_regularization_strength,
                         l2_regularization_strength=l2_regularization_strength,
                         l2_shrinkage_regularization_strength=l2_shrinkage_regularization_strength,
                         beta=beta)
        
    elif optimizer_name == 'nadam':
        if learning_rate is not None: learning_rate = 0.001
        if beta_1 is None: beta_1 = 0.9
        if beta_2 is None: beta_2 = 0.999
        if epsilon is None: epsilon = 1e-7
        optimizer = Nadam(learning_rate=learning_rate,
                          beta_1=beta_1,
                          beta_2=beta_2,
                          epsilon=epsilon)
    elif optimizer_name == 'rmsprop':
        if learning_rate is not None: learning_rate = 0.001
        if rho is None: rho = 0.9
        if momentum is None: momentum = 0.0
        if epsilon is None: epsilon = 1e-7
        if centered is None: centered = False
        optimizer = RMSprop(learning_rate=learning_rate,
                            rho=rho,
                            momentum=momentum,
                            epsilon=epsilon,
                            centered=centered)
    elif optimizer_name == 'sgd':
        if learning_rate is not None: learning_rate = 0.001
        if momentum is None: momentum = 0.0
        if nesterov is None: nesterov = False
        optimizer = SGD(learning_rate=learning_rate,
                        momentum=momentum,
                        nesterov=nesterov)
    else:
        # This is the default value for the Tensorflow keras compile
        # function optimizer argument
        optimizer = 'rmsprop' 

    return optimizer

def _handle_dim_ordering():
    global CONV_DIM1
    global CONV_DIM2
    global CONV_DIM3
    global CHANNEL_AXIS
    if K.image_data_format() == 'channels_last':
        CONV_DIM1 = 1
        CONV_DIM2 = 2
        CONV_DIM3 = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        CONV_DIM1 = 2
        CONV_DIM2 = 3
        CONV_DIM3 = 4


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


# 组合模型
class Densenet3DBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs):
        print('original input shape:', input_shape)
        _handle_dim_ordering()
        if len(input_shape) != 4:
            raise Exception("Input shape should be a tuple (nb_channels, kernel_dim1, kernel_dim2, kernel_dim3)")

        print('original input shape:', input_shape)
        # orignal input shape: 1,7,7,200

        print(f'Image data format: {K.image_data_format()}')
        if K.image_data_format() == 'channels_last':
            input_shape = (input_shape[1], input_shape[2], input_shape[3], input_shape[0])
        print('change input shape:', input_shape)

        # 张量流输入
        input = Input(shape=input_shape)

        # 3D Convolution and pooling
        conv1 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME', kernel_initializer='he_normal')(
            input)
        pool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(conv1)

        # Dense Block1
        # x = dense_block_3d(pool1, 6, name='conv1')
        # x = transition_block_3d(x, 0.5, name='pool1')
        # x = dense_block_3d(x, 6, name='conv2')
        # x = transition_block_3d(x, 0.5, name='pool2')
        # x = dense_block_3d(x, 6, name='conv3')
        x = dense_block_3d(pool1, 3, name='conv1')
        x = transition_block_3d(x, 0.5, name='pool1')
        x = dense_block_3d(x, 3, name='conv2')
        x = transition_block_3d(x, 0.5, name='pool2')
        x = dense_block_3d(x, 3, name='conv3')
        print(x.shape)
        x = GlobalAveragePooling3D(name='avg_pool')(x)
        print(x.shape)
        # x = Dense(16, activation='softmax')(x)

        # 输入分类器
        # Classifier block
        dense = Dense(units=num_outputs, activation="softmax", kernel_initializer="he_normal")(x)

        model = Model(inputs=input, outputs=dense, name='3D-DenseNet')
        return model

    @staticmethod
    def build_resnet_8(input_shape, num_outputs):
        # (1,7,7,200),16
        return Densenet3DBuilder.build(input_shape, num_outputs)

def densenet_3d_model(img_rows, img_cols, img_channels, nb_classes):

    model = Densenet3DBuilder.build_resnet_8(
        (1, img_rows, img_cols, img_channels), nb_classes)

    return model
class CNN3DBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs):
        print('original input shape:', input_shape)
        _handle_dim_ordering()
        if len(input_shape) != 4:
            raise Exception("Input shape should be a tuple (nb_channels, kernel_dim1, kernel_dim2, kernel_dim3)")

        print('original input shape:', input_shape)
        # orignal input shape: 1,7,7,200

        print(f'Image data format: {K.image_data_format()}')
        if K.image_data_format() == 'channels_last':
            input_shape = (input_shape[1], input_shape[2], input_shape[3], input_shape[0])
        print('change input shape:', input_shape)

        input = Input(shape=input_shape)

        conv1 = Conv3D(filters=128, kernel_size=(3, 3, 20), strides=(1, 1, 5), padding='same',
                       kernel_regularizer=regularizers.l2(0.01))(input)
        act1 = Activation('relu')(conv1)
        pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')(act1)

        conv2 = Conv3D(filters=192, kernel_size=(2, 2, 3), strides=(1, 1, 2), padding='same',
                       kernel_regularizer=regularizers.l2(0.01))(pool1)
        act2 = Activation('relu')(conv2)
        drop1 = Dropout(0.5)(act2)
        pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')(drop1)

        conv3 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 2), padding='same',
                       kernel_regularizer=regularizers.l2(0.01))(pool2)
        act3 = Activation('relu')(conv3)
        drop2 = Dropout(0.5)(act3)

        flatten1 = Flatten()(drop2)
        fc1 = Dense(200, kernel_regularizer=regularizers.l2(0.01))(flatten1)
        act3 = Activation('relu')(fc1)

        # conv1 = Conv3D(filters=32, kernel_size=(3, 3, 20), strides=(1, 1, 5), padding='same',
        #                kernel_regularizer=regularizers.l2(0.01))(input)
        # act1 = Activation('relu')(conv1)
        # pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')(act1)

        # conv2 = Conv3D(filters=64, kernel_size=(2, 2, 3), strides=(1, 1, 2), padding='same',
        #                kernel_regularizer=regularizers.l2(0.01))(pool1)
        # act2 = Activation('relu')(conv2)
        # drop1 = Dropout(0.5)(act2)
        # pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')(drop1)

        # conv3 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 2), padding='same',
        #                kernel_regularizer=regularizers.l2(0.01))(pool2)
        # act3 = Activation('relu')(conv3)
        # drop2 = Dropout(0.5)(act3)

        # flatten1 = Flatten()(drop2)
        # fc1 = Dense(num_outputs*2, kernel_regularizer=regularizers.l2(0.01))(flatten1)
        # act3 = Activation('relu')(fc1)


        # Classifier block
        dense = Dense(units=num_outputs, activation="softmax", kernel_initializer="he_normal")(act3)

        model = Model(inputs=input, outputs=dense, name='3D-CNN')
        return model

    @staticmethod
    def build_resnet_8(input_shape, num_outputs):
        # (1,7,7,200),16
        return CNN3DBuilder.build(input_shape, num_outputs)


def cnn_3d_model(img_rows, img_cols, img_channels, nb_classes):

    model = CNN3DBuilder.build_resnet_8(
        (1, img_rows, img_cols, img_channels), nb_classes)

    return model

class CNN2DBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs):
        print('original input shape:', input_shape)
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, kernel_dim1, kernel_dim2)")

        print('original input shape:', input_shape)

        print(f'Image data format: {K.image_data_format()}')
        if K.image_data_format() == 'channels_last':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])
        print('change input shape:', input_shape)

        input = Input(shape=input_shape)

        conv1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                       kernel_regularizer=regularizers.l2(0.01))(input)
        act1 = Activation('relu')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(act1)

        conv2 = Conv2D(filters=192, kernel_size=(2, 2), strides=(1, 1), padding='same',
                       kernel_regularizer=regularizers.l2(0.01))(pool1)
        act2 = Activation('relu')(conv2)
        drop1 = Dropout(0.5)(act2)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(drop1)

        conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                       kernel_regularizer=regularizers.l2(0.01))(pool2)
        act3 = Activation('relu')(conv3)
        drop2 = Dropout(0.5)(act3)

        flatten1 = Flatten()(drop2)
        fc1 = Dense(200, kernel_regularizer=regularizers.l2(0.01))(flatten1)
        act3 = Activation('relu')(fc1)

        # Classifier block
        dense = Dense(units=num_outputs, activation="softmax", kernel_initializer="he_normal")(act3)

        model = Model(inputs=input, outputs=dense, name='2D-CNN')
        return model

    @staticmethod
    def build_resnet_8(input_shape, num_outputs):
        return CNN2DBuilder.build(input_shape, num_outputs)


def cnn_2d_model(img_rows, img_cols, img_channels, nb_classes):

    model = CNN2DBuilder.build_resnet_8(
        (img_channels, img_rows, img_cols), nb_classes)

    return model

def baseline_cnn_model(img_rows, img_cols, img_channels, 
                       patch_size, nb_filters, nb_classes):
    """
    Generates baseline CNN model for classifying HSI dataset.

    Parameters
    ----------
    img_rows : int
        Number of rows in neighborhood patch.
    img_cols : int
        Number of columns in neighborhood patch.
    img_channels : int
        Number of spectral bands.
    nb_classes : int
        Number of label categories.
    lr : float
        Learning rate for the model
    momentum : float
        Momentum value for optimizer

    Returns
    -------
    model : Model
        A keras API model of the constructed ML network.
    """

    model_input = Input(shape=(img_rows, img_cols, img_channels))
    conv_layer = Conv2D(nb_filters, (patch_size, patch_size), 
                        strides=(1, 1),name='2d_convolution_layer', padding='same',
                        kernel_regularizer=regularizers.l2(0.01))(model_input)
    activation_layer = Activation('relu', name='activation_layer')(conv_layer)
    max_pool_layer = MaxPooling2D(pool_size=(2, 2), name='2d_max_pooling_layer', padding='same')(activation_layer)
    flatten_layer = Flatten(name='flatten_layer')(max_pool_layer)
    dense_layer = Dense(units=nb_classes, name='dense_layer')(flatten_layer)
    classifier_layer = Activation('softmax', name='classifier_layer')(dense_layer)

    model = Model(model_input, classifier_layer, name='baseline_cnn_model')

    return model

def fusion_fcn_conv_block(x, branch_num, block_num):
    """
    """
    
    if x.shape[0] == 3:
        height = x.shape[0]
        width = x.shape[1]
    else:
        height = x.shape[1]
        width = x.shape[2]
    
    x = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same',
                            name=f'Branch_{branch_num}_Conv2D_{block_num}')(x)
    x = Activation('relu', name=f'Branch_{branch_num}_ReLU_{block_num}')(x)
    x = AveragePooling2D(pool_size=(2,2), padding='same',
                            name=f'Branch_{branch_num}_AveragePool2D_{block_num}')(x)
    x = Resizing(height, width, interpolation='nearest',
                            name=f'Branch_{branch_num}_Resizing_{block_num}')(x)

    return x

def fusion_fcn_model(branch_1_shape, branch_2_shape, branch_3_shape, nb_classes):

    # Initialize Inputs
    branch_1_input = Input(shape=branch_1_shape)
    branch_2_input = Input(shape=branch_2_shape)
    branch_3_input = Input(shape=branch_3_shape)

    print(f'Branch 1 shape: {branch_1_input.shape}')
    print(f'Branch 2 shape: {branch_2_input.shape}')
    print(f'Branch 3 shape: {branch_3_input.shape}')

    # Set channel axis
    channel_axis = len(branch_1_input.shape) - 1 if K.image_data_format() == 'channels_last' else 1

    # First branch
    branch_1_a = fusion_fcn_conv_block(branch_1_input, 1, 1)
    branch_1_b = fusion_fcn_conv_block(branch_1_a, 1, 2)
    branch_1_c = fusion_fcn_conv_block(branch_1_b, 1, 3)

    branch_1 = Add(name='Branch_1_Add')([branch_1_a, branch_1_b, branch_1_c])

    # Second branch
    branch_2_a = fusion_fcn_conv_block(branch_2_input, 2, 1)
    branch_2_b = fusion_fcn_conv_block(branch_2_a, 2, 2)
    branch_2_c = fusion_fcn_conv_block(branch_2_b, 2, 3)

    branch_2 = Add(name='Branch_2_Add')([branch_2_a, branch_2_b, branch_2_c])

    # Third branch
    branch_3 = branch_3_input

    # Branch fusion
    fusion = Concatenate(axis=channel_axis, name='Fusion_Concatenate')([branch_1, branch_2, branch_3])
    fusion = Conv2D(nb_classes, (1,1), strides=(1,1), padding='same',
                        name='Fusion_Conv2D')(fusion)
    fusion = Activation('relu', name='Fusion_ReLU')(fusion)
    out = Activation('softmax', name='Fusion_Softmax')(fusion)

    model = Model(inputs=[branch_1_input, branch_2_input, branch_3_input], 
                  outputs=out,
                  name='Fusion-FCN')

    return model

def fusion_fcn_v2_model(branch_1_shape, branch_2_shape, branch_3_shape, nb_classes):

    # Initialize Inputs
    branch_1_input = Input(shape=branch_1_shape)
    branch_2_input = Input(shape=branch_2_shape)
    branch_3_input = Input(shape=branch_3_shape)

    print(f'Branch 1 shape: {branch_1_input.shape}')
    print(f'Branch 2 shape: {branch_2_input.shape}')
    print(f'Branch 3 shape: {branch_3_input.shape}')

    # Set channel axis
    channel_axis = len(branch_1_input.shape) - 1 if K.image_data_format() == 'channels_last' else 1

    # First branch
    branch_1_a = fusion_fcn_conv_block(branch_1_input, 1, 1)
    branch_1_b = fusion_fcn_conv_block(branch_1_a, 1, 2)
    branch_1_c = fusion_fcn_conv_block(branch_1_b, 1, 3)

    branch_1 = Add(name='Branch_1_Add')([branch_1_a, branch_1_b, branch_1_c])

    # Second branch
    branch_2_a = fusion_fcn_conv_block(branch_2_input, 2, 1)
    branch_2_b = fusion_fcn_conv_block(branch_2_a, 2, 2)
    branch_2_c = fusion_fcn_conv_block(branch_2_b, 2, 3)

    branch_2 = Add(name='Branch_2_Add')([branch_2_a, branch_2_b, branch_2_c])

    # Third branch
    branch_3_a = fusion_fcn_conv_block(branch_3_input, 3, 1)
    branch_3_b = fusion_fcn_conv_block(branch_3_a, 3, 2)
    branch_3_c = fusion_fcn_conv_block(branch_3_b, 3, 3)

    branch_3 = Add(name='Branch_3_Add')([branch_3_a, branch_3_b, branch_3_c])

    # Branch fusion
    fusion = Concatenate(axis=channel_axis, name='Fusion_Concatenate')([branch_1, branch_2, branch_3])
    fusion = Conv2D(fusion.shape[channel_axis], (1,1), strides=(1,1), padding='same',
                        name='Fusion_Conv2D')(fusion)
    fusion = Activation('relu', name='Fusion_ReLU')(fusion)
    fusion = Flatten(name='Fusion_Flatten')(fusion)
    fusion = Dense(units=nb_classes, name='Fusion_Dense')(fusion)
    out = Activation('softmax', name='Fusion_Softmax')(fusion)

    model = Model(inputs=[branch_1_input, branch_2_input, branch_3_input], 
                  outputs=out,
                  name='Fusion-FCN-V2')

    return model

def dense_block_2d(x, blocks, name):
    """A dense block.

    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block_2d(x, 32, name=name + '_block' + str(i + 1))
    return x


def conv_block_2d(x, growth_rate, name):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1, use_bias=False,
                name=name + '_1_conv', padding='same')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3, padding='same', use_bias=False,
                name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def transition_block_2d(x, reduction, name):
    """A transition block.

    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias=False,
               name=name + '_conv', padding='same')(x)
    x = AveragePooling2D(1, strides=(2, 2), name=name + '_pool', padding='same')(x)
    return x


class Densenet2DBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs):
        print('original input shape:', input_shape)
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, kernel_dim1, kernel_dim2)")

        print('original input shape:', input_shape)

        print(f'Image data format: {K.image_data_format()}')
        if K.image_data_format() == 'channels_last':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])
        print('change input shape:', input_shape)

        input = Input(shape=input_shape)

        # 2D Convolution and pooling
        conv1 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', kernel_initializer='he_normal')(
            input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)

        # Dense Block1
        x = dense_block_2d(pool1, 6, name='conv1')
        x = transition_block_2d(x, 0.5, name='pool1')
        x = dense_block_2d(x, 6, name='conv2')
        x = transition_block_2d(x, 0.5, name='pool2')
        x = dense_block_2d(x, 6, name='conv3')
        x = GlobalAveragePooling2D(name='avg_pool')(x)

        # Classifier block
        dense = Dense(units=num_outputs, activation="softmax", kernel_initializer="he_normal")(x)

        model = Model(inputs=input, outputs=dense, name='2D-DenseNet')
        return model

    @staticmethod
    def build_resnet_8(input_shape, num_outputs):
        return Densenet2DBuilder.build(input_shape, num_outputs)

def densenet_2d_model(img_rows, img_cols, img_channels, nb_classes):

    model = Densenet2DBuilder.build_resnet_8(
        (img_channels, img_rows, img_cols), nb_classes)

    return model

def densenet_2d_multi_model(branch_1_shape, branch_2_shape, branch_3_shape, branch_4_shape, nb_classes):
    pass


def nin_block(x, filters, kernel_size, block_num, strides=(1,1), num_mlp_layers=2):
    """
    """

    for layer in range(num_mlp_layers):
        x = Conv2D(x.shape[-1], kernel_size=(1,1), strides=(1,1), 
                   name=f'MLPConv_{block_num}_layer_{layer}',
                   activation='relu', padding='valid', 
                   kernel_regularizer=regularizers.l2(0.01))(x)
    
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, 
               name=f'Conv_{block_num}', activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(0.01))(x)
    
    return x

def nin_model(img_rows, img_cols, img_channels, num_classes, num_mlp_layers=2):
    """
    """

    model_input = Input(shape=(img_rows, img_cols, img_channels))

    # Convolution block 1
    x = nin_block(model_input, img_channels, (5,5), 1, num_mlp_layers=num_mlp_layers)
    x = Dropout(0.5)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same',
                     name='Spatial_Pooling_1')(x)
    
    # Convolution block 2
    x = nin_block(x, img_channels, (3,3), 2, num_mlp_layers=num_mlp_layers)
    x = Dropout(0.5)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same',
                     name='Spatial_Pooling_1')(x)
    
    # Convolution block 3
    x = nin_block(x, num_classes, (3,3), 3, num_mlp_layers=num_mlp_layers)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same',
                     name='Spatial_Pooling_1')(x)

    # Global Average Pooling
    x = GlobalAveragePooling2D(name='Global_Average_Pooling')(x)
    x = Activation('softmax', name='Softmax_Classification')(x)

    # Model creation
    model = Model(model_input, x, name='nin_model')

    return model


def nin_band_selection_model(nb_channels, nb_classes, nb_layers=2):
    """
    """
    model_input = Input(shape=(1, 1, nb_channels))
    x = model_input

    for layer in range(nb_layers):
        x = Conv2D(nb_channels, kernel_size=(1,1), strides=(1,1), name=f'mlp_conv_{layer}',
               padding='same', kernel_regularizer=regularizers.l2(0.01))(x)

    x = GlobalAveragePooling2D(name='global_average_pooling')(x)
    x = Activation('softmax', name='softmax_classification')(x)


    model = Model(model_input, x, name='nin_band_selection_model')

    return model