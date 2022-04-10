### Built-in Imports ###
import math
import typing

### Other Library Imports ###
import numpy as np
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
    Layer,
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

def nin_band_selection_model(nb_channels, nb_classes, nb_layers=2):
    """
    """
    model_input = Input(shape=(1, 1, nb_channels))
    x = model_input

    for layer in range(nb_layers):
        x = Conv2D(nb_channels, kernel_size=(1,1), strides=(1,1), name=f'mlp_conv_{layer}',
               padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    
    x = Conv2D(nb_classes, kernel_size=(1,1), strides=(1,1), name=f'mlp_conv_{layer}',
               padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = GlobalAveragePooling2D(name='global_average_pooling')(x)
    x = Activation('softmax', name='softmax_classification')(x)


    model = Model(model_input, x, name='nin_band_selection_model')

    return model

if __name__ == '__main__':
