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

def nin_conv_op(inputs, w, b):
    pass

class Linear(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, input_shape, num_layers, activation='relu'):
        super(MLPBlock, self).__init__()
        self.num_inputs = math.prod(input_shape)
        self.num_layers = num_layers
        self.linear_block = [Linear(self.num_inputs) for _ in range(self.num_layers)]
        # self.linear_1 = Linear(self.num_inputs)
        # self.linear_2 = Linear(self.num_inputs)
        # self.linear_3 = Linear(1)
        self.linear_output = Linear(1)

        # Get appropriate activation function
        self.activation = tf.keras.activations.get(activation)
        

    def call(self, inputs):
        x = tf.reshape(inputs, [self.num_inputs, -1])
        # x = self.linear_1(x)
        # x = self.activation(x)
        # x = self.linear_2(x)
        # x = self.activation(x)
        # return self.linear_3(x)
        for layer in range(self.num_layers):
            x = self.linear_block[layer](x)
        return self.linear_output(x)


def normalize_tuple(value, n, name):
    """Transforms a single integer or iterable of integers into an integer tuple.

    Args:
        value: The value to validate and convert. Could an int, or any iterable of
        ints.
        n: The size of the tuple to be returned.
        name: The name of the argument being validated, e.g. "strides" or
        "kernel_size". This is only used to format error messages.

    Returns:
        A tuple of n integers.

    Raises:
        ValueError: If something else than an int/long or iterable thereof was
        passed.
    """
    if isinstance(value, int):
        return (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError('The `' + name + '` argument must be a tuple of ' +
                            str(n) + ' integers. Received: ' + str(value))
        if len(value_tuple) != n:
            raise ValueError('The `' + name + '` argument must be a tuple of ' +
                            str(n) + ' integers. Received: ' + str(value))
        for single_value in value_tuple:
            try:
                int(single_value)
            except (ValueError, TypeError):
                raise ValueError('The `' + name + '` argument must be a tuple of ' +
                                str(n) + ' integers. Received: ' + str(value) + ' '
                                'including element ' + str(single_value) + ' of type' +
                                ' ' + str(type(single_value)))
        return value_tuple

def normalize_padding(value):
    if isinstance(value, (list, tuple)):
        return value
    padding = value.lower()
    if padding not in {'valid', 'same', 'causal'}:
        raise ValueError('The `padding` argument must be a list/tuple or one of '
                            '"valid", "same" (or "causal", only for `Conv1D). '
                            'Received: ' + str(padding))
    return padding

def normalize_data_format(value):
    if value is None:
        value = K.image_data_format()
    data_format = value.lower()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('The `data_format` argument must be one of '
                            '"channels_first", "channels_last". Received: ' +
                            str(value))
    return data_format

def convert_data_format(data_format, ndim):
    if data_format == 'channels_last':
        if ndim == 3:
            return 'NWC'
        elif ndim == 4:
            return 'NHWC'
        elif ndim == 5:
            return 'NDHWC'
        else:
            raise ValueError('Input rank not supported:', ndim)
    elif data_format == 'channels_first':
        if ndim == 3:
            return 'NCW'
        elif ndim == 4:
            return 'NCHW'
        elif ndim == 5:
            return 'NCDHW'
        else:
            raise ValueError('Input rank not supported:', ndim)
    else:
        raise ValueError('Invalid data_format:', data_format)

def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
    """Determines output length of a convolution given input length.

    Args:
        input_length: integer.
        filter_size: integer.
        padding: one of "same", "valid", "full", "causal"
        stride: integer.
        dilation: dilation rate, integer.

    Returns:
        The output length (integer).
    """
    if input_length is None:
        return None
    assert padding in {'same', 'valid', 'full', 'causal'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if padding in ['same', 'causal']:
        output_length = input_length
    elif padding == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif padding == 'full':
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride

def squeeze_batch_dims(inp, op, inner_rank):
    """Returns `unsqueeze_batch(op(squeeze_batch(inp)))`.

    Where `squeeze_batch` reshapes `inp` to shape
    `[prod(inp.shape[:-inner_rank])] + inp.shape[-inner_rank:]`
    and `unsqueeze_batch` does the reverse reshape but on the output.

    Args:
        inp: A tensor with dims `batch_shape + inner_shape` where `inner_shape`
        is length `inner_rank`.
        op: A callable that takes a single input tensor and returns a single.
        output tensor.
        inner_rank: A python integer.

    Returns:
        `unsqueeze_batch_op(squeeze_batch(inp))`.
    """
    shape = inp.shape

    inner_shape = shape[-inner_rank:]
    if not inner_shape.is_fully_defined():
        inner_shape = tf.shape(inp)[-inner_rank:]

    batch_shape = shape[:-inner_rank]
    if not batch_shape.is_fully_defined():
        batch_shape = tf.shape(inp)[:-inner_rank]

    if isinstance(inner_shape, tf.TensorShape):
        inp_reshaped = tf.reshape(inp, [-1] + inner_shape.as_list())
    else:
        inp_reshaped = tf.reshape(
            inp, tf.concat(([-1], inner_shape), axis=-1))

    out_reshaped = op(inp_reshaped)

    out_inner_shape = out_reshaped.shape[-inner_rank:]
    if not out_inner_shape.is_fully_defined():
        out_inner_shape = tf.shape(out_reshaped)[-inner_rank:]

    out = tf.reshape(
        out_reshaped, tf.concat((batch_shape, out_inner_shape), axis=-1))

    out.set_shape(inp.shape[:-inner_rank] + out.shape[-inner_rank:])
    return out

class NINConv(Layer):
    """
    """
    def __init__(self, 
                rank,
                filters,
                kernel_size,
                num_mlp_layers=2,
                strides=(1, 1),
                padding='valid',
                data_format=None,
                dilation_rate=(1, 1),
                groups=1,
                activation=None,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                trainable=True,
                name=None,
                **kwargs):

        super(NINConv, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=tf.keras.regularizers.get(activity_regularizer)
            **kwargs)
        

        self.rank = rank
        self.num_mlp_layers = num_mlp_layers

        if isinstance(filters, float):
            filters = int(filters)
        if filters is not None and filters < 0:
            raise ValueError(f'Received a negative value for `filters`.'
                            f'Was expecting a positive value, got {filters}.')
        self.filters = filters
        self.groups = groups or 1
        self.kernel_size = normalize_tuple(
            kernel_size, self.rank, 'kernel_size')
        self.strides = normalize_tuple(strides, self.rank, 'strides')
        self.padding = normalize_padding(padding)
        self.data_format = normalize_data_format(data_format)
        self.dilation_rate = normalize_tuple(
            dilation_rate, self.rank, 'dilation_rate')

        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=self.rank + 2)

        self.nin_layers = []
        for layer in range(self.num_mlp_layers):
            self.nin_layers.append(tf.keras.layers.Dense(
                units=math.prod(self.kernel_size),
                activation='relu',
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constrain=None,
                name=f'NIN_layer_{layer}',
            ))
        self.nin_output = tf.keras.layers.Dense(
            units=filters,
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constrain=None,
            name=f'NIN_output',
        )

        self._validate_init()
        self._is_causal = self.padding == 'causal'
        self._channels_first = self.data_format == 'channels_first'
        self._tf_data_format = convert_data_format(
            self.data_format, self.rank + 2)

    def _validate_init(self):
        if self.filters is not None and self.filters % self.groups != 0:
            raise ValueError(
                'The number of filters must be evenly divisible by the number of '
                'groups. Received: groups={}, filters={}'.format(
                    self.groups, self.filters))

        if not all(self.kernel_size):
            raise ValueError('The argument `kernel_size` cannot contain 0(s). '
                            'Received: %s' % (self.kernel_size,))

        if not all(self.strides):
            raise ValueError('The argument `strides` cannot contains 0(s). '
                            'Received: %s' % (self.strides,))

        if (self.padding == 'causal' and self.rank > 1):
            raise ValueError('Causal padding is only supported for `NINConv1D`.')

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        if input_channel % self.groups != 0:
            raise ValueError(
                'The number of input channels must be evenly divisible by the number '
                'of groups. Received groups={}, but the input has {} channels '
                '(full input shape is {}).'.format(self.groups, input_channel,
                                                    input_shape))
        kernel_shape = self.kernel_size + (input_channel // self.groups,
                                        self.filters)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        channel_axis = self._get_channel_axis()
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=self.rank + 2,
                                    axes={channel_axis: input_channel})

        # Convert Keras formats to TF native formats.
        if self.padding == 'causal':
            tf_padding = 'VALID'  # Causal padding handled in `call`.
        elif isinstance(self.padding, str):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding
        tf_dilations = list(self.dilation_rate)
        tf_strides = list(self.strides)

        tf_op_name = self.__class__.__name__

        # self._convolution_op = functools.partial(
        #     nn_ops.convolution_v2,
        #     strides=tf_strides,
        #     padding=tf_padding,
        #     dilations=tf_dilations,
        #     data_format=self._tf_data_format,
        #     name=tf_op_name)

        self.built = True

    def call(self, inputs):
        input_shape = inputs.shape

        if self._is_causal:  # Apply causal padding to inputs for Conv1D.
            inputs = tf.pad(inputs, self._compute_causal_padding(inputs))

        for layer in range(self.num_mlp_layers):
            x = self.nin_layers[layer](x)
        outputs = self.nin_output(x)
    
        outputs = tf.convert_to_tensor(outputs)

        if self.use_bias:
            output_rank = outputs.shape.rank
        if self.rank == 1 and self._channels_first:
            # nn.bias_add does not accept a 1D input tensor.
            bias = tf.reshape(self.bias, (1, self.filters, 1))
            outputs += bias
        else:
            # Handle multiple batch dimensions.
            if output_rank is not None and output_rank > 2 + self.rank:

                def _apply_fn(o):
                    return tf.nn.bias_add(o, self.bias, data_format=self._tf_data_format)

                outputs = squeeze_batch_dims(
                    outputs, _apply_fn, inner_rank=self.rank + 1)
            else:
                outputs = tf.nn.bias_add(
                    outputs, self.bias, data_format=self._tf_data_format)

        if not tf.executing_eagerly():
            # Infer the static output shape:
            out_shape = self.compute_output_shape(input_shape)
            outputs.set_shape(out_shape)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _spatial_output_shape(self, spatial_input_shape):
        return [
            conv_output_length(
                length,
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            for i, length in enumerate(spatial_input_shape)
        ]

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        batch_rank = len(input_shape) - self.rank - 1
        if self.data_format == 'channels_last':
            return tf.TensorShape(
                input_shape[:batch_rank]
                + self._spatial_output_shape(input_shape[batch_rank:-1])
                + [self.filters])
        else:
            return tf.TensorShape(
                input_shape[:batch_rank] + [self.filters] +
                self._spatial_output_shape(input_shape[batch_rank + 1:]))

    def get_config(self):
        config = {
            'filters':
                self.filters,
            'kernel_size':
                self.kernel_size,
            'strides':
                self.strides,
            'padding':
                self.padding,
            'data_format':
                self.data_format,
            'dilation_rate':
                self.dilation_rate,
            'groups':
                self.groups,
            'activation':
                tf.keras.activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                tf.keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer':
                tf.keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
                tf.keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                tf.keras.regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint':
                tf.keras.constraints.serialize(self.bias_constraint)
        }
        base_config = super(NINConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _compute_causal_padding(self, inputs):
        """Calculates padding for 'causal' option for 1-d conv layers."""
        left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
        if getattr(inputs.shape, 'ndims', None) is None:
            batch_rank = 1
        else:
            batch_rank = len(inputs.shape) - 2
        if self.data_format == 'channels_last':
            causal_padding = [[0, 0]] * batch_rank + [[left_pad, 0], [0, 0]]
        else:
            causal_padding = [[0, 0]] * batch_rank + [[0, 0], [left_pad, 0]]
        return causal_padding

    def _get_channel_axis(self):
        if self.data_format == 'channels_first':
            return -1 - self.rank
        else:
            return -1

    def _get_input_channel(self, input_shape):
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                            'should be defined. Found `None`.')
        return int(input_shape[channel_axis])

    def _get_padding_op(self):
        if self.padding == 'causal':
            op_padding = 'valid'
        else:
            op_padding = self.padding
        if not isinstance(op_padding, (list, tuple)):
            op_padding = op_padding.upper()
        return op_padding

# class NINConv2D(NINConv):
#     """
#     """
#     def __init__(self,
#                filters,
#                kernel_size,
#                strides=1,
#                padding='valid',
#                data_format='channels_last',
#                dilation_rate=1,
#                groups=1,
#                activation=None,
#                use_bias=True,
#                kernel_initializer='glorot_uniform',
#                bias_initializer='zeros',
#                kernel_regularizer=None,
#                bias_regularizer=None,
#                activity_regularizer=None,
#                kernel_constraint=None,
#                bias_constraint=None,
#                **kwargs):
#         super(NINConv2D, self).__init__(
#             rank=2,
#             filters=filters,
#             kernel_size=kernel_size,
#             strides=strides,
#             padding=padding,
#             data_format=data_format,
#             dilation_rate=dilation_rate,
#             groups=groups,
#             activation=tf.keras.activations.get(activation),
#             use_bias=use_bias,
#             kernel_initializer=tf.keras.initializers.get(kernel_initializer),
#             bias_initializer=tf.keras.initializers.get(bias_initializer),
#             kernel_regularizer=tf.keras.regularizers.get(kernel_regularizer),
#             bias_regularizer=tf.keras.regularizers.get(bias_regularizer),
#             activity_regularizer=tf.keras.regularizers.get(activity_regularizer),
#             kernel_constraint=tf.keras.constraints.get(kernel_constraint),
#             bias_constraint=tf.keras.constraints.get(bias_constraint),
#             **kwargs)

class MLPConvFilter(Layer):
    """
    """
    def __init__(self, input_shape, num_layers=2, 
                 activation='relu', name='MLPConvFilter', **kwargs):
        super(MLPConvFilter, self).__init__(name, trainable=True, **kwargs)
        self.num_inputs = math.prod(input_shape)
        self.num_layers = num_layers
        self.mlp_block = [tf.keras.layers.Dense(units=math.prod(self.kernel_size),
                                                activation='relu',
                                                use_bias=True,
                                                kernel_initializer='glorot_uniform',
                                                bias_initializer='zeros',
                                                kernel_regularizer=None,
                                                bias_regularizer=None,
                                                activity_regularizer=None,
                                                kernel_constraint=None,
                                                bias_constrain=None,
                                                name=f'{name}_{layer}',
                            ) for layer in range(self.num_layers)]
        self.mlp_output = tf.keras.layers.Dense(
            units=1,
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constrain=None,
            name=f'{name}_output',
        )

        # Get appropriate activation function
        self.activation = tf.keras.activations.get(activation)

    def call(self, inputs, training=False):
        # Flatten the input before feeding it into the mlp
        x = tf.reshape(inputs, [self.num_inputs, -1])

        # Loop through the mlp layers, feeding the input through each
        for layer in range(self.num_layers):
            x = self.mlp_block[layer](x, training=training)

        # Return the output of the output layer
        return self.mlp_output(x, training=training)

class NINConv2D(Conv2D):
    """
    """
    def __init__(self, filters, kernel_size, num_mlp_layers=2, name='NINConv2D', **kwargs):
        """
        """
        super(NINConv2D, self).__init__(filters, kernel_size=kernel_size, name=name, **kwargs)

        self.num_mlp_layers = num_mlp_layers

        self.filters = [MLPConvFilter(self.kernel_size, 
                                      num_mlp_layers, 
                                      name=f'{self.name}_MLPConvFilter_{filter}') 
                        for filter in range(filters)]
        
        print(f'kernel_size: {self.kernel_size}')

    def convolution_op(self, inputs, kernel):
        """
        """
        pass

    def call(self, inputs, training=False):
        """
        """

        # Determine the shape and dimensional indices
        if K.image_data_format() == 'channels_last':
            batches, height, width, channels = inputs.shape
            BATCHES_DIM = 0
            HEIGHT_DIM = 1
            WIDTH_DIM = 2
            CHANNELS_DIM = 3
            x_inputs = inputs
        else:
            batches, channels, height, width = inputs.shape
            BATCHES_DIM = 0
            CHANNELS_DIM = 1
            HEIGHT_DIM = 2
            WIDTH_DIM = 3
            x_inputs = tf.transpose(inputs, perm=[BATCHES_DIM, HEIGHT_DIM, WIDTH_DIM, CHANNELS_DIM])


def sliding_windows(input, kernel, padding='VALID'):
    """
    """
    pass


def nin_conv_layer(x, num_filters, kernel_shape, strides, activation, name):
    layer = Conv2D(num_filters, 
                   kernel_shape, 
                   strides=strides, 
                   activation=activation,
                   name=name, 
                   padding='valid',
                   kernel_regularizer=regularizers.l2(0.01))(x)

    return layer

def nin_model(img_rows, img_cols, img_channels, 
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
    x = nin_conv_layer(x, 
                       num_filters=nb_filters, 
                       kernel_shape=(3, 3), 
                       strides=(1,1), 
                       activation='relu',
                       name='nin_conv_layer_1')

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