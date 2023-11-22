import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from tensorflow.keras.regularizers import l2

import numpy as np
import warnings
import os
from functools import partial


CONV_INIT = 'he_normal'

NORM_DICT = {'bn': layers.BatchNormalization,
             'gn': layers.GroupNormalization,
             'ln': layers.LayerNormalization}

def points_nms(x):
    x_max_pool = tf.nn.max_pool2d(x, ksize=2, strides=1, padding=[[0, 0], [1, 1], [1, 1], [0, 0]])[:, :-1, :-1, :]
    x = tf.where(tf.equal(x, x_max_pool), x, 0)
    return x

def pad_with_coord(data):
    data_shape = tf.shape(data)
    batch_size, height, width = data_shape[0], data_shape[1], data_shape[2]
    x = tf.cast(tf.linspace(-1, 1, num=width), data.dtype)
    x = tf.tile(x[tf.newaxis, tf.newaxis, ..., tf.newaxis], [batch_size, height, 1, 1])
    y = tf.cast(tf.linspace(-1, 1, num=height), data.dtype)
    y = tf.tile(y[tf.newaxis, ..., tf.newaxis, tf.newaxis], [batch_size, 1, width, 1])
    data = tf.concat([data, x, y], axis=-1)
    return data


def convblock(filters_in,
              filters_out,
              nconv=1,
              activation='gelu',
              kernel_size=(3, 3),
              separable=False,
              dilation_rate=1,
              groups=1,
              group_width=32,
              name="",
              strides=1,
              normalization='gn',
              normalization_kw={'groups': 32},
              weight_decay=0,
              bias=False,
              residual=False,
              preact=True,
              **kwargs):

    """Conv block with n convolutions
    Can be used to generate residual blocks with pre or post activation, using grouped convolution"""

    input_tensor = tf.keras.Input(shape=(None, None, filters_in), name="input")

    NORM = NORM_DICT.get(normalization.lower(), NORM_DICT['gn'])
    NORM = partial(NORM, **normalization_kw)

    if groups is None:
        if input_tensor.shape[-1] % group_width != 0:
            warnings.warn("number of channels of input image not divisible by group_width !")

        groups = input_tensor.shape[-1] // group_width

    x = input_tensor

    if preact:
        x = NORM(name=name + "norm")(x)
        x = layers.Activation(activation, name=name + 'activation')(x)

    if residual:
        shortcut = x

    if separable:
        conv = layers.SeparableConv2D
    else:
        conv = partial(layers.Conv2D, groups=groups)

    for i in range(nconv):

        if nconv == 1:
            convname = name + 'conv'
        else:
            convname = name + 'conv{}'.format(i+1)

        x = conv(filters_out,
                 kernel_size,
                 padding='same',
                 use_bias=bias,
                 strides=strides,
                 dilation_rate=dilation_rate,
                 kernel_initializer=CONV_INIT,
                 kernel_regularizer=l2(weight_decay),
                 name=convname)(x)

    if residual:
        if strides > 1:
            shortcut = layers.MaxPool2D(pool_size=(strides, strides),
                                        strides=strides,
                                        name=name + 'shortcut_pool',
                                        padding='same')(shortcut)
        if filters_in != filters_out:
            shortcut = layers.Conv2D(filters_out,
                                     kernel_size=1,
                                     strides=1,
                                     kernel_initializer=CONV_INIT,
                                     kernel_regularizer=l2(weight_decay),
                                     name=name + 'shortcut_projection',
                                     use_bias=bias)(shortcut)

        x = layers.Add(name=name+"add")([shortcut, x])

    if not preact:
        x = NORM(name=name + "norm")(x)
        x = layers.Activation(activation, name=name + 'activation')(x)

    return tf.keras.Model(inputs=input_tensor, outputs=x, name=name + 'Convblock')


def squeeze_excite_block(filters_in, ratio=16, activation='relu', name='SE_Block', weight_decay=0):
    ''' Create a channel-wise squeeze-excite block
    Args:
        input: input tensor
        ratio: project to a low dim feature space of dimension input_filters//ratio
    Returns: a keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    https://github.com/titu1994/keras-squeeze-excite-network/blob/master/se.py
    '''
    input_tensor = tf.keras.Input(shape=(None, None, filters_in), name=name+"_input")

    se_shape = (1, 1, filters_in)
    se = layers.GlobalAveragePooling2D(name=name + '_se_squeeze')(input_tensor)
    se = layers.Reshape(se_shape, name=name + '_se_reshape')(se)

    se = layers.Conv2D(filters_in // ratio,
                       kernel_size=1,
                       padding='same',
                       activation=activation,
                       kernel_regularizer=l2(weight_decay),
                       kernel_initializer=CONV_INIT,
                       name=name + '_se_reduce')(se)
    se = layers.Conv2D(filters_in, 1,
                       padding='same',
                       activation='sigmoid',
                       kernel_regularizer=l2(weight_decay),
                       kernel_initializer=CONV_INIT,
                       name=name + '_se_expand')(se)

    x = layers.multiply([input_tensor, se], name=name + '_se_excite')
    return tf.keras.Model(inputs=input_tensor, outputs=x, name=name)

def bottleneckblock(filters_in,
                    filters_out,
                    activation='relu',
                    reduction=4,
                    kernel_size=(3, 3),
                    separable=False,
                    dilation_rate=1,
                    groups=1,
                    group_width=32,
                    name="",
                    strides=1,
                    normalization='gn',
                    normalization_kw={'groups': 32},
                    weight_decay=0,
                    bias=False,
                    preact=True,
                    se_block=True,
                    se_ratio=4,
                    **kwargs):
    """Bottleneck block with pre or post activation
    if groups > 1, this block can be used as base block for a ResneXt network
    if separable, a separable convolution is used instead of standard conv"""

    input_tensor = tf.keras.Input(shape=(None, None, filters_in), name=name+"_input")

    NORM = NORM_DICT.get(normalization.lower(), NORM_DICT['gn'])
    NORM = partial(NORM, **normalization_kw)

    if groups is None:
        if input_tensor.shape[-1] % group_width != 0:
            warnings.warn("number of channels of input image not divisible by group_width !")

        groups = input_tensor.shape[-1] // group_width

    if separable:
        conv = layers.SeparableConv2D
    else:
        conv = partial(layers.Conv2D, groups=groups)

    x = input_tensor

    if preact:
        x = NORM(name=name + "norm")(x)
        x = layers.Activation(activation, name=name + 'activation')(x)

    shortcut = x

    x = layers.Conv2D(min(filters_out//reduction, filters_in),
                      kernel_size=1,
                      padding='same',
                      use_bias=bias,
                      strides=1,
                      dilation_rate=dilation_rate,
                      kernel_initializer=CONV_INIT,
                      kernel_regularizer=l2(weight_decay),
                      name=name + 'reduction')(x)

    x = conv(filters_out//reduction,
             kernel_size=kernel_size,
             padding='same',
             use_bias=bias,
             strides=strides,
             dilation_rate=dilation_rate,
             kernel_initializer=CONV_INIT,
             kernel_regularizer=l2(weight_decay),
             name=name + 'conv')(x)

    if se_block:
        x = squeeze_excite_block(filters_in=filters_out//reduction, name=name, ratio=se_ratio)(x)

    x = layers.Conv2D(filters_out,
                      kernel_size=1,
                      padding='same',
                      use_bias=bias,
                      strides=1,
                      dilation_rate=dilation_rate,
                      kernel_initializer=CONV_INIT,
                      kernel_regularizer=l2(weight_decay),
                      name=name + 'expansion')(x)

    if strides > 1:
        shortcut = layers.MaxPool2D(pool_size=(strides, strides),
                                    strides=strides,
                                    name=name + 'shortcut_pool',
                                    padding='same')(shortcut)
    if filters_in != filters_out:
        shortcut = layers.Conv2D(filters_out,
                                 kernel_size=1,
                                 strides=1,
                                 kernel_initializer=CONV_INIT,
                                 kernel_regularizer=l2(weight_decay),
                                 name=name + 'shortcut_projection',
                                 use_bias=bias)(shortcut)

    x = layers.Add(name=name+"add")([shortcut, x])

    if not preact:
        x = NORM(name=name + "norm")(x)
        x = layers.Activation(activation, name=name + 'activation')(x)

    return tf.keras.Model(inputs=input_tensor, outputs=x, name=name + 'Convblock')


class LayerScale(layers.Layer):
    """Layer scale module.
    References:
      - https://arxiv.org/abs/2103.17239
    Args:
      init_values (float): Initial value for layer scale. Should be within
        [0, 1].
      projection_dim (int): Projection dimensionality.
    Returns:
      Tensor multiplied to the scale.
    """

    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.gamma = tf.Variable(
            self.init_values * tf.ones((self.projection_dim,))
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
            }
        )
        return config


class StochasticDepth(layers.Layer):
    """Stochastic Depth module.
    It performs batch-wise dropping rather than sample-wise. In libraries like
    `timm`, it's similar to `DropPath` layers that drops residual paths
    sample-wise.
    References:
      - https://github.com/rwightman/pytorch-image-models
    Args:
      drop_path_rate (float): Probability of dropping paths. Should be within
        [0, 1].
    Returns:
      Tensor either with the residual path dropped or kept.
    """

    def __init__(self, drop_path_rate, **kwargs):
        super().__init__(**kwargs)
        self.drop_path_rate = drop_path_rate

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path_rate
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"drop_path_rate": self.drop_path_rate})
        return config


def ConvNextBlock(filters_in,
                  filters_out,
                  layer_scale_init_value=1,
                  strides=1,
                  expansion=4,
                  kernel_size=7,
                  activation='gelu',
                  drop_path_rate=0,
                  weight_decay=0,
                  normalization='ln',
                  normalization_kw={'epsilon': 1e-6},
                  bias=False,
                  name='ConvNextBlock',
                  **kwargs):
    """Generic convnext block as in https://arxiv.org/pdf/2201.03545.pdf
    """

    NORM = NORM_DICT.get(normalization.lower(), NORM_DICT['gn'])
    NORM = partial(NORM, **normalization_kw)

    assert strides >= 1, "strides must be >=1"

    input_tensor = tf.keras.Input(shape=(None, None, filters_in))

    x = layers.DepthwiseConv2D(kernel_size=kernel_size,
                               strides=1,
                               kernel_initializer=CONV_INIT,
                               kernel_regularizer=l2(weight_decay),
                               padding='same',
                               use_bias=bias)(x)
    x = NORM()(x)
    x = layers.Conv2D(filters_out*expansion,
                      kernel_size=1,
                      strides=1,
                      kernel_initializer=CONV_INIT,
                      kernel_regularizer=l2(weight_decay),
                      padding='same',
                      use_bias=bias)(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv2D(filters_out,
                      kernel_size=1,
                      strides=1,
                      kernel_initializer=CONV_INIT,
                      kernel_regularizer=l2(weight_decay),
                      padding='same',
                      use_bias=bias)(x)

    if layer_scale_init_value is not None:
        x = LayerScale(
            layer_scale_init_value,
            filters_out,
            name=name + "_layer_scale")(x)

    if drop_path_rate is not None and drop_path_rate > 0:
        x = StochasticDepth(drop_path_rate, name=name + "_stochastic_depth")(x)

    x = x + input_tensor

    return tf.keras.Model(inputs=input_tensor, outputs=x, name=name)
