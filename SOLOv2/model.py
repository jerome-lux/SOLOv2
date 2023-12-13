# coding: utf-8

import tensorflow as tf
from tensorflow.keras.regularizers import l2

import numpy as np
import os
from functools import partial

from . import utils
from .backbone import build_backbone
from .layers import convblock, pad_with_coord, points_nms
from .loss import compute_image_loss
from .metrics import BinaryRecall, BinaryPrecision

import tensorflow.keras.layers as layers

CONV_INIT = 'he_normal'

NORM_DICT = {
    'bn': layers.BatchNormalization,
    'gn': layers.GroupNormalization,
    'ln': layers.LayerNormalization
}


def SOLOv2_head(ncls,
                filters_in,
                conv_filters=256,
                kernel_filters=256,
                head_layers=4,
                activation='gelu',
                name="SOLO_head",
                normalization='gn',
                normalization_kw={'groups': 32},
                block_params=None,
                **kwargs):

    # class and kernel shared heads

    NORM = NORM_DICT.get(normalization.lower(), NORM_DICT['gn'])
    NORM = partial(NORM, **normalization_kw)

    if name is None:
        name = "SOLO_head"

    input_tensor = tf.keras.Input(shape=(None, None, filters_in),
                                  name=name + "_input")

    # ct_head = input_tensor
    # No preactivation if groupnorm, because depth of input feature map with coord is not a power of 2
    if block_params is None:
        block_params = {
            'preact': False,
            'groups': 1,
            'activation': activation,
            'normalization': normalization,
            'normalization_kw': normalization_kw
        }

    class_head = input_tensor[..., :
                              -2]  # no need for coordinates in class head
    kernel_head = input_tensor

    for i in range(head_layers):

        if i == 0:
            conv_filters_in = filters_in - 2
        else:
            conv_filters_in = conv_filters

        class_head = convblock(filters_in=conv_filters_in,
                               filters_out=conv_filters,
                               kernel_initializer=CONV_INIT,
                               name=name + "_class_{}_".format(i + 1),
                               **block_params)(class_head)

    class_head = NORM(name=name + "_class_final_norm")(class_head)
    class_head = layers.Activation(activation,
                                   name=name + "_class_final_act")(class_head)

    class_head = layers.Conv2D(filters=ncls,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               name=name + "_class_logits",
                               kernel_initializer=CONV_INIT)(class_head)

    for i in range(head_layers):

        if i == 0:
            conv_filters_in = filters_in
        else:
            conv_filters_in = conv_filters

        kernel_head = convblock(filters_in=conv_filters_in,
                                filters_out=conv_filters,
                                name=name + "_kernel_{}_".format(i + 1),
                                **block_params)(kernel_head)

    kernel_head = NORM(name=name + "_kernel_final_norm")(kernel_head)
    kernel_head = layers.Activation(activation,
                                    name=name +
                                    "_kernel_final_act")(kernel_head)
    kernel_head = layers.Conv2D(filters=kernel_filters,
                                kernel_size=3,
                                strides=1,
                                padding='same',
                                name=name + "_pred_kernels",
                                kernel_initializer=CONV_INIT)(kernel_head)

    return tf.keras.Model(inputs=input_tensor,
                          outputs=[class_head, kernel_head],
                          name=name)

def SOLOv2_mask_head(fpn_features,
                     in_ch=258,
                     mid_ch=128,
                     out_ch=256,
                     activation='gelu',
                     normalization='gn',
                     normalization_kw={'groups': 32}):

    # each FPN level is upscaled to size H/2 x W/2, where H and W are the dims of the input image.
    # TODO: add an option for upscaling only to H/4 x W/4 (like in the original SOLOv2 implementation) to limit memory consumption...
    # Then all levels are added and conv

    outputs = []
    for level, fpn_feature in enumerate(fpn_features):

        if level == len(fpn_features) - 1:
            feature = pad_with_coord(fpn_feature)
            channels = in_ch + 2
        else:
            feature = fpn_feature
            channels = in_ch

        for i in range(level + 1):
            if i > 0:
                feature = layers.UpSampling2D(
                    interpolation="bilinear")(feature)
                channels = mid_ch

            feature = convblock(filters_in=channels,
                                filters_out=mid_ch,
                                activation=activation,
                                normalization=normalization,
                                normalization_kw=normalization_kw,
                                preact=False,
                                name='mask_head_P{}{}'.format(
                                    level + 2, i + 1))(feature)

        outputs.append(feature)

    seg_outputs = layers.Add()(outputs)
    seg_outputs = convblock(filters_in=mid_ch,
                            filters_out=out_ch,
                            kernel_size=1,
                            activation=activation,
                            normalization=normalization,
                            normalization_kw=normalization_kw,
                            preact=False,
                            name='mask_head_output')(seg_outputs)

    return seg_outputs


def SOLOv2(config):
    """Create the SOLOv2 model using the given config object
    """

    if config.load_backbone:
        backbone = tf.keras.models.load_model(config.backbone)
    else:
        backbone = build_backbone(config.backbone,
                                  input_shape=config.imshape,
                                  classes=config.ncls,
                                  normalization=config.normalization,
                                  normalization_kw=config.normalization_kw)

    FPN_inputs = {}
    # use get_output_at(0) instead of .output to avoid Graph disconnected error...
    for lvl, layer in config.connection_layers.items():
        FPN_inputs[lvl] = backbone.get_layer(layer).get_output_at(0)

    fpn_features = FPN(FPN_inputs,
                       pyramid_filters=config.FPN_filters,
                       extra_layers=config.extra_FPN_layers)

    head_model = SOLOv2_head(ncls=config.ncls,
                             filters_in=config.FPN_filters + 2,
                             head_layers=config.head_layers,
                             conv_filters=config.head_filters,
                             kernel_filters=config.mask_output_filters *
                             config.kernel_size**2,
                             activation=config.activation,
                             normalization=config.normalization,
                             normalization_kw=config.normalization_kw)

    maxdim = max(config.imshape)
    H = config.imshape[0]
    W = config.imshape[1]

    feat_kernel_list, feat_cls_list = [], []
    for level, (feature,
                grid_size) in enumerate(zip(fpn_features, config.grid_sizes)):
        if level == 0:
            # add maxpool for P2 as in fastetimator implementation https://github.com/fastestimator
            feature = layers.MaxPool2D()(feature)
        feature = pad_with_coord(feature)
        feature = tf.image.resize(feature,
                                  size=(grid_size * H // maxdim,
                                        grid_size * W // maxdim))
        feat_cls, feat_kernel = head_model(feature)
        feat_kernel_list.append(feat_kernel)
        feat_cls_list.append(feat_cls)

    # to build the mask output, we resize and add all FPN levels, except the extra levels > P5
    mask_output = SOLOv2_mask_head(fpn_features[:-config.extra_FPN_layers],
                                   in_ch=config.FPN_filters,
                                   mid_ch=config.mask_mid_filters,
                                   out_ch=config.mask_output_filters,
                                   activation=config.activation,
                                   normalization=config.normalization,
                                   normalization_kw=config.normalization_kw)

    model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[mask_output, feat_cls_list, feat_kernel_list],
        name=config.model_name)
    return model


def FPN(connection_layers,
        pyramid_filters=256,
        name="",
        extra_layers=0,
        interpolation="bilinear",
        weight_decay=0):

    if name is None:
        name = ""
    """Feature Pyramid Network using Group Normalization
    connection_layers: dict{"C2":layernameX,"C3":layernameY... to C5} dict of keras layers
    if C2 is not provided, P2 layer is not built.
    activation: activation function default 'relu'
    name: base name of the fpn
    interpolation: type of interpolation when upscaling feature maps (default: "bilinear")
    """
    try:
        C5 = connection_layers["C5"]
        C4 = connection_layers["C4"]
        C3 = connection_layers["C3"]

    except Exception as e:
        print(
            "Error when building FPN: can't get backbone network connection layers"
        )
        print(e, e.args)

    C2 = connection_layers.get("C2", None)

    outputs = []

    P5 = layers.Conv2D(pyramid_filters,
                       kernel_size=1,
                       padding='same',
                       use_bias=True,
                       strides=1,
                       kernel_initializer=CONV_INIT,
                       kernel_regularizer=l2(weight_decay),
                       name=name + "FPN_C5P5_")(C5)

    P5up = layers.UpSampling2D(size=(2, 2),
                               interpolation=interpolation,
                               name=name + "FPN_P5upsampled")(P5)

    C4P4 = layers.Conv2D(pyramid_filters,
                         kernel_size=1,
                         padding='same',
                         use_bias=True,
                         strides=1,
                         kernel_initializer=CONV_INIT,
                         kernel_regularizer=l2(weight_decay),
                         name=name + "FPN_C4P4_")(C4)

    P4 = layers.Add(name=name + "FPN_P4add")([P5up, C4P4])

    P4up = layers.UpSampling2D(size=(2, 2),
                               interpolation=interpolation,
                               name=name + "FPN_P4upsampled")(P4)

    C3P3 = layers.Conv2D(pyramid_filters,
                         kernel_size=1,
                         padding='same',
                         use_bias=True,
                         strides=1,
                         kernel_initializer=CONV_INIT,
                         kernel_regularizer=l2(weight_decay),
                         name=name + "FPN_C3P3_")(C3)
    P3 = layers.Add(name=name + "FPN_P3add")([P4up, C3P3])

    if C2 is not None:
        P3up = layers.UpSampling2D(size=(2, 2),
                                   interpolation=interpolation,
                                   name=name + "FPN_P3upsampled")(P3)
        P3up = layers.Conv2D(pyramid_filters,
                             kernel_size=1,
                             padding='same',
                             use_bias=True,
                             strides=1,
                             kernel_initializer=CONV_INIT,
                             kernel_regularizer=l2(weight_decay),
                             name=name + "FPN_P3upconv_")(P3up)
        C2P2 = layers.Conv2D(pyramid_filters,
                             kernel_size=1,
                             padding='same',
                             use_bias=True,
                             strides=1,
                             kernel_initializer=CONV_INIT,
                             kernel_regularizer=l2(weight_decay),
                             name=name + "FPN_C2P2_")(C2)
        P2 = layers.Add(name=name + "FPN_P2add")([P3up, C2P2])

        outputs.append(
            layers.Conv2D(pyramid_filters,
                          kernel_size=3,
                          padding='same',
                          use_bias=True,
                          strides=1,
                          kernel_initializer=CONV_INIT,
                          kernel_regularizer=l2(weight_decay),
                          name=name + "FPN_P2_")(P2))

    outputs.append(
        layers.Conv2D(pyramid_filters,
                      kernel_size=3,
                      padding='same',
                      use_bias=True,
                      strides=1,
                      kernel_initializer=CONV_INIT,
                      kernel_regularizer=l2(weight_decay),
                      name=name + "FPN_P3_")(P3))

    outputs.append(
        layers.Conv2D(pyramid_filters,
                      kernel_size=3,
                      padding='same',
                      use_bias=True,
                      strides=1,
                      kernel_initializer=CONV_INIT,
                      kernel_regularizer=l2(weight_decay),
                      name=name + "FPN_P4_")(P4))

    outputs.append(
        layers.Conv2D(pyramid_filters,
                      kernel_size=3,
                      padding='same',
                      use_bias=True,
                      strides=1,
                      kernel_initializer=CONV_INIT,
                      kernel_regularizer=l2(weight_decay),
                      name=name + "FPN_P5_")(P5))

    for i in range(extra_layers):
        x = layers.Conv2D(pyramid_filters,
                          kernel_size=3,
                          padding='same',
                          use_bias=True,
                          strides=1,
                          kernel_initializer=CONV_INIT,
                          kernel_regularizer=l2(weight_decay),
                          name=name + "FPN_P{}_".format(i + 6))(outputs[-1])
        outputs.append(x)

    return outputs


def FPNv2(connection_layers,
          pyramid_filters=256,
          activation='gelu',
          name="",
          extra_layers=0,
          interpolation="bilinear",
          normalization='gn',
          normalization_kw={'groups': 32}):

    if name is None:
        name = ""
    """Feature Pyramid Network using Group Normalization
    connection_layers: dict{"C2":layernameX,"C3":layernameY... to C5} dict of keras layers
    if C2 is not provided, P2 layer is not built.
    activation: activation function default 'relu'
    name: base name of the fpn
    interpolation: type of interpolation when upscaling feature maps (default: "bilinear")
    A difference with classic FPN is that we add activation and normalization layer
    """
    try:
        C5 = connection_layers["C5"]
        C4 = connection_layers["C4"]
        C3 = connection_layers["C3"]

    except Exception as e:
        print(
            "Error when building FPN: can't get backbone network connection layers"
        )
        print(e, e.args)
        raise

    input_shapes = [C5.shape[-1], C4.shape[-1], C3.shape[-1]]
    C2 = connection_layers.get("C2", None)
    if C2 is not None:
        input_shapes.append(C2.shape[-1])
    outputs = []

    # tf.print(input_shapes)

    # We must first add Normalization + activation for models using preactivation

    P5 = convblock(filters_in=input_shapes[0],
                   filters_out=pyramid_filters,
                   kernel_size=(1, 1),
                   preact=False,
                   activation=activation,
                   name=name + "FPN_C5P5_",
                   normalization=normalization,
                   normalization_kw=normalization_kw)(C5)

    P5up = layers.UpSampling2D(size=(2, 2),
                               interpolation=interpolation,
                               name=name + "FPN_P5upsampled")(P5)

    C4P4 = convblock(filters_in=input_shapes[1],
                     filters_out=pyramid_filters,
                     kernel_size=(1, 1),
                     preact=False,
                     activation=activation,
                     name=name + "FPN_C4P4_",
                     normalization=normalization,
                     normalization_kw=normalization_kw)(C4)

    P4 = layers.Add(name=name + "FPN_P4add")([P5up, C4P4])

    P4up = layers.UpSampling2D(size=(2, 2),
                               interpolation=interpolation,
                               name=name + "FPN_P4upsampled")(P4)
    C3P3 = convblock(filters_in=input_shapes[2],
                     filters_out=pyramid_filters,
                     kernel_size=(1, 1),
                     preact=False,
                     activation=activation,
                     name=name + "FPN_C3P3_",
                     normalization=normalization,
                     normalization_kw=normalization_kw)(C3)
    P3 = layers.Add(name=name + "FPN_P3add")([P4up, C3P3])

    if C2 is not None:
        P3up = layers.UpSampling2D(size=(2, 2),
                                   interpolation=interpolation,
                                   name=name + "FPN_P3upsampled")(P3)
        P3up = convblock(filters_in=pyramid_filters,
                         filters_out=pyramid_filters,
                         kernel_size=(3, 3),
                         preact=False,
                         activation=activation,
                         name=name + "FPN_P3upconv_",
                         normalization=normalization,
                         normalization_kw=normalization_kw)(P3up)
        C2P2 = convblock(filters_in=input_shapes[3],
                         filters_out=pyramid_filters,
                         kernel_size=1,
                         preact=False,
                         activation=activation,
                         name=name + "FPN_C2P2_",
                         normalization=normalization,
                         normalization_kw=normalization_kw)(C2)
        P2 = layers.Add(name=name + "FPN_P2add")([P3up, C2P2])

        outputs.append(
            convblock(filters_in=pyramid_filters,
                      filters_out=pyramid_filters,
                      kernel_size=3,
                      preact=False,
                      activation=activation,
                      name=name + "FPN_P2_",
                      normalization=normalization,
                      normalization_kw=normalization_kw,
                      attention_kernel=7)(P2))

    outputs.append(
        convblock(filters_in=pyramid_filters,
                  filters_out=pyramid_filters,
                  kernel_size=3,
                  preact=False,
                  activation=activation,
                  name=name + "FPN_P3_",
                  normalization=normalization,
                  normalization_kw=normalization_kw)(P3))

    outputs.append(
        convblock(filters_in=pyramid_filters,
                  filters_out=pyramid_filters,
                  kernel_size=3,
                  preact=False,
                  activation=activation,
                  name=name + "FPN_P4_",
                  normalization=normalization,
                  normalization_kw=normalization_kw)(P4))

    outputs.append(
        convblock(filters_in=pyramid_filters,
                  filters_out=pyramid_filters,
                  kernel_size=3,
                  preact=False,
                  activation=activation,
                  name=name + "FPN_P5_",
                  normalization=normalization,
                  normalization_kw=normalization_kw)(P5))

    for i in range(extra_layers):
        x = convblock(filters_in=pyramid_filters,
                      filters_out=pyramid_filters,
                      kernel_size=3,
                      preact=False,
                      activation=activation,
                      strides=2,
                      name=name + "FPN_P{}_".format(i + 6),
                      normalization=normalization,
                      normalization_kw=normalization_kw)(outputs[-1])
        outputs.append(x)

    return outputs


@tf.function()
def flatten_predictions(predictions, ncls=1, kernel_depth=256):
    """Flatten and concat solo head predictions for each FPN level
    predictions: [pred_cls_list, pred_kernel_list] each list contains predictions by level
    TODO: add strides tensor as output to filter binary masks by size
    """

    num_lvl = len(predictions[0])

    flat_pred_cls = [[]] * num_lvl
    flat_pred_kernel = [[]] * num_lvl

    # Extract and flatten the i-th predicted tensors
    for lvl in range(num_lvl):
        x, y = tf.shape(predictions[0][lvl])[0], tf.shape(
            predictions[0][lvl])[1]
        # tf.print(tf.shape(predictions[0][lvl]), x*y)
        flat_pred_cls[lvl] = tf.reshape(predictions[0][lvl], [x * y, ncls])
        flat_pred_kernel[lvl] = tf.reshape(predictions[1][lvl],
                                           [x * y, kernel_depth])

    # Concat predictions -> one big vector for all scales
    flat_pred_cls = tf.concat(flat_pred_cls, 0)
    flat_pred_kernel = tf.concat(flat_pred_kernel, 0)

    return flat_pred_cls, flat_pred_kernel


@tf.function
def compute_one_image_masks(inputs,
                            cls_threshold=0.5,
                            mask_threshold=0.5,
                            nms_threshold=0.5,
                            kernel_size=1,
                            kernel_depth=256,
                            max_inst=400):
    """given predicted class results and predicted kernels, compute the output one encoded mask tensor
        How to implement mask size filtering by strides whereas all inputs are flattened?
    """

    flat_pred_cls = inputs[0]
    flat_pred_kernel = inputs[1]
    masks_head_output = inputs[2]

    # Only one prediction by pixel
    cls_labels = tf.argmax(flat_pred_cls, axis=-1)
    flat_pred_cls = tf.reduce_max(flat_pred_cls, axis=-1)
    # cls_preds = tf.sigmoid(flat_pred_cls)
    positive_idx = tf.where(flat_pred_cls >= cls_threshold)

    cls_scores = tf.gather_nd(flat_pred_cls, positive_idx)
    cls_labels = tf.gather_nd(cls_labels, positive_idx)

    kernel_preds = tf.gather(
        flat_pred_kernel,
        positive_idx[:, 0])  # shape [N, kernel_depth*kernel_size**2]
    kernel_preds = tf.transpose(
        kernel_preds)  # [kernel_depth*kernel_size**2, N]
    kernel_preds = tf.reshape(
        kernel_preds, (kernel_size, kernel_size, kernel_depth, -1)
    )  # SHAPE [ks, ks, cin, cout] where cin=kernel_depth, cout=N isntances

    seg_preds = tf.sigmoid(
        tf.nn.conv2d(masks_head_output[tf.newaxis, ...],
                     kernel_preds,
                     strides=1,
                     padding="SAME"))[0]  # results is shape [H, W, ninstances]
    seg_preds = tf.transpose(seg_preds,
                             perm=[2, 0, 1])  # reshape to [ninstances, H, W]
    binary_masks = tf.where(seg_preds >= mask_threshold, 1., 0.)  # [N, H, W]

    mask_sum = tf.reduce_sum(
        binary_masks,
        axis=[1, 2])  # area of each instance (one instance per slice) -> [N]

    # scale the category score by mask confidence
    mask_scores = tf.math.divide_no_nan(
        tf.reduce_sum(seg_preds * binary_masks, axis=[1, 2]), mask_sum)  # [N]

    scores = cls_scores * mask_scores  # [N]

    seg_preds, scores, cls_labels = matrix_nms(cls_labels,
                                               scores,
                                               seg_preds,
                                               binary_masks,
                                               mask_sum,
                                               post_nms_k=max_inst,
                                               score_threshold=nms_threshold)
    # seg_preds = tf.RaggedTensor.from_tensor(tf.reshape(seg_preds,[-1,tf.shape(seg_preds)[1]*tf.shape(seg_preds)[2]]))
    seg_preds = tf.RaggedTensor.from_tensor(seg_preds)
    # scores = tf.RaggedTensor.from_tensor(scores)
    # cls_labels = tf.RaggedTensor.from_tensor(cls_labels)

    return seg_preds, scores, cls_labels


@tf.function
def matrix_nms(cls_labels,
               scores,
               seg_preds,
               binary_masks,
               mask_sum,
               sigma=0.5,
               pre_nms_k=800,
               post_nms_k=300,
               score_threshold=0.5):

    # Select only first pre_nms_k instances (sorted by scores)
    num_selected = tf.minimum(pre_nms_k, tf.shape(scores)[0])
    indices = tf.argsort(scores, direction='DESCENDING')[:num_selected]
    # keep the selected masks, scores and labels (and mask areas)
    seg_preds = tf.gather(seg_preds, indices)
    seg_masks = tf.gather(binary_masks, indices)
    cls_labels, scores = tf.gather(cls_labels,
                                   indices), tf.gather(scores, indices)
    mask_sum = tf.gather(mask_sum, indices)  # [N]

    # calculate iou between different masks
    seg_masks = tf.reshape(seg_masks, shape=(num_selected, -1))  # [N, H*W]
    intersection = tf.matmul(seg_masks, seg_masks, transpose_b=True)  # [N, N]
    mask_sum = tf.tile(mask_sum[tf.newaxis, ...], multiples=[num_selected,
                                                             1])  # [N,N]
    union = mask_sum + tf.transpose(mask_sum) - intersection
    iou = tf.math.divide_no_nan(intersection, union)
    iou = tf.linalg.band_part(iou, 0, -1) - tf.linalg.band_part(
        iou, 0, 0)  # equivalent of np.triu(diagonal=1)

    # iou decay and compensation
    labels_match = tf.tile(cls_labels[tf.newaxis, ...],
                           multiples=[num_selected, 1])
    labels_match = tf.where(labels_match == tf.transpose(labels_match), 1.0,
                            0.0)
    labels_match = tf.linalg.band_part(
        labels_match, 0, -1) - tf.linalg.band_part(labels_match, 0, 0)

    decay_iou = iou * labels_match  # iou with any object from same class
    compensate_iou = tf.reduce_max(decay_iou, axis=0)
    compensate_iou = tf.tile(compensate_iou[..., tf.newaxis],
                             multiples=[1, num_selected])
    # matrix nms
    inv_sigma = 1. / sigma
    decay_coefficient = tf.reduce_min(tf.exp(
        -inv_sigma * (decay_iou**2 - compensate_iou**2)),
                                      axis=0)

    scores = scores * decay_coefficient
    # scores = tf.where(scores >= score_threshold, scores, 0)
    indices = tf.where(scores >= score_threshold)
    scores = tf.gather_nd(scores, indices)
    seg_preds = tf.gather(seg_preds, tf.reshape(indices, [-1]))

    num_selected = tf.minimum(post_nms_k, tf.shape(scores)[0])
    # select the final predictions
    sorted_indices = tf.argsort(scores, direction='DESCENDING')[:num_selected]
    scores = tf.gather(scores, tf.reshape(sorted_indices, [-1]))
    cls_labels = tf.gather(cls_labels, tf.reshape(sorted_indices, [-1]))

    return seg_preds, scores, cls_labels


@tf.function
def compute_masks(flat_cls_pred,
                  flat_kernel_pred,
                  mask_features,
                  cls_threshold=0.5,
                  mask_threshold=0.5,
                  nms_threshold=0.5,
                  kernel_depth=256,
                  kernel_size=1,
                  max_inst=400):
    """Compute mask
    inputs:
        list of predicted class features form SOLO head
        list of predicted kernel features  from SOLO head
        feature maps from mask_head
        ...

    """
    prediction_function = partial(compute_one_image_masks,
                                  cls_threshold=cls_threshold,
                                  mask_threshold=mask_threshold,
                                  nms_threshold=nms_threshold,
                                  kernel_size=kernel_size,
                                  kernel_depth=kernel_depth,
                                  max_inst=max_inst)

    seg_preds, scores, cls_labels = tf.map_fn(
        prediction_function, [flat_cls_pred, flat_kernel_pred, mask_features],
        fn_output_signature=(tf.RaggedTensorSpec(shape=(None, None, None),
                                                 dtype=tf.float32,
                                                 ragged_rank=1),
                             tf.RaggedTensorSpec(shape=(None),
                                                 dtype=tf.float32,
                                                 ragged_rank=0),
                             tf.RaggedTensorSpec(shape=(None),
                                                 dtype=tf.int64,
                                                 ragged_rank=0)))
    # return predicted segmentation masks as [B, N , H, W] ragged tensor
    return seg_preds, scores, cls_labels


class SOLOv2Model(tf.keras.Model):

    def __init__(self, config, **kwargs):

        super(SOLOv2Model, self).__init__(**kwargs)
        self.config = config
        self.model = SOLOv2(config)
        self.kernel_depth = config.mask_output_filters * config.kernel_size**2
        self.ncls = config.ncls
        self.strides = config.strides

        # losses
        self.seg_loss = tf.keras.metrics.Mean(name="seg_loss",
                                              dtype=tf.float32)
        self.cls_loss = tf.keras.metrics.Mean(name="cls_loss",
                                              dtype=tf.float32)
        self.total_loss = tf.keras.metrics.Mean(name="total_loss",
                                                dtype=tf.float32)

        # Metrics
        self.precision = BinaryPrecision(name="precision")
        self.recall = BinaryRecall(name="recall")

    @property
    def metrics(self):

        return [
            self.precision, self.recall, self.cls_loss, self.seg_loss,
            self.total_loss
        ]

    def call(self, inputs, training=False, **kwargs):

        default_kwargs = {
            "score_threshold": 0.5,
            "seg_threshold": 0.5,
            "nms_threshold": 0.5,
            "max_detections": 400
        }

        default_kwargs.update(kwargs)

        mask_head_output, feat_cls_list, feat_kernel_list = self.model(
            inputs, training=training)

        # Apply Points NMS in inference
        for lvl in range(len(feat_cls_list)):
            feat_cls_list[lvl] = points_nms(tf.sigmoid(feat_cls_list[lvl]))

        flatten_predictions_func = partial(flatten_predictions,
                                           ncls=self.ncls,
                                           kernel_depth=self.kernel_depth)
        flat_cls_pred, flat_kernel_pred = tf.map_fn(
            flatten_predictions_func, [feat_cls_list, feat_kernel_list],
            fn_output_signature=(tf.float32, tf.float32))

        seg_preds, scores, cls_labels = compute_masks(
            flat_cls_pred,
            flat_kernel_pred,
            mask_head_output,
            cls_threshold=default_kwargs["score_threshold"],
            mask_threshold=default_kwargs["seg_threshold"],
            nms_threshold=default_kwargs["nms_threshold"],
            kernel_depth=self.config.mask_output_filters,
            kernel_size=self.config.kernel_size,
            max_inst=default_kwargs["max_detections"])

        return seg_preds, scores, cls_labels

    @tf.function
    def train_step(self, inputs):

        gt_img = inputs[1]
        gt_mask_img = inputs[2]
        gt_boxes = inputs[3]
        gt_cls_ids = inputs[4]
        gt_labels = inputs[5]

        nx = tf.shape(inputs[1])[1]
        ny = tf.shape(inputs[1])[2]

        # Compute flattened targets

        compute_targets_func = partial(utils.compute_solo_cls_targets,
                                       shape=(nx, ny),
                                       strides=self.strides,
                                       grid_sizes=self.config.grid_sizes,
                                       scale_ranges=self.config.scale_ranges,
                                       offset_factor=self.config.offset_factor)
        class_targets, label_targets = tf.map_fn(
            compute_targets_func,
            (gt_boxes, gt_labels, gt_cls_ids, gt_mask_img),
            fn_output_signature=(tf.int32, tf.int32))

        # OHE class targets and delete bg
        class_targets = tf.one_hot(class_targets, self.ncls + 1)[..., 1:]

        with tf.GradientTape() as tape:

            mask_head_output, feat_cls_list, feat_kernel_list = self.model(
                gt_img, training=True)

            flatten_predictions_func = partial(flatten_predictions,
                                               ncls=self.ncls,
                                               kernel_depth=self.kernel_depth)
            # Flattened tensor over locations and levels [B, locations, ncls] and [B, locations, kern_depth]
            flat_cls_pred, flat_kernel_pred = tf.map_fn(
                flatten_predictions_func, [feat_cls_list, feat_kernel_list],
                fn_output_signature=(tf.float32, tf.float32))

            flat_cls_pred = tf.sigmoid(flat_cls_pred)

            loss_function = partial(
                compute_image_loss,
                weights=self.config.lossweights,
                kernel_size=self.config.kernel_size,
                kernel_depth=self.config.mask_output_filters)
            cls_loss, seg_loss = tf.map_fn(loss_function, [
                class_targets, label_targets, gt_mask_img, flat_cls_pred,
                flat_kernel_pred, mask_head_output
            ],
                                           fn_output_signature=(tf.float32,
                                                                tf.float32))

            # avg over batch size
            cls_loss = tf.reduce_mean(cls_loss)
            seg_loss = tf.reduce_mean(seg_loss)

            # Update loss
            self.seg_loss.update_state(seg_loss)
            self.cls_loss.update_state(cls_loss)

            total_loss = cls_loss + seg_loss
            self.total_loss.update_state(total_loss)

            grads = tape.gradient(total_loss, self.trainable_variables)

            self.optimizer.apply_gradients(
                (grad, var)
                for (grad, var) in zip(grads, self.trainable_variables)
                if grad is not None)

        # Update Metrics
        if self.ncls == 1:
            class_targets = class_targets[..., tf.newaxis]
        self.precision.update_state(class_targets, flat_cls_pred)
        self.recall.update_state(class_targets, flat_cls_pred)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):

        gt_img = inputs[1]
        gt_mask_img = inputs[2]
        gt_boxes = inputs[3]
        gt_cls_ids = inputs[4]
        gt_labels = inputs[5]

        nx = tf.shape(inputs[1])[1]
        ny = tf.shape(inputs[1])[2]

        # Compute targets
        compute_targets_func = partial(utils.compute_solo_cls_targets,
                                       shape=(nx, ny),
                                       strides=self.strides,
                                       grid_sizes=self.config.grid_sizes,
                                       scale_ranges=self.config.scale_ranges,
                                       offset_factor=self.config.offset_factor)
        class_targets, label_targets = tf.map_fn(
            compute_targets_func,
            (gt_boxes, gt_labels, gt_cls_ids, gt_mask_img),
            fn_output_signature=(tf.int32, tf.int32))

        # OHE class targets and delete bg
        class_targets = tf.one_hot(class_targets, self.ncls + 1)[..., 1:]

        mask_head_output, feat_cls_list, feat_kernel_list = self.model(
            gt_img, training=True)

        flatten_predictions_func = partial(flatten_predictions,
                                           ncls=self.ncls,
                                           kernel_depth=self.kernel_depth)
        # Flattened tensor over locations and levels [B, locations, ncls] and [B, locations, kern_depth]
        flat_cls_pred, flat_kernel_pred = tf.map_fn(
            flatten_predictions_func, [feat_cls_list, feat_kernel_list],
            fn_output_signature=(tf.float32, tf.float32))

        flat_cls_pred = tf.sigmoid(flat_cls_pred)

        loss_function = partial(compute_image_loss,
                                weights=self.config.lossweights,
                                kernel_size=self.config.kernel_size,
                                kernel_depth=self.config.mask_output_filters)
        cls_loss, seg_loss = tf.map_fn(loss_function, [
            class_targets, label_targets, gt_mask_img, flat_cls_pred,
            flat_kernel_pred, mask_head_output
        ],
                                       fn_output_signature=(tf.float32,
                                                            tf.float32))

        # avg over batch size
        cls_loss = tf.reduce_mean(cls_loss)
        seg_loss = tf.reduce_mean(seg_loss)

        # Update Metrics
        if self.ncls == 1:
            class_targets = class_targets[..., tf.newaxis]
        self.precision.update_state(class_targets, flat_cls_pred)
        self.recall.update_state(class_targets, flat_cls_pred)

        # Update loss
        self.seg_loss.update_state(seg_loss)
        self.cls_loss.update_state(cls_loss)
        self.total_loss.update_state(cls_loss + seg_loss)

        return {m.name: m.result() for m in self.metrics}


def train(model,
          train_dataset,
          epochs,
          batch_size=1,
          val_dataset=None,
          steps_per_epoch=None,
          validation_steps=None,
          optimizer=None,
          callbacks=None,
          initial_epoch=0,
          prefetch=tf.data.AUTOTUNE,
          buffer=None):

    # Dataset returns name, image, mask, bboxes, classes, labels

    if buffer is None:
        buffer = len(train_dataset)

    train_dataset = train_dataset.shuffle(buffer,
                                          reshuffle_each_iteration=True)
    train_dataset = train_dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size))
    train_dataset = train_dataset.repeat(epochs)

    if val_dataset is not None and (validation_steps is None
                                    or validation_steps is None > 0):
        val_dataset = val_dataset.apply(
            tf.data.experimental.dense_to_ragged_batch(batch_size))
        validation_steps = len(val_dataset)
        val_dataset = val_dataset.repeat(epochs)

    if steps_per_epoch is None:
        steps_per_epoch = len(train_dataset)

    print("Length of the batched dataset:", len(train_dataset))
    print("number of epochs:", epochs)
    print("number of training steps per epoch", steps_per_epoch)
    print("number of validation steps per epoch", validation_steps)

    train_dataset = train_dataset.prefetch(prefetch)
    if val_dataset is not None:
        val_dataset = val_dataset.prefetch(prefetch)

    # Here lr can be a scheduler
    if optimizer is not None:
        model.optimizer = optimizer
        model.compile(optimizer=model.optimizer)
    else:
        if model.optimizer is None:
            model.optimizer = tf.keras.optimizers.Adam()
            model.compile(optimizer=model.optimizer)

    print("Training using {} optimizer with lr0={}".format(
        model.optimizer._name, model.optimizer.lr.numpy()))

    history = model.fit(train_dataset,
                        steps_per_epoch=steps_per_epoch,
                        initial_epoch=initial_epoch,
                        epochs=epochs,
                        validation_data=val_dataset,
                        validation_steps=validation_steps,
                        callbacks=callbacks)
    return history


def eval(model, dataset, maxiter, threshold=0.75, **kwargs):
    """compute AP score (for now it does not compute AP per class, but for all classes)
    inputs:
    model: SOLOv2 model
    dataset: tf.dataset (e.g. dataset attribute of the DataLoader class)
    maxiter: number of images used
    threshold: iou threshold

    outputs:
    AP
    sorted scores
    precision
    interpolated precision
    recall

    TODO: add per class AP
    TODO: deal with multiple positive predictions for the same GT object
    """

    from sklearn.metrics import auc

    default_kwargs = {
        "score_threshold": 0.5,
        "seg_threshold": 0.5,
        "nms_threshold": 0.5,
        "max_detections": 400
    }

    default_kwargs.update(kwargs)

    for i, data in enumerate(dataset):

        if i >= maxiter:
            break

        imgname = data[0].numpy()[0]
        gt_img = data[1]
        gt_mask_img = data[2]
        gt_boxes = data[3]
        gt_cls_ids = data[4]
        gt_labels = data[5]

        seg_preds, scores, cls_labels = model(gt_img,
                                              training=False,
                                              **default_kwargs)
        seg_preds = seg_preds[0, ...].to_tensor()
        scores = scores[0, ...]

        labels, _ = tf.unique(tf.reshape(gt_mask_img, [-1]))
        ngt = tf.size(labels) - 1

        gt_masks = tf.one_hot(gt_mask_img, ngt + 1)[..., 1:]
        gt_masks = tf.reshape(gt_masks, [-1, ngt])
        gt_masks = tf.transpose(gt_masks, [1, 0])

        pred_masks = tf.where(seg_preds > default_kwargs["seg_threshold"], 1,
                              0)
        pred_masks = tf.reshape(pred_masks, shape=(pred_masks.shape[0], -1))  # [Npred, H*W]
        print("Processing image {}/{}: {} containing {} objets. Detected : {}".
              format(i + 1, maxiter, imgname, ngt, pred_masks.shape[0]))
        gt_masks = tf.cast(gt_masks, tf.int32)

        intersection = tf.matmul(gt_masks, pred_masks, transpose_b=True)

        gt_sums = tf.reduce_sum(gt_masks, axis=1)
        gt_sums = tf.tile(gt_sums[tf.newaxis, ...],
                          multiples=[pred_masks.shape[0], 1])  # [Npred x NGT]

        pred_sums = tf.reduce_sum(pred_masks, axis=1)
        pred_sums = tf.tile(pred_sums[tf.newaxis, ...],
                            multiples=[ngt, 1])  # [NGT, Npred]

        union = tf.transpose(gt_sums) + pred_sums - intersection
        iou = tf.math.divide_no_nan(tf.cast(intersection, tf.float32),
                                    tf.cast(union, tf.float32))

        # iou représente la matrice des ious entre GT et PRED (nGT lignes, npred colonnes)

        inds = tf.argmax(iou,
                         axis=0)  # indices des meilleures iou entre pred et GT

        if i == 0:
            all_iou = tf.reduce_max(iou, axis=0)
            TP = tf.where(all_iou > threshold, True,
                          False)  # storing if a predicted box is TP or FP
            all_scores = tf.gather(scores, inds)  # Get predicted box scores
        else:
            all_iou = tf.concat([all_iou, tf.reduce_max(iou, axis=0)], axis=-1)
            TP = tf.concat([TP, tf.where(all_iou > threshold, True, False)],
                           axis=-1)
            all_scores = tf.concat(
                [all_scores, tf.gather(scores, inds)], axis=-1)

    # Sort by score
    sorted_indices = tf.argsort(all_scores, direction='DESCENDING')
    sorted_TP = tf.gather(TP, sorted_indices)
    sorted_scores = tf.gather(all_scores, sorted_indices)

    npred = sorted_TP.shape[0]

    TP = 0
    FP = 0
    recall = np.zeros(npred, dtype=float)
    precision = np.zeros(npred, dtype=float)

    for i in range(npred):

        if sorted_TP[i]:
            TP += 1
        else:
            FP += 1

        recall[i] = TP / npred
        precision[i] = TP / (TP + FP)

    decreasing_max_precision = np.maximum.accumulate(precision[::-1])[::-1]

    AP = auc(recall, decreasing_max_precision)

    return AP, sorted_scores, precision, decreasing_max_precision, recall