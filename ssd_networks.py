from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Reshape, ZeroPadding2D, concatenate, Activation, Dropout, BatchNormalization
from keras.models import Model
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import keras.backend as K
import numpy as np
import random
import sys

from ssd_layers import Normalize
from ssd_layers import PriorBox

def ssd300(input_shape=(300, 300, 3),
           num_classes=21,
           min_scale=0.1,
           max_scale=0.9,
           scales=None,
           aspect_ratios_global=None,
           aspect_ratios_per_layer=None,
           two_boxes_for_ar1=True,
           limit_boxes=True,
           variances=[0.1, 0.1, 0.2, 0.2],
           weights_path=None,
           frozen_layers=None,
           summary=False,
           plot=False):

    n_predictor_layers = 6  # The number of predictor conv layers in the network is 6 for the original SSD300
    default_aspect_ratios = [[0.5, 1.0, 2.0],
                             [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
                             [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
                             [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
                             [0.5, 1.0, 2.0],
                             [0.5, 1.0, 2.0]]

    # Get a few exceptions out of the way first
    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        print(
            "`aspect_ratios_global` and `aspect_ratios_per_layer` both are None. Default aspect ratios of the paper implementation are used.")

    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError(
                "It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(
                    n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers + 1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(
                n_predictor_layers + 1, len(scales)))
    else:  # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        aspect_ratios = default_aspect_ratios
    elif aspect_ratios_per_layer and aspect_ratios_global is None:
        aspect_ratios = aspect_ratios_per_layer
    elif aspect_ratios_per_layer is None and aspect_ratios_global:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    aspect_ratios_conv4_3 = aspect_ratios[0]
    aspect_ratios_fc7     = aspect_ratios[1]
    aspect_ratios_conv6_2 = aspect_ratios[2]
    aspect_ratios_conv7_2 = aspect_ratios[3]
    aspect_ratios_conv8_2 = aspect_ratios[4]
    aspect_ratios_conv9_2 = aspect_ratios[5]

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios:
        n_boxes = []
        for aspect_ratio in aspect_ratios:
            if (1 in aspect_ratio) & two_boxes_for_ar1:
                n_boxes.append(len(aspect_ratio) + 1)  # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(aspect_ratio))
        n_boxes_conv4_3 = n_boxes[0]
        n_boxes_fc7     = n_boxes[1]
        n_boxes_conv6_2 = n_boxes[2]
        n_boxes_conv7_2 = n_boxes[3]
        n_boxes_conv8_2 = n_boxes[4]
        n_boxes_conv9_2 = n_boxes[5]

    input_layer = Input(shape=input_shape)
    img_height, img_width, img_channels = input_shape[0], input_shape[1], input_shape[2]

    # Block 1 -----------------------------------------------
    conv1_1 = Conv2D(64, (3, 3),
                     name='conv1_1',
                     padding='same',
                     activation='relu')(input_layer)

    conv1_2 = Conv2D(64, (3, 3),
                     name='conv1_2',
                     padding='same',
                     activation='relu')(conv1_1)

    pool1 = MaxPooling2D(name='pool1',
                         pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same', )(conv1_2)

    # Block 2 ----------------------------------------------
    conv2_1 = Conv2D(128, (3, 3),
                     name='conv2_1',
                     padding='same',
                     activation='relu')(pool1)

    conv2_2 = Conv2D(128, (3, 3),
                     name='conv2_2',
                     padding='same',
                     activation='relu')(conv2_1)

    pool2 = MaxPooling2D(name='pool2',
                         pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same')(conv2_2)

    # Block 3 ----------------------------------------------
    conv3_1 = Conv2D(256, (3, 3),
                     name='conv3_1',
                     padding='same',
                     activation='relu')(pool2)

    conv3_2 = Conv2D(256, (3, 3),
                     name='conv3_2',
                     padding='same',
                     activation='relu')(conv3_1)

    conv3_3 = Conv2D(256, (3, 3),
                     name='conv3_3',
                     padding='same',
                     activation='relu')(conv3_2)

    pool3 = MaxPooling2D(name='pool3',
                         pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same')(conv3_3)

    # Block 4 ---------------------------------------------
    conv4_1 = Conv2D(512, (3, 3),
                     name='conv4_1',
                     padding='same',
                     activation='relu')(pool3)

    conv4_2 = Conv2D(512, (3, 3),
                     name='conv4_2',
                     padding='same',
                     activation='relu')(conv4_1)

    conv4_3 = Conv2D(512, (3, 3),
                     name='conv4_3',
                     padding='same',
                     activation='relu')(conv4_2)

    pool4 = MaxPooling2D(name='pool4',
                         pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same')(conv4_3)

    # Block 5 --------------------------------------------
    conv5_1 = Conv2D(512, (3, 3),
                     name='conv5_1',
                     padding='same',
                     activation='relu')(pool4)

    conv5_2 = Conv2D(512, (3, 3),
                     name='conv5_2',
                     padding='same',
                     activation='relu')(conv5_1)

    conv5_3 = Conv2D(512, (3, 3),
                     name='conv5_3',
                     padding='same',
                     activation='relu')(conv5_2)

    pool5 = MaxPooling2D(name='pool5',
                         pool_size=(3, 3),
                         strides=(1, 1),
                         padding='same')(conv5_3)

    # Block 6 --------------------------------------------
    fc6 = Conv2D(1024, (3, 3),
                 name='fc6',
                 dilation_rate=(6, 6),
                 padding='same',
                 activation='relu'
                 )(pool5)

    # Block 7 --------------------------------------------
    fc7 = Conv2D(1024, (1, 1),
                 name='fc7',
                 padding='same',
                 activation='relu'
                 )(fc6)

    # EXTRAS
    # Block 8 --------------------------------------------
    conv6_1 = Conv2D(256, (1, 1),
                     name='conv6_1',
                     padding='same',
                     activation='relu')(fc7)

    conv6_1z = ZeroPadding2D(name='conv6_1z')(conv6_1)

    conv6_2 = Conv2D(512, (3, 3),
                     name='conv6_2',
                     strides=(2, 2),
                     padding='valid',
                     activation='relu')(conv6_1z)

    # Block 9 --------------------------------------------
    conv7_1 = Conv2D(128, (1, 1),
                     name='conv7_1',
                     padding='same',
                     activation='relu')(conv6_2)

    conv7_1z = ZeroPadding2D(name='conv7_1z')(conv7_1)

    conv7_2 = Conv2D(256, (3, 3),
                     name='conv7_2',
                     padding='valid',
                     strides=(2, 2),
                     activation='relu')(conv7_1z)

    # Block 10 -------------------------------------------
    conv8_1 = Conv2D(128, (1, 1),
                     name='conv8_1',
                     padding='same',
                     activation='relu')(conv7_2)

    conv8_2 = Conv2D(256, (3, 3),
                     name='conv8_2',
                     padding='valid',
                     strides=(1, 1),
                     activation='relu')(conv8_1)

    # Block 11 -------------------------------------------
    conv9_1 = Conv2D(128, (1, 1),
                     name='conv9_1',
                     padding='same',
                     activation='relu')(conv8_2)

    conv9_2 = Conv2D(256, (3, 3),
                     name='conv9_2',
                     padding='valid',
                     strides=(1, 1),
                     activation='relu')(conv9_1)

    # Last Pool ------------------------------------------
    #pool6 = GlobalAveragePooling2D(name='pool6')(conv8_2)

    # Prediction from conv4_3 ----------------------------
    conv4_3_norm = Normalize(20, name='conv4_3_norm')(conv4_3)

    conv4_3_norm_mbox_loc = Conv2D(n_boxes_conv4_3 * 4, (3, 3),
                                   name='conv4_3_norm_mbox_loc',
                                   padding='same')(conv4_3_norm)

    conv4_3_norm_mbox_loc_flat = Flatten(name='conv4_3_norm_mbox_loc_flat')(conv4_3_norm_mbox_loc)

    conv4_3_norm_mbox_conf = Conv2D(n_boxes_conv4_3 * num_classes, (3, 3),
                                    name='conv4_3_norm_mbox_conf',
                                    padding='same')(conv4_3_norm)

    conv4_3_norm_mbox_conf_flat = Flatten(name='conv4_3_norm_mbox_conf_flat')(conv4_3_norm_mbox_conf)

    conv4_3_norm_mbox_priorbox = PriorBox(img_height, img_width,
                                          this_scale=scales[0], next_scale=scales[1],
                                          aspect_ratios=aspect_ratios_conv4_3,
                                          two_boxes_for_ar1=two_boxes_for_ar1,
                                          limit_boxes=limit_boxes,
                                          variances=variances,
                                          name='conv4_3_norm_mbox_priorbox')(conv4_3_norm)

    # Prediction from fc7 ---------------------------------
    fc7_mbox_conf = Conv2D(n_boxes_fc7 * num_classes, (3, 3),
                           padding='same',
                           name='fc7_mbox_conf')(fc7)

    fc7_mbox_conf_flat = Flatten(name='fc7_mbox_conf_flat')(fc7_mbox_conf)

    fc7_mbox_loc = Conv2D(n_boxes_fc7 * 4, (3, 3),
                          name='fc7_mbox_loc',
                          padding='same')(fc7)

    fc7_mbox_loc_flat = Flatten(name='fc7_mbox_loc_flat')(fc7_mbox_loc)

    fc7_mbox_priorbox = PriorBox(img_height, img_width,
                                 this_scale=scales[1], next_scale=scales[2],
                                 aspect_ratios=aspect_ratios_fc7,
                                 two_boxes_for_ar1=two_boxes_for_ar1,
                                 limit_boxes=limit_boxes,
                                 variances=variances,
                                 name='fc7_mbox_priorbox')(fc7)

    # Prediction from conv6_2 ------------------------------
    conv6_2_mbox_conf = Conv2D(n_boxes_conv6_2 * num_classes, (3, 3),
                               padding='same',
                               name='conv6_2_mbox_conf')(conv6_2)

    conv6_2_mbox_conf_flat = Flatten(name='conv6_2_mbox_conf_flat')(conv6_2_mbox_conf)

    conv6_2_mbox_loc = Conv2D(n_boxes_conv6_2 * 4, (3, 3,),
                              name='conv6_2_mbox_loc',
                              padding='same')(conv6_2)

    conv6_2_mbox_loc_flat = Flatten(name='conv6_2_mbox_loc_flat')(conv6_2_mbox_loc)

    conv6_2_mbox_priorbox = PriorBox(img_height, img_width,
                                     this_scale=scales[2], next_scale=scales[3],
                                     aspect_ratios=aspect_ratios_conv6_2,
                                     two_boxes_for_ar1=two_boxes_for_ar1,
                                     limit_boxes=limit_boxes,
                                     variances=variances,
                                     name='conv6_2_mbox_priorbox')(conv6_2)

    # Prediction from conv7_2 --------------------------------
    conv7_2_mbox_conf = Conv2D(n_boxes_conv7_2 * num_classes, (3, 3),
                               padding='same',
                               name='conv7_2_mbox_conf')(conv7_2)

    conv7_2_mbox_conf_flat = Flatten(name='conv7_2_mbox_conf_flat')(conv7_2_mbox_conf)

    conv7_2_mbox_loc = Conv2D(n_boxes_conv7_2 * 4, (3, 3),
                              padding='same',
                              name='conv7_2_mbox_loc')(conv7_2)

    conv7_2_mbox_loc_flat = Flatten(name='conv7_2_mbox_loc_flat')(conv7_2_mbox_loc)

    conv7_2_mbox_priorbox = PriorBox(img_height, img_width,
                                     this_scale=scales[3], next_scale=scales[4],
                                     aspect_ratios=aspect_ratios_conv7_2,
                                     two_boxes_for_ar1=two_boxes_for_ar1,
                                     limit_boxes=limit_boxes,
                                     variances=variances,
                                     name='conv7_2_mbox_priorbox')(conv7_2)

    # Prediction from conv8_2 -------------------------------
    conv8_2_mbox_conf = Conv2D(n_boxes_conv8_2 * num_classes, (3, 3),
                               padding='same',
                               name='conv8_2_mbox_conf')(conv8_2)

    conv8_2_mbox_conf_flat = Flatten(name='conv8_2_mbox_conf_flat')(conv8_2_mbox_conf)

    conv8_2_mbox_loc = Conv2D(n_boxes_conv8_2 * 4, (3, 3),
                              padding='same',
                              name='conv8_2_mbox_loc')(conv8_2)

    conv8_2_mbox_loc_flat = Flatten(name='conv8_2_mbox_loc_flat')(conv8_2_mbox_loc)

    conv8_2_mbox_priorbox = PriorBox(img_height, img_width,
                                     this_scale=scales[4], next_scale=scales[5],
                                     aspect_ratios=aspect_ratios_conv8_2,
                                     two_boxes_for_ar1=two_boxes_for_ar1,
                                     limit_boxes=limit_boxes,
                                     variances=variances,
                                     name='conv8_2_mbox_priorbox')(conv8_2)

    # Prediction from conv9_2 --------------------------------------------
    conv9_2_mbox_conf = Conv2D(n_boxes_conv9_2 * num_classes, (3, 3),
                               padding='same',
                               name='conv9_2_mbox_conf')(conv9_2)

    conv9_2_mbox_conf_flat = Flatten(name='conv9_2_mbox_conf_flat')(conv9_2_mbox_conf)

    conv9_2_mbox_loc = Conv2D(n_boxes_conv9_2 * 4, (3, 3),
                              padding='same',
                              name='conv9_2_mbox_loc')(conv9_2)

    conv9_2_mbox_loc_flat = Flatten(name='conv9_2_mbox_loc_flat')(conv9_2_mbox_loc)

    conv9_2_mbox_priorbox = PriorBox(img_height, img_width,
                                     this_scale=scales[5], next_scale=scales[6],
                                     aspect_ratios=aspect_ratios_conv9_2,
                                     two_boxes_for_ar1=two_boxes_for_ar1,
                                     limit_boxes=limit_boxes,
                                     variances=variances,
                                     name='conv9_2_mbox_priorbox')(conv9_2)


    # Gather all predictions -------------------------------------------
    mbox_loc = concatenate([conv4_3_norm_mbox_loc_flat,
                            fc7_mbox_loc_flat,
                            conv6_2_mbox_loc_flat,
                            conv7_2_mbox_loc_flat,
                            conv8_2_mbox_loc_flat,
                            conv9_2_mbox_loc_flat],
                           axis=1,
                           name='mbox_loc')

    mbox_conf = concatenate([conv4_3_norm_mbox_conf_flat,
                             fc7_mbox_conf_flat,
                             conv6_2_mbox_conf_flat,
                             conv7_2_mbox_conf_flat,
                             conv8_2_mbox_conf_flat,
                             conv9_2_mbox_conf_flat],
                            axis=1,
                            name='mbox_conf')

    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    conv4_3_norm_mbox_priorbox_reshape = Reshape((-1, 8), name='conv4_3_norm_mbox_priorbox_reshape')(conv4_3_norm_mbox_priorbox)
    fc7_mbox_priorbox_reshape = Reshape((-1, 8), name='fc7_mbox_priorbox_reshape')(fc7_mbox_priorbox)
    conv6_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv6_2_mbox_priorbox_reshape')(conv6_2_mbox_priorbox)
    conv7_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv7_2_mbox_priorbox_reshape')(conv7_2_mbox_priorbox)
    conv8_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv8_2_mbox_priorbox_reshape')(conv8_2_mbox_priorbox)
    conv9_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv9_2_mbox_priorbox_reshape')(conv9_2_mbox_priorbox)

    mbox_priorbox = concatenate([conv4_3_norm_mbox_priorbox_reshape,
                                 fc7_mbox_priorbox_reshape,
                                 conv6_2_mbox_priorbox_reshape,
                                 conv7_2_mbox_priorbox_reshape,
                                 conv8_2_mbox_priorbox_reshape,
                                 conv9_2_mbox_priorbox_reshape],
                                axis=1, name='mbox_priorbox')

    if hasattr(mbox_loc, '_keras_shape'):
        num_boxes = mbox_loc._keras_shape[-1] // 4
    elif hasattr(mbox_loc, 'int_shape'):
        num_boxes = K.int_shape(mbox_loc)[-1] // 4
    mbox_loc = Reshape((num_boxes, 4),
                       name='mbox_loc_final')(mbox_loc)
    mbox_conf = Reshape((num_boxes, num_classes),
                        name='mbox_conf_logits')(mbox_conf)
    mbox_conf = Activation('softmax',
                           name='mbox_conf_final')(mbox_conf)
    predictions = concatenate([mbox_loc,
                               mbox_conf,
                               mbox_priorbox],
                              axis=2,
                              name='predictions')

    model = Model(inputs=input_layer, outputs=predictions)

    if weights_path is not None:
        model.load_weights(weights_path, by_name=True)

    if frozen_layers is not None:
        for layer in model.layers:
            if layer.name in frozen_layers:
                layer.trainable = False

    if summary:
        model.summary()

    if plot:
        plot_model(model, to_file='SSD300.png')
        SVG(model_to_dot(model).create(prog='dot', format='svg'))

    return model


def ssd512(input_shape=(512, 512, 3),
           num_classes=21,
           min_scale=0.1,
           max_scale=0.9,
           scales=None,
           aspect_ratios_global=None,
           aspect_ratios_per_layer=None,
           two_boxes_for_ar1=True,
           limit_boxes=True,
           variances=[0.1, 0.1, 0.2, 0.2],
           weights_path=None,
           frozen_layers=None,
           summary=False,
           plot=False):

    n_predictor_layers = 7  # The number of predictor conv layers in the network is 6 for the original SSD300
    default_aspect_ratios = [[0.5, 1.0, 2.0],
                             [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
                             [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
                             [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
                             [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
                             [0.5, 1.0, 2.0],
                             [0.5, 1.0, 2.0]]

    # Get a few exceptions out of the way first
    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        print(
            "`aspect_ratios_global` and `aspect_ratios_per_layer` both are None. Default aspect ratios of the paper implementation are used.")

    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError(
                "It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(
                    n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers + 1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(
                n_predictor_layers + 1, len(scales)))
    else:  # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        aspect_ratios = default_aspect_ratios
    elif aspect_ratios_per_layer and aspect_ratios_global is None:
        aspect_ratios = aspect_ratios_per_layer
    elif aspect_ratios_per_layer is None and aspect_ratios_global:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    aspect_ratios_conv4_3 = aspect_ratios[0]
    aspect_ratios_fc7     = aspect_ratios[1]
    aspect_ratios_conv6_2 = aspect_ratios[2]
    aspect_ratios_conv7_2 = aspect_ratios[3]
    aspect_ratios_conv8_2 = aspect_ratios[4]
    aspect_ratios_conv9_2 = aspect_ratios[5]
    aspect_ratios_conv10_2 = aspect_ratios[6]

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios:
        n_boxes = []
        for aspect_ratio in aspect_ratios:
            if (1 in aspect_ratio) & two_boxes_for_ar1:
                n_boxes.append(len(aspect_ratio) + 1)  # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(aspect_ratio))
        n_boxes_conv4_3 = n_boxes[0]
        n_boxes_fc7 = n_boxes[1]
        n_boxes_conv6_2 = n_boxes[2]
        n_boxes_conv7_2 = n_boxes[3]
        n_boxes_conv8_2 = n_boxes[4]
        n_boxes_conv9_2 = n_boxes[5]
        n_boxes_conv10_2 = n_boxes[6]

    input_layer = Input(shape=input_shape)
    img_height, img_width, img_channels = input_shape[0], input_shape[1], input_shape[2]

    # Block 1 -----------------------------------------------
    conv1_1 = Conv2D(64, (3, 3),
                     name='conv1_1',
                     padding='same',
                     activation='relu')(input_layer)

    conv1_2 = Conv2D(64, (3, 3),
                     name='conv1_2',
                     padding='same',
                     activation='relu')(conv1_1)

    pool1 = MaxPooling2D(name='pool1',
                         pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same', )(conv1_2)

    # Block 2 ----------------------------------------------
    conv2_1 = Conv2D(128, (3, 3),
                     name='conv2_1',
                     padding='same',
                     activation='relu')(pool1)

    conv2_2 = Conv2D(128, (3, 3),
                     name='conv2_2',
                     padding='same',
                     activation='relu')(conv2_1)

    pool2 = MaxPooling2D(name='pool2',
                         pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same')(conv2_2)

    # Block 3 ----------------------------------------------
    conv3_1 = Conv2D(256, (3, 3),
                     name='conv3_1',
                     padding='same',
                     activation='relu')(pool2)

    conv3_2 = Conv2D(256, (3, 3),
                     name='conv3_2',
                     padding='same',
                     activation='relu')(conv3_1)

    conv3_3 = Conv2D(256, (3, 3),
                     name='conv3_3',
                     padding='same',
                     activation='relu')(conv3_2)

    pool3 = MaxPooling2D(name='pool3',
                         pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same')(conv3_3)

    # Block 4 ---------------------------------------------
    conv4_1 = Conv2D(512, (3, 3),
                     name='conv4_1',
                     padding='same',
                     activation='relu')(pool3)

    conv4_2 = Conv2D(512, (3, 3),
                     name='conv4_2',
                     padding='same',
                     activation='relu')(conv4_1)

    conv4_3 = Conv2D(512, (3, 3),
                     name='conv4_3',
                     padding='same',
                     activation='relu')(conv4_2)

    pool4 = MaxPooling2D(name='pool4',
                         pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same')(conv4_3)

    # Block 5 --------------------------------------------
    conv5_1 = Conv2D(512, (3, 3),
                     name='conv5_1',
                     padding='same',
                     activation='relu')(pool4)

    conv5_2 = Conv2D(512, (3, 3),
                     name='conv5_2',
                     padding='same',
                     activation='relu')(conv5_1)

    conv5_3 = Conv2D(512, (3, 3),
                     name='conv5_3',
                     padding='same',
                     activation='relu')(conv5_2)

    pool5 = MaxPooling2D(name='pool5',
                         pool_size=(3, 3),
                         strides=(1, 1),
                         padding='same')(conv5_3)

    # Block 6 --------------------------------------------
    fc6 = Conv2D(1024, (3, 3),
                 name='fc6',
                 dilation_rate=(6, 6),
                 padding='same',
                 activation='relu'
                 )(pool5)

    # Block 7 --------------------------------------------
    fc7 = Conv2D(1024, (1, 1),
                 name='fc7',
                 padding='same',
                 activation='relu'
                 )(fc6)

    # EXTRAS
    # Block 8 --------------------------------------------
    conv6_1 = Conv2D(256, (1, 1),
                     name='conv6_1',
                     padding='same',
                     activation='relu')(fc7)

    conv6_1z = ZeroPadding2D(name='conv6_1z')(conv6_1)

    conv6_2 = Conv2D(512, (3, 3),
                     name='conv6_2',
                     strides=(2, 2),
                     padding='valid',
                     activation='relu')(conv6_1z)

    # Block 9 --------------------------------------------
    conv7_1 = Conv2D(128, (1, 1),
                     name='conv7_1',
                     padding='same',
                     activation='relu')(conv6_2)

    conv7_1z = ZeroPadding2D(name='conv7_1z')(conv7_1)

    conv7_2 = Conv2D(256, (3, 3),
                     name='conv7_2',
                     padding='valid',
                     strides=(2, 2),
                     activation='relu')(conv7_1z)

    # Block 10 -------------------------------------------
    conv8_1 = Conv2D(128, (1, 1),
                     name='conv8_1',
                     padding='same',
                     activation='relu')(conv7_2)

    conv8_2 = Conv2D(256, (3, 3),
                     name='conv8_2',
                     padding='valid',
                     strides=(1, 1),
                     activation='relu')(conv8_1)

    # Block 11 -------------------------------------------
    conv9_1 = Conv2D(128, (1, 1),
                     name='conv9_1',
                     padding='same',
                     activation='relu')(conv8_2)

    conv9_2 = Conv2D(256, (3, 3),
                     name='conv9_2',
                     padding='valid',
                     strides=(1, 1),
                     activation='relu')(conv9_1)

    # Block 12 -------------------------------------------
    conv10_1 = Conv2D(128, (1, 1),
                     name='conv10_1',
                     padding='same',
                     activation='relu')(conv9_2)

    conv10_2 = Conv2D(256, (4, 4),
                     name='conv10_2',
                     padding='valid',
                     strides=(1, 1),
                     activation='relu')(conv10_1)

    # Last Pool ------------------------------------------
    # pool6 = GlobalAveragePooling2D(name='pool6')(conv8_2)

    # Prediction from conv4_3 ----------------------------
    conv4_3_norm = Normalize(20, name='conv4_3_norm')(conv4_3)

    conv4_3_norm_mbox_loc = Conv2D(n_boxes_conv4_3 * 4, (3, 3),
                                   name='conv4_3_norm_mbox_loc',
                                   padding='same')(conv4_3_norm)

    conv4_3_norm_mbox_loc_flat = Flatten(name='conv4_3_norm_mbox_loc_flat')(conv4_3_norm_mbox_loc)

    conv4_3_norm_mbox_conf = Conv2D(n_boxes_conv4_3 * num_classes, (3, 3),
                                    name='conv4_3_norm_mbox_conf',
                                    padding='same')(conv4_3_norm)

    conv4_3_norm_mbox_conf_flat = Flatten(name='conv4_3_norm_mbox_conf_flat')(conv4_3_norm_mbox_conf)

    conv4_3_norm_mbox_priorbox = PriorBox(img_height, img_width,
                                          this_scale=scales[0], next_scale=scales[1],
                                          aspect_ratios=aspect_ratios_conv4_3,
                                          two_boxes_for_ar1=two_boxes_for_ar1,
                                          limit_boxes=limit_boxes,
                                          variances=variances,
                                          name='conv4_3_norm_mbox_priorbox')(conv4_3_norm)

    # Prediction from fc7 ---------------------------------
    fc7_mbox_conf = Conv2D(n_boxes_fc7 * num_classes, (3, 3),
                           padding='same',
                           name='fc7_mbox_conf')(fc7)

    fc7_mbox_conf_flat = Flatten(name='fc7_mbox_conf_flat')(fc7_mbox_conf)

    fc7_mbox_loc = Conv2D(n_boxes_fc7 * 4, (3, 3),
                          name='fc7_mbox_loc',
                          padding='same')(fc7)

    fc7_mbox_loc_flat = Flatten(name='fc7_mbox_loc_flat')(fc7_mbox_loc)

    fc7_mbox_priorbox = PriorBox(img_height, img_width,
                                 this_scale=scales[1], next_scale=scales[2],
                                 aspect_ratios=aspect_ratios_fc7,
                                 two_boxes_for_ar1=two_boxes_for_ar1,
                                 limit_boxes=limit_boxes,
                                 variances=variances,
                                 name='fc7_mbox_priorbox')(fc7)

    # Prediction from conv6_2 ------------------------------
    conv6_2_mbox_conf = Conv2D(n_boxes_conv6_2 * num_classes, (3, 3),
                               padding='same',
                               name='conv6_2_mbox_conf')(conv6_2)

    conv6_2_mbox_conf_flat = Flatten(name='conv6_2_mbox_conf_flat')(conv6_2_mbox_conf)

    conv6_2_mbox_loc = Conv2D(n_boxes_conv6_2 * 4, (3, 3,),
                              name='conv6_2_mbox_loc',
                              padding='same')(conv6_2)

    conv6_2_mbox_loc_flat = Flatten(name='conv6_2_mbox_loc_flat')(conv6_2_mbox_loc)

    conv6_2_mbox_priorbox = PriorBox(img_height, img_width,
                                     this_scale=scales[2], next_scale=scales[3],
                                     aspect_ratios=aspect_ratios_conv6_2,
                                     two_boxes_for_ar1=two_boxes_for_ar1,
                                     limit_boxes=limit_boxes,
                                     variances=variances,
                                     name='conv6_2_mbox_priorbox')(conv6_2)

    # Prediction from conv7_2 --------------------------------
    conv7_2_mbox_conf = Conv2D(n_boxes_conv7_2 * num_classes, (3, 3),
                               padding='same',
                               name='conv7_2_mbox_conf')(conv7_2)

    conv7_2_mbox_conf_flat = Flatten(name='conv7_2_mbox_conf_flat')(conv7_2_mbox_conf)

    conv7_2_mbox_loc = Conv2D(n_boxes_conv7_2 * 4, (3, 3),
                              padding='same',
                              name='conv7_2_mbox_loc')(conv7_2)

    conv7_2_mbox_loc_flat = Flatten(name='conv7_2_mbox_loc_flat')(conv7_2_mbox_loc)

    conv7_2_mbox_priorbox = PriorBox(img_height, img_width,
                                     this_scale=scales[3], next_scale=scales[4],
                                     aspect_ratios=aspect_ratios_conv7_2,
                                     two_boxes_for_ar1=two_boxes_for_ar1,
                                     limit_boxes=limit_boxes,
                                     variances=variances,
                                     name='conv7_2_mbox_priorbox')(conv7_2)

    # Prediction from conv8_2 -------------------------------
    conv8_2_mbox_conf = Conv2D(n_boxes_conv8_2 * num_classes, (3, 3),
                               padding='same',
                               name='conv8_2_mbox_conf')(conv8_2)

    conv8_2_mbox_conf_flat = Flatten(name='conv8_2_mbox_conf_flat')(conv8_2_mbox_conf)

    conv8_2_mbox_loc = Conv2D(n_boxes_conv8_2 * 4, (3, 3),
                              padding='same',
                              name='conv8_2_mbox_loc')(conv8_2)

    conv8_2_mbox_loc_flat = Flatten(name='conv8_2_mbox_loc_flat')(conv8_2_mbox_loc)

    conv8_2_mbox_priorbox = PriorBox(img_height, img_width,
                                     this_scale=scales[4], next_scale=scales[5],
                                     aspect_ratios=aspect_ratios_conv8_2,
                                     two_boxes_for_ar1=two_boxes_for_ar1,
                                     limit_boxes=limit_boxes,
                                     variances=variances,
                                     name='conv8_2_mbox_priorbox')(conv8_2)

    # Prediction from conv9_2 -------------------------------
    conv9_2_mbox_conf = Conv2D(n_boxes_conv9_2 * num_classes, (3, 3),
                               padding='same',
                               name='conv9_2_mbox_conf')(conv9_2)

    conv9_2_mbox_conf_flat = Flatten(name='conv9_2_mbox_conf_flat')(conv9_2_mbox_conf)

    conv9_2_mbox_loc = Conv2D(n_boxes_conv9_2 * 4, (3, 3),
                              padding='same',
                              name='conv9_2_mbox_loc')(conv9_2)

    conv9_2_mbox_loc_flat = Flatten(name='conv9_2_mbox_loc_flat')(conv9_2_mbox_loc)

    conv9_2_mbox_priorbox = PriorBox(img_height, img_width,
                                     this_scale=scales[5], next_scale=scales[6],
                                     aspect_ratios=aspect_ratios_conv9_2,
                                     two_boxes_for_ar1=two_boxes_for_ar1,
                                     limit_boxes=limit_boxes,
                                     variances=variances,
                                     name='conv9_2_mbox_priorbox')(conv9_2)

    # Prediction from conv10_2 --------------------------------------------
    conv10_2_mbox_conf = Conv2D(n_boxes_conv10_2 * num_classes, (3, 3),
                               padding='same',
                               name='conv10_2_mbox_conf')(conv10_2)

    conv10_2_mbox_conf_flat = Flatten(name='conv10_2_mbox_conf_flat')(conv10_2_mbox_conf)

    conv10_2_mbox_loc = Conv2D(n_boxes_conv10_2 * 4, (3, 3),
                              padding='same',
                              name='conv10_2_mbox_loc')(conv10_2)

    conv10_2_mbox_loc_flat = Flatten(name='conv10_2_mbox_loc_flat')(conv10_2_mbox_loc)

    conv10_2_mbox_priorbox = PriorBox(img_height, img_width,
                                      this_scale=scales[6], next_scale=scales[7],
                                      aspect_ratios=aspect_ratios_conv10_2,
                                      two_boxes_for_ar1=two_boxes_for_ar1,
                                      limit_boxes=limit_boxes,
                                      variances=variances,
                                      name='conv10_2_mbox_priorbox')(conv10_2)


    # Gather all predictions -------------------------------------------
    mbox_loc = concatenate([conv4_3_norm_mbox_loc_flat,
                            fc7_mbox_loc_flat,
                            conv6_2_mbox_loc_flat,
                            conv7_2_mbox_loc_flat,
                            conv8_2_mbox_loc_flat,
                            conv9_2_mbox_loc_flat,
                            conv10_2_mbox_loc_flat],
                           axis=1,
                           name='mbox_loc')

    mbox_conf = concatenate([conv4_3_norm_mbox_conf_flat,
                             fc7_mbox_conf_flat,
                             conv6_2_mbox_conf_flat,
                             conv7_2_mbox_conf_flat,
                             conv8_2_mbox_conf_flat,
                             conv9_2_mbox_conf_flat,
                             conv10_2_mbox_conf_flat],
                            axis=1,
                            name='mbox_conf')

    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    conv4_3_norm_mbox_priorbox_reshape = Reshape((-1, 8), name='conv4_3_norm_mbox_priorbox_reshape')(conv4_3_norm_mbox_priorbox)
    fc7_mbox_priorbox_reshape = Reshape((-1, 8), name='fc7_mbox_priorbox_reshape')(fc7_mbox_priorbox)
    conv6_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv6_2_mbox_priorbox_reshape')(conv6_2_mbox_priorbox)
    conv7_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv7_2_mbox_priorbox_reshape')(conv7_2_mbox_priorbox)
    conv8_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv8_2_mbox_priorbox_reshape')(conv8_2_mbox_priorbox)
    conv9_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv9_2_mbox_priorbox_reshape')(conv9_2_mbox_priorbox)
    conv10_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv10_2_mbox_priorbox_reshape')(conv10_2_mbox_priorbox)

    mbox_priorbox = concatenate([conv4_3_norm_mbox_priorbox_reshape,
                                 fc7_mbox_priorbox_reshape,
                                 conv6_2_mbox_priorbox_reshape,
                                 conv7_2_mbox_priorbox_reshape,
                                 conv8_2_mbox_priorbox_reshape,
                                 conv9_2_mbox_priorbox_reshape,
                                 conv10_2_mbox_priorbox_reshape],
                                axis=1, name='mbox_priorbox')

    if hasattr(mbox_loc, '_keras_shape'):
        num_boxes = mbox_loc._keras_shape[-1] // 4
    elif hasattr(mbox_loc, 'int_shape'):
        num_boxes = K.int_shape(mbox_loc)[-1] // 4
    mbox_loc = Reshape((num_boxes, 4),
                       name='mbox_loc_final')(mbox_loc)
    mbox_conf = Reshape((num_boxes, num_classes),
                        name='mbox_conf_logits')(mbox_conf)
    mbox_conf = Activation('softmax',
                           name='mbox_conf_final')(mbox_conf)
    predictions = concatenate([mbox_loc,
                               mbox_conf,
                               mbox_priorbox],
                              axis=2,
                              name='predictions')

    model = Model(inputs=input_layer, outputs=predictions)

    if weights_path is not None:
        model.load_weights(weights_path, by_name=True)

    if frozen_layers is not None:
        for layer in model.layers:
            if layer.name in frozen_layers:
                layer.trainable = False

    if summary:
        model.summary()

    if plot:
        plot_model(model, to_file='SSD512.png')
        SVG(model_to_dot(model).create(prog='dot', format='svg'))

    return model