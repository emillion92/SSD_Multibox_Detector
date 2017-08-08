"""SSD training utils."""

import tensorflow as tf
import keras.backend as K
import csv
import pickle
from random import shuffle
import os

class MultiboxLoss_old(object):
    """Multibox loss with some helper functions.

    # Arguments
        num_classes: Number of classes including background.
        alpha: Weight of L1-smooth loss.
        neg_pos_ratio: Max ratio of negative to positive boxes in loss.
        background_label_id: Id of background label.
        negatives_for_hard: Number of negative boxes to consider
            it there is no positive boxes in batch.

    # References
        https://arxiv.org/abs/1512.02325

    # TODO
        Add possibility for background label id be not zero
    """
    def __init__(self, num_classes, alpha=1.0, neg_pos_ratio=3.0,
                 background_label_id=0, negatives_for_hard=100.0):
        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        self.background_label_id = background_label_id
        self.negatives_for_hard = negatives_for_hard

    #def get_config(self):
    #    config = {'num_classes': self.num_classes,
    #              'alpha': self.alpha,
    #              'neg_pos_ratio': self.neg_pos_ratio,
    #              'background_label_id': self.background_label_id,
    #              'negatives_for_hard': self.negatives_for_hard
    #              }
    #    base_config = super(MultiboxLoss, self).get_config()
    #    return dict(list(base_config.items()) + list(config.items()))

    def _l1_smooth_loss(self, y_true, y_pred):
        """Compute L1-smooth loss for localizaion.

        # Arguments
            y_true: Ground truth bounding boxes,
                tensor of shape (?, num_boxes, 4).
            y_pred: Predicted bounding boxes,
                tensor of shape (?, num_boxes, 4).

        # Returns
            l1_loss: L1-smooth loss, tensor of shape (?, num_boxes).

        # References (Fast R-CNN paper)
            https://arxiv.org/abs/1504.08083
        # L1 smooth loss is less sensitive to outliers than the L2 loss used in R-CNN and SPPnet.
          When the regression targets  are  unbounded,  training  with L2 loss  can  require careful tuning
          of learning rates in order to prevent exploding gradients. L1 smooth loss eliminates this sensitivity.
        """
        absolute_value_loss = tf.abs(y_true - y_pred) - 0.5          # |x|-0.5
        square_loss = 0.5 * (y_true - y_pred) ** 2        # 0.5x^2
        absolute_value_condition = K.less(absolute_value_loss, 1.0) # if |x| < 1: sq_loss, otherwise
        l1_smooth_loss = tf.where(absolute_value_condition, square_loss,
                                  absolute_value_loss)
        return K.sum(l1_smooth_loss, axis=-1)

    def _softmax_loss(self, y_true, y_pred):
        """Compute softmax loss for confidence.

        # Arguments
            y_true: Ground truth targets,
                tensor of shape (?, num_boxes, num_classes).
            y_pred: Predicted logits,
                tensor of shape (?, num_boxes, num_classes).

        # Returns
            softmax_loss: Softmax loss, tensor of shape (?, num_boxes).

        # standard loss for final layer of NN for classification task
        """
        y_pred = K.maximum(K.minimum(y_pred, 1 - 1e-15), 1e-15)
        softmax_loss = - K.sum(y_true * K.log(y_pred), axis=-1)
        return softmax_loss

    def compute_loss(self, y_true, y_pred):
        """Compute mutlibox loss.

        # Arguments
            y_true: Ground truth targets,
                tensor of shape (?, num_boxes, 4 + num_classes + 8),
                priors in ground truth are fictitious,
                y_true[:, :, -8] has 1 if prior should be penalized
                    or in other words is assigned to some ground truth box,
                y_true[:, :, -7:] are all 0.
            y_pred: Predicted logits,
                tensor of shape (?, num_boxes, 4 + num_classes + 8).

        # Returns
            loss: Loss for prediction, tensor of shape (?,).
        """
        batch_size = tf.shape(y_true)[0]
        num_boxes = tf.to_float(tf.shape(y_true)[1])

        y_pred_localization = y_pred[:, :, :4]
        y_true_localization = y_true[:, :, :4]
        y_pred_classification = y_pred[:, :, 4:(4 + self.num_classes)]
        y_true_classification = y_true[:, :, 4:(4 + self.num_classes)]

        # loss for all priors
        conf_loss = self._softmax_loss(y_true_classification,   # compute softmax loss for predicted class labels
                                       y_pred_classification)
        loc_loss = self._l1_smooth_loss(y_true_localization,    # compute L1-smoothe loss for predicted bounding boxes
                                        y_pred_localization)

        # get positives loss from assigned flag at y_true[:, :, -8]
        num_pos = tf.reduce_sum(y_true[:, :, -8], axis=-1)                  # count nr. of gt bb which were assigned
        pos_loc_loss = tf.reduce_sum(loc_loss * y_true[:, :, -8], axis=1)   # sum up all the loc_loss of each assigned bb, eq. 2 Lconf
        pos_conf_loss = tf.reduce_sum(conf_loss * y_true[:, :, -8], axis=1) # sum up all the conf_loss of each assigned bb, eq. 3.1

        # get negatives loss, we penalize only confidence here, for eq. 3
        num_neg = tf.minimum(self.neg_pos_ratio * num_pos, num_boxes - num_pos) # check if number of 3 x neg_pos_ratio
                                                                                #  bb or number of negative bb is
                                                                                # bigger and take smaller value to get
                                                                                # neg/pos # ratio of neg_pos_ratio:1

        pos_num_neg_mask = tf.greater(num_neg, 0)
        has_min = tf.to_float(tf.reduce_any(pos_num_neg_mask))
        num_neg = tf.concat(axis=0, values=[num_neg, [(1 - has_min) * self.negatives_for_hard]])
        num_neg_batch = tf.reduce_min(tf.boolean_mask(num_neg, tf.greater(num_neg, 0)))
        num_neg_batch = tf.to_int32(num_neg_batch)
        confs_start = 4 + self.background_label_id + 1              # index where class predictions start in y (excl. 'background')
        confs_end = confs_start + self.num_classes - 1              # index where class predictions end in y
        max_confs = tf.reduce_max(y_pred[:, :, confs_start:confs_end], axis=2)
        _, indices = tf.nn.top_k(max_confs * (1 - y_true[:, :, -8]), k=num_neg_batch)
        batch_idx = tf.expand_dims(tf.range(0, batch_size), 1)
        batch_idx = tf.tile(batch_idx, (1, num_neg_batch))
        full_indices = (tf.reshape(batch_idx, [-1]) * tf.to_int32(num_boxes) + tf.reshape(indices, [-1]))
        # full_indices = tf.concat(2, [tf.expand_dims(batch_idx, 2),
        #                              tf.expand_dims(indices, 2)])
        # neg_conf_loss = tf.gather_nd(conf_loss, full_indices)
        neg_conf_loss = tf.gather(tf.reshape(conf_loss, [-1]), full_indices)
        neg_conf_loss = tf.reshape(neg_conf_loss, [batch_size, num_neg_batch])
        neg_conf_loss = tf.reduce_sum(neg_conf_loss, axis=1)

        # loss is sum of positives and negatives
        total_loss = pos_conf_loss + neg_conf_loss
        total_loss /= (num_pos + tf.to_float(num_neg_batch))
        num_pos = tf.where(tf.not_equal(num_pos, 0), num_pos, tf.ones_like(num_pos))
        total_loss += (self.alpha * pos_loc_loss) / num_pos     # 1/N = 1/num_pos wehere N is number of matched
                                                                # default bounding boxes
        return total_loss

class MultiboxLoss(object):
    def __init__(self, num_classes, alpha=1.0, neg_pos_ratio=3.0,
                 background_id=0, negatives_for_hard=100.0):
        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        self.background_id = background_id
        self.negatives_for_hard = negatives_for_hard

    def _l1_smooth_loss(self, y_true, y_pred):
        absolute_value_loss = tf.abs(y_true - y_pred) - 0.5
        square_loss = 0.5 * (y_true - y_pred)**2
        absolute_value_condition = K.less(absolute_value_loss, 1.0)
        l1_smooth_loss = tf.where(absolute_value_condition, square_loss,
                                                    absolute_value_loss)
        return K.sum(l1_smooth_loss, axis=-1)

    def _softmax_loss(self, y_true, y_pred):
        y_pred = K.maximum(K.minimum(y_pred, 1 - 1e-15), 1e-15)
        softmax_loss = - K.sum(y_true * K.log(y_pred), axis=-1)
        return softmax_loss

    def compute_loss(self, y_true, y_pred):
        batch_size = K.shape(y_true)[0]
        num_prior_boxes = K.cast(K.shape(y_true)[1], 'float')

        y_pred_localization = y_pred[:, :, :4]
        y_true_localization = y_true[:, :, :4]
        y_pred_classification = y_pred[:, :, 4:(4 + self.num_classes)]
        y_true_classification = y_true[:, :, 4:(4 + self.num_classes)]

        localization_loss = self._l1_smooth_loss(y_true_localization,
                                                 y_pred_localization)
        classification_loss = self._softmax_loss(y_true_classification,
                                                 y_pred_classification)

        int_positive_mask = 1 - y_true[:, :, 4 + self.background_id]
        num_positives = tf.reduce_sum(int_positive_mask, axis=-1)
        positive_localization_losses = (localization_loss * int_positive_mask)
        positive_classification_losses = (classification_loss *
                                          int_positive_mask)
        positive_classification_loss = K.sum(positive_classification_losses, 1)
        positive_localization_loss = K.sum(positive_localization_losses, 1)

        num_negatives_1 = self.neg_pos_ratio * num_positives
        num_negatives_2 = num_prior_boxes - num_positives
        num_negatives = tf.minimum(num_negatives_1, num_negatives_2)

        num_positive_mask = tf.greater(num_negatives, 0)
        has_a_positive = tf.to_float(tf.reduce_any(num_positive_mask))
        num_negatives = tf.concat([num_negatives,
                        [(1 - has_a_positive) * self.negatives_for_hard]], 0)
        num_positive_mask = tf.greater(num_negatives, 0)
        num_neg_batch = tf.reduce_min(tf.boolean_mask(num_negatives,
                                                num_positive_mask))
        num_neg_batch = tf.to_int32(num_neg_batch)

        pred_class_values = K.max(y_pred_classification[:, :, 1:], axis=2)
        int_negatives_mask = y_true[:, :, 4 + self.background_id]
        pred_negative_class_values = pred_class_values * int_negatives_mask
        top_k_negative_indices = tf.nn.top_k(pred_negative_class_values,
                                                    k=num_neg_batch)[1]

        batch_indices = K.expand_dims(K.arange(0, batch_size), 1)
        batch_indices = K.tile(batch_indices, (1, num_neg_batch))
        batch_indices = K.flatten(batch_indices) * K.cast(num_prior_boxes,
                                                                'int32')
        full_indices = batch_indices + K.flatten(top_k_negative_indices)

        negative_classification_loss = K.gather(K.flatten(classification_loss),
                                                                full_indices)
        negative_classification_loss = K.reshape(negative_classification_loss,
                                                    [batch_size, num_neg_batch])
        negative_classification_loss = K.sum(negative_classification_loss, 1)

        # loss is sum of positives and negatives
        total_loss = (positive_classification_loss +
                      negative_classification_loss)
        num_prior_boxes_per_batch = (num_positives +
                                     K.cast(num_neg_batch, 'float'))
        total_loss = total_loss / num_prior_boxes_per_batch
        num_positives = tf.where(K.not_equal(num_positives, 0), num_positives,
                                                   K.ones_like(num_positives))
        positive_localization_loss = self.alpha * positive_classification_loss
        positive_localization_loss = positive_localization_loss / num_positives
        total_loss = total_loss + positive_localization_loss
        return total_loss

def split_data(gts_path, gts, training_ratio=.8):
    gt = {}
    for gt_file in gts:                                # read in ground truth data from pickle file
        file_path = os.path.join(gts_path,gt_file)
        gt.update(pickle.load(open(file_path, 'rb')))
    keys = sorted(gt.keys())                           # sort gt filenames is ascending order
    shuffle(keys)
    num_train = int(round(training_ratio* len(keys)))  # size of training data
    train_keys = keys[:num_train]                      # split dataset in 90% training
    val_keys = keys[num_train:]                        # and 10% validation data
    num_val = len(val_keys)                            # size of validation data
    print('# training images  : {}'.format(num_train))
    print('# validation images: {}\n'.format(num_val))
    return gt, train_keys, val_keys

def scheduler(epoch, decay=0.9, base_lr=3e-4):
    #if epoch > int(round(0.5*total_epochs)):
    #    lr = 4e-4
    #elif epoch > int(round(0.853*total_epochs)):
    #    lr = 4e-5
    #else:
    #    lr = 4e-3
    lr = base_lr * decay ** (epoch)
    #print('Learning rate: {}'.format(str(lr)))
    #export_lr_csv(lr, epoch, log_path)
    return lr

def export_lr_csv(lr, epoch, path):
    # writing
    with open(path + 'lr.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow((epoch, lr))