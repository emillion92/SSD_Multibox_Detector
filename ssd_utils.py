"""Some utils for SSD."""

import numpy as np
import tensorflow as tf
from ssd_layers import convert_coordinates

class BBoxUtility(object):
    """Utility class to do some stuff with bounding boxes and priors.
    # Arguments
        num_classes: Number of classes including background.
        priors: Priors and variances, numpy tensor of shape (num_priors, 8),
            priors[i] = [xmin, ymin, xmax, ymax, varxc, varyc, varw, varh].
        overlap_threshold: Threshold to assign box to a prior.
        nms_thresh: Nms threshold.
        top_k: Number of total bboxes to be kept per image after nms step.
    # References
        https://arxiv.org/abs/1512.02325
    """
    # TODO add setter methods for nms_thresh and top_K
    def __init__(self, num_classes, input_shape, priors=None, overlap_threshold=0.5,
                 nms_thresh=0.3, top_k=400):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.priors = priors
        self.num_priors = 0 if priors is None else len(priors)
        self.overlap_threshold = overlap_threshold
        self._nms_thresh = nms_thresh
        self._top_k = top_k
        self.boxes = tf.placeholder(dtype='float32', shape=(None, 4))
        self.scores = tf.placeholder(dtype='float32', shape=(None,))
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=self._nms_thresh)
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

    @property
    def nms_thresh(self):
        return self._nms_thresh

    @nms_thresh.setter
    def nms_thresh(self, value):
        self._nms_thresh = value
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=self._nms_thresh)

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, value):
        self._top_k = value
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=self._nms_thresh)

    def iou(self, box):
        """Compute intersection over union for the box with all priors.
        # Arguments
            box: Box, numpy tensor of shape (4,).
        # Return
            iou: Intersection over union,
                numpy tensor of shape (num_priors).
        """
        # compute intersection

        priors = self.cvt_boxes(self.priors, conversion='centroids2minmax')
        box_cvt = self.cvt_boxes(box, conversion='centroids2minmax')

        inter_upleft = np.maximum(priors[:, :2], box_cvt[:2])
        inter_botright = np.minimum(priors[:, 2:4], box_cvt[2:])
        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # compute union
        area_pred = (box_cvt[2] - box_cvt[0]) * (box_cvt[3] - box_cvt[1])
        area_gt = (priors[:, 2] - priors[:, 0])
        area_gt *= (priors[:, 3] - priors[:, 1])
        union = area_pred + area_gt - inter
        # compute iou
        iou = inter / union
        return iou

    def assign_boxes(self, boxes):
        """Assign boxes to priors for training.
        # Arguments
            boxes: Box, numpy tensor of shape (num_boxes, 4 + num_classes),
                num_classes without background.
        # Return
            assignment: Tensor with assigned boxes,
                numpy tensor of shape (num_boxes, 4 + num_classes + 8),
                priors in ground truth are fictitious,
                assignment[:, -8] has 1 if prior should be penalized
                    or in other words is assigned to some ground truth box,
                assignment[:, -7:] are all 0. See loss for more details.
        """
        assignment = np.zeros((self.num_priors, 4 + self.num_classes + 8))
        assignment[:, 4] = 1.0
        if len(boxes) == 0:
            return assignment
        centroid_boxes = self.cvt_boxes(boxes, conversion='minmax2centroids')
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, centroid_boxes[:, :4])
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]
        assign_num = len(best_iou_idx)
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx,
                                                         np.arange(assign_num),
                                                         :4]
        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 5:-8][best_iou_mask] = boxes[best_iou_idx, 4:]    # add one-hot encoded classes to assignment
        assignment[:, -8][best_iou_mask] = 1                            # set to 1 for loss function, only penalizing
                                                                        # the assigned boxes
        return assignment

    def encode_box(self, box, return_iou=True):
        """Encode box for training, do it only for assigned priors.
        # Arguments
            box: Box, numpy tensor of shape (4,).
            return_iou: Whether to concat iou to encoded values.
        # Return
            encoded_box: Tensor with encoded box
                numpy tensor of shape (num_priors, 4 + int(return_iou)).
        """
        iou = self.iou(box)
        encoded_box = np.zeros((self.num_priors, 4 + return_iou))
        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]
        assigned_priors = self.priors[assign_mask]

        # regress gt box and assigned prior box
        encoded_box[:, :2][assign_mask] = box[:2] - assigned_priors[:, :2]  # eq. (2.2) and (2.3)
        encoded_box[:, :2][assign_mask] /= assigned_priors[:, 2:4]          # eq. (2.2) and (2.3)
        # encode variance: 0.1 for box centre coordinates, 0.2 for box width and height
        encoded_box[:, :2][assign_mask] /= assigned_priors[:, -4:-2]                  # encode box centers by variance 0.1
        encoded_box[:, 2:4][assign_mask] = np.log(box[2:] / assigned_priors[:, 2:4])  # eq. (2.4) and (2.5)
        encoded_box[:, 2:4][assign_mask] /= assigned_priors[:, -2:]  # encode box centers by variance 0.2
        return encoded_box.ravel()

    def decode_boxes(self, mbox_loc, mbox_priorbox, variances):
        """Convert bboxes from local predictions to shifted priors.
        # Arguments
            mbox_loc: Numpy array of predicted locations.
            mbox_priorbox: Numpy array of prior boxes.
            variances: Numpy array of variances.
        # Return
            decode_bbox: Shifted priors.
        """
        prior_center_x = mbox_priorbox[:, 0]
        prior_center_y = mbox_priorbox[:, 1]
        prior_width = mbox_priorbox[:, 2]
        prior_height = mbox_priorbox[:, 3]

        decode_bbox_center_x = mbox_loc[:, 0] * prior_width * variances[:, 0]
        decode_bbox_center_x +=  prior_center_x

        decode_bbox_center_y = mbox_loc[:, 1] * prior_width * variances[:, 1]
        decode_bbox_center_y += prior_center_y

        decode_bbox_width = np.exp(mbox_loc[:, 2] * variances[:, 2])
        decode_bbox_width *= prior_width

        decode_bbox_height = np.exp(mbox_loc[:, 3] * variances[:, 3])
        decode_bbox_height *= prior_height

        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), axis=-1)
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)

        return decode_bbox

    def detection_out(self, predictions, background_label_id=0, keep_top_k=200,
                      confidence_threshold=0.01):
        """Do non maximum suppression (nms) on prediction results.
        # Arguments
            predictions: Numpy array of predicted values.
            num_classes: Number of classes for prediction.
            background_label_id: Label of background class.
            keep_top_k: Number of total bboxes to be kept per image
                after nms step.
            confidence_threshold: Only consider detections,
                whose confidences are larger than a threshold.
        # Return
            results: List of predictions for every picture. Each prediction is:
                [label, confidence, xmin, ymin, xmax, ymax]
        """
        mbox_loc = predictions[:, :, :4]
        variances = predictions[:, :, -4:]
        mbox_priorbox = predictions[:, :, -8:-4]
        mbox_conf = predictions[:, :, 4:-8]
        results = []
        for i in range(len(mbox_loc)):
            results.append([])
            decode_bbox = self.decode_boxes(mbox_loc[i],
                                            mbox_priorbox[i], variances[i])
            for c in range(self.num_classes):
                if c == background_label_id:
                    continue
                c_confs = mbox_conf[i, :, c]
                c_confs_m = c_confs > confidence_threshold
                if len(c_confs[c_confs_m]) > 0:
                    boxes_to_process = decode_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]
                    feed_dict = {self.boxes: boxes_to_process,
                                 self.scores: confs_to_process}
                    idx = self.sess.run(self.nms, feed_dict=feed_dict)
                    good_boxes = boxes_to_process[idx]
                    confs = confs_to_process[idx][:, None]
                    labels = c * np.ones((len(idx), 1))
                    c_pred = np.concatenate((labels, confs, good_boxes),
                                            axis=1)
                    results[-1].extend(c_pred)
            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                argsort = np.argsort(results[-1][:, 1])[::-1]
                results[-1] = results[-1][argsort]
                results[-1] = results[-1][:keep_top_k]
        return results

    def cvt_boxes(self, boxes, conversion):
        converted_boxes = np.copy(boxes)
        if boxes.size > 4:
            if conversion=='minmax2centroids':
                converted_boxes[:, 2] = (boxes[:, 2] - boxes[:, 0]) #w
                converted_boxes[:, 3] = (boxes[:, 3] - boxes[:, 1]) #h
                converted_boxes[:, 0] = boxes[:, 0]+ (converted_boxes[:, 2] / 2)  #cx
                converted_boxes[:, 1] = boxes[:, 1]+ (converted_boxes[:, 3] / 2)  #cy
            elif conversion == 'centroids2minmax':
                converted_boxes[:, 0] = boxes[:, 0] - 0.5*boxes[:, 2] # xmin
                converted_boxes[:, 1] = boxes[:, 1] - 0.5*boxes[:, 3] # ymin
                converted_boxes[:, 2] = boxes[:, 0] + 0.5*boxes[:, 2] # xmax
                converted_boxes[:, 3] = boxes[:, 1] + 0.5*boxes[:, 3] # ymax
            else:
                print('Invalid conversion selected! Only "minmax2centroids" and "centroids2minmax" are supportet as conversion arguments!')
        else:
            if conversion=='minmax2centroids':
                converted_boxes[2] = (boxes[2] - boxes[0]) #w
                converted_boxes[3] = (boxes[3] - boxes[1]) #h
                converted_boxes[0] = boxes[0]+ (converted_boxes[2] / 2)  #cx
                converted_boxes[1] = boxes[1]+ (converted_boxes[3] / 2)  #cy
            elif conversion == 'centroids2minmax':
                converted_boxes[0] = boxes[0] - 0.5*boxes[2] # xmin
                converted_boxes[1] = boxes[1] - 0.5*boxes[3] # ymin
                converted_boxes[2] = boxes[0] + 0.5*boxes[2] # xmax
                converted_boxes[3] = boxes[1] + 0.5*boxes[3] # ymax
            else:
                print('Invalid conversion selected! Only "minmax2centroids" and "centroids2minmax" are supportet as conversion arguments!')
                quit()
        return converted_boxes

def load_model_configurations(model):
    model_configurations = []
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        if layer_type == 'PriorBox':
            layer_data = {}
            layer_data['img_height'] = layer.img_height
            layer_data['img_width'] = layer.img_width
            layer_data['layer_width'] = layer.input_shape[1]
            layer_data['layer_height'] = layer.input_shape[2]
            layer_data['this_scale'] = layer.this_scale
            layer_data['next_scale'] = layer.next_scale
            layer_data['aspect_ratios'] = layer.aspect_ratios
            layer_data['two_boxes_for_ar1'] = layer.two_boxes_for_ar1
            layer_data['limit_boxes'] = layer.limit_boxes
            layer_data['variances'] = layer.variances
            model_configurations.append(layer_data)
    return model_configurations

def create_prior_boxes(model):
    boxes_parameters = []
    model_configurations = load_model_configurations(model)
    for layer_config in model_configurations:
        img_height = layer_config['img_height']
        img_width = layer_config['img_width']
        feature_map_height = layer_config['layer_height']
        feature_map_width = layer_config['layer_width']
        this_scale = layer_config['this_scale']
        next_scale = layer_config['next_scale']
        aspect_ratios = layer_config['aspect_ratios']
        two_boxes_for_ar1 = layer_config['two_boxes_for_ar1']
        limit_boxes = layer_config['limit_boxes']
        variances = layer_config['variances']

        if two_boxes_for_ar1:
            n_boxes = len(aspect_ratios)+1
        else:
            n_boxes = len(aspect_ratios)

        aspect_ratios = np.sort(aspect_ratios)
        size = min(img_height, img_width)
        # Compute the box widths and and heights for all aspect ratios
        wh_list = []
        for ar in aspect_ratios:
            if (ar == 1) & two_boxes_for_ar1:
                # Compute the regular default box for aspect ratio 1 and...
                w = this_scale * size * np.sqrt(ar)
                h = this_scale * size / np.sqrt(ar)
                wh_list.append((w, h))
                # ...also compute one slightly larger version using the geometric mean of this scale value and the next
                w = np.sqrt(this_scale * next_scale) * size * np.sqrt(ar)
                h = np.sqrt(this_scale * next_scale) * size / np.sqrt(ar)
                wh_list.append((w, h))
            else:
                w = this_scale * size * np.sqrt(ar)
                h = this_scale * size / np.sqrt(ar)
                wh_list.append((w, h))
        wh_list = np.array(wh_list)

        # Compute the grid of box center points. They are identical for all aspect ratios
        cell_height = img_height / feature_map_height
        cell_width = img_width / feature_map_width
        cx = np.linspace(cell_width / 2, img_width - cell_width / 2, feature_map_width)
        cy = np.linspace(cell_height / 2, img_height - cell_height / 2, feature_map_height)
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1)  # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1)  # This is necessary for np.tile() to do what we want further down

        # Create a 4D tensor template of shape `(n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        prior_boxes = np.zeros((feature_map_height, feature_map_width, n_boxes, 4))

        prior_boxes[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes))  # Set cx
        prior_boxes[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes))  # Set cy
        prior_boxes[:, :, :, 2] = wh_list[:, 0]  # Set w
        prior_boxes[:, :, :, 3] = wh_list[:, 1]  # Set h

        # Convert `(cx, cy, w, h)` to `(xmin, xmax, ymin, ymax)`
        boxes_tensor = convert_coordinates(prior_boxes, start_index=0, conversion='centroids2minmax')

        # If `limit_boxes` is enabled, clip the coordinates to lie within the image boundaries
        if limit_boxes:
            x_coords = boxes_tensor[:, :, :, [0, 1]]
            x_coords[x_coords >= img_width] = img_width - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:, :, :, [0, 1]] = x_coords
            y_coords = boxes_tensor[:, :, :, [2, 3]]
            y_coords[y_coords >= img_height] = img_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:, :, :, [2, 3]] = y_coords

        boxes_tensor[:, :, :, :2] /= img_width
        boxes_tensor[:, :, :, 2:] /= img_height

        # TODO: Implement box limiting directly for `(cx, cy, w, h)` so that we don't have to unnecessarily convert back and forth
        # Convert `(xmin, xmax, ymin, ymax)` back to `(cx, cy, w, h)`
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='minmax2centroids')
        boxes_tensor = boxes_tensor.reshape(-1, 4)
        # 4: Create a tensor to contain the variances and append it to `boxes_tensor`. This tensor has the same shape
        #    as `boxes_tensor` and simply contains the same 4 variance values for every position in the last axis.
        variances_tensor = np.zeros_like(
            boxes_tensor)  # Has shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        variances_tensor += variances  # Long live broadcasting
        # Now `boxes_tensor` becomes a tensor of shape `(feature_map_height, feature_map_width, n_boxes, 8)`
        boxes_tensor = np.concatenate((boxes_tensor, variances_tensor), axis=-1)

        boxes_parameters.append(boxes_tensor)

    return np.concatenate(boxes_parameters, axis=0)
