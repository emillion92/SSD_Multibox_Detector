import cv2
import keras
import random
import numpy as np
from scipy.misc import imresize
from keras.applications.imagenet_utils import preprocess_input

from dataset_utils import vizualize_img
from dataset_utils import vizualize_gtassigned_boxes

class BatchGenerator(object):
    def __init__(self, input_size, gt, gt_format,bbox_util,
                 batch_size,
                 train_keys, val_keys,
                 priors,
                 satur_prob = 0.5,
                 brightn_prob = 0.5,
                 contr_prob = 0.5,
                 lighting_prob = 0.5,
                 crop_prob = 0.5,
                 scale_prob = 0.5,
                 blur_prob = 0.5,
                 split_prob = 0.5,
                 rotate_prob = 0.5,
                 hflip_prob = 0.5,
                 vflip_prob = 1,
                 aug_prob = 0.5,
                 do_crop=False,
                 do_blur=False,
                 do_scale=False,
                 do_noise=False,
                 do_split=False,
                 do_plot=False,
                 do_rotate=False,
                 crop_area_range=[0.7, 1.0],
                 rot_ang_range=[-3, 3],
                 aspect_ratio_range=[1./2., 2.]):
        self.input_size = input_size
        self.gt = gt
        self.bbox_util = bbox_util
        self.batch_size = batch_size
        self.train_keys = train_keys
        self.val_keys = val_keys
        self.train_batches = len(train_keys)
        self.val_batches = len(val_keys)
        self.priors = priors
        self.gt_format = gt_format
        self.color_jitter = []
        if satur_prob:
            self.saturation_var = satur_prob
            self.color_jitter.append(self.saturation)
        if brightn_prob:
            self.brightness_var = brightn_prob
            self.color_jitter.append(self.brightness)
        if contr_prob:
            self.contrast_var = contr_prob
            self.color_jitter.append(self.contrast)
        if lighting_prob:
            self.lighting_prob = lighting_prob
            self.color_jitter.append(self.lighting)
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.do_crop = do_crop
        self.crop_prob = crop_prob
        self.do_scale = do_scale
        self.scale_prob = scale_prob
        self.do_blur = do_blur
        self.blur_prob = blur_prob
        self.do_noise = do_noise
        self.do_split = do_split
        self.do_rotate = do_rotate
        self.split_prob = split_prob
        self.crop_area_range = crop_area_range
        self.rot_ang_range = rot_ang_range
        self.rotate_prob = rotate_prob
        self.aspect_ratio_range = aspect_ratio_range
        self.aug_prob = aug_prob
        self.do_plot = do_plot
        self.counter = -1

    def saturation(self, img):
        if (1- np.random.random()) > self.saturation_var:
            rand_contrast_scale = random.uniform(0.8, 1.2)
            img_hsv = cv2.cvtColor(img, code=cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(img_hsv)
            s = s * rand_contrast_scale
            s[np.where(v > 255)] = 255
            s = np.array(s, dtype='f')
            s = s.astype(np.uint8)  # make the data types of h,s and v same
            final_hsv = cv2.merge((h, s, v))
            img = cv2.cvtColor(final_hsv, code=cv2.COLOR_HSV2BGR)
        return np.clip(img.astype('uint8'), 0, 255)

    def brightness(self, img):
        if (1- np.random.random()) > self.brightness_var:
            rand_brightness_scale = random.uniform(0.8, 1.2)
            img_hsv = cv2.cvtColor(img, code=cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(img_hsv)
            v = v * rand_brightness_scale
            v[np.where(v > 255)] = 255
            v = np.array(v, dtype='f')
            v = v.astype(np.uint8)  # make the data types of h,s and v same
            final_hsv = cv2.merge((h, s, v))
            img = cv2.cvtColor(final_hsv, code=cv2.COLOR_HSV2BGR)
        return np.clip(img.astype('uint8'), 0, 255)

    def contrast(self, img):
        if (1- np.random.random()) > self.contrast_var:
            img = img.astype('int16')
            alpha = random.uniform(0.8,1.2)
            beta = random.randint(-60, 60)
            mul_img = cv2.multiply(img, np.array([alpha]))  # mul_img = img*alpha
            img = cv2.add(mul_img, (np.ones(img.shape)*beta).astype('int16'))
            img[np.where(img > 255)] = 255
            img[np.where(img < 0)] = 0
        return img.astype('uint8')

    def lighting(self, img):
        if (1- np.random.random()) > self.lighting_prob:
            img = img.astype('float32')
            row, col, ch = img.shape
            mean = 0
            var = 0.1
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            #img = img + gauss
        return np.clip(img.astype('uint8'), 0, 255)

    def blur(self, img):
        if (1- np.random.random()) > self.blur_prob:
            random_kernel_size = 3
            img = cv2.blur(img,(random_kernel_size, random_kernel_size))
        return img

    def rotate(self, img, targets):
        if (1- np.random.random()) > self.rotate_prob:
            rand_angle = random.randint(self.rot_ang_range[0], self.rot_ang_range[1])

            if rand_angle != 0:
                (img_h,img_w) = img.shape[:2]
                (cX, cY) = (img_w // 2, img_h // 2)

                # grab the rotation matrix (applying the negative of the
                # angle to rotate clockwise), then grab the sine and cosine
                # (i.e., the rotation components of the matrix)
                M = cv2.getRotationMatrix2D((cX, cY), rand_angle, 1.0)
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])

                # compute the new bounding dimensions of the image
                nW = int((img_h * sin) + (img_w * cos))
                nH = int((img_h * cos) + (img_w * sin))

                # adjust the rotation matrix to take into account translation
                M[0, 2] += (nW / 2) - cX
                M[1, 2] += (nH / 2) - cY

                # perform the actual rotation and return the image
                img = cv2.warpAffine(img, M, (nW, nH))

                offset_x = int(np.round(0.5 * (nW - img_w)))
                offset_y = int(np.round(0.5 * (nH - img_h)))

                dx = (nW - img_w) / float(nW)
                dy = (nH - img_h) / float(nH)

                img = img[offset_y:offset_y+img_h, offset_x:offset_x+img_w]

                new_targets = []
                for box in targets:
                    xmin = box[0]
                    ymin = box[1]
                    xmax = box[2]
                    ymax = box[3]

                    lever = 1
                    # clockwise and bb on left side
                    if rand_angle > 0 and xmin < 0.5:
                        lever = (0.5 - xmin) / 0.5
                        xmin += xmin * lever * dx
                        ymin += ymin * lever * dy
                        xmax += xmax * lever * dx
                        ymax += ymax * lever * dy
                    # clockwise and bb on right side
                    elif rand_angle > 0 and xmin > 0.5:
                        lever = (0.5 -(1.0 - xmin)) / 0.5
                        xmin -= xmin * lever * dx
                        ymin -= ymin * lever * dy
                        xmax -= xmax * lever * dx
                        ymax -= ymax * lever * dy
                    # counter clockwise and bb on left side
                    elif rand_angle < 0 and xmin < 0.5:
                        lever = (0.5 - xmin) / 0.5
                        xmin -= xmin * lever * dx
                        ymin -= ymin * lever * dy
                        xmax -= xmax * lever * dx
                        ymax -= ymax * lever * dy
                    # counter clockwise and bb on right side
                    if rand_angle < 0 and xmin > 0.5:
                        lever = (0.5 -(1.0 - xmin)) / 0.5
                        xmin += xmin * lever * dx
                        ymin += ymin * lever * dy
                        xmax += xmax * lever * dx
                        ymax += ymax * lever * dy

                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(1, xmax)
                    ymax = min(1, ymax)

                    #if xmin < 0.5:
                    #    xmin = max(0 + dx, xmin)
                    #    ymin = max(0 + dy, ymin)
                    #    xmax = min(1, xmax)
                    #    ymax = min(1, ymax)
                    #else:
                    #    xmin = max(0, xmin)
                    #    ymin = max(0, ymin)
                    #    xmax = min(1 - dx, xmax)
                    #    ymax = min(1 - dy, ymax)

                    box[:4] = [xmin, ymin, xmax, ymax]
                    new_targets.append(box)

                new_targets = np.asarray(new_targets).reshape(-1, targets.shape[1])

                return img, new_targets
            else:
                return img, targets
        #plot_generator_img(img, y)
        return img, targets

    def horizontal_flip(self, img, y):
        if (1- np.random.random()) > self.hflip_prob:
            img = img[:, ::-1]
            y[:, [0, 2]] = 1 - y[:, [2, 0]]
        return img, y

    def vertical_flip(self, img, y):
        if (1- np.random.random()) > self.vflip_prob:
            img = img[::-1]
            y[:, [1, 3]] = 1 - y[:, [3, 1]]
        return img, y

    def random_sized_crop(self, img, boxes):
        if (1- np.random.random()) > self.crop_prob:
            (img_h, img_w) = img.shape[:2]
            img_area = img_w * img_h
            random_scale = np.random.random()
            random_scale *= (self.crop_area_range[1] -
                             self.crop_area_range[0])
            random_scale += self.crop_area_range[0]
            target_area = random_scale * img_area
            random_ratio = np.random.random()
            random_ratio *= (self.aspect_ratio_range[1] -
                             self.aspect_ratio_range[0])
            random_ratio += self.aspect_ratio_range[0]
            w = np.round(np.sqrt(target_area * random_ratio))
            h = np.round(np.sqrt(target_area / random_ratio))
            if np.random.random() < 0.5:
                w, h = h, w
            w = min(w, img_w)
            w_rel = w / img_w
            w = int(w)
            h = min(h, img_h)
            h_rel = h / img_h
            h = int(h)
            x = np.random.random() * (img_w - w)
            x_rel = x / img_w
            x = int(x)
            y = np.random.random() * (img_h - h)
            y_rel = y / img_h
            y = int(y)
            img = img[y:y + h, x:x + w]
            new_targets = []
            for box in boxes:
                cx = 0.5 * (box[0] + box[2])
                cy = 0.5 * (box[1] + box[3])
                if (x_rel < cx < x_rel + w_rel and y_rel < cy < y_rel + h_rel):
                    xmin = (box[0] - x_rel) / w_rel
                    ymin = (box[1] - y_rel) / h_rel
                    xmax = (box[2] - x_rel) / w_rel
                    ymax = (box[3] - y_rel) / h_rel
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(1, xmax)
                    ymax = min(1, ymax)
                    box[:4] = [xmin, ymin, xmax, ymax]
                    new_targets.append(box)
            boxes = np.asarray(new_targets).reshape(-1, boxes.shape[1])
            return img, boxes
        else:
            return img, boxes

    def scale(self, img, y):
        if (1- np.random.random()) > self.scale_prob:
            random_scale = random.uniform(0.8, 1.5)

            img_w = img.shape[1]
            img_h = img.shape[0]

            resized_image = cv2.resize(img,
                                       dsize=(int(np.round(random_scale * img_w)), int(np.round(random_scale * img_h))),
                                       interpolation=cv2.INTER_CUBIC)

            img_w_res = float(resized_image.shape[1])
            img_h_res = float(resized_image.shape[0])

            offset_x = int(np.round(0.5 * (img_w_res - img_w)))
            offset_y = int(np.round(0.5 * (img_h_res - img_h)))

            #img_mean = np.mean(img, axis=(0, 1))

            if random_scale <= 1:
                img = cv2.copyMakeBorder(resized_image,
                                                   top=int(np.ceil(0.5 * (img_h - img_h_res))),
                                                   bottom=int(np.floor(0.5 * (img_h - img_h_res))),
                                                   left=int(np.ceil(0.5 * (img_w - img_w_res))),
                                                   right=int(np.floor(0.5 * (img_w - img_w_res))),
                                                   borderType=cv2.BORDER_CONSTANT)

            else:
                img = resized_image[offset_y:offset_y + img_h, offset_x:offset_x + img_w]

            new_targets = []
            if y.size:
                for box in y:

                    # orig relative gt bb coordinates
                    bb_xmin = box[0]
                    bb_ymin = box[1]
                    bb_xmax = box[2]
                    bb_ymax = box[3]

                    # orig relative gt bb centers
                    cx_rel = 0.5 * (bb_xmin + bb_xmax)
                    cy_rel = 0.5 * (bb_ymin + bb_ymax)

                    x_rel = offset_x / img_w_res
                    y_rel = offset_y / img_h_res

                    w_rel = img_w / img_w_res
                    h_rel = img_h / img_h_res

                    if (x_rel < cx_rel < x_rel + w_rel and y_rel < cy_rel < y_rel + h_rel):

                        xmin = (bb_xmin - x_rel) / w_rel
                        ymin = (bb_ymin - y_rel) / h_rel
                        xmax = (bb_xmax - x_rel) / w_rel
                        ymax = (bb_ymax - y_rel) / h_rel

                        bb_xmin = max(0, xmin)
                        bb_ymin = max(0, ymin)
                        bb_xmax = min(1, xmax)
                        bb_ymax = min(1, ymax)
                        box[:4] = [bb_xmin, bb_ymin, bb_xmax, bb_ymax]
                        new_targets.append(box)

            y = np.asarray(new_targets).reshape(-1, y.shape[1])

        return img, y

    def random_noise(self, img, gt, img_noise):

        width, height, channels = img.shape
        img_noise_width, img_noise_heigth, img_noise_channels = img_noise.shape
        img_noise_copy = img_noise.copy()

        rnd_x = max(0, random.randint(0, img_noise_width) - width)
        rnd_y = max(0, random.randint(0, img_noise_heigth) - height)

        noise_crop = img_noise_copy[rnd_y:rnd_y+height, rnd_x:rnd_x+width]

        if np.random.random() > 0.5:
            # horizontal flip
            noise_crop = noise_crop[:, ::-1]
        if np.random.random() > 0.5:
            # vertical flip
            noise_crop = noise_crop[::-1]

        for box in gt:
            # orig relative gt bb coordinates
            bb_xmin = box[0] * width
            bb_ymin = box[1] * height
            bb_xmax = box[2] * width
            bb_ymax = box[3] * height

            pad_x = random.randint(2,15)
            pad_y = random.randint(2,15)

            y = max(0, int(bb_ymin)-pad_y)
            x = max(0, int(bb_xmin)-pad_x)
            h = min(height, int(bb_ymax - bb_ymin) + 2*pad_y)
            w = min(width, int(bb_xmax - bb_xmin) + 2*pad_x)
            noise_crop[y:y+h, x:x+w] = img[y:y + h, x:x + w]

        return noise_crop

    def split_img(self, img, boxes):
        if (1- np.random.random()) > self.split_prob:
            part = np.random.random()
            height, width,_ = img.shape
            step_width = int(width / 3)
            y = 0
            h = height
            w = step_width
            if part <= 0.25:
                x = 0
            elif part >= 0.75:
                x = 2*step_width
            else:
                x = step_width

            img = img[y:y + h, x:x + w]

            img_w_res = float(img.shape[1])

            new_targets = []
            if boxes.size:
                for box in boxes:
                    cx = 0.5 * (box[0] + box[2])
                    x_rel = x / float(width)
                    w_rel = img_w_res / width

                    if (x_rel < cx < x_rel + w_rel):
                        xmin = (box[0] - x_rel) / w_rel
                        xmax = (box[2] - x_rel) / w_rel

                        xmin = max(0, xmin)
                        ymin = box[1]
                        xmax = min(1, xmax)
                        ymax = box[3]
                        box[:4] = [xmin, ymin, xmax, ymax]
                        new_targets.append(box)

                boxes = np.asarray(new_targets).reshape(-1, boxes.shape[1])

        return img, boxes

    def preprocess_input(self, input):

        data_format = keras.backend.image_data_format()
        assert data_format in {'channels_last', 'channels_first'}

        if data_format == 'channels_first':
            # 'RGB'->'BGR'
            input = input[:, ::-1, :, :]
            # Zero-center by mean pixel
            input[:, 0, :, :] -= 127
            input[:, 1, :, :] -= 127
            input[:, 2, :, :] -= 127
        else:
            # 'RGB'->'BGR'
            input = input[:, :, :, ::-1]
            # Zero-center by mean pixel
            input[:, :, :, 0] -= 127
            input[:, :, :, 1] -= 127
            input[:, :, :, 2] -= 127

        #if data_format == 'channels_first':
            # 'RGB'->'BGR'
            #tmp_inp = input[:, ::-1, :, :]
            # Zero-center by mean pixel
            #input[:, 0, :, :] = input[:, 0, :, :] / 127
            #input[:, 1, :, :] = input[:, 1, :, :] / 127
            #input[:, 2, :, :] = input[:, 2, :, :] / 127
        #else:
            # 'RGB'->'BGR'
            #input = input[:, :, :, ::-1]
            # Zero-center by mean pixel
            #input[:, :, :, 0] = input[:, :, :, 0] / 127
            #input[:, :, :, 1] = input[:, :, :, 1] / 127
            #input[:, :, :, 2] = input[:, :, :, 2] / 127

        return input

    def generate(self, train=True, aug=False):

        if self.do_noise:
            img_noise = []
            width, height , channels = (1000, 1000, 3)
            nr_pixels = width * height
            for i in range(0, nr_pixels):
                    # randomize indices between 0 & 255
                    img_noise.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255),))
            img_noise = np.reshape(img_noise, (width, height, channels)).astype('uint8')

        while True:
            if train:
                if self.counter < self.train_batches-1 and self.counter < len(self.train_keys)-1:
                    self.counter += self.batch_size
                else:
                    random.shuffle(self.train_keys)
                    self.counter = 0
                #print('\nCounter: ' + str(self.counter))
                keys = self.train_keys[self.counter:self.counter + self.batch_size]
            else:
                random.shuffle(self.val_keys)
                keys = self.val_keys
            inputs = []
            targets = []
            for key in keys:
                try_aug = True
                img_path = key
                img = cv2.imread(img_path)
                y = self.gt[key].copy()
                if self.do_plot:
                    vizualize_img(img, y, bb_format=self.gt_format, img_path=img_path)
                if train:
                    tries = 0
                    img_ = cv2.imread(img_path)
                    while try_aug and tries < 50:
                        img = img_
                        y = self.gt[key].copy()
                        if aug:
                            if self.do_split:
                                if "KITTI" in img_path:
                                    img, y = self.split_img(img, y)
                            if np.random.random() > self.aug_prob:
                                random.shuffle(self.color_jitter)
                                if self.do_scale:
                                    img, y = self.scale(img, y)
                                if self.do_crop:
                                    img, y = self.random_sized_crop(img, y)
                                if self.do_rotate:
                                    img, y = self.rotate(img, y)
                                img = imresize(img, self.input_size)
                                for jitter in self.color_jitter:
                                    img = jitter(img)
                                if self.do_blur:
                                    img = self.blur(img)
                                if self.hflip_prob > 0:
                                    img, y = self.horizontal_flip(img, y)
                                if self.vflip_prob > 0:
                                    img, y = self.vertical_flip(img, y)
                                #img = self.random_noise(img, y, img_noise)
                                if bool(np.any(y)):
                                    try_aug = False
                                else:
                                    tries += 1
                        else:
                            try_aug=False
                img = imresize(img, self.input_size).astype('float32')
                if self.do_plot:
                    vizualize_img(img, y, bb_format=self.gt_format, img_path=img_path)
                if not train:
                    img = imresize(img, self.input_size).astype('float32')
                assigned_boxes = self.bbox_util.assign_boxes(y)
                if self.do_plot:
                    vizualize_gtassigned_boxes(img, y, assigned_boxes, self.priors, img_path)
                img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
                inputs.append(img)
                targets.append(assigned_boxes)
                del img, assigned_boxes
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    prep_input = preprocess_input(tmp_inp)
                    if self.do_plot:
                        vizualize_img(prep_input[-1], y, bb_format=self.gt_format, img_path=img_path)
                    yield prep_input, tmp_targets
