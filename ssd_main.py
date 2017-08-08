import keras
import keras.backend as K
import datetime
import random
from keras import metrics

from ssd_networks import ssd300, ssd512
from ssd_training import MultiboxLoss_old
from ssd_training import export_lr_csv
from ssd_utils import BBoxUtility
from ssd_utils import create_prior_boxes
from ssd_generator import BatchGenerator

from dataset_utils import read_gts
from dataset_utils import check_paths
from dataset_utils import cpu_gpu_swith
from ssd_evaluation import evaluate_model

now = datetime.datetime.now()
#random.seed(1)
debug = False

# path variables
path = './training_with_new_weights/SSD300/own_implementation/udacity/no_bg_gtfile/'
training_path = path + 'training_{}'.format(now.strftime("%Y%m%d_%H:%M"))
checkpoint_path = training_path + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
log_path = training_path + '/logs/'

check_paths(paths=[path, training_path, log_path])

classes = ['background', 'Vehicle']

# load ground truth information from dataset in form of pickle file
gt_path = './gts/'
gt_files = [#'kitti_backgroundCar_min40_medium_minmax.pkl']#,
            #'udacity_backgroundVehicle_min40_all_minmax.pkl']#,
            #'caltech_Pedestrian_all_minmax.pkl']#,
            #'kitti_Car_min40_all_minmax.pkl']#,
            'udacity_Vehicle_min60_all_minmax.pkl']
gt, gt_format, train_keys, num_train, val_keys, num_val = read_gts(gt_files, gt_path, ratio=0.90, train=True)

# some constants
NUM_CLASSES = len(classes)
batch_size = 8
base_lr = 1e-5
decay = 5e-4
iterations = 150000
epochs = 100
steps_per_epoch = (num_train + num_val)/batch_size
USE_CPU = False

cpu_gpu_swith(USE_CPU)

freeze = ['conv1_1', 'conv1_2', 'pool1',
          'conv2_1', 'conv2_2', 'pool2',
          'conv3_1', 'conv3_2', 'conv3_3', 'pool3',
          'conv4_1', 'conv4_2', 'conv4_3', 'pool4']#,
          #'conv5_1', 'conv5_2', 'conv5_3', 'pool5']

model = ssd300(num_classes=NUM_CLASSES, weights_path='SSD300_core.hdf5', min_scale=0.05, max_scale=0.7, aspect_ratios_global=[0.33, 0.5, 1, 2, 3])
#model = ssd512(num_classes=NUM_CLASSES, weights_path='SSD512_core.hdf5', frozen_layers=freeze, min_scale=0.05, max_scale=0.7)#, aspect_ratios_global=[0.33, 0.5, 1, 2, 3])

priors = create_prior_boxes(model)

bbox_util = BBoxUtility(NUM_CLASSES, (model.input_shape[1:3]), priors, overlap_threshold=.5)

gen = BatchGenerator((model.input_shape[1:3]), gt, gt_format, bbox_util, batch_size,
                train_keys, val_keys, priors,
                do_crop=True, do_blur=False, do_scale=True,
                do_rotate=False, do_split=True, do_plot=False)

def schedule(epoch, decay=0.9):
    lr = base_lr * decay ** (epoch)
    print('Learning rate: {}'.format(str(lr)))
    export_lr_csv(lr, epoch, log_path)
    return lr

def loc_acc(y_true, y_pred):
    return K.mean(K.equal(y_true[:, :, 4:4+NUM_CLASSES], y_pred[:, :, 4:4+NUM_CLASSES]), axis=-1)

def cls_acc(y_true, y_pred):
    return K.mean(K.equal(y_true[:, :, :4], y_pred[:, :, :4]), axis=-1)

if debug:
    callbacks = []
else:
    callbacks = [keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, save_best_only=True),
                 keras.callbacks.LearningRateScheduler(schedule),
                 keras.callbacks.TensorBoard(log_dir=log_path+'tensorboard_training_{}'.format(now.strftime("%Y%m%d_%H:%M")), histogram_freq=1, write_graph=True),
                 keras.callbacks.CSVLogger(log_path+'log.csv', separator=';', append=True),
                 keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=0,epsilon=0.001,cooldown=0)
                 ]

#optim = keras.optimizers.Adam(lr=base_lr)
optim = keras.optimizers.SGD(lr=base_lr, momentum=0.9, decay=decay, nesterov=True)

model.compile(optimizer=optim,loss=MultiboxLoss_old(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss)

model.fit_generator(gen.generate(train=True, aug=True),
                    steps_per_epoch,
                    epochs, verbose=1,
                    callbacks=callbacks,
                    validation_data=gen.generate(train=False, aug=False),
                    validation_steps=gen.val_batches,
                    initial_epoch=0)

#new_weights = np.array(sorted(glob.glob(path + '*.hdf5')))
#model.load_weights(new_weights[-1])

#evaluate_model(model, save_path=path+ 'evaluation/results/txt/data/', classes=classes, img_test=img_test, conf_thresh=0.75, plot=False, save=True)
