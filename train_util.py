import numpy as np
import pandas as pd
import os

from tqdm import tqdm

from datetime import datetime as dt
from keras.wrappers.scikit_learn import KerasClassifier
from keras import optimizers

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from keras.models import Model
from keras.models import load_model

from keras.callbacks import LambdaCallback
from operator import itemgetter
from skimage.transform import resize
import cv2
from glob import glob
from keras import backend as K
from keras.losses import binary_crossentropy
import keras

import tensorflow as tf

# from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/61720
BAD_MASKS =[
'1eaf42beee','33887a0ae7','33dfce3a76','3975043a11','39cd06da7d','483b35d589','49336bb17b','4ef0559016','4fbda008c7','4fdc882e4b','50d3073821','53e17edd83','5b217529e7','5f98029612','608567ed23','62aad7556c','62d30854d7','6460ce2df7','6bc4c91c27','7845115d01','7deaf30c4a','80a458a2b6','81fa3d59b8','8367b54eac','849881c690','876e6423e6','90720e8172','916aff36ae','919bc0e2ba','a266a2a9df','a6625b8937','a9ee40cf0d','aeba5383e4','b63b23fdc9','baac3469ae','be7014887d','be90ab3e56','bfa7ee102e','bfbb9b9149','c387a012fc','c98dfd50ba','caccd6708f','cb4f7abe67','d0bbe4fd97','d4d2ed6bd2','de7202d286','f0c401b64b','f19b7d20bb','f641699848','f75842e215','00950d1627','0280deb8ae','06d21d76c4','09152018c4','09b9330300','0b45bde756','130229ec15','15d76f1672','182bfc6862','23afbccfb5','24522ec665','285f4b2e82','2bc179b78c','2f746f8726','3cb59a4fdc','403cb8f4b3','4f5df40ab2','50b3aef4c4','52667992f8','52ac7bb4c1','56f4bcc716','58de316918','640ceb328a','71f7425387','7c0b76979f','7f0825a2f0','834861f1b6','87afd4b1ca','88a5c49514','9067effd34','93a1541218','95f6e2b2d1','96216dae3b','96523f824a','99ee31b5bc','9a4b15919d','9b29ca561d','9eb4a10b98','ad2fa649f7','b1be1fa682','b24d3673e1','b35b1b412b','b525824dfc','b7b83447c4','b8a9602e21','ba1287cb48','be18a24c49','c27409a765','c2973c16f1','c83d9529bd','cef03959d8','d4d34af4f7','d9a52dc263','dd6a04d456','ddcb457a07','e12cd094a6','e6e3e58c43','e73ed6e7f2','f6e87c1458','f7380099f6','fb3392fee0','fb47e8e74e','febd1d2a67']

def compute_iou(y_true, y_pred):
    '''
    compute iou with consiferation of no salt mask

    Arg
    --------------------------
        y_true: np array (h, w, 1) with value (0, 1)
        y_pred: np array (h, w, 1) with value (0, 1)

    Ret
    --------------------------
        iou: 0.0 ~ 1.0 float
    '''
    true_objects = 2
    pred_objects = 2

    # deal with no salt mask
    if np.max(y_true) == 0:
        if np.max(y_pred) == 1:
            return 0.
        elif np.max(y_pred) == 0:
            return 1.

    intersection = np.histogram2d(y_true.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(y_true, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union
    return iou[0][0]
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def iou_coef(y_true, y_pred, smooth=0.0001, thresh=0.5):
    """
    IoU = (|X & Y|)/ (|X or Y|)
    """
    # y_true = tf.to_int32(y_true)
    # y_pred = np.int32(y_pred > K.variable(thresh))
    y_pred = K.cast(K.greater(y_pred, thresh), K.floatx())
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true,[1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    return (intersection + smooth) / ( union + smooth)

# it looks like this is not right..
# def mean_iou(y_true, y_pred):
#     prec = []
#     for t in np.arange(0.5, 1.0, 0.05):
#         y_pred_ = tf.to_int32(y_pred > t)
#         score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
#         K.get_session().run(tf.local_variables_initializer())
#         with tf.control_dependencies([up_opt]):
#             score = tf.identity(score)
#         prec.append(score)
#     return K.mean(K.stack(prec), axis=0)
def mean_iou_threshold(y_true, y_pred):
    '''
    ref:https://www.kaggle.com/pestipeti/explanation-of-scoring-metric
    This is for kaggle metrics for PB
    '''
    prec_list = []
    iou = iou_coef(y_true, y_pred)
    for t in np.arange(0.5, 1.0, 0.05):
        prec = K.mean(K.cast(K.greater(iou, t), K.floatx()))
        prec_list.append(prec)
    return K.mean(K.stack(prec_list), axis=0)
def dice_p_bce(in_gt, in_pred):
    """combine DICE and BCE"""
    # return 0.01*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)
    return binary_crossentropy(in_gt, in_pred) + 1 - dice_coef(in_gt, in_pred)

def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)

def get_data(data_type='train', size=(101, 101, 3)):
    base_image_path = './data/{}/images/*.png'.format(data_type)
    base_depth_path = './data/depth_prop_cls.csv'
    depth_csv = pd.read_csv(base_depth_path)
    paths = glob(base_image_path)

    # exclude all bad mask ids
    if data_type == 'train':
        not_bad_masks = []
        for p in paths:
            if p.split('/')[-1].split('.')[0] not in BAD_MASKS:
                not_bad_masks.append(p)
        paths = not_bad_masks


    X = np.zeros((len(paths), size[0], size[1], size[2]), dtype=np.float32)
    X_feat = np.zeros((len(paths), 1), dtype=np.float32)

#     for mask
    base_mask_path = './data/train/masks/'
    Y = np.zeros((len(paths), size[0], size[1], 1), dtype=np.float32)
    Y_no_salt = []
    Y_salt_cls = []
    for i, p in tqdm(enumerate(paths), total=len(paths)):
        im_id = p.split('/')[-1]
        im = cv2.imread(p)
        im = cv2.resize(im, (size[0], size[1]))
        im = np.array(im, dtype='float32')
        im /= 255
        X[i] = im
        X_feat[i] = depth_csv[depth_csv['id']==im_id.split('.')[0]]['z'].iloc[0]
        if data_type == 'train':
            mask_path = base_mask_path + im_id
            mask = cv2.imread(mask_path)
            mask = cv2.resize(mask, (size[0], size[1]))
            mask = np.array(mask, dtype='float32')
            mask = mask / 255
            Y[i] = resize(mask[:, :, 0], (size[0], size[1], 1))
            Y_no_salt.append(depth_csv[depth_csv['id']==im_id.split('.')[0]]['no_salt'].iloc[0])
            Y_salt_cls.append(depth_csv[depth_csv['id']==im_id.split('.')[0]]['salt_propotion_class'].iloc[0])
    Y_no_salt = np.array(Y_no_salt, dtype=np.uint8)
    return X, X_feat, Y, Y_no_salt, Y_salt_cls

def norm_X_feat(X_feat_train, X_feat_valid):

    mean = X_feat_train.mean(axis=0, keepdims=True)
    std = X_feat_train.std(axis=0, keepdims=True)
    X_feat_train -= mean
    X_feat_train /= std

    X_feat_valid -= mean
    X_feat_valid /= std

    return X_feat_train, X_feat_valid


def _epochOutput(epoch, logs):

    print("Finished epoch: " + str(epoch))
    print(logs)

    # if os.listdir(dirname)

def _delete_oldest_weightfile(dirname):
    weight_files = []
    for file in os.listdir(dirname):
        base, ext = os.path.splitext(file)
        if ext == 'hdf5':
            weight_files.append([file, os.path.getctime(file)])

    weight_files.sort(key=itemgetter(1), reverse=True)
    os.remove(weight_files[-1][0])

def _get_date_str():
    tdatetime = dt.now()
    tstr = tdatetime.strftime('%Y_%m%d_%H%M')
    return tstr

def _make_dir(dir_name):
    if not os.path.exists('dir_name'):
        os.makedirs(dir_name)

def train(model, train_gen, val_gen, steps_per_epoch=None, optimizer='adam', log_dir='./log', epochs=100, loss='binary_crossentropy', metrics=['accuracy'], reduce_lr=True, reduce_lr_factor=0.2, reduce_lr_patience=10, validation_steps=None):
    if steps_per_epoch is None:
        steps_per_epoch = len(train_gen)

    sub_dir = _get_date_str()
    log_dir = log_dir + '/' + sub_dir
    # make log dir
    _make_dir(log_dir)
    # saved model path
    fpath = log_dir + '/weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
    # callback
    tb_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)
    cp_cb = ModelCheckpoint(filepath=log_dir+'/best_weights.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    batchLogCallback = LambdaCallback(on_epoch_end=_epochOutput)
    # csv_logger = CSVLogger(log_dir + '/training.log')
    csv_logger = AllLogger(log_dir + '/training.log')
    callbacks = [batchLogCallback, csv_logger, cp_cb]
    if reduce_lr:
        callbacks.append(ReduceLROnPlateau(factor=reduce_lr_factor, patience=reduce_lr_patience, verbose=1))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print(model.summary())
    model.fit_generator(
        train_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=2,
        validation_data=val_gen,
        workers=8,
        validation_steps=validation_steps,
        callbacks=callbacks,
        )

    model.save(log_dir + '/' + str(epochs) + 'epochs_final_save')

class AllLogger(keras.callbacks.Callback):

    def __init__(self, log_file_path):
        super(AllLogger, self).__init__()
        self.log_file_path = log_file_path

    def on_train_begin(self, logs={}):
        self.logs_list = []
        self.epochs = []
        # leraning rates
        self.lrs = []
    def on_epoch_end(self, epoch, logs={}):
        self.epochs.append(epoch)
        # dictionary is mutable and keras is going to modify 'logs' over this training.
        # so copy logs and then append it to the list.
        self.logs_list.append(logs.copy())
        self.lrs.append(K.eval(self.model.optimizer.lr))
        self.save_logs()

    def save_logs(self):
        log_df = pd.DataFrame(self.logs_list)
        epoch_lrs_df = pd.DataFrame({'epoch':self.epochs, 'learning_rate':self.lrs})
        all_log_df = epoch_lrs_df.merge(log_df, left_index=True, right_index=True)
        all_log_df.to_csv(self.log_file_path)

