# simple U-net model
# paper: https://arxiv.org/abs/1505.04597

from keras.models import Model
from keras.layers.core import Dropout, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import Input, Concatenate, RepeatVector, Reshape, BatchNormalization, ELU
from keras.layers.pooling import MaxPooling2D
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    # This code doesn't deal with no salt mask!!
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
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
    print('^^^^^^^^^^^^^^^^^^')
    print(iou)

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)


# [TODO] handle with train and test.
def show_image_mask(df, row=30):
    '''
    show image and mask from data frame.

    ARG
    -------------------
        df: DataFrame which has 'id' and 'z'
    '''

    # [['id', z], ['id', z], ,,,]
    target_list = df.values

    image_path = './data/train/images/'
    mask_path = './data/train/masks/'

    counter = 0
    n = 4

    for pair in target_list:
        if counter % n == 0:
            plt.figure(figsize=(10, 5))
        im = cv2.imread(image_path + pair[0] + '.png')
        mask = cv2.imread(mask_path + pair[0] + '.png')
        plt.subplot(1, n, counter%n + 1)
        title = pair[0] + '-' + str(pair[1])
        plt.title(title)
        plt.imshow(im)

        plt.subplot(1, n, counter%n + 2)
        plt.title(title)
        plt.imshow(mask)

        counter +=2

        if counter > row*4:
            break




def show_true_pred(X, Y, pred, row=30, randomed=True):
    '''
    show X_train, Y_train (mask as ground truth) and pred.
    X, Y and pred should have the same order.

    **USAGE**
        X, X_feat, Y = train_util.get_data(size=(128, 128, 3))

        model = load_model('./log/2018_0727_1535/best_weights.hdf5', custom_objects={
            'mean_iou': train_util.mean_iou, 
            'dice_p_bce':train_util.dice_p_bce, 
            'dice_coef':train_util.dice_coef,
            'true_positive_rate':train_util.true_positive_rate,
        })

        pred= model.predict(([X,X_feat]), verbose=1)

        util.show_true_pred(X, Y, pred)

    ARG
    --------------------------
        X: (*, h, w, ch) numpy
        Y: (*, h, w, 1) numpy as ground truth mask
        pred: (*, h, w, 1) numpy as prediction mask
        randomed: Boolean, if it's True, the index of the shown images are randomized
    '''
    counter = 0
    n = 3
    if randomed:
        idx = random.sample(range(X.shape[0]), row)
    else:
        idx = range(0, row)
    # for i, x in enumerate(X):
    for i in idx:
        if counter % n == 0:
            plt.figure(figsize=(10, 5))
        # x
        plt.subplot(1, n, counter%n + 1)
        plt.title('Image')
        plt.imshow(X[i])

        # mask
        plt.subplot(1, n, counter%n + 2)
        plt.title('Ground True')
        plt.imshow(np.squeeze(Y[i]))

        # pred
        plt.subplot(1, n, counter%n + 3)
        plt.title('Prediction')
        plt.imshow(np.squeeze(pred[i]))

        counter +=3
        if counter  > row * 3:
            break



def unet_model(input_shape=(128, 128, 3), min_filter_num=16, kernel_size=(3, 3), up_down_size=(2, 2), strides=(1, 1), activation='elu', offset=2, kernel_initializer='he_normal', with_vec=False, dropout=0.5, vec_shape=8):
    '''
    U-net model. Encoder - Decoder with skipped connections.
    Arguments
    ------------------
        input_shape: input shape of image (h, w, channel)
        min_filter: the minimum number of filters in each convolution layer
        kernel_size: kernel size of filters
        up_down_size: up and down sampling size
        strides: strides size of each conv
        activation: activation function of each conv
        offset: the number of encoder layers should be conv num(log2_shape) - offset
        with_vec: Boolean. If it true, the u net would learn mask data and feature vec
        dropout: propotion of dropout
        vec_shape: vector should be reshaped (h, w, 1) at the concat layer. This value would be the 'h' and 'w'
    Returns
    ------------------
        unet_model, keras model
    '''

    # a num of conv = log_2(im_height or width)
    # log_a(X) = log_e(X) / log_e(a)
    # use 'e' to compute
    min_shape = min(input_shape[:-1])
    conv_num = int(np.floor(np.log(min_shape)/np.log(2))) - offset

    # filter number list; ex) [64, 128, 256, 512, 512, 512....]
    filter_nums = [min_filter_num*min(2**i, 16) for i in range(conv_num)]

    # Input layer
    input_l = Input(shape=input_shape, name='input_layer')

    # Input layer for feature vec
    if with_vec:
        input_l_f = Input((1,), name='feature')

    # first encoder
    first_encoder = _encoder_block(input_l, filter_nums[0], strides=strides, kernel_size=kernel_size, dropout=dropout, name='encoder_block_1', activation=activation)
    x = MaxPooling2D(up_down_size)(first_encoder)

    # make the rest of encoders
    encoders = [first_encoder]

    # concat vector feature to 4th convolution block
    # compute right index for (vec_shape, vec_shape, 1)
    inner_layer_num = input_shape[0] // vec_shape
    index = int(np.floor(np.log(inner_layer_num)/np.log(2)))
    for i, filter_num in enumerate(filter_nums[1:]):
        x = _encoder_block(x, filter_num, name='encoder_block_'+str(i+2), strides=strides, kernel_size=kernel_size, dropout=dropout, activation=activation)
        encoders.append(x)
        if not i == (len(filter_nums) - 2):
            x = MaxPooling2D(up_down_size)(x)

        # since i layer is actually (i + 2)th layer
        if (i == index-2) and with_vec:
            f_repeat = RepeatVector(vec_shape*vec_shape)(input_l_f)
            f_conv = Reshape((vec_shape, vec_shape, 1))(f_repeat)
            x = Concatenate()([f_conv, x])

    # Decoders
    # revers filter nums for decoders
    # do not use the last filter num
    decoder_filter_nums = filter_nums[::-1][1:]

    for i, filter_num in enumerate(decoder_filter_nums):
        # [NOTE] first decoder is concated with the second last encoders!
        x = _decoder_block(x, encoders[-(i+2)], decoder_filter_nums[i], name='decoder_block_'+str(i+1), strides=strides, kernel_size=kernel_size, dropout=dropout, activation=activation)

    # for segmentation, apply sigmoid to every pixel
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = Model(inputs=[input_l, input_l_f], outputs=[outputs])

    return model





def _encoder_block(x, filter_num, name='encoder_block', strides=(1, 1), kernel_size=(3, 3), dropout=0.1, activation='elu', down_size=(2, 2), kernel_initializer='he_normal'):
    '''
    U-net encoder cnn block
    Conv -> activation -> dropout -> conv -> activation
    '''

    x = Conv2D(filter_num, kernel_size, strides=strides, name=name, padding='same', kernel_initializer=kernel_initializer)(x)
    # x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(dropout)(x)
    x = Conv2D(filter_num, kernel_size, strides=strides, padding='same', kernel_initializer=kernel_initializer)(x)
    # x = BatchNormalization()(x)
    x = ELU()(x)
    # out = MaxPooling2D(down_size)(x)

    return x

def _decoder_block(x, skip_connect, filter_num, name='decoder_block', strides=(1, 1), kernel_size=(3, 3), dropout=0.1, activation='elu', up_size=(2, 2), kernel_initializer='he_normal'):
    '''
    U-net decoder cnn block
    transpose conv -> concat skip_connect -> conv -> dropout -> conv
    '''

    x = Conv2DTranspose(filter_num, up_size, strides=up_size, padding='same', name=name)(x)
    # x = BatchNormalization()(x)
    x = ELU()(x)
    x = Concatenate()([x, skip_connect])
    x = Conv2D(filter_num, kernel_size, strides=strides, padding='same', kernel_initializer=kernel_initializer)(x)
    # x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(dropout)(x)
    x = Conv2D(filter_num, kernel_size, strides=strides, padding='same', kernel_initializer=kernel_initializer)(x)
    # x = BatchNormalization()(x)
    x = ELU()(x)
    return x