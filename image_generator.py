import numpy as np
import pandas as pd
from glob import glob
from random import shuffle
from skimage.transform import resize
from imgaug import augmenters as iaa
import cv2

class Generator(object):
    def __init__(self, train_ids, depth_dict, val_ids, mask_base_path='./data/train/masks/', image_base_path='./data/train/images/', augment=True, augmenters=None, batch_size=32, size=(128, 128, 3), feature_norm=False, feature_out=True, id_out=False):
        '''

        Arguments
        ------------------------
        train_ids: id list for train data set
        depth_dict: {'id': depth} depth is not need to be normalized and it should contain all id depth pair for train and val data set
        val_ids: id list for val data set
        mask_base_path: mask path
        image_base_path: image path
        augment: set True if you need some data augmentation
        augmenters: it assume iaa.Sequential object. if it's none, set default iaa.Sequential in constructor
        batch_size: batch size
        size: load size for images and masks


        Usage
        -------------------------
        train_ids, val_ids = train_test_split(ids, test_size=0.2, random_state=43, stratify=classes)
        batch_size = 32
        tgs_generator = Generator(train_ids = train_ids, depth_dict=depth_dict, val_ids=val_ids, batch_size=batch_size)
        next(tgs_generator.generate(False))[0][1]

        Note
        -------------------------
        Output shuffle data for train data set and sequential data for val data set
        all images are normalized between 0. ~ 1. and depth is also normalized with 0 mean and 1 std



        '''
        self.train_ids = train_ids
        self.depth_dict = depth_dict
        self.mask_base_path = mask_base_path
        self.image_base_path = image_base_path
        self.augment = augment
        self.size = size
        self.val_ids = val_ids
        self.batch_size = batch_size
        self.val_index = 0
        self.feature_norm = feature_norm
        self.feature_out = feature_out
        self.id_out = id_out

        # compute train depth mean and std for depth normalization
        train_depth_list = []
        for train_id in train_ids:
            train_depth_list.append(depth_dict[train_id])
        self.depth_mean = np.mean(train_depth_list)
        self.depth_std = np.std(train_depth_list)

        # Define augmentations
        if augment and augmenters is None:
            affine_seq = iaa.Sequential([
            # General
            iaa.SomeOf((1, 2),
                       [iaa.Fliplr(0.5),
                        iaa.Affine(rotate=(-10, 10),
                                   translate_percent={"x": (-0.25, 0.25)}, mode='symmetric'),
                        ]),
            # Deformations
            iaa.Sometimes(0.3, iaa.PiecewiseAffine(scale=(0.04, 0.08))),
            iaa.Sometimes(0.3, iaa.PerspectiveTransform(scale=(0.05, 0.1))),
        ], random_order=True)

            seq = iaa.Sequential([iaa.Sometimes(0.8, affine_seq), iaa.Sometimes(0.5, iaa.Crop(px=(0, 10)))], random_order=False)
            self.augmenters = seq
        else:
            if augment and augmenters is not None:
                self.augmenters = augmenters

    def generate(self, train=True):
        while True:
            if train:
                shuffle(self.train_ids)
                ids = self.train_ids
            else:
                # use index instead of shuffle for val ids
                # shuffle(self.train_ids)
                ids  = self.val_ids[self.val_index*self.batch_size:(self.val_index+1)*self.batch_size]
                self.val_index += 1
                if self.val_index > len(self.val_ids):
                    self.val_index = 0
                # if val, batch_size is going to be len(val_ids)
                # self.batch_size = len(self.val_ids)
            image_list = []
            depth_list = []
            mask_list = []

            # for test
            id_list =[]
            for im_id in ids:
                # get depth
                depth = self.depth_dict[im_id]
                # it looks like original depth value perform better than normed one
                if self.feature_norm:
                    depth -= self.depth_mean
                    depth /= self.depth_std
                depth_list.append(depth)
                # load image
                # neet to be int8 for augmentation
                path = self.image_base_path + im_id + '.png'
                im = cv2.imread(path)
                im = cv2.resize(im, (self.size[0], self.size[1]))
                # load mask
                mask_p = self.mask_base_path + im_id + '.png'
                mask = cv2.imread(mask_p)
                # keep 3 channel for augmentation
                # never interpolation for masks
                mask = cv2.resize(mask, (self.size[0], self.size[1]), interpolation=cv2.INTER_NEAREST)
                image_list.append(im)
                mask_list.append(mask)

                # for test
                id_list.append(im_id)
                if len(image_list) == self.batch_size:
                    images = np.array(image_list)
                    depths = np.array(depth_list)
                    masks = np.array(mask_list)

                    # for test
                    ids = np.array(id_list)
                    # for next generation
                    image_list = []
                    depth_list = []
                    mask_list = []

                    # for test
                    id_list = []
                    # Augmentation
                    if self.augment and train:
                        aug_det = self.augmenters.to_deterministic()
                        images = aug_det.augment_images(images)
                        masks = aug_det.augment_images(masks)
                    masks = np.array((masks > 127.5), dtype='float32')
                    # (batch_size, h, w, 3) -> (batch_size, h, w, 1)
                    # never interpolation for masks
                    # this will not work
                    # masks = resize(masks[:, :, :, 0], (self.batch_size, self.size[0], self.size[1], 1), order=0)
                    masks = np.expand_dims(masks[:, :, :, 0], axis=-1)
                    images =np.array(images, dtype='float32')
                    images /= 255
                    if self.id_out:
                        if self.feature_out:
                            yield [images, depths], masks, ids
                        else:
                            yield images, masks, ids
                    else:
                        if self.feature_out:
                            yield [images, depths], masks
                        else:
                            yield images, masks




