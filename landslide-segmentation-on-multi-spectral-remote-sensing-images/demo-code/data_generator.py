import os
import math
import h5py
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from PIL import ImageEnhance
from natsort import natsorted
from scipy import ndimage

from util import *
from hyper_parameter import *


class DataGenerator(object):
    def __init__(self, data_dir, batch_size, train_ratio=0.1, band_opt=1, is_multi_resolution=True):
        self.data_dir        = data_dir
        self.images_dir      = data_dir + '/images/'
        self.masks_dir       = data_dir + '/masks/'

        self.n_img_train     = math.ceil( len(os.listdir(self.images_dir)) * train_ratio )
        self.n_img_val       = len(os.listdir(self.images_dir)) - self.n_img_train

        self.img_list        = os.listdir(self.images_dir)
        self.mask_list       = os.listdir(self.masks_dir)
        self.img_list        = natsorted(self.img_list)
        self.mask_list       = natsorted(self.mask_list)

        self.landslide_mask, self.landslide_image  = findLandslideImage(data_dir, train_ratio)

        self.batch_size      = batch_size
        self.band_opt        = band_opt # 0->14; 1->19; 2->9; 3->20
        self.class_num       = HyperParameter().class_num
        self.label_dict      = HyperParameter().label_dict
        self.img_mean        = np.reshape(np.array(HyperParameter().img_mean), (1, 1, -1))
        self.img_max         = np.reshape(np.array(HyperParameter().img_max), (1, 1, -1))
        self.is_multi_resolution = is_multi_resolution

    def getImgList(self):
        return self.img_list

    def getNumBatch(self, op='train'):
        batch_total = self.getNumImg(op)
        if batch_total % self.batch_size == 0:
            return (int(batch_total/self.batch_size), 0)
        return (int(batch_total/self.batch_size) + 1, batch_total % self.batch_size)

    def getNumImg(self, op='train'):
        if op == 'train':
            return self.n_img_train
        return self.n_img_val

    def getBatch(self, batch_num, is_aug=False, is_train=True, is_cutmix=True):
        if is_train:  # training
            img_list  = self.img_list[:self.n_img_train]
            mask_list = self.mask_list[:self.n_img_train]
            is_aug    = is_aug
            op        = "train"
        else:         # validation
            img_list  = self.img_list[self.n_img_train:]
            mask_list = self.mask_list[self.n_img_train:]
            is_aug    = False
            is_cutmix = False
            op        = "val"

        n_image = 0

        ## handle last batch
        if (batch_num == (self.getNumBatch(op=op)[0]-1)) and (self.getNumBatch(op=op)[1] != 0):
            start_ind = batch_num*self.batch_size
            end_ind   = start_ind + self.getNumBatch(op=op)[1]
        else:
            start_ind = batch_num*self.batch_size
            end_ind   = (batch_num+1)*self.batch_size
        
        ## loop to get batch of images and masks
        for ind in range(start_ind, end_ind):
            mask_name = mask_list[ind]
            mask_open = os.path.join(self.masks_dir, mask_name)

            img_name = img_list[ind]
            img_open = os.path.join(self.images_dir, img_name)

            ## load mask
            f_mask    = h5py.File(mask_open, 'r')
            # print(list(f_mask.keys()))
            one_mask  = f_mask[list(f_mask.keys())[0]]
            one_mask  = np.asarray(one_mask) # (128,128)
            f_mask.close()

            ## load image
            f_image   = h5py.File(img_open, 'r')
            # print(list(f_image.keys()))
            one_image = f_image[list(f_image.keys())[0]]
            one_image = np.asarray(one_image) # (128,128,14)
            f_image.close()

            # ## multiply with band's mean values (pixel wise multiply)
            # one_image = np.multiply(one_image, self.img_mean)

            ## add feature
            if self.band_opt == 1:   #  output shape: 128x128xC
                one_image = addRGB(one_image, is_rgb=False)
                #one_image = addRGB_2(one_image, is_rgb=False)
                one_image = addNDVI(one_image)
                one_image = addNBR(one_image)
                one_image = addVegetationIndex(one_image)
                one_image = addGray(one_image)
                # one_image = addEdge(one_image)
                one_image = addBlur(one_image)
                #one_image = addGradient(one_image)
            elif self.band_opt == 2: # 3 + 3 + 3 bands (rgb only) -> output shape: 128x128x6
                one_image = one_image[:,:,1:4]               # bgr
                one_image = addRGB(one_image, is_rgb=True)   # rgb normalize -> BGRRGB -> 128x128x6
                one_image = addRGB_2(one_image, is_rgb=True) # rgb normalize -> BGRRGB -> 128x128x9

            ## do augmentation
            if is_aug:
                one_image, one_mask = self.augmentateImage(one_image, one_mask)

            ## apply cutmix
            if is_cutmix:
                num_cut = np.random.randint(0, 3)
                for cut in range(num_cut):
                    one_image, one_mask = self.cutmixImgMask(one_image, one_mask)

            ## apply multi resolution
            if self.is_multi_resolution:
                one_image, mask_256, mask_128, mask_64 = self.applyMultiResolution(one_image, one_mask)
                ## put all image together to form a 4-d tensor
                if (n_image == 0):
                    seq_x = one_image
                    seq_y_256 = mask_256
                    seq_y_128 = mask_128
                    seq_y_64  = mask_64
                else:
                    seq_x = np.concatenate((seq_x, one_image), axis=0)
                    seq_y_256 = np.concatenate((seq_y_256, mask_256), axis=0)
                    seq_y_128 = np.concatenate((seq_y_128, mask_128), axis=0)
                    seq_y_64  = np.concatenate((seq_y_64,  mask_64),  axis=0)
            else:
                nW, nH, nC  = one_image.shape  # (128,128,#bands)
                nw, nh      = one_mask.shape   # (128,128)
                ## reshape
                one_image   = np.reshape(one_image, (1, nW, nH, nC))
                one_mask    = np.reshape(one_mask, (1, nw, nh))
                ## put all image together to form a 4-d tensor
                if (n_image == 0):
                    seq_x = one_image
                    seq_y = one_mask
                else:
                    seq_x = np.concatenate((seq_x, one_image), axis=0)
                    seq_y = np.concatenate((seq_y, one_mask), axis=0)
            
            n_image += 1

        seq_x = tf.convert_to_tensor(seq_x, dtype=tf.float32)
        if self.is_multi_resolution:
            seq_y_256 = tf.convert_to_tensor(seq_y_256, dtype=tf.float32)
            seq_y_128 = tf.convert_to_tensor(seq_y_128, dtype=tf.float32)
            seq_y_64  = tf.convert_to_tensor(seq_y_64,  dtype=tf.float32)
            return seq_x, seq_y_256, seq_y_128, seq_y_64, n_image
        else:
            seq_y = tf.convert_to_tensor(seq_y, dtype=tf.float32)
            return seq_x, seq_y, n_image

    def applyMultiResolution(self, one_image, one_mask):
        w,h,c = one_image.shape
        ## resize
        mask_256 = cv2.resize(one_mask, (w*2, h*2), interpolation=cv2.INTER_NEAREST)
        mask_128 = one_mask
        mask_64  = cv2.resize(one_mask, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
        ## reshape
        one_image = np.reshape(one_image, (1, 128, 128, c))                       
        mask_256 = np.reshape(mask_256, (1, 256, 256))            
        mask_128 = np.reshape(mask_128, (1, 128, 128))            
        mask_64  = np.reshape(mask_64,  (1, 64, 64))    

        return one_image, mask_256, mask_128, mask_64

    def augmentateImage(self, one_image, one_mask):
        ## rotate both image and mask
        if np.random.uniform() >= 0.3:
            one_image, one_mask = self.rotateImgMask(one_image, one_mask)

        return one_image, one_mask

    def rotateImgMask(self, one_image, one_mask):
        angle = np.random.randint(1,4)             # 1,2,3 -> 90,180,270
        rotated_image = np.rot90(one_image, angle)
        rotated_mask  = np.rot90(one_mask, angle)

        return rotated_image, rotated_mask

    def cutmixImgMask(self, img_1, mask_1, is_rgb=False):
        """
            # input:
                - an image
                - a mask
            # output:
                - new image:
                - new mask:
        """
        ## pick random image with sum(mask) > 0
        rand_ind  = np.random.randint(0, len(self.landslide_mask))
        mask_open = os.path.join(self.masks_dir, self.landslide_mask[rand_ind])
        img_open  = os.path.join(self.images_dir, self.landslide_image[rand_ind])

        ## load mask
        f_mask   = h5py.File(mask_open, 'r')
        mask_2   = f_mask[list(f_mask.keys())[0]]
        mask_2   = np.asarray(mask_2) # (128,128)
        f_mask.close()

        ## load data
        f_image  = h5py.File(img_open, 'r')
        img_2    = f_image[list(f_image.keys())[0]]
        img_2    = np.asarray(img_2) # (128,128,14)
        f_image.close()

        # ## multiply with band's mean values (pixel wise multiply)
        # img_2 = np.multiply(img_2, self.img_mean)

        if self.band_opt == 1:       # output shape: 128x128x26
            img_2 = addRGB(img_2, is_rgb=False)
            #img_2 = addRGB_2(img_2, is_rgb=False)
            img_2 = addNDVI(img_2)
            img_2 = addNBR(img_2)
            img_2 = addVegetationIndex(img_2)
            img_2 = addGray(img_2)
            # img_2 = addEdge(img_2)
            img_2 = addBlur(img_2)
            #img_2 = addGradient(img_2)
        elif self.band_opt == 2:   # output shape: 128x128x9
            img_2 = img_2[:,:,1:4]               # bgr
            img_2 = addRGB(img_2, is_rgb=True)   # rgb normalize -> BGRRGB -> 128x128x6
            img_2 = addRGB_2(img_2, is_rgb=True) # rgb normalize -> BGRRGB -> 128x128x9

        ## do cutmix
        mask_2   = np.expand_dims(mask_2, axis=-1) # 128x128x1
        new_img  = img_1 * (1 - mask_2)      #
        new_img  = new_img + img_2 * mask_2  #
        new_mask = np.logical_or(mask_1, np.squeeze(mask_2, axis=-1)) # (128x128) || (128x128) => (128x128)
        new_mask = new_mask.astype(int)

        return new_img, new_mask



if __name__ == '__main__':
    ### TEST GET BATCH ###
    # generator = DataGenerator(data_dir='../dataset/train', batch_size=20, train_ratio=0.8, band_opt=0)
    # print(generator.getNumBatch(op='train'))
    # print(generator.getNumBatch(op='val'))
    # imgs,masks,n_imgs = generator.getBatch(batch_num=1, is_aug=True, is_train=True)
    # print(n_imgs, imgs.shape, masks.shape, type(imgs))

    # generator = DataGenerator(data_dir='../dataset/train', batch_size=20, train_ratio=0.8, band_opt=1)
    # print(generator.getNumBatch(op='train'))
    # print(generator.getNumBatch(op='val'))
    # imgs,masks,n_imgs = generator.getBatch(batch_num=1, is_aug=True, is_train=True)
    # print(n_imgs, imgs.shape, masks.shape, type(imgs))

    # generator = DataGenerator(data_dir='../dataset/train', batch_size=20, train_ratio=0.8, band_opt=2)
    # print(generator.getNumBatch(op='train'))
    # print(generator.getNumBatch(op='val'))
    # imgs,masks,n_imgs = generator.getBatch(batch_num=1, is_aug=True, is_train=True)
    # print(n_imgs, imgs.shape, masks.shape, type(imgs))

    # del generator,masks,imgs,n_imgs

    generator = DataGenerator(data_dir='../dataset/train', batch_size=12, train_ratio=0.8, band_opt=1, is_multi_resolution=True)
    print(generator.getNumBatch(op='train'))
    print(generator.getNumBatch(op='val'))

    imgs, masks_256, masks_128, masks_64, n_imgs = generator.getBatch(batch_num=1, is_aug=False, is_train=True)
    visualizeOneImg(imgs[1], op=0)     
    visualizeOneMask(masks_128[1]) 

    imgs, masks_256, masks_128, masks_64, n_imgs = generator.getBatch(batch_num=1, is_aug=True, is_train=True)
    print(n_imgs, imgs.shape, masks_256.shape, masks_128.shape, masks_64.shape, type(imgs))
    visualizeOneImg(imgs[1], op=0)     
    # visualizeOneMask(masks_256[1]) 
    visualizeOneMask(masks_128[1]) 
    # visualizeOneMask(masks_64[1]) 
    del generator,masks_256,masks_128,masks_64,imgs



    ### DRAW EVERYTHING - REMEMBER TO UNCOMMENT ###
    # generator = DataGenerator(data_dir='../dataset/train', batch_size=1, train_ratio=0.8, band_opt=1)
    # print(generator.getNumBatch(op='train'))
    # print(generator.getNumBatch(op='val'))
    # imgs,masks,n_imgs = generator.getBatch(batch_num=1, is_aug=True, is_train=True, is_cutmix=True)
    # print(n_imgs, imgs.shape, masks.shape, type(imgs))
    # one_img  = imgs[0]
    # one_mask = masks[0]
    # visualizeOneImg(one_img, op=0)     # rgb - min max scaling
    # visualizeOneImg(one_img, op=1)     # rgb - max scaling
    # visualizeOneMask(one_img[...,23])  # gray scale image
    # visualizeOneMask(one_img[...,24])  # apply Canny Edge
    # visualizeOneMask(one_img[...,25])  # apply gausian blur
    # visualizeOneMask(one_img[...,26])  # apply median blur
    # visualizeOneMask(one_mask)


    ### TEST CUTMIX - NON SLIDING IMAGE ### 
    # generator = DataGenerator(data_dir='../dataset/train', batch_size=1, train_ratio=0.8, band_opt=1)
    # print(generator.getNumBatch(op='train'))
    # print(generator.getNumBatch(op='val'))
    # imgs,masks,n_imgs = generator.getBatch(batch_num=2, is_aug=True, is_train=True, is_cutmix=False)
    # print(n_imgs, imgs.shape, masks.shape, type(imgs))
    # one_img  = imgs[0]
    # one_mask = masks[0]
    # visualizeOneImg(one_img, op=0)
    # visualizeOneImg(one_img, op=1)
    # visualizeOneMask(one_mask)
    # print(tf.reduce_sum(one_mask))

    # generator = DataGenerator(data_dir='../dataset/train', batch_size=1, train_ratio=0.8, band_opt=1)
    # print(generator.getNumBatch(op='train'))
    # print(generator.getNumBatch(op='val'))
    # imgs,masks,n_imgs = generator.getBatch(batch_num=2, is_aug=True, is_train=True, is_cutmix=True)
    # print(n_imgs, imgs.shape, masks.shape, type(imgs))
    # one_img  = imgs[0]
    # one_mask = masks[0]
    # visualizeOneImg(one_img, op=0)
    # visualizeOneImg(one_img, op=1)
    # visualizeOneMask(one_mask)
    # print(tf.reduce_sum(one_mask))


    ### TEST AUGMENTATION AND CUTMIX (A LANDSLIDE IMAGE) ###
    # generator = DataGenerator(data_dir='../dataset/train', batch_size=1, train_ratio=0.8, band_opt=1)
    # print(generator.getNumBatch(op='train'))
    # print(generator.getNumBatch(op='val'))

    # imgs,masks,n_imgs = generator.getBatch(batch_num=236, is_aug=False, is_train=True, is_cutmix=False)
    # print(n_imgs, imgs.shape, masks.shape, type(imgs))
    # one_img  = imgs[0]
    # one_mask = masks[0]
    # visualizeOneImg(one_img, op=0)
    # visualizeOneImg(one_img, op=1)
    # visualizeOneMask(one_mask)
    # print(tf.reduce_sum(one_mask))

    # imgs,masks,n_imgs = generator.getBatch(batch_num=236, is_aug=True, is_train=True, is_cutmix=False)
    # print(n_imgs, imgs.shape, masks.shape, type(imgs))
    # print(generator.getImgList()[:10],'\n')
    # one_img  = imgs[0]
    # one_mask = masks[0]
    # visualizeOneImg(one_img, op=0)
    # visualizeOneImg(one_img, op=1)
    # visualizeOneMask(one_mask)
    # print(tf.reduce_sum(one_mask))

    # imgs,masks,n_imgs = generator.getBatch(batch_num=236, is_aug=True, is_train=True, is_cutmix=True)
    # print(n_imgs, imgs.shape, masks.shape, type(imgs))
    # one_img  = imgs[0]
    # one_mask = masks[0]
    # visualizeOneImg(one_img, op=0)
    # visualizeOneImg(one_img, op=1)
    # visualizeOneMask(one_mask)
    # print(tf.reduce_sum(one_mask))


    ### TEST getBatch RGB ONLY ###
    # generator = DataGenerator(data_dir='../dataset/train', batch_size=20, train_ratio=0.8, band_opt=2)
    # print(generator.getNumBatch(op='train'))
    # print(generator.getNumBatch(op='val'))

    # imgs,masks,n_imgs = generator.getBatch(batch_num=1, is_aug=True, is_train=True,is_cutmix=False)
    # print(n_imgs, imgs.shape, masks.shape, type(imgs))
    # one_img  = imgs[3]
    # one_mask = masks[3]
    # visualizeOneImg(one_img, op=0, is_rgb=True)
    # visualizeOneImg(one_img, op=1, is_rgb=True)
    # visualizeOneMask(one_mask)
    # print(tf.reduce_sum(one_mask))

    # imgs,masks,n_imgs = generator.getBatch(batch_num=1, is_aug=True, is_train=True,is_cutmix=True)
    # print(n_imgs, imgs.shape, masks.shape, type(imgs))
    # one_img  = imgs[3]
    # one_mask = masks[3]
    # visualizeOneImg(one_img, op=0, is_rgb=True)
    # visualizeOneImg(one_img, op=1, is_rgb=True)
    # visualizeOneMask(one_mask)
    # print(tf.reduce_sum(one_mask))





    # del generator,masks,imgs,n_imgs,one_img,one_mask,tf,plt
