from typing import Union
import numpy as np
import cv2
import glob
import os
import argparse
import natsort
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import math
import random

name = 'matting_human_dataset'

parser = argparse.ArgumentParser()
parser.add_argument("--rgb_path",     type=str,   help="raw image path", default='./data_augmentation/raw_data/raw_datasets/{0}/clip_img/'.format(name))
parser.add_argument("--mask_path",     type=str,   help="raw mask path", default='./data_augmentation/raw_data/raw_datasets/{0}/matting/'.format(name))
parser.add_argument("--test",     type=str, default=False)
parser.add_argument("--output_path",     type=str,   help="Path to save the conversion result", default='./data_augmentation/raw_data/raw_datasets/{0}/select/'.format(name))

args = parser.parse_args()

class ImageAugmentationLoader():
    def __init__(self, args):
        """
        Args
            args  (argparse) : inputs (rgb, mask)
                >>>    rgb : RGB image.
                >>>    mask : Segmentation mask.
        """
        self.RGB_PATH = args.rgb_path
        self.MASK_PATH = args.mask_path
        
        self.OUTPUT_PATH = args.output_path

        self.OUT_RGB_PATH = self.OUTPUT_PATH + 'rgb/'
        self.OUT_MASK_PATH = self.OUTPUT_PATH + 'gt/'
        
        os.makedirs(self.OUT_RGB_PATH, exist_ok=True)
        os.makedirs(self.OUT_MASK_PATH, exist_ok=True)
        
        self.rgb_dir_list = glob.glob(os.path.join(self.RGB_PATH + '*'))
        self.rgb_dir_list = natsort.natsorted(self.rgb_dir_list,reverse=False)



    def image_resize(self, rgb: np.ndarray, mask: np.ndarray, size=(1600, 900)) -> Union[np.ndarray, np.ndarray]:
        """
            Image resizing function    
            Args:
                rgb      (np.ndarray) : (H,W,3) Image.
                mask     (np.ndarray) : (H,W,1) Image.
                size     (tuple)      : Image size to be adjusted.
        """
        resized_rgb = tf.image.resize(images=rgb, size=size, method=tf.image.ResizeMethod.BILINEAR)
        resized_rgb = resized_rgb.numpy().astype(np.uint8)

        resized_mask = tf.image.resize(images=mask, size=size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        resized_mask = resized_mask.numpy().astype(np.uint8)


        return resized_rgb, resized_mask

    def save_images(self, rgb, mask, prefix):
        cv2.imwrite(self.OUT_RGB_PATH + prefix +'_rgb.jpg', rgb)
        cv2.imwrite(self.OUT_MASK_PATH + prefix + '_mask.png', mask)

                                                              
if __name__ == '__main__':
    """
    Image augmentation can be selected according to the option using the internal function of ImageAugmentationLoader.
    """
    from tqdm import tqdm

    image_loader = ImageAugmentationLoader(args=args)
    rgb_dir_list = image_loader.rgb_dir_list

    rgb_dir_list_len = len(rgb_dir_list)
    sample_idx = 0
    for dir_idx in tqdm(range(rgb_dir_list_len), total=rgb_dir_list_len):
        # print(rgb_dir_list[dir_idx])

        inner_list = glob.glob(os.path.join(rgb_dir_list[dir_idx] + '/*'))

        for inner in inner_list:
            print(inner)
            rgb_list = glob.glob(os.path.join(inner + '/*'))

            for rgb_name in rgb_list:
                sample_idx += 1

                mask_name = rgb_name.replace('clip_img', 'matting')
                mask_name = mask_name.replace('clip_', 'matting_')
                mask_name = mask_name.replace('jpg', 'png')
                img = cv2.imread(rgb_name)

                mask = cv2.imread(mask_name, cv2.IMREAD_UNCHANGED) 
                
                # get alpha channel
                mask = mask[:,:,3]
                
                mask = np.where(mask>1. , 255, 0)

                image_loader.save_images(rgb=img, mask=mask, prefix='matting_human_dataset_{0}'.format(sample_idx))

                