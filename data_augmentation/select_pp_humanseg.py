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

name = 'pp_human_dataset'

parser = argparse.ArgumentParser()
parser.add_argument("--rgb_path",     type=str,   help="raw image path", default='./raw_data/raw_datasets/{0}/images/'.format(name))
parser.add_argument("--mask_path",     type=str,   help="raw mask path", default='./raw_data/raw_datasets/{0}/annotations/'.format(name))
parser.add_argument("--test",     type=str, default=False)
parser.add_argument("--output_path",     type=str,   help="Path to save the conversion result", default='./raw_data/raw_datasets/{0}/select/'.format(name))

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
        os.makedirs(self.OUTPUT_PATH, exist_ok=True)

        self.OUT_RGB_PATH = self.OUTPUT_PATH + 'rgb/'
        self.OUT_MASK_PATH = self.OUTPUT_PATH + 'gt/'
        
        os.makedirs(self.OUT_RGB_PATH, exist_ok=True)
        os.makedirs(self.OUT_MASK_PATH, exist_ok=True)
        
        self.rgb_dir_list = glob.glob(os.path.join(self.RGB_PATH + '*'))
        self.rgb_dir_list = natsort.natsorted(self.rgb_dir_list,reverse=True)



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
        print(rgb_dir_list[dir_idx])

        inner_list = glob.glob(os.path.join(rgb_dir_list[dir_idx] + '/*'))

        for inner in inner_list:
            print(inner)
            rgb_list = glob.glob(os.path.join(inner + '/*'))

            for rgb_name in rgb_list:
                sample_idx += 1

                mask_name = rgb_name.replace('images', 'annotations')
                mask_name = mask_name.replace('jpg', 'png')

                
                img = cv2.imread(rgb_name)
                
                mask = cv2.imread(mask_name) 

                try:
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    mask = np.where(mask>1. , 255, 0)
                    mask = np.expand_dims(mask, axis=-1)
                
                    # if args.test:                    
                    # test_mask = np.concatenate([mask, mask, mask], axis=-1)
                    # masked_image = img * (test_mask / 255)
                    # masked_image = masked_image.astype(np.uint8)
                    # img = img.astype(np.uint8)
                    # test_mask = test_mask.astype(np.uint8)
                    # masked_image = masked_image.astype(np.uint8)
                    # concat_img = cv2.hconcat([img, test_mask, masked_image]) # original_rgb * (original_mask/255)
                    # cv2.imshow('test', concat_img)
                    # cv2.waitKey(0)



                    image_loader.save_images(rgb=img, mask=mask, prefix='matting_human_dataset_{0}'.format(sample_idx))
                except:
                    continue

                