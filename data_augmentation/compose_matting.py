from typing import Union
import numpy as np
import cv2
import glob
import os
import argparse
import natsort
import random

name = 'yebin_fashion_dataset'
parser = argparse.ArgumentParser()
parser.add_argument("--rgb_path",     type=str,   help="raw image path", default='./raw_data/raw_datasets/{0}/select/rgb/'.format(name))
parser.add_argument("--mask_path",     type=str,   help="raw mask path", default='./raw_data/raw_datasets/{0}/select/gt/'.format(name))
parser.add_argument("--bg_path",     type=str,   help="bg image path, Convert raw rgb image using mask area", default='./raw_data/raw_datasets/coex_bg/select/rgb/')
parser.add_argument("--output_path",     type=str,   help="Path to save the conversion result", default='./raw_data/raw_datasets/{0}/augmented/'.format(name))

args = parser.parse_args()

class ImageAugmentationLoader():
    def __init__(self, args):
        """
        Args
            args  (argparse) : inputs (rgb, mask segObj, bg)
                >>>    rgb : RGB image.
                >>>    mask : segmentation mask.
                >>>    segObj : segmentation object mask.
                >>>    label_map : segmentation mask(label) information.
                >>>    bg : Background image.
        """
        self.RGB_PATH = args.rgb_path
        self.MASK_PATH = args.mask_path
        self.BG_PATH = args.bg_path
        self.OUTPUT_PATH = args.output_path

        self.OUT_RGB_PATH = self.OUTPUT_PATH + 'rgb/'
        self.OUT_MASK_PATH = self.OUTPUT_PATH + 'gt/'
        os.makedirs(self.OUT_RGB_PATH, exist_ok=True)
        os.makedirs(self.OUT_MASK_PATH, exist_ok=True)

        self.rgb_list = glob.glob(os.path.join(self.RGB_PATH+'*.jpg'))
        self.rgb_list = natsort.natsorted(self.rgb_list,reverse=True)

        self.mask_list = glob.glob(os.path.join(self.MASK_PATH+'*.png'))
        self.mask_list = natsort.natsorted(self.mask_list,reverse=True)

        self.bg_list = glob.glob(os.path.join(self.BG_PATH +'*.jpg' ))
        
        # Check your data (RGB file samples = Mask file samples)
        self.check_image_len() 


    def check_image_len(self):
        """
            Check rgb, mask, obj mask sample counts
        """
        rgb_len = len(self.rgb_list)
        mask_len = len(self.mask_list)

        if rgb_len != mask_len:
            raise Exception('RGB Image files : {0}, Mask Image files : {1}. Check your image and mask files '
                            .format(rgb_len, mask_len))


    def get_rgb_list(self) -> list:
        """
            return rgb list instance
        """
        return self.rgb_list

    def get_mask_list(self) -> list:
        """
            return mask list instance
        """
        return self.mask_list

    def get_bg_list(self) -> list:
        """
            return bg image list instance
        """
        return self.bg_list
    
    def resize_bg_image(self, bg_image: np.ndarray, rgb_shape: tuple):
        h, w = rgb_shape[:2]
        bg_image = cv2.resize(bg_image, (w, h))

        return bg_image


    def bg_area_blurring(self, rgb: np.ndarray, mask: np.ndarray,
                         gaussian_min: int = 5, gaussian_max: int = 17) -> np.ndarray:
        rgb_area = np.where(mask >= 1, rgb, 0)
        bg_area = np.where(mask>=1, 0, rgb)

        k = random.randrange(gaussian_min, gaussian_max, 2)    
        bg_area = cv2.GaussianBlur(bg_area, (k, k), 0)

        blurred_rgb = cv2.add(rgb_area, bg_area)

        return blurred_rgb

        
    def image_random_translation(self, rgb: np.ndarray, mask: np.ndarray,
                                 min_dx: int, min_dy: int,
                                 max_dx: int, max_dy: int) -> Union[np.ndarray, np.ndarray]:
        """
            Random translation function   
            Args:
                rgb        (np.ndarray) : (H,W,3) Image.
                mask       (np.ndarray) : (H,W,1) Image.
                min_dx  (int)      : Minimum value of pixel movement distance based on the x-axis when translating an image.
                min_dy  (int)      : Minimum value of pixel movement distance based on the y-axis when translating an image.
                max_dx  (int)      : Maximum value of pixel movement distance based on the x-axis when translating an image.
                max_dy  (int)      : Maximum value of pixel movement distance based on the y-axis when translating an image.
                
        """
        
        random_dx = random.randint(min_dx, max_dx)
        random_dy = random.randint(min_dy, max_dy)
        random_axis = random.randint(0, 1)
        
        if max_dx == 0:
            random_dx = 1

        if max_dy == 0:
            random_dy = 1

        if random_axis == 1:
            random_dx *= -1

        # if tf.random.uniform([]) > 0.5:
        #     random_dy *= -1

        rows, cols = rgb.shape[:2]
        trans_mat = np.float64([[1, 0, random_dx], [0, 1, random_dy]])

        trans_rgb = cv2.warpAffine(rgb, trans_mat, (cols, rows))
        trans_mask = cv2.warpAffine(mask, trans_mat, (cols, rows))

        return trans_rgb, trans_mask

    def save_images(self, rgb: np.ndarray, mask: np.ndarray, prefix: str):
        """
            Save image and mask
            Args:
                rgb     (np.ndarray) : (H,W,3) Image.
                mask    (np.ndarray) : (H,W,1) Image.
                prefix  (str)        : The name of the image to be saved.
        """
        cv2.imwrite(self.OUT_RGB_PATH + prefix +'_.jpg', rgb)
        cv2.imwrite(self.OUT_MASK_PATH + prefix + '_mask.png', mask)

                                                              
if __name__ == '__main__':
    """
    Image augmentation can be selected according to the option using the internal function of ImageAugmentationLoader.
    """
    from tqdm import tqdm

    image_loader = ImageAugmentationLoader(args=args)
    rgb_list = image_loader.get_rgb_list()
    mask_list = image_loader.get_mask_list()
    bg_list = image_loader.get_bg_list()

    if len(rgb_list) <= 3000:
        max_aug = 1
    elif len(rgb_list) <= 8000:
        max_aug = 2
    elif len(rgb_list) <= 12000:
        max_aug = 3
    elif len(rgb_list) <= 18000:
        max_aug = 4
    elif len(rgb_list) <= 20000:
        max_aug = 5
    else:
        max_aug = 6

    
    print('dataset name : {0}, max_aug : {1}'.format(name, max_aug))
    
    # 1. rgb ????????? ?????? ?????? ?????????
    
    rgb_len = len(rgb_list)
    # for idx in range(len(rgb_list)):

    for idx in tqdm(range(rgb_len), total=rgb_len):
        random_select = random.randint(0, max_aug)
        if random_select == 0:
            original_rgb = cv2.imread(rgb_list[idx])
            original_mask = cv2.imread(mask_list[idx])
            
            original_rgb_shape = original_rgb.shape[:2]
            original_mask_shape = original_mask.shape[:2]

            if original_rgb_shape != original_mask_shape:
                print('not match shape!  resize mask to rgb shape')
                h, w = original_rgb_shape
                original_mask = cv2.resize(original_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # random shiftq
            h, w = original_rgb_shape
            max_dx = int(w/3)
            max_dy = int(h/4)

            """compose background image with multiple objects"""
            background_shape = (1280, 720, 3)

            # 1. select background
            bg_rnd_idx = random.randint(0, len(bg_list)-1)
            original_bg = cv2.imread(bg_list[bg_rnd_idx])
            original_bg = image_loader.resize_bg_image(bg_image=original_bg, rgb_shape=background_shape) # for matting_dataset
            # ????????? background image, mask ??????
            bg_image = original_bg.copy()
            
            compose_mask = np.zeros(background_shape)

            # 2. load multiple objects
            # center 
            # rgb_rnd_idx = random.randint(0, rgb_len-1)
            # rnd_rgb = cv2.imread(rgb_list[rgb_rnd_idx]) # shape = 800, 600 ,3 
            # rnd_mask = cv2.imread(mask_list[rgb_rnd_idx]) # shape = 800, 600 ,3 
            # rnd_mask = np.where(rnd_mask>=1, 1, 0)
            # zero_rgb = np.zeros((1440, 810, 3))
            # zero_mask = np.zeros((1440, 810, 3))
            # zero_rgb[240:240+800, 105:105+600] = rnd_rgb
            # zero_mask[240:240+800, 105:105+600] += rnd_mask
            # compose_mask[240:240+800, 105:105+600] += rnd_mask
            # bg_image = np.where(zero_mask >= 1, zero_rgb, original_bg)
            # bg_image = bg_image.astype(np.uint8)

            # left
            rgb_rnd_idx = random.randint(0, rgb_len-1)
            rnd_rgb = cv2.imread(rgb_list[rgb_rnd_idx]) # shape = 800, 600 ,3
            rnd_rgb = cv2.resize(rnd_rgb, (600, 800))
            # get rgb shape
            rnd_rgb_shape = rnd_rgb.shape[:2]
            rgb_h, rgb_w = rnd_rgb_shape

            rgb_darker_factor = random.random()
            if rgb_darker_factor <= 0.7:
                rgb_darker_factor = 1.0
            rnd_rgb = (rnd_rgb * rgb_darker_factor).astype(np.uint8)
            rnd_mask = cv2.imread(mask_list[rgb_rnd_idx]) # shape = 800, 600 ,3 
            rnd_mask = cv2.resize(rnd_mask, (600, 800), cv2.INTER_NEAREST)
            rnd_mask = np.where(rnd_mask>=1, 1, 0)
            zero_rgb = np.zeros(background_shape)
            zero_mask = np.zeros(background_shape)
            zero_rgb[480:480+rgb_h, 0:0+rgb_w] = rnd_rgb
            zero_mask[480:480+rgb_h, 0:0+rgb_w] += rnd_mask
            compose_mask[480:480+rgb_h, 0:0+rgb_w] += rnd_mask
            zero_rgb = zero_rgb.astype(np.uint8)
            bg_image = np.where(zero_mask >= 1, zero_rgb, original_bg)
            bg_image = bg_image.astype(np.uint8)

            # right
            rgb_rnd_idx = random.randint(0, rgb_len-1)
            rnd_rgb = cv2.imread(rgb_list[rgb_rnd_idx]) # shape = 800, 600 ,3 
            rnd_rgb = cv2.resize(rnd_rgb, (600, 800))
            rgb_darker_factor = random.random()
            if rgb_darker_factor <= 0.5:
                rgb_darker_factor = 0.5
            rnd_rgb = (rnd_rgb * rgb_darker_factor).astype(np.uint8)
            rnd_mask = cv2.imread(mask_list[rgb_rnd_idx]) # shape = 800, 600 ,3 
            rnd_mask = cv2.resize(rnd_mask, (600, 800), cv2.INTER_NEAREST)
            rnd_mask = np.where(rnd_mask>=1, 1, 0)
            zero_rgb = np.zeros(background_shape)
            zero_mask = np.zeros(background_shape)
            zero_rgb[480:480+rgb_h, 120:120+rgb_w] = rnd_rgb
            zero_mask[480:480+rgb_h, 120:120+rgb_w] += rnd_mask
            compose_mask[480:480+rgb_h, 120:120+rgb_w] += rnd_mask
            zero_rgb = zero_rgb.astype(np.uint8)
            bg_image = np.where(zero_mask >= 1, zero_rgb, bg_image)

            output_mask = np.where(compose_mask >= 1, 255, 0)
            bg_image = bg_image.astype(np.uint8)
            output_mask = output_mask.astype(np.uint8)
            bg_image = cv2.resize(bg_image, (540, 960))
            output_mask = cv2.resize(output_mask, (540, 960), cv2.INTER_NEAREST)

            image_loader.save_images(rgb=bg_image, mask=output_mask, prefix='{0}_idx_{1}_multiple_objects'.format(name, idx))

            # masked_image = bg_image * (output_mask / 255)
            # masked_image = masked_image.astype(np.uint8)
            # concat_img = cv2.hconcat([bg_image, output_mask, masked_image]) # original_rgb * (original_mask/255)
            # cv2.imshow('test', concat_img)
            # cv2.waitKey(0)
            


            # """2. change augmented bg (color aug + rgb shift)"""
            # # random shift original rgb and mask
            
            sift_rgb = original_rgb.copy()
            sift_mask = original_mask.copy()
            sift_rgb , sift_mask = image_loader.image_random_translation(rgb=sift_rgb, mask=sift_mask, min_dx=0, min_dy=0, max_dx=max_dx, max_dy=max_dy)
            # get random background idx
            bg_rnd_idx = random.randint(0, len(bg_list)-1)
            # load bg img
            original_bg = cv2.imread(bg_list[bg_rnd_idx])

            rgb_darker_factor = random.random()
            if rgb_darker_factor <= 0.8:
                rgb_darker_factor = 1.0
            original_bg = (original_bg * rgb_darker_factor).astype(np.uint8)

            rgb_darker_factor = random.random()
            if rgb_darker_factor <= 0.7:
                rgb_darker_factor = 1.0
            sift_rgb = (sift_rgb * rgb_darker_factor).astype(np.uint8)

            # resize bg img
            bg_image = image_loader.resize_bg_image(bg_image=original_bg, rgb_shape=original_rgb.shape)
            
            bg_img_whitout_rgb = np.where(
                        sift_mask == 255, 0, bg_image)

            rgb_img_only_object = np.where(sift_mask == 255, sift_rgb, 0)

            compose_aug_rgb = cv2.add(bg_img_whitout_rgb, rgb_img_only_object)

            image_loader.save_images(rgb=compose_aug_rgb, mask=sift_mask.copy(), prefix='{0}_idx_{1}_change_bg_augmented'.format(name, idx))