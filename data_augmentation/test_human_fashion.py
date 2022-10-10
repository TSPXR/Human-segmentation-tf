import numpy as np
import cv2
import glob
import os
import argparse
import natsort
import matplotlib.pyplot as plt

name = 'human_fahsion_1_dataset'

parser = argparse.ArgumentParser()
parser.add_argument("--rgb_path",     type=str,   help="raw image path", default='./raw_data/raw_datasets/{0}/rgb/'.format(name))
parser.add_argument("--mask_path",     type=str,   help="raw mask path", default='./raw_data/raw_datasets/{0}/mask/'.format(name))
parser.add_argument("--test",     type=str, default=False)
parser.add_argument("--output_path",     type=str,   help="Path to save the conversion result", default='./raw_data/raw_datasets/{0}/select/'.format(name))

args = parser.parse_args()

class ImageAugmentationLoader():
    def __init__(self, args):
        """
        Args
            args  (argparse) : inputs (rgb, mask)
                >>>    rgb : RGB image.
                >>>    mask : segmentation mask.
        """
        self.RGB_PATH = args.rgb_path
        self.MASK_PATH = args.mask_path
        
        self.OUTPUT_PATH = args.output_path

        self.OUT_RGB_PATH = self.OUTPUT_PATH + 'rgb/'
        self.OUT_MASK_PATH = self.OUTPUT_PATH + 'gt/'
        
        os.makedirs(self.OUT_RGB_PATH, exist_ok=True)
        os.makedirs(self.OUT_MASK_PATH, exist_ok=True)
        

        self.rgb_list = glob.glob(os.path.join(self.RGB_PATH+'*.jpg'))
        self.rgb_list = natsort.natsorted(self.rgb_list,reverse=True)

        self.mask_list = glob.glob(os.path.join(self.MASK_PATH+'*.png'))
        self.mask_list = natsort.natsorted(self.mask_list,reverse=True)

    def save_images(self, rgb, mask, prefix):
        cv2.imwrite(self.OUT_RGB_PATH + prefix +'_rgb.jpg', rgb)
        cv2.imwrite(self.OUT_MASK_PATH + prefix + '_mask.png', mask)

                                                              
if __name__ == '__main__':
    """
    Image augmentation can be selected according to the option using the internal function of ImageAugmentationLoader.
    """
    from tqdm import tqdm

    image_loader = ImageAugmentationLoader(args=args)
    rgb_list = image_loader.rgb_list
    mask_list = image_loader.mask_list


    # for idx in range(len(rgb_list)):
    for idx in tqdm(range(len(rgb_list)), total=len(rgb_list)):
        # print(idx)
        original_rgb = cv2.imread(rgb_list[idx])
        original_mask = cv2.imread(mask_list[idx])
        if name == 'human_fahsion_1_dataset':
            original_mask = np.where(original_mask==(90, 6 ,69), 0, 255)
        
        kernel_size_row = 3
        kernel_size_col = 3
        kernel = np.ones((kernel_size_row, kernel_size_col), np.uint8)
        
        original_rgb_shape = original_rgb.shape[:2]
        original_mask_shape = original_mask.shape[:2]

        if original_rgb_shape != original_mask_shape:
            h, w = original_rgb_shape
            original_mask = cv2.resize(original_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        original_mask = original_mask.astype(np.uint8)
        original_mask = cv2.cvtColor(original_mask, cv2.COLOR_BGR2GRAY)
        original_mask = np.where(original_mask >= 1, 1, 0).astype(np.uint8)

        contours, _ = cv2.findContours(
                original_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 컨투어 전체 병합
        contour_list = []
        len_contour = len(contours)
        
        for i in range(len_contour):
            drawing = np.zeros_like(original_mask, np.uint8)  # create a black image
            img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)            
            contour_list.append(img_contour)  
        original_mask = sum(contour_list)
        

        if len(contour_list) != 0:

        # 병합된 컨투어 마스크에서 외부 노이즈 제거
            compose_contours, _ = cv2.findContours(
                original_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 컨투어가 두 개 이상일 때만
        
            compose_contours_len = len(compose_contours)
            if compose_contours_len == 1:
            # for i in range(len(compose_contours)):
            #     contour_area = cv2.contourArea(compose_contours[i])
                
            #     if contour_area >= 50:
            #         original_mask = cv2.drawContours(original_mask, compose_contours, i, (255, 255, 255), -1)

            #     # 넓이가 50 이하의 작은 컨투어만 0으로 칠함
            #     else:
            #         print('draw zeros')
            #         original_mask = cv2.drawContours(original_mask, compose_contours, i, (0, 0, 0), -1)

            #         test_mask = np.expand_dims(original_mask, axis=-1)
            #         test_mask = np.concatenate([test_mask, test_mask, test_mask], axis=-1)
            #         masked_image = original_rgb * (test_mask / 255)
            #         masked_image = masked_image.astype(np.uint8)
            #         concat_img = cv2.hconcat([test_mask, masked_image]) # original_rgb * (original_mask/255)
            #         cv2.imshow('test', concat_img)
            #         cv2.waitKey(0)



                original_mask = cv2.erode(original_mask, kernel, iterations=1)  #// make dilation image

                # zero_maks = np.zeros(original_mask.shape, np.uint8)
                # zero_maks = cv2.drawContours(zero_maks, draw_contours, -1, 1, thickness=-1)


                # original_mask += zero_maks
                original_mask = np.where(original_mask>=1, 255, 0).astype(np.uint8)
                original_mask = np.expand_dims(original_mask, axis=-1)
                
                
                if args.test:
                    test_mask = np.concatenate([original_mask, original_mask, original_mask], axis=-1)
                    masked_image = original_rgb * (test_mask / 255)
                    masked_image = masked_image.astype(np.uint8)
                    concat_img = cv2.hconcat([original_rgb, test_mask, masked_image]) # original_rgb * (original_mask/255)
                    cv2.imshow('test', concat_img)
                    cv2.waitKey(0)

                image_loader.save_images(rgb=original_rgb, mask=original_mask, prefix='{0}_{1}'.format(name, idx))