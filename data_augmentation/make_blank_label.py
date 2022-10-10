import numpy as np
import cv2
import glob
import os
import argparse

parser = argparse.ArgumentParser()
name = 'remove_human_nyu'
parser.add_argument("--rgb_path",     type=str,   help="raw image path", default='./raw_data/raw_datasets/bg_img/{0}/rgb/'.format(name))
parser.add_argument("--output_path",     type=str,   help="raw image path", default='./raw_data/raw_datasets/bg_img/{0}/select/'.format(name))

args = parser.parse_args()


if __name__ == '__main__':
    
    os.makedirs(args.output_path, exist_ok=True)
    resized_rgb_path = args.output_path + 'rgb/'
    resized_mask_path = args.output_path + 'mask/'
    os.makedirs(resized_rgb_path, exist_ok=True)
    os.makedirs(resized_mask_path, exist_ok=True)

    rgb_path = os.path.join(args.rgb_path, '*.jpg')
    rgb_list = glob.glob(rgb_path)

    idx = 0
    for rgb_idx in rgb_list:
        print(rgb_idx)
        idx += 1
        file_name = rgb_idx.split('/')[5]
        file_name = file_name.split('.')[0]
        
        # if not os.path.isfile(args.obj_mask_path + file_name + '.png'):
        rgb_img = cv2.imread(rgb_idx)

        h, w = rgb_img.shape[:2]
        print(h)
        if h >= 1500:
            resize_factor = 2
        elif h >= 2000:
            resize_factor = 2.5
        elif h >= 3000:
            resize_factor = 3
        else:
            resize_factor = 1
        
        if resize_factor != 1:
            new_w = int(w / resize_factor)
            new_h = int(h / resize_factor)
            rgb_img = cv2.resize(rgb_img, (new_w, new_h))
        else:
            new_w = w
            new_h = h
        

        if name == 'remove_human_nyu':
            rgb_img = rgb_img[10:new_h-10, 10:new_w-10]
        
        zero_mask = np.zeros(rgb_img.shape[:2], dtype=np.uint8)
        zero_mask = np.expand_dims(zero_mask, axis=-1)

        cv2.imwrite(resized_rgb_path + '{0}_{1}_'.format(name, idx) + 'rgb.jpg', rgb_img)
        cv2.imwrite(resized_mask_path + '{0}_{1}_'.format(name, idx) + 'mask.png', zero_mask)
        # cv2.imwrite(args.obj_mask_path + file_name + '.png', zero_mask)
