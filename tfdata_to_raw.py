import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import os


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--dataset_type", type=str, help="테스트할 데이터셋 선택  'binary' or 'semantic'", default='semantic')
parser.add_argument("--dataset_nums", type=int, help="테스트 이미지 개수  'binary' or 'semantic'", default=100)
args = parser.parse_args()

DATASET_DIR = args.dataset_dir
DATASET_TYPE = args.dataset_type
DATASET_NUMS = args.dataset_nums
IMAGE_SIZE = (640, 360)

if __name__ == "__main__":
    # tf.data.experimental.enable_debug_mode()
    train_data = tfds.load('human_segmentation',
                               data_dir=DATASET_DIR, split='train')
    # number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()

    os.makedirs('./datasets/new_human_segmentation/background/rgb/', exist_ok=True)
    os.makedirs('./datasets/new_human_segmentation/background/mask/', exist_ok=True)
    os.makedirs('./datasets/new_human_segmentation/foreground/rgb/', exist_ok=True)
    os.makedirs('./datasets/new_human_segmentation/foreground/mask/', exist_ok=True)
    number_train = 84926
    print("Nuber of train dataset = {0}".format(number_train))
    idx = 0
    train_data.take(number_train)
    for sample in tqdm(train_data, total=number_train):
    # for sample in train_data:
        idx += 1
        img = sample['rgb'].numpy()
        mask = sample['gt'].numpy()


        is_object = np.any(mask>0)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 라벨에 객체가 없을 때 (bg 이미지)
        if is_object == False:
            

            cv2.imwrite('./datasets/new_human_segmentation/background/rgb/' + str(idx) + '_bg_rgb.jpg', img)
            cv2.imwrite('./datasets/new_human_segmentation/background/mask/' + str(idx) + 'bg_mask.png', mask)

        else:
            cv2.imwrite('./datasets/new_human_segmentation/foreground/rgb/' + str(idx) + '_bg_rgb.jpg', img)
            cv2.imwrite('./datasets/new_human_segmentation/foreground/mask/' + str(idx) + 'bg_mask.png', mask)
