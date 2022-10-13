import tensorflow as tf
import numpy as np
import cv2
import os
from models.model_builder import ModelBuilder
import glob
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--video_dir",    type=str,
                    help="Dataset directory", default='./datasets/new_bg')
parser.add_argument("--video_result_dir", type=str,
                    help="Test result save directory", default='/home/park/0704_capture/videos/results/')


args = parser.parse_args()

if __name__ == '__main__':
    video_list = os.path.join(args.video_dir, '*.mp4')
    video_list = glob.glob(video_list)

    os.makedirs(args.video_result_dir, exist_ok=True)

    print(video_list)

    for video_idx, video_file in enumerate(video_list):
        video_idx += 1

        if os.path.isfile(video_file):	
            cap = cv2.VideoCapture(video_file)
        else:
            raise('cannot find file : {0}'.format(video_file))

        # Get camera FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 30
        # Frame width size
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # Frame height size
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_size = (frameWidth, frameHeight)
        print('frame_size={0}'.format(frame_size))
        

        frame_idx = 0
        while True:
            print(frame_idx)
            retval, frame = cap.read()

            frame_idx+=1

            if not(retval):
                break
            
            original_frame_shape = frame.shape

            
            cv2.imwrite('./datasets/new_bg/video_frame/' + str(frame_idx) + 'coex_bg_rgb.jpg', frame)


        if cap.isOpened():
            cap.release()
            

