import cv2
import argparse
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
from models.model_builder import ModelBuilder
import timeit

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",     type=int,
                    help="Evaluation batch size", default=1)
parser.add_argument("--num_classes",     type=int,
                    help="Model num classes", default=2)
parser.add_argument("--image_size",     type=tuple,
                    help="Model image size (input resolution)", default=(640, 360))
parser.add_argument("--video_dir",    type=str,
                    help="Dataset directory", default='/home/park/0708_capture/videos')
parser.add_argument("--video_result_dir", type=str,
                    help="Test result save directory", default='/home/park/0708_capture/videos/results/')
parser.add_argument("--checkpoint_dir", type=str,
                    help="Setting the model storage directory", default='./checkpoints/')
parser.add_argument("--weight_name", type=str,
                    help="Saved model weights directory", default='1011/_1011_pidnet-b16-ep100-lr0.005-focal+aux+boundary-adam-640x360-multigpu-semanticSeg_best_loss.h5')

args = parser.parse_args()


if __name__ == '__main__':
    # model = ModelBuilder(image_size=args.image_size, num_classes=args.num_classes).build_model()
    

    model = ModelBuilder(image_size=args.image_size,
                                  num_classes=args.num_classes, use_weight_decay=False, weight_decay=0)
    model = model.build_model(model_name='pidnet', training=False)

    model.load_weights(args.checkpoint_dir + args.weight_name, by_name=True)
    model.summary()


    # Camera
    frame_width = 720
    frame_height = 1280
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    while cv2.waitKey(1) < 0:
        ret, frame = capture.read()
        
        start_t = timeit.default_timer()
        
        frame = frame[40:40+640, 360:360+360]
        h, w = frame.shape[:2]
        # print(frame.shape)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = tf.image.resize(img, size=args.image_size,
                method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32)
        # img = preprocess_input(x=img, mode='torch')
        img /= 255
        
        img = tf.expand_dims(img, axis=0)

        output = model.predict_on_batch(img)
        
        terminate_t = timeit.default_timer()
        
        FPS = int(1./(terminate_t - start_t ))

        # output = tf.where(output>=0.9, 1, 0)        
        
        output = tf.expand_dims(output, axis=-1)
        output = output[0]

        output = tf.image.resize(output, (h, w), tf.image.ResizeMethod.NEAREST_NEIGHBOR).numpy().astype(np.uint8)
        frame *= output
        output = output * 255
        
        cv2.putText(output, 'FPS : {0}'.format(str(FPS)),(50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (200, 50, 0), 3, cv2.LINE_AA)
        output_concat = np.concatenate([output, output, output], axis=-1)

        concat_img = cv2.hconcat([frame, output_concat])
        cv2.imshow("VideoFrame", concat_img)

    capture.release()
    cv2.destroyAllWindows()
