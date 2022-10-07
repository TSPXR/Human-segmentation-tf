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
                    help="Model num classes", default=1)
parser.add_argument("--image_size",     type=tuple,
                    help="Model image size (input resolution)", default=(640, 360))
parser.add_argument("--video_dir",    type=str,
                    help="Dataset directory", default='/home/park/0708_capture/videos')
parser.add_argument("--video_result_dir", type=str,
                    help="Test result save directory", default='/home/park/0708_capture/videos/results/')
parser.add_argument("--checkpoint_dir", type=str,
                    help="Setting the model storage directory", default='./checkpoints/')
parser.add_argument("--weight_name", type=str,
                    help="Saved model weights directory", default='/1007/_1007_b16-e50-lr0.001-adam-640x360-no-augment-multiGPU-normalBCE_best_loss.h5')

args = parser.parse_args()


if __name__ == '__main__':
    # model = ModelBuilder(image_size=args.image_size, num_classes=args.num_classes).build_model()
    from models.model_zoo.PIDNet import PIDNet

    model = PIDNet(input_shape=(*args.image_size, 3), m=2, n=3, num_classes=args.num_classes,
                planes=32, ppm_planes=96, head_planes=128, augment=False, training=False).build()

    # from models.model_zoo.pidnet.pidnet import PIDNet
        
    # model = PIDNet(input_shape=(*args.image_size, 3), m=2, n=3, num_classes=args.num_classes,
    #                    planes=32, ppm_planes=96, head_planes=128, augment=False)
    # model.build((None, *args.image_size, 3))


    model.load_weights(args.checkpoint_dir + args.weight_name, by_name=True)
    model.summary()


    # Camera
    frame_width = 1280
    frame_height = 720
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    while cv2.waitKey(1) < 0:
        ret, frame = capture.read()
        

        
        start_t = timeit.default_timer()
        
        frame = frame[40:40+640, 180:180+360]
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

        
        output = output[0]
        
        output = tf.where(output>0.5, 1., 0.)
        
        output = tf.image.resize(output, (h, w), tf.image.ResizeMethod.NEAREST_NEIGHBOR).numpy().astype(np.uint8)
        frame *= output

        cv2.putText(output, 'FPS : {0}'.format(str(FPS)),(50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (200, 50, 0), 3, cv2.LINE_AA)
        cv2.imshow("VideoFrame", frame)

    capture.release()
    cv2.destroyAllWindows()
