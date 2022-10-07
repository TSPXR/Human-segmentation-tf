from models.model_builder import ModelBuilder
from utils.load_datasets import DatasetGenerator
import argparse
import time
import os
import tensorflow as tf
from tqdm import tqdm
from utils.get_flops import get_flops

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",      type=int,
                    help="Evaluation batch size", default=1)
parser.add_argument("--num_classes",     type=int,
                    help="Model num classes", default=1)
parser.add_argument("--image_size",      type=tuple,
                    help="Model image size (input resolution H,W)", default=(640, 360))
parser.add_argument("--dataset_dir",     type=str,
                    help="Dataset directory", default='./datasets/')
parser.add_argument("--dataset_name",     type=str,
                    help="Dataset directory", default='human_segmentation')
parser.add_argument("--checkpoint_dir",  type=str,
                    help="Setting the model storage directory", default='./checkpoints/')
parser.add_argument("--weight_path",     type=str,
                    help="Saved model weights directory", default='1007/_1007_b16-e100-lr0.001-adam-640x360-no-augment-multiGPU_best_loss.h5')

# Prediction results visualize options
parser.add_argument("--visualize",  help="Whether to image and save inference results", action='store_true')
parser.add_argument("--result_dir",      type=str,
                    help="Test result save directory", default='./results/')

args = parser.parse_args()

if __name__ == '__main__':
    # Create result plot image path
    os.makedirs(args.result_dir, exist_ok=True)
    
    # Configuration test(valid) datasets
    dataset_config = DatasetGenerator(data_dir=args.dataset_dir, image_size=args.image_size,
                                       batch_size=args.batch_size, dataset_name=args.dataset_name)
    dataset = dataset_config.get_testData(valid_data=dataset_config.valid_data)
    test_steps = dataset_config.number_valid // args.batch_size

    # Model build and load pre-trained weights
    # model = ModelBuilder(image_size=args.image_size, num_classes=args.num_classes).build_model()
    from models.model_zoo.PIDNet import PIDNet
    
    model = PIDNet(input_shape=(*args.image_size, 3), m=2, n=3, num_classes=args.num_classes,
                       planes=32, ppm_planes=96, head_planes=128, augment=False, training=False).build()
    model.load_weights(args.checkpoint_dir + args.weight_path)
    model.summary()

    # Model warm up
    _ = model.predict(tf.zeros((1, args.image_size[0], args.image_size[1], 3)))

    # Set evaluate metrics and Color maps
    # TODO: ADD metrics

    # Set plot configs
    rows = 1
    cols = 2
    batch_idx = 0
    avg_duration = 0
    batch_index = 0

    # Predict
    for x, gt, original_img in tqdm(dataset, total=test_steps):
        # Check inference time
        start = time.process_time()
        prediction = model.predict_on_batch(x)
        duration = (time.process_time() - start)

        # Argmax prediction
        # pred = tf.math.argmax(prediction, axis=-1, output_type=tf.int32)


        tf.keras.preprocessing.image.save_img(args.result_dir + str(batch_index)+'.png', prediction[0])
        batch_index += 1

        avg_duration += duration
        batch_idx += 1

    print('Model FLOPs {0}'.format(get_flops(model=model, batch_size=1)))
    print('avg inference time : {0}sec.'.format((avg_duration / dataset_config.number_valid)))
    