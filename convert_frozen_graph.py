import tensorflow as tf
from models.model_builder import ModelBuilder
from tensorflow.keras.models import Model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from models.model_zoo.pidnet.pidnet import PIDNet
import argparse 
import os

# ONNX Convert
# 1. pip install tf2onnx
# (Frozen graph)
# 2. python -m tf2onnx.convert --input ./checkpoints/new_tfjs_frozen/frozen_graph.pb --output ./checkpoints/onnx_test.onnx --inputs x:0 --outputs Identity:0 --opset 10

# quantize_uint8
# tensorflowjs_converter ./checkpoints/frozen_result/frozen_graph.pb ./checkpoints/converted_tfjs/ --input_format=tf_frozen_model --output_node_names='Identity' --quantize_float16 
# tensorflowjs_converter ./checkpoints/frozen_result/frozen_graph.pb ./checkpoints/converted_tfjs/ --input_format=tf_frozen_model --output_node_names='Identity' --quantize_uint8 '*' 
# tensorflowjs_converter ./checkpoints/new_tfjs_frozen/frozen_graph.pb ./checkpoints/converted_tfjs/ --input_format=tf_frozen_model --output_node_names='Identity' --quantize_uint8 '*'
# tensorflowjs_converter --input_format=tf_frozen_model --output_node_names='Identity' ./checkpoints/new_tfjs_frozen/frozen_graph.pb ./checkpoints/converted_tfjs/


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir",   type=str,    help="Set the model storage directory",
                    default='./checkpoints/')
parser.add_argument("--model_weights", type=str,     help="Saved model weights directory",
                    default='./checkpoints/1014/_1014_pidnet-b16-ep30-lr0.001-ce2+aux0.4+boundary2-adam-640x360-singleGpu-semanticSeg-shift_aug_bn0.9-single_best_loss.h5')
parser.add_argument("--num_classes",          type=int,    help="Set num classes for model and post-processing",
                    default=2)  
parser.add_argument("--image_size",          type=tuple,    help="Set image size for priors and post-processing",
                    default=(640, 360))
parser.add_argument("--gpu_num",          type=int,    help="Set GPU number to use(When without distribute training)",
                    default=0)
parser.add_argument("--frozen_dir",   type=str,    help="Path to save frozen graph transformation result",
                    default='./checkpoints/frozen_result/')
parser.add_argument("--frozen_name",   type=str,    help="Frozen graph file name to save",
                    default='frozen_graph')
            
args = parser.parse_args()

if __name__ == '__main__':
    tf.config.set_soft_device_placement(True)
    # tf.config.run_functions_eagerly(True)
    gpu_number = '/device:GPU:' + str(args.gpu_num)
    with tf.device(gpu_number):
        model = ModelBuilder(image_size=args.image_size,
                                  num_classes=args.num_classes, use_weight_decay=False, weight_decay=0)
        model = model.build_model(model_name='pidnet', training=False)
            
        model.load_weights(args.model_weights, by_name=True)
        model.summary()

        export_path = os.path.join('checkpoints', 'export_path', '1')
        os.makedirs(export_path, exist_ok=True)

        tf.keras.models.save_model(
            model,
            export_path,
            overwrite=True,
            include_optimizer=False,
            save_format=None,
            signatures=None,
            options=None
        )
        
        print(model)
        #path of the directory where you want to save your model
        frozen_out_path = args.frozen_dir
        # name of the .pb file
        frozen_graph_filename = args.frozen_name
        # Convert Keras model to ConcreteFunction
        full_model = tf.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
        # Get frozen ConcreteFunction
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        layers = [op.name for op in frozen_func.graph.get_operations()]
        
        print("Frozen model layers: ")
        for layer in layers:
            print(layer)
        
        print("Frozen model inputs: {0}".format(frozen_func.inputs))
        print("Frozen model outputs: {0}".format(frozen_func.outputs))
        
        # Save frozen graph to disk
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                        logdir=frozen_out_path,
                        name=f"{frozen_graph_filename}.pb",
                        as_text=False)
        # Save its text representation
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                        logdir=frozen_out_path,
                        name=f"{frozen_graph_filename}.pbtxt",
                        as_text=True)