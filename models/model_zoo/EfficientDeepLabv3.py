import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from .efficientDeeplab.efficientnetv2 import EfficientNetV2S
from .efficientDeeplab.light_deeplabv3 import deepLabV3Plus
bn_mom = 0.1

class EfficientDeepLabV3(object):
    def __init__(self, input_shape=(640, 360, 3), num_classes=2, training=True):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.training = training
        
    def classifier(self, x: tf.Tensor, num_classes: int = 2, upper: int = 8,
                   prefix: str = None, activation: str = None) -> tf.Tensor:
        """
            Segmentation 출력을 logits 형태로 만듭니다. Conv2D + Upsampling (resize)
            Args: 
                x            (tf.Tensor)  : Input tensor (segmentation model output)
                num_classes  (int)        : Number of classes to classify
                upper        (int)        : Upsampling2D layer extension factor
                prefix       (str)        : Set the final output graph name
                activation   (activation) : Set the activation function of the Conv2D layer.
                                            logits do not apply activation. ( Default : None)
                                         
            Returns:
                x            (tf.Tensor)  : Output tensor (logits output)
        """
        x = layers.Conv2D(num_classes, kernel_size=1, strides=1, activation=activation)(x)
        x = layers.UpSampling2D(size=(upper, upper),
                                interpolation='bilinear',
                                name=prefix)(x)
        if self.training == False:
            x = tf.math.argmax(x, axis=-1)
        return x
        
    def build(self) -> models.Model:
        
        base = EfficientNetV2S(input_shape=self.input_shape, num_classes=0)
        model_input = base.input
        
        skip_feature = base.get_layer('add_7').output
        x = base.get_layer('add_34').output

        # 'add_7' : 80, 45 1/8
        # 'add_34' : 20, 12, 1/32
        

        output = deepLabV3Plus(features=[skip_feature, x], base_channel=256, activation='swish')

        model_output = self.classifier(x=output, num_classes=self.num_classes, upper=8, prefix='output')

        model = models.Model(model_input, model_output)

        return model
    

if __name__ == '__main__':
    print('Test EfficientNet model')

    model = EfficientDeepLabV3(input_shape=(640, 360, 3), num_classes=2, augment=False).build()

    model.summary()
