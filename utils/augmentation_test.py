"""
Test augmentation (test ok)
"""
import tensorflow as tf
import tensorflow_addons as tfa

# Random rotate
upper = 20 * (self.pi/180.0) # Degrees to Radian
rand_degree = tf.random.uniform([], minval=0., maxval=upper)
img = tfa.image.rotate(img, rand_degree, interpolation='bilinear')
labels = tfa.image.rotate(labels, rand_degree, interpolation='nearest')


# Random shift
shift_x_max = self.image_size[1] / 3
shift_y_max = self.image_size[0] / 4
max_x = tf.random.uniform([], minval=0., maxval=shift_x_max)
max_y = tf.random.uniform([], minval=0., maxval=shift_y_max)
max_x = tf.cast(max_x, dtype=tf.int32)
max_y = tf.cast(max_y, dtype=tf.int32)
if tf.random.uniform([]) > 0.5:
    max_x *= -1
if tf.random.uniform([]) > 0.5:
    max_y *= -1
img = tfa.image.translate_xy(image=img, translate_to=[max_x, max_y], replace=0)
concat_labels = tf.concat([labels, labels, labels], axis=-1)
concat_labels = tfa.image.translate_xy(image=concat_labels, translate_to=[max_x, max_y], replace=0)
concat_labels = concat_labels[:, :, 0]
concat_labels = tf.expand_dims(concat_labels, axis=-1)