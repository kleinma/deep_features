import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten

"""
def dense_init(shape, dtype=None):
  return tf.constant(tf.reshape([tf.dtypes.cast(i,tf.dtypes.float32) for i in range(72)],(18,4)))

model_dense = tf.keras.Sequential([Flatten(input_shape=(3,3,2)), Dense(4, kernel_initializer=dense_init)], name='dense')

model_fcn = tf.keras.Sequential([Conv2D(4,(3,3), input_shape=(3,3,2)), Flatten()], name='fcn')

model_dense.summary()
model_fcn.summary()

filters_dense, biases_dense = model_dense.layers[1].get_weights()
filters_fcn, biases_fcn = model_fcn.layers[0].get_weights()

print('filters_dense.shape=\n{}\n'.format(filters_dense.shape))
print('filters_dense=\n{}\n'.format(filters_dense))
print('biases_dense=\n{}\n'.format(biases_dense))
print('filters_fcn.shape=\n{}\n'.format(filters_fcn.shape))
print('filters_fcn=\n{}\n'.format(filters_fcn))
print('biases_fcn=\n{}\n'.format(biases_fcn))

model_fcn.layers[0].set_weights([tf.reshape(filters_dense,filters_fcn.shape), biases_dense])

filters_fcn, biases_fcn = model_fcn.layers[0].get_weights()

print('filters_fcn.shape\n= {}\n'.format(filters_fcn.shape))
print('filters_fcn=\n{}\n'.format(filters_fcn))
print('biases_fcn=\n{}\n'.format(biases_fcn))

import numpy as np
input_array = np.expand_dims(tf.dtypes.cast(np.random.rand(3,3,2),tf.dtypes.float32),0)

print('input_array =\n{}\n'.format(input_array))

print('Try model_dense')
print(model_dense(input_array))
print('Try model_fcn')
print(model_fcn(input_array))

"""

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
model_dense = VGG16()

# Get the input
inp = model_dense.input
# Get the output of the last layer before flattening (block5_pool)
mid = model_dense.layers[18].output

# Get the weights and activations of the three fully connected layers
filters_fc1, biases_fc1 = model_dense.layers[20].get_weights()
filters_fc2, biases_fc2 = model_dense.layers[21].get_weights()
filters_pred, biases_pred = model_dense.layers[22].get_weights()

act_fc1 = model_dense.layers[20].activation
act_fc2 = model_dense.layers[21].activation
act_pred = model_dense.layers[22].activation

# Create the new convolutional layers
conv1 = Conv2D(4096, 7, activation=act_fc1, name='fc1')
conv2 = Conv2D(4096, 1, activation=act_fc2, name='fc2')
conv3 = Conv2D(1000, 1, activation=act_pred, name='predictions')
flatten = Flatten(name='flatten')

# Add these new layers on top of the block5_pool layer
out = flatten(conv3(conv2(conv1(mid))))

# And create the new model
model_fcn = Model(inp, out)

# Now, replace the convolutional layer weights with the fully connected weights
# First, get the filters from the conv layers so we can get the shape to easily
# reshape the fc weights
filters_conv1, _ = conv1.get_weights()
filters_conv2, _ = conv2.get_weights()
filters_conv3, _ = conv3.get_weights()

# Now set the weights, reshaping the fc weights
conv1.set_weights([tf.reshape(filters_fc1, filters_conv1.shape).numpy(), biases_fc1])
conv2.set_weights([tf.reshape(filters_fc2, filters_conv2.shape).numpy(), biases_fc2])
conv3.set_weights([tf.reshape(filters_pred, filters_conv3.shape).numpy(), biases_pred])

# Now check to make sure the output of both networks run on imagenet2012
# validation data have the same result
import tensorflow_datasets as tfds
data = tfds.load("imagenet2012", data_dir='data')
data_val = data['validation']

tot_right = 0
tot = 0
for ex in data_val.take(100):
  image = ex['image']
  label = ex['label']

  image_proc = preprocess_input(tf.expand_dims(tf.image.resize(image,(224,224)),0))

  tot = tot + 1
  if tf.math.argmax(model_dense(image_proc),1)[0] == tf.math.argmax(model_fcn(image_proc),1)[0]:
    tot_right = tot_right + 1

print('percentage right = {}%'.format(100*tot_right/tot))
