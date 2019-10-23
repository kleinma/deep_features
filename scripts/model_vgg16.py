import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from model_base import ModelBase

class ModelVGG16(ModelBase):
  """ The VGG16 network model """
  def build_fully_convolutional_network(self):
    # Start with the model with fully connected (dense) layers
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
    # Note: Don't add the flatten layer at the top. That is done in the
    # build_classifier_network() function in ModelBase. We want to leave this
    # portion of the network fully convolutional.
    conv1 = Conv2D(4096, 7, activation=act_fc1, name='fc1')
    conv2 = Conv2D(4096, 1, activation=act_fc2, name='fc2')
    conv3 = Conv2D(1000, 1, activation=act_pred, name='predictions')
    # (See note above) flatten = Flatten(name='flatten')

    # Add these new layers on top of the block5_pool layer
    # (See note above) out = flatten(conv3(conv2(conv1(mid))))
    out = conv3(conv2(conv1(mid)))

    # And create the new model
    model_fcn = Model(inp, out, name='fcn')

    # Now, replace the convolutional layer weights with the fully connected
    # weights. First, get the filters from the conv layers so we can get the
    # shape to easily reshape the fc weights.
    filters_conv1, _ = conv1.get_weights()
    filters_conv2, _ = conv2.get_weights()
    filters_conv3, _ = conv3.get_weights()

    # Now set the weights, reshaping the fc weights
    conv1.set_weights([tf.reshape(filters_fc1, filters_conv1.shape).numpy(), biases_fc1])
    conv2.set_weights([tf.reshape(filters_fc2, filters_conv2.shape).numpy(), biases_fc2])
    conv3.set_weights([tf.reshape(filters_pred, filters_conv3.shape).numpy(), biases_pred])

    return model_fcn

  def init_desired_output_layers(self):
    """ Grab the last convolutional layer from each block """
    """
    return ['input_1',
            'block1_conv1', 'block1_conv2', 'block1_pool',
            'block2_conv1', 'block2_conv2', 'block2_pool',
            'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_pool',
            'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_pool',
            'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_pool']
    """
    return ['block5_pool','fc1','fc2','predictions']

  def init_checkpoint_dir(self):
    return './checkpoints/vgg16'

  def preprocess_data(self, input_data):
    """
    Use preprocess_input function in tensorflow.keras.applications.vgg16.

    Subtract mean of all training images [103.939, 116.779, 123.68]. Do not
    normalize by dividing by standard deviation.
    https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py#L18

    Parameters
    ----------
    input_data : np.ndarray
      The input image

    Returns
    -------
    np.ndarray
      The preprocessed input image (default returns input)
    """
    return preprocess_input(input_data)


if __name__ == "__main__":
  import matplotlib.pyplot as plt
  from scipy import misc

  model = ModelVGG16()
  model.fcn.summary()
  model.classifier.summary()

  import numpy as np
  image = np.random.rand(1024,768,3)
  network_output = model.fcn_pass(image)

  for name, output, _ in network_output:
    print('layer_name = {}, output_value\'s shape = {} '.format(name, output.shape))

  img = misc.face()
  network_output = model.fcn_pass(img)


  for name, output, scale in network_output:
    num_channels = output.shape[-1]
    print('Layer {} has {} channels and a scale of {}'.format(name, num_channels, scale))
    for channel in range(num_channels):
      # _ = plt.hist(output[:,channel].flatten(), bins='auto')
      # plt.title(name + ", channel = {}".format(channel))
      # plt.show()
      print('name = {}, channel = {}, max = {}'.format(name, channel, np.max(output[:,:,:,channel])))

  # Now check to make sure the output of both networks run on imagenet2012
  # validation data have the same result
  import tensorflow_datasets as tfds
  data = tfds.load("imagenet2012", data_dir='data')
  data_val = data['validation']

  tot_right = 0
  tot = 0
  for ex in data_val.take(1000):
    image = ex['image']
    label = ex['label']

    image = tf.image.resize(image,(224,224))
    image_proc = model.expand_to_bhwc(model.preprocess_data(image))

    tot = tot + 1
    # print('{} = {} ?'.format(tf.math.argmax(model.classifier(image_proc),1).numpy()[0], label.numpy()))
    if tf.math.argmax(model.classifier(image_proc),1).numpy()[0] == label.numpy():
      tot_right = tot_right + 1

  print('percentage right = {}%'.format(100*tot_right/tot))

  # Just look at some of the feature maps
  from plotting_utils import plot_layer_output

  img = misc.face()
  network_output = model.fcn_pass(img)

  for layer_output in network_output:
    plot_layer_output(layer_output, 20, 39)
