import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from model_base import ModelBase

class ModelVGG16(ModelBase):
  """ The VGG16 network model """
  def build_fully_convolutional_network(self):
    return VGG16()

  def init_desired_output_layers(self):
    """ Grab the last convolutional layer from each block """
    return ['block5_conv3']

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
  model = ModelVGG16()
  model.fcn.summary()
  model.classifier.summary()

  import numpy as np
  image = np.random.rand(448,224,3)
  network_output = model.fcn_pass(image)

  for name, output, _ in network_output:
    print('layer_name = {}, output_value\'s shape = {} '.format(name, output.shape))

  from scipy import misc
  img = misc.face()
  network_output = model.fcn_pass(img)

  import matplotlib.pyplot as plt

  for name, output, scale in network_output:
    num_channels = output.shape[-1]
    print('Layer {} has {} channels and a scale of {}'.format(name, num_channels, scale))
    for channel in range(num_channels):
      # _ = plt.hist(output[:,channel].flatten(), bins='auto')
      # plt.title(name + ", channel = {}".format(channel))
      # plt.show()
      print('name = {}, channel = {}, max = {}'.format(name, channel, np.max(output[:,:,:,channel])))
