import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from model_base import ModelBase

class ModelVGG16(ModelBase):
  """ The VGG16 network model """
  def build_fully_convolutional_network(self):
    return VGG16()

  def init_desired_output_layers(self):
    """ Grab the last convolutional layer from each block """
    return [2, 5, 9, 13, 17, 19]

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

  for name, output in network_output:
    print('layer_name = {}, output_value\'s shape = {} '.format(name, output.shape))
