import tensorflow as tf
import os
from collections import OrderedDict

class ModelBase():
  """
  The base class of a CNN model

  The model class specifies the fully convolutional and classifier networks. It
  also speficies where checkpoints are saved and loaded from. Finally, it
  defines a function `fcn_pass(input_data)` which returns the output of the
  fully convultional network (fcn), given the input data. This consists of an
  ordered dictionary with the keys being the layer name and the value being a
  numpy array containing the network output values. This is what is sent to the
  feature detector and feature descriptor.

  Attributes
  ----------
  fcn : tf.keras.Model
    Fully Convolutional Network
  classifier : tf.keras.Model
    The fully convolutional network `fcn` extended so that it is trainable
  desired_output_layers : List[str or int] or None
    List of indices or names of layers to include in network output
  output_functors : List[tf.keras.backend.function]
    Functions to compute the output of the desired network layers
  output_layer_names : List[str]
    The names of the desired output network layers
  latest_checkpoint_path : str
    Filename of the latest checkpoint
  best_checkpoint_path :str
    Filename of the best checkpoint
  checkpoint_dir : str
    The directory containing the checkpoint files

  """
  def __init__(self, is_trainable=False):
    """
    Build the models, their outputs, and set the checkpoint paths
    """
    self.fcn = self.build_fully_convolutional_network()
    self.classifier = self.build_classifier_network(self.fcn)

    # Define the output layers that we are interested in and create a list of the
    # functions to calculate the output and a list of the layers' names.
    self.desired_output_layers = self.init_desired_output_layers()
    self.define_output_functors_and_layer_names(self.desired_output_layers)

    # Used to save the latest checkpoint and the checkpoint with the best test
    # accuracy
    checkpoint_dir = self.init_checkpoint_dir()
    self.latest_checkpoint_path = checkpoint_dir + '/cp-{epoch:05d}.ckpt'
    self.best_checkpoint_path = checkpoint_dir + '/cp-best.ckpt'
    self.checkpoint_dir = os.path.dirname(self.latest_checkpoint_path)

  def init_checkpoint_dir(self):
    """Override to set checkpoint directory"""
    return './checkpoints/base'

  def build_fully_convolutional_network(self):
    """
    Override to create the fully connected network portion of the total model.
    Return tf.keras.Model
    """
    inputs = tf.keras.Input(shape=(None, None, None))
    return tf.keras.Model(inputs=inputs, outputs=inputs, name='fcn')

  def build_classifier_network(self, base_fcn):
    """
    Override to create the trainable network portion of the total model.

    Parameters
    ----------
    base_fcn : tf.keras.Model
      The fully convolutional portion of the total model

    Returns
    -------
    tf.keras.Model
      A trainable model with the fcn as the base
    """
    return tf.keras.Sequential([base_fcn, tf.keras.layers.Flatten()], name='classifier')

  def init_desired_output_layers(self):
    """
    Override to redefine default layers to be used in the network output.

    Returns
    -------
    List[str or int] or None
      List of indices or names of layers to include in network output
    """
    return None

  def define_output_functors_and_layer_names(self, desired_output_layers=None):
    """
    Define the functions to compute the desired output layers and their names.

    Parameters
    ----------
    desired_output_layers : List[int or str]
      List of indices or names of layers to include in network output

    Data Attributes Modified
    ------------------------
    output_functors : List[tf.keras.backend.function]
      Functions to compute the output of the desired network layers
    output_layer_names : List[str]
      The names of the desired output network layers
    """

    all_layer_names = [layer.name for layer in self.fcn.layers]

    fcn_input = self.fcn.input
    if desired_output_layers == None:
      # Use all the layers
      outputs = [layer.output for layer in self.fcn.layers]
      layer_names = [layer.name for layer in self.fcn.layers]
    else:
      outputs = []
      layer_names = []
      for layer_ind_or_name in desired_output_layers:
        if isinstance(layer_ind_or_name, str):
          # layer represented by its name
          # check if the name is in the list of names
          try:
            idx = all_layer_names.index(layer_ind_or_name)
          except ValueError as err:
            print(err)
            continue
        elif isinstance(layer_ind_or_name, int):
          # layer represented as an index number
          idx = layer_ind_or_name
        else:
          print('Entry of desired_output_layers, {}, is of type {}. Should be of type int or str. Not using.'.format(layer_ind_or_name, type(layer_ind_or_name)))
          continue
        # check if the index in the range
        if 0 <= idx <= len(self.fcn.layers)-1:
          outputs.append(self.fcn.layers[idx].output)
          layer_names.append(self.fcn.layers[idx].name)
        else:
          print('{} is out of range - MK'.format(idx))

    functors = [tf.keras.backend.function([fcn_input], [out]) for out in outputs]

    # Set data attributes
    self.output_functors = functors
    self.output_layer_names = layer_names

  def get_desired_output_layers(self):
    """
    Get the indices of the output layers we want

    Returns
    -------
    List of integers or strings (or None)
      Indices or names of layers we want in the output. Use None if want all.

    """
    return self.desired_output_layers

  def fcn_pass(self, input_data):
    """
    Get the names and outputs of each layer of the fully convolutional network

    Parameters
    ----------
    input_data : np.array
      input to the network

    Returns
    -------
    collections.OrderedDict (key: str, value: np.array)
      layer names (keys) matched with layer outputs (values)
    """
    # TODO - allow the following sizes of input_data:
    # size = 2: single channel image (HW format)
    # size = 3: N channel image (HWC format)
    # size = 4: Batch of M N channel images (BHWC format)
    layer_outs = [func([input_data]) for func in self.output_functors]
    layer_dict = OrderedDict()
    for name, output in zip(self.output_layer_names, layer_outs):
      layer_dict[name] = output[0]
    return layer_dict

  def save_best_weights(self):
    """
    Save a checkpoint designated as the best.

    This should only be called in save weights when the test_accuracy exceeds
    the self.best_test_accuracy.
    """
    self.classifier.save_weights(self.best_checkpoint_path)

  def save_weights(epoch, test_accuracy=None):
    """
    Saves a checkpoint for the given epoch.
    """
    self.classifier.save_weights(self.latest_checkpoint_path.format(epoch=epoch))
    if test_accuracy and test_accuracy > self.best_test_accuracy:
      self.best_test_accuracy = test_accuracy
      save_best_weights()

  def load_latest_weights(self):
    latest = tf.train.latest_checkpoint(self.latest_checkpoint_dir)
    self.classifier.load_weights(latest)

  def load_best_weights(self):
    self.classifier.load_weights(self.best_checkpoint_path)

if __name__ == "__main__":
  """ Test script to demonstrate model base class """
  model = ModelBase()
  model.fcn.summary()
  model.classifier.summary()

  import numpy as np
  image = np.random.rand(1,2,2,1)
  output_dict = model.fcn_pass(image)
  print(output_dict)
