import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from model_base import ModelBase

class ModelMnist(ModelBase):
  ''' Define a small network to classify MNIST digits '''
  def build_fully_convolutional_network(self):
    return tf.keras.Sequential([Conv2D(32, 3, activation='relu',
                                       input_shape=(None,None,1),
                                       name='conv2d_0'),
                                Conv2D(128, 28-2, activation='relu',
                                       name='conv2d_1'),
                                Conv2D(10, 1, activation='softmax',
                                       name='conv2d_2')],
                               name='fcn')

  def init_checkpoint_dir(self):
    return './checkpoints/mnist'

if __name__ == "__main__":
  model = ModelMnist()
  model.fcn.summary()
  model.classifier.summary()

  import numpy as np
  image = np.random.rand(1,50,50,1)
  output_dict = model.fcn_pass(image)
  # print(output_dict)

  for key in output_dict.keys():
    print('key = {}, value\'s shape = {} '.format(key, output_dict[key].shape))
