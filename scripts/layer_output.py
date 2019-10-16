from collections import namedtuple

"""
A layer output is a named tuple consisting of the layer's name and the output of\
each neuron in that layer. A network output is consists of a list of layer
outputs.
"""

LayerOutput = namedtuple('LayerOutput', ['layer_name', 'output_values', 'layer_scale'])
"""
Parameters
----------
layer_name : string
  The name of the layer
output_values : np.ndarray
  The output values of the layer
layer_scale : float
  Difference in scale between the input image and the feature map
"""
