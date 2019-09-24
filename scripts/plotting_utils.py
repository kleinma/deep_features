import matplotlib.pyplot as plt
import numpy as np
from test_result import TestResult
from transform_utils import transform_feature_locations

def get_feature_map_and_feature_locations(network_output, feature_locations,
                                          layer_name, channel):
  # Get all the output values that match the layer_name
  output_values_list = [layer_output.output_values for layer_output in network_output if layer_output.layer_name == layer_name]
  output_values = output_values_list[0]
  print('output_values.shape = {}'.format(output_values.shape))
  feature_map = output_values[0,:,:,channel]
  print('feature_map.shape = {}'.format(feature_map.shape))


  trimmed_feature_locations = [feature_location for feature_location in feature_locations if feature_location.network_loc.layer_name == layer_name and feature_location.network_loc.channel == channel]

  return feature_map, trimmed_feature_locations

def plot_feature_map_with_feature_locations(fmap, flocs):
  flocs_x = [floc.width for floc in flocs]
  flocs_y = [floc.height for floc in flocs]
  plt.imshow(fmap, cmap='gray')
  plt.scatter(flocs_x, flocs_y)
  plt.show()

def plot_two_feature_maps_with_feature_locations(fmap_0, flocs_0, fmap_1, flocs_1, img_0, img_1):
  flocs_0_x = [floc.network_loc.width for floc in flocs_0]
  flocs_0_y = [floc.network_loc.height for floc in flocs_0]
  flocs_1_x = [floc.network_loc.width for floc in flocs_1]
  flocs_1_y = [floc.network_loc.height for floc in flocs_1]

  print('fmap_0.shape = {} and fmap_1.shape = {}'.format(fmap_0.shape, fmap_1.shape))
  print('len(flocs_0) = {} and len(flocs_1) = {}'.format(len(flocs_0), len(flocs_1)))

  plt.subplot(221)
  plt.imshow(img_0.astype(np.dtype('uint8')))
  print('img_0: dtype = {}, max = {}, min = {}'.format(img_0.dtype, np.max(img_0), np.min(img_0)))

  plt.subplot(222)
  plt.imshow(img_1.astype(np.dtype('uint8')))
  print('img_1: dtype = {}, max = {}, min = {}'.format(img_1.dtype, np.max(img_1), np.min(img_1)))

  plt.subplot(223)
  plt.imshow(fmap_0, cmap='gray')
  plt.scatter(flocs_0_x, flocs_0_y)

  plt.subplot(224)
  plt.imshow(fmap_1, cmap='gray')
  plt.scatter(flocs_1_x, flocs_1_y)
  plt.scatter(flocs_0_x_tf, flocs_0_y_tf, 'r')

  plt.show()

def plot_test_result(test_result, layer_name, channel):
  img_0 = test_result.transformed_image_0
  img_1 = test_result.transformed_image_1
  tf_0 = test_result.transform_0
  tf_1 = test_result.transform_1
  network_output_0 = test_result.network_output_0
  network_output_1 = test_result.network_output_1
  feature_locations_0 = test_result.feature_locations_0
  feature_locations_1 = test_result.feature_locations_1

  fmap_0, flocs_0 = get_feature_map_and_feature_locations(network_output_0,
                                                          feature_locations_0,
                                                          layer_name, channel)
  fmap_1, flocs_1 = get_feature_map_and_feature_locations(network_output_1,
                                                          feature_locations_1,
                                                          layer_name, channel)
  flocs_0_x = [floc.network_loc.width for floc in flocs_0]
  flocs_0_y = [floc.network_loc.height for floc in flocs_0]
  flocs_1_x = [floc.network_loc.width for floc in flocs_1]
  flocs_1_y = [floc.network_loc.height for floc in flocs_1]

  flocs_0_tf = transform_feature_locations(flocs_0, tf_1)
  flocs_0_x_tf = [floc.network_loc.width for floc in flocs_0_tf]
  flocs_0_y_tf = [floc.network_loc.height for floc in flocs_0_tf]

  print('fmap_0.shape = {} and fmap_1.shape = {}'.format(fmap_0.shape, fmap_1.shape))
  print('len(flocs_0) = {} and len(flocs_1) = {}'.format(len(flocs_0), len(flocs_1)))

  plt.subplot(221)
  plt.imshow(img_0.astype(np.dtype('uint8')))
  print('img_0: dtype = {}, max = {}, min = {}'.format(img_0.dtype, np.max(img_0), np.min(img_0)))

  plt.subplot(222)
  plt.imshow(img_1.astype(np.dtype('uint8')))
  print('img_1: dtype = {}, max = {}, min = {}'.format(img_1.dtype, np.max(img_1), np.min(img_1)))

  plt.subplot(223)
  plt.imshow(fmap_0, cmap='gray')
  plt.scatter(flocs_0_x, flocs_0_y)

  plt.subplot(224)
  plt.imshow(fmap_1, cmap='gray')
  plt.scatter(flocs_1_x, flocs_1_y)
  plt.scatter(flocs_0_x_tf, flocs_0_y_tf, marker='x')

  plt.show()
