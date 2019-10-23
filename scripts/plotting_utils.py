from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import factorial
from test_result import TestResult
from transform_utils import transform_feature_locations

def get_feature_map_and_feature_locations(network_output, feature_locations,
                                          layer_name, channel):
  # Get all the output values that match the layer_name
  output_values_list = [layer_output.output_values for layer_output in network_output if layer_output.layer_name == layer_name]
  output_values = output_values_list[0]
  feature_map = output_values[0,:,:,channel]


  trimmed_feature_locations = [feature_location for feature_location in feature_locations if feature_location.network_loc.layer_name == layer_name and feature_location.network_loc.channel == channel]

  return feature_map, trimmed_feature_locations

def plot_feature_map_with_feature_locations(fmap, flocs):
  flocs_x = [floc.width for floc in flocs]
  flocs_y = [floc.height for floc in flocs]
  plt.imshow(fmap, cmap='gray')
  plt.scatter(flocs_x, flocs_y)
  plt.show()

FeatureLocationMatch = namedtuple('FeatureLocationMatch',['feature_location','closest_match','distance','distance_scaled'])

def plot_two_feature_maps_with_feature_locations(fmap_0, flocs_0, fmap_1, flocs_1, img_0, img_1):
  flocs_0_x = [floc.network_loc.width for floc in flocs_0]
  flocs_0_y = [floc.network_loc.height for floc in flocs_0]
  flocs_1_x = [floc.network_loc.width for floc in flocs_1]
  flocs_1_y = [floc.network_loc.height for floc in flocs_1]

  plt.subplot(221)
  plt.imshow(img_0.astype(np.dtype('uint8')))

  plt.subplot(222)
  plt.imshow(img_1.astype(np.dtype('uint8')))

  plt.subplot(223)
  plt.imshow(fmap_0, cmap='gray')
  plt.scatter(flocs_0_x, flocs_0_y)

  plt.subplot(224)
  plt.imshow(fmap_1, cmap='gray')
  plt.scatter(flocs_1_x, flocs_1_y)
  plt.scatter(flocs_0_x_tf, flocs_0_y_tf, 'r')

  plt.show()

def plot_matches_in_layer(test_result, layer_num, min_channel=None, max_channel=None):
  num_channels = test_result.network_output_0[layer_num].output_values.shape[-1]

  if min_channel is not None:
    min_channel = max(min_channel,0)
    min_channel = min(min_channel, num_channels-1)
  else:
    min_channel = 0

  if max_channel is not None:
    max_channel = max(max_channel,0)
    max_channel = min(max_channel, num_channels-1)
  else:
    max_channel = num_channels-1

  tot_matches_0 = []
  tot_matches_1 = []
  for i in range(min_channel, max_channel):
    matches_0, matches_1 = plot_test_result(test_result, layer_num, i, plot=False)
    tot_matches_0.extend(matches_0)
    tot_matches_1.extend(matches_1)

  flocs_0_value = [match.feature_location.pixel_value.sub_pixel_value for match in tot_matches_0]
  flocs_1_value = [match.feature_location.pixel_value.sub_pixel_value for match in tot_matches_1]

  flocs_0_min_dist = [match.distance for match in tot_matches_0]
  flocs_1_min_dist = [match.distance for match in tot_matches_1]

  plt.subplot(121)
  plt.hist2d(flocs_0_value, flocs_0_min_dist, bins=100)
  plt.xlabel('feature value')
  plt.ylabel('distance to closest feature')

  plt.subplot(122)
  plt.hist2d(flocs_1_value, flocs_1_min_dist, bins=100)
  plt.xlabel('feature value')
  plt.ylabel('distance to closest feature')

  plt.show()

  value_threshold = 15
  true_matches_0 = [match for match in tot_matches_0 if match.feature_location.pixel_value.sub_pixel_value > value_threshold]
  false_matches_0 = [match for match in tot_matches_0 if match.feature_location.pixel_value.sub_pixel_value <= value_threshold]
  true_matches_1 = [match for match in tot_matches_1 if match.feature_location.pixel_value.sub_pixel_value > value_threshold]
  false_matches_1 = [match for match in tot_matches_1 if match.feature_location.pixel_value.sub_pixel_value <= value_threshold]

  true_match_dists_0 = [match.distance for match in true_matches_0]
  false_match_dists_0 = [match.distance for match in false_matches_0]
  true_match_dists_1 = [match.distance for match in true_matches_1]
  false_match_dists_1 = [match.distance for match in false_matches_1]

  # Plot histograms with poisson fit
  # https://stackoverflow.com/a/25828558/5525775
  ranges = [-0.5, 30.5]
  bins = int(ranges[1]-ranges[0])
  plt.subplot(221)
  entries_true_0, bin_edges, _ = plt.hist(true_match_dists_0, bins=bins, range=ranges, alpha=0.5, density=True, label='true')
  entries_false_0, _, _ = plt.hist(false_match_dists_0, bins=bins, range=ranges, alpha=0.5, density=True, label='false')
  plt.legend()

  bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])

  # poisson function, parameter lamb is the fit parameter
  def poisson(k, lamb):
    return (lamb**k/factorial(k)) * np.exp(-lamb)

  # fit with curve_fit
  parameters_true_0, _ = curve_fit(poisson, bin_middles, entries_true_0)
  parameters_false_0, _ = curve_fit(poisson, bin_middles, entries_false_0)

  x_plot = np.linspace(0, 30, 1000)
  plt.plot(x_plot, poisson(x_plot, *parameters_true_0), 'b-', lw=1)
  plt.plot(x_plot, poisson(x_plot, *parameters_false_0), 'r-', lw=1)

  plt.subplot(222)
  entries_true_1, _, _ = plt.hist(true_match_dists_1, bins=bins, range=ranges, alpha=0.5, density=True, label='true')
  entries_false_1, _, _ = plt.hist(false_match_dists_1, bins=bins, range=ranges, alpha=0.5, density=True, label='false')
  plt.legend()

  # fit with curve_fit
  parameters_true_1, _ = curve_fit(poisson, bin_middles, entries_true_1)
  parameters_false_1, _ = curve_fit(poisson, bin_middles, entries_false_1)

  plt.plot(x_plot, poisson(x_plot, *parameters_true_1), 'b-', lw=1)
  plt.plot(x_plot, poisson(x_plot, *parameters_false_1), 'r-', lw=1)

  ranges = [-0.5, 30.5]
  bins = 10*int(ranges[1]-ranges[0])

  plt.subplot(223)
  plt.hist(true_match_dists_0, bins=bins, range=ranges, alpha=0.5, label='true')
  plt.hist(false_match_dists_0, bins=bins, range=ranges, alpha=0.5, label='false')
  plt.legend()

  plt.subplot(224)
  plt.hist(true_match_dists_1, bins=bins, range=ranges, alpha=0.5, label='true')
  plt.hist(false_match_dists_1, bins=bins, range=ranges, alpha=0.5, label='false')
  plt.legend()

  plt.show()

def plot_test_result(test_result, layer_num, channel, plot=True):
  layer_name = test_result.network_output_0[layer_num].layer_name
  layer_scale = test_result.network_output_0[layer_num].layer_scale
  img_0 = test_result.transformed_image_0
  img_1 = test_result.transformed_image_1
  tf_0 = test_result.transform_0
  tf_1 = test_result.transform_1
  offset_tf = test_result.offset_transform
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

  flocs_0_x_scaled = [(1./layer_scale)*num for num in flocs_0_x]
  flocs_0_y_scaled = [(1./layer_scale)*num for num in flocs_0_y]
  flocs_1_x_scaled = [(1./layer_scale)*num for num in flocs_1_x]
  flocs_1_y_scaled = [(1./layer_scale)*num for num in flocs_1_y]

  tf0_2_tf1 = np.matmul(offset_tf, tf_1) @ np.linalg.inv(np.matmul(offset_tf, tf_0))

  flocs_0_tf = transform_feature_locations(flocs_0, tf0_2_tf1)
  flocs_0_x_tf = [floc.network_loc.width for floc in flocs_0_tf]
  flocs_0_y_tf = [floc.network_loc.height for floc in flocs_0_tf]

  flocs_0_x_tf_scaled = [(1./layer_scale)*num for num in flocs_0_x_tf]
  flocs_0_y_tf_scaled = [(1./layer_scale)*num for num in flocs_0_y_tf]

  matches_0, matches_1 = get_feature_min_distances(flocs_0_tf, flocs_1)

  flocs_0_value = [match.feature_location.pixel_value.sub_pixel_value for match in matches_0]
  flocs_1_value = [match.feature_location.pixel_value.sub_pixel_value for match in matches_1]

  flocs_0_min_dist = [match.distance for match in matches_0]
  flocs_1_min_dist = [match.distance for match in matches_1]

  if plot:
    plt.subplot(321)
    plt.imshow(img_0.astype(np.dtype('uint8')))
    plt.scatter(flocs_0_x_scaled, flocs_0_y_scaled)

    plt.subplot(322)
    plt.imshow(img_1.astype(np.dtype('uint8')))
    plt.scatter(flocs_1_x_scaled, flocs_1_y_scaled)
    plt.scatter(flocs_0_x_tf_scaled, flocs_0_y_tf_scaled)

    plt.subplot(323)
    plt.imshow(fmap_0, cmap='gray')
    plt.scatter(flocs_0_x, flocs_0_y)

    plt.subplot(324)
    plt.imshow(fmap_1, cmap='gray')
    plt.scatter(flocs_1_x, flocs_1_y)
    plt.scatter(flocs_0_x_tf, flocs_0_y_tf, marker='x')

    plt.subplot(325)
    plt.scatter(flocs_0_value, flocs_0_min_dist)
    plt.xlabel('feature value')
    plt.ylabel('distance to closest feature')

    plt.subplot(326)
    plt.scatter(flocs_1_value, flocs_1_min_dist)
    plt.xlabel('feature value')
    plt.ylabel('distance to closest feature')

    plt.show()

  return matches_0, matches_1

def get_feature_min_distances(feat_locs_0, feat_locs_1):
  """
  Find closest match in each list of feature locations

  Parameters
  ----------
  feat_locs_0, feat_locs_1 : List[FeatureLocation]
    Set of feature locations
  dist_thresh : float
    Distance between features to be determined a match

  Return
  ------
  Tuple[List[FeatureLocationMatch]]
    List of feature location matches for each set of feature locations
  """
  matches_0 = []
  for loc_0 in feat_locs_0:
    _, h_0, w_0, _, scale = loc_0.network_loc
    pt_0 = np.array(((w_0), (h_0)))
    min_dist = np.inf # Set min dist at infinity before search
    closest_match = None
    for loc_1 in feat_locs_1:
      _, h_1, w_1, _, scale = loc_1.network_loc
      pt_1 = np.array(((w_1), (h_1)))
      dist = np.linalg.norm(pt_1-pt_0)
      if dist < min_dist:
        min_dist = dist
        closest_match = loc_1
    if closest_match is not None:
      matches_0.append(FeatureLocationMatch(loc_0, closest_match, min_dist, (1./scale)*min_dist))

  matches_1 = []
  for loc_1 in feat_locs_1:
    _, h_1, w_1, _, scale = loc_1.network_loc
    pt_1 = np.array(((w_1), (h_1)))
    min_dist = np.inf # Set min dist at infinity before search
    closest_match = None
    for loc_0 in feat_locs_0:
      _, h_0, w_0, _, scale = loc_0.network_loc
      pt_0 = np.array(((w_0), (h_0)))
      dist = np.linalg.norm(pt_0-pt_1)
      if dist < min_dist:
        min_dist = dist
        closest_match = loc_0
    if closest_match is not None:
      matches_1.append(FeatureLocationMatch(loc_1, closest_match, min_dist, (1./scale)*min_dist))

  return (matches_0, matches_1)

def plot_layer_output(layer_output, channel_min=0, channel_max=-1, grid_height = 5, grid_width = 4):
  """
  Plot channels of a layer output.

  Parameters
  ----------
  layer_output : LayerOutput
    Layer output to be plotted.
  channel_min, channel_max : int
    Min and max channels to be plotted.
  grid_height, grid_width : int
    Dimensions of the subplot grid.

  Raises
  ------
  ValueError
    If channel_min or channel_max are out of range.
  """
  name, output, scale = layer_output
  num_channels = output.shape[-1]

  if channel_min < 0:
    raise ValueError('channel_min must be 0 or greater')
  if channel_min >= num_channels:
    raise ValueError('channel_min must be less than the number of channels')
  if channel_max >= num_channels:
    raise ValueError('channel_max must be less than the number of channels')
  if channel_max < channel_min:
    raise ValueError('channel_max must be greater than or equal to channel_min')
  if channel_max < 0:
    channel_max = channel_min # Just look at channel_min

  output = np.squeeze(output, 0)
  num_channels_per_set = grid_height * grid_width
  channels = list(range(channel_min, channel_max+1))

  # Split channels up into sets that will be plotted at once
  channel_sets = [channels[x:x+num_channels_per_set]
                  for x in range(0, len(channels), num_channels_per_set)]

  print(channel_sets)

  for channel_set in channel_sets:
    ix = 0
    jx = 0
    fig, axs = plt.subplots(grid_height, grid_width, constrained_layout=True)
    fig.suptitle('layer name = {}'.format(name))
    print(len(axs))
    print(len(axs[0]))
    for channel in channel_set:
      print('ix = {}, jx = {}'.format(ix,jx))
      fmap = output[:,:,channel]
      axs[jx,ix].imshow(fmap, cmap='gray')
      axs[jx,ix].set_title('channel = {}'.format(channel))
      axs[jx,ix].set_xticks([])
      axs[jx,ix].set_yticks([])
      ix += 1
      if ix >= grid_width:
        ix = 0
        jx += 1
    plt.show()

def plot_layer_output_and_feature_locations(layer_output, feature_locations, channel_min=0, channel_max=-1, grid_height = 5, grid_width = 4):
  """
  Plot channels of a layer output.

  Parameters
  ----------
  layer_output : LayerOutput
    Layer output to be plotted.
  feature_locations : List[FeatureLocation]
    Feature locations to be plotted.
  channel_min, channel_max : int
    Min and max channels to be plotted.
  grid_height, grid_width : int
    Dimensions of the subplot grid.

  Raises
  ------
  ValueError
    If channel_min or channel_max are out of range.
  """
  name, output, scale = layer_output
  num_channels = output.shape[-1]

  if channel_min < 0:
    raise ValueError('channel_min must be 0 or greater')
  if channel_min >= num_channels:
    raise ValueError('channel_min must be less than the number of channels')
  if channel_max >= num_channels:
    raise ValueError('channel_max must be less than the number of channels')
  if channel_max < channel_min:
    raise ValueError('channel_max must be greater than or equal to channel_min')
  if channel_max < 0:
    channel_max = channel_min # Just look at channel_min

  output = np.squeeze(output, 0)
  num_channels_per_set = grid_height * grid_width
  channels = list(range(channel_min, channel_max+1))

  # Split channels up into sets that will be plotted at once
  channel_sets = [channels[x:x+num_channels_per_set]
                  for x in range(0, len(channels), num_channels_per_set)]

  for channel_set in channel_sets:
    ix = 0
    jx = 0
    fig, axs = plt.subplots(grid_height, grid_width, constrained_layout=True)
    for channel in channel_set:
      trimmed_feature_locations = [feature_location for feature_location in feature_locations if feature_location.network_loc.layer_name == name and feature_location.network_loc.channel == channel]
      flocs_x = [floc.network_loc.width for floc in trimmed_feature_locations]
      flocs_y = [floc.network_loc.height for floc in trimmed_feature_locations]
      fmap = output[:,:,channel]
      axs[jx,ix].imshow(fmap, cmap='gray')
      axs[jx,ix].scatter(flocs_x, flocs_y, marker='x', s = 20, color='tab:red')
      axs[jx,ix].set_title('channel = {}'.format(channel))
      axs[jx,ix].set_xticks([])
      axs[jx,ix].set_yticks([])
      ix += 1
      if ix >= grid_width:
        ix = 0
        jx += 1
    plt.show()
