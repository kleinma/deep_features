import numpy as np
import cv2
from collections import Sequence
from numbers import Number
import math
import matplotlib.pyplot as plt

def transform_image(img, tf, dsize=None):
  """
  Transform the image using the given transform

  Parameters
  ----------
  img : np.ndarray
    Image to be transformed
  tf : np.ndarray or Sequence of Numbers
    2x3 or 3x3 affine transform matrix
  dsize : tuple of int
    Size of the output image

  Returns
  ------
  np.ndarray
    Transformed image

  Notes
  -----
  Uses warpAffine from cv2 (link to documentation below).
  https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#warpaffine
  """
  if dsize is None:
    dsize=(img.shape[1], img.shape[0])
  tf = tf.flatten()[0:6].reshape(2,3).astype(np.float64)
  return cv2.warpAffine(src=img, M=tf, dsize=dsize, flags=cv2.INTER_CUBIC)

def transform_feature_locations(feature_locs, tf):
  """
  Transform the feature locations using the given transform

  Parameters
  ----------
  feature_locs : List[FeatureLocs]
    Feature locations to be transformed
  tf : np.ndarray or Sequence of Numbers
    2x3 or 3x3 affine transform matrix

  Returns
  -------
  List[FeatureLocs]
    Transformed feature locations
  """
  if tf.size == 6:
    tf = np.append(np.array(tf), (0,0,1)).reshape(3,3)
  elif tf.size == 9:
    tf = np.array(tf).reshape(3,3)
  else:
    raise Exception('tf must be of size 6 or 9')

  transformed_feature_locs = []
  for feature_loc in feature_locs:
    _, h, w, _, scale = feature_loc.network_loc
    x = np.array(((w), (h), (1)))
    scale_up = np.array(((1./scale, 0., 0.),(0., 1./scale, 0.),(0., 0., 1.)))
    scale_down = np.array(((scale, 0., 0.),(0., scale, 0.),(0., 0., 1.)))
    x_prime = scale_down @ tf @ scale_up @ x
    # x_prime = np.matmul(scale_down,np.matmul(tf,np.matmul(scale_up,x)))
    network_loc = feature_loc.network_loc._replace(width=x_prime[0], height=x_prime[1])
    transformed_feature_locs.append(feature_loc._replace(network_loc=network_loc))
  return transformed_feature_locs

def transform_points(points, tf, scale=1):
  """
  Transform an array on 2D points given a 3x3 transform matrix

  Parameters
  ----------
  points : np.ndarray
    2xN array of points
  tf : np.ndarray
    2x3 or 3x3 tranformation matrix
  scale : float
    Used to adjust the transform of the image for a smaller feature map

  Returns
  -------
  np.ndarray
    2xN array of transformed points
  """
  if tf.shape[0] == 2:
    tf_exp = np.identity(3)
    tf_exp[:-1,:] = tf
  elif tf.shape[0] == 3:
    tf_exp = tf
  else:
    raise Exception('Transform should have 2 or 3 rows.')
  points_exp = np.ones((points.shape[0]+1,points.shape[1]))
  points_exp[:-1,:] = points
  points_tf = tf_exp @ points_exp
  return points_tf[:-1,:]

def is_affine_transform(tf):
  """ An affine transform should contain six or 9 numeric elements """
  if isinstance(tf, np.ndarray): # If it is a numpy array
    if tf.size == 6 or tf.size == 9:
      return np.issubdtype(tf.dtype, np.number)
  elif isinstance(tf, Sequence): # If it is a Sequence
    if len(tf) == 6 or len(tf) == 9:
      return all(isinstance(el, Number) for el in tf)
  else: # If it is anything else
    return False

def make_3x3_affine_tf(tf):
  """
  Turn a sequence of numbers or numpy array with 6 or 9 elements into a 3x3

  Parameters
  ----------
  tf : Sequence of numbers or np.ndarray
    transform to turn into 3x3 array

  Returns
  -------
  np.ndarray
    3x3 matrix for use in matrix multiplication
  """
  tf_new = np.array(tf).reshape((-1,3))
  if tf_new.shape[0] == 2:
    tf_temp = np.identity(3)
    tf_temp[:-1,:] = tf_new
    tf_new = tf_temp.copy()
  elif tf_new.shape[0] == 3:
    pass
  else:
    raise RuntimeError('Transform should be 2x3 or 3x3. This one is {}x{}.'.format(tf_new.shape[0],tf_new.shape[1]))
  return tf_new



def make_full_frame_transformed_images(orig_img, tf_0, tf_1, plot=False):
  tf_0 = make_3x3_affine_tf(tf_0)
  tf_1 = make_3x3_affine_tf(tf_1)

  # Get the points at the corners of a box around the image
  orig_img_height = orig_img.shape[0]
  orig_img_width = orig_img.shape[1]
  orig_box = np.array(((0.0, orig_img_width-1, orig_img_width-1, 0.0),
                       (0.0, 0.0, orig_img_height-1, orig_img_height-1)))

  # Look at the transformed boxes and find a box that encompasses both
  tf_box_0 = transform_points(orig_box, tf_0)
  tf_box_1 = transform_points(orig_box, tf_1)

  # Find the min and max of each box
  tf_box_0_max_x = np.max(tf_box_0[0,:])
  tf_box_0_min_x = np.min(tf_box_0[0,:])

  tf_box_0_max_y = np.max(tf_box_0[1,:])
  tf_box_0_min_y = np.min(tf_box_0[1,:])

  tf_box_1_max_x = np.max(tf_box_1[0,:])
  tf_box_1_min_x = np.min(tf_box_1[0,:])

  tf_box_1_max_y = np.max(tf_box_1[1,:])
  tf_box_1_min_y = np.min(tf_box_1[1,:])

  # Find a box that encompasses both images
  full_box_max_x = np.max(np.append(tf_box_0[0,:],tf_box_1[0,:]))
  full_box_min_x = np.min(np.append(tf_box_0[0,:],tf_box_1[0,:]))

  full_box_max_y = np.max(np.append(tf_box_0[1,:],tf_box_1[1,:]))
  full_box_min_y = np.min(np.append(tf_box_0[1,:],tf_box_1[1,:]))

  full_box = np.array(((full_box_min_x, full_box_max_x, full_box_max_x, full_box_min_x),(full_box_min_y, full_box_min_y, full_box_max_y, full_box_max_y)))

  orig_img_exp = 0*np.ones((math.ceil(full_box_max_y)-math.floor(full_box_min_y), math.ceil(full_box_max_x)-math.floor(full_box_min_x),3), dtype=orig_img.dtype)
  orig_img_exp_at_0 = orig_img_exp.copy()

  orig_img_exp_at_0[0:orig_img_height, 0:orig_img_width] = orig_img

  offset_tf = np.array(((1, 0, -math.floor(full_box_min_x)),(0, 1, -math.floor(full_box_min_y)),(0, 0, 1)))

  orig_img_exp = transform_image(orig_img_exp_at_0, offset_tf)
  tf_img_0_exp = transform_image(orig_img_exp_at_0, np.matmul(offset_tf, tf_0))
  tf_img_1_exp = transform_image(orig_img_exp_at_0, np.matmul(offset_tf, tf_1))

  orig_box_exp = transform_points(orig_box, offset_tf)
  tf_box_0_exp = transform_points(orig_box, np.matmul(offset_tf, tf_0))
  tf_box_1_exp = transform_points(orig_box, np.matmul(offset_tf, tf_1))

  tf0_2_tf1 = np.matmul(offset_tf, tf_1) @ np.linalg.inv(np.matmul(offset_tf, tf_0))

  tf_box_0_exp_tf_1 = transform_points(tf_box_0_exp, tf0_2_tf1)

  full_box_exp = transform_points(full_box, offset_tf)

  if plot:
    plt.subplot(131)
    plt.imshow(orig_img_exp)
    plt.scatter(orig_box_exp[0,:],orig_box_exp[1,:])
    plt.scatter(full_box_exp[0,:],full_box_exp[1,:])

    plt.subplot(132)
    plt.imshow(tf_img_0_exp)
    plt.scatter(tf_box_0_exp[0,:],tf_box_0_exp[1,:])
    plt.scatter(full_box_exp[0,:],full_box_exp[1,:])

    plt.subplot(133)
    plt.imshow(tf_img_1_exp)
    plt.scatter(tf_box_1_exp[0,:],tf_box_1_exp[1,:])
    plt.scatter(full_box_exp[0,:],full_box_exp[1,:])
    plt.scatter(tf_box_0_exp_tf_1[0,:], tf_box_0_exp_tf_1[1,:],color='red', marker='x')
    plt.show()

  return orig_img_exp, tf_img_0_exp, tf_img_1_exp, orig_box_exp, tf_box_0_exp, tf_box_1_exp, offset_tf
