import numpy as np
import cv2
from collections import Sequence
from numbers import Number

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
  print('tf =\n{}'.format(tf))
  return cv2.warpAffine(src=img, M=tf, dsize=dsize, flags=cv2.INTER_CUBIC)

def transform_feature_locations(feature_locs, tf):
  """
  Transform the feature locations using the given transform

  Parameters
  ----------
  img : np.ndarray
    Image to be transformed
  tf : np.ndarray or Sequence of Numbers
    2x3 or 3x3 affine transform matrix

  Returns
  -------
  np.ndarray
    Transformed image
  """
  if tf.size == 6:
    tf = np.append(np.array(tf), (0,0,1)).reshape(3,3)
  elif tf.size == 9:
    tf = np.array(tf).reshape(3,3)
  else:
    raise Exception('tf must be of size 6 or 9')

  transformed_feature_locs = []
  for feature_loc in feature_locs:
    _, h, w, _ = feature_loc.network_loc
    x = np.array(((w), (h), (1)))
    x_prime = np.matmul(tf,x)
    network_loc = feature_loc.network_loc._replace(width=x_prime[0], height=x_prime[1])
    transformed_feature_locs.append(feature_loc._replace(network_loc=network_loc))
  return transformed_feature_locs

def transform_points(points, tf):
  """
  Transform an array on 2D points given a 3x3 transform matrix

  Parameters
  ----------
  points : np.ndarray
    2xN array of points
  tf : np.ndarray
    3x3 tranformation matrix

  Returns
  -------
  np.ndarray
    2xN array of transformed points
  """
  points_exp = np.ones((points.shape[0]+1,points.shape[1]))
  points_exp[:-1,:] = points
  points_tf = tf @ points_exp
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
