from collections import namedtuple
import numpy as np
from transform_utils import transform_image, transform_feature_locations, is_affine_transform

TestResult = namedtuple('TestResult', ['transform_0', 'transform_1', 'feature_loc_0', 'feature_loc_1', 'repeatability'])

class TestDetectorAffine():
  """
  Tests the repeatability of detected features after an affine transformation.

  Attributes
  ----------
  orig_img : np.ndarray
    Orignal image that transforms are applied to
  model : ModelBase or derived
    Neural network model used
  detector : FeatureDetectorBase or derived
    Feature detector used
  test_results : List[TestResult]]
    Test result consists of the transfoms used on the original image, the feature
    locations in original image and tranformed image, and repeatability score
  """
  def __init__(self, orig_img, model, detector):
    """
    Paramters
    ---------
    orig_img : np.ndarray
      Orignal image that transforms are applied to
    model : ModelBase or derived
      Neural network model used
    detector : FeatureDetectorBase or derived
      Feature detector used
    """
    self.orig_img = orig_img
    self.model = model
    self.detector = detector

    self.test_results = []

  def run_test(self, tf_0, tf_1=None):
    # Convert the transforms into 2x3 matrices
    if is_affine_transform(tf_0):
      tf_0 = np.array(tf_0).reshape((2,3))
    # If only one transform is sent, make the first transform the identity matrix
    transform_img_0 = True
    if not tf_1:
      transform_img_0 = False
      tf_1 = np.copy(tf_0)
      tf_0 = np.array([1, 0, 0, 0, 1, 0]).reshape((2,3))
    elif is_affine_transform(tf_1):
      tf_1 = np.array(tf_1).reshape((2,3))

    # Apply transforms to self.orig_img
    if transform_img_0: # Don't transform if using the original img
      img_0 = transform_image(self.orig_img, tf_0)
    else:
      img_0 = np.copy(self.orig_img)
    img_1 = transform_image(self.orig_img, tf_1)

    # Find feature locations in both images
    feat_locs_0 = self.find_features(img_0)
    feat_locs_1 = self.find_features(img_1)

    # Apply transform to features from image 0 and compare to those from image 1
    feat_locs_0_transformed = transform_feature_locations(feat_locs_0, tf_1)

    # Calculate repeatability
    repeatability = self.compare_feature_locations(feat_locs_0_transformed,
                                                   feat_locs_1)

    # Add score to self.test_results
    self.test_results.append(TestResult(tf_0, tf_1, feat_locs_0, feat_locs_1, repeatability))

  def find_features(self, img):
    """
    Run the image through the model and find the features in the output

    Parameters
    ----------
    img : np.ndarray
      Image to search for features

    Returns
    -------
    List[FeatureLocation]
      A list of the feature locations
    """
    network_output = self.model.fcn_pass(img)
    feature_locs, _ = self.detector.detect(network_output)
    print('len(feature_locs) = {}'.format(len(feature_locs)))
    return feature_locs

  def compare_feature_locations(self, feature_locs_0, feature_locs_1):
    """
    Run the image through the model and find the features in the output

    Parameters
    ----------
    feature_locs_0 : List[FeatureLocation]
      A list of the feature locations from the first image
    feature_locs_1 : List[FeatureLocation]
      A list of the feature locations from the second image

    Returns
    -------
    float
      Repeatability score
    """
    return 0

  def save_test_results(self, filename, method='csv'):
    # save as a csv or yaml file?
    if method == 'csv':
      pass

if __name__ == "__main__":
  from model_vgg16 import ModelVGG16
  from feature_detector_max import FeatureDetectorMax
  import numpy as np

  from tensorflow.keras.applications.vgg16 import preprocess_input
  orig_img = np.random.randint(0,256,(224,224,3))
  model = ModelVGG16()
  detector = FeatureDetectorMax()
  test = TestDetectorAffine(orig_img=orig_img, model=model, detector=detector)

  tf_0 = [1, 0, 0, 0, 1, 0]
  test.run_test(tf_0=tf_0)
  tf_1 = [1, 0, 0, 0, 1, 1]
  test.run_test(tf_0=tf_0, tf_1=tf_1)
