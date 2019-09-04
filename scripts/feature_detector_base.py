from layer_output import LayerOutput
from feature import FeatureLocation, DescriptorType
'''
Base class:
- FeatureDetector
-- detect(network_ouput) -> returns list of FeatureLocation
'''

class FeatureDetectorBase():
  '''
  Base class for feature detectors. Derived classes must override detect().
  '''
  def __init__(self):
    pass

  def detect(self, network_output):
    list_of_locations = []
    return list_of_locations
