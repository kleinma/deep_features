from collections import namedtuple
from enum import Enum

'''
A Feature contains a location and a descriptor. The location describes where
the feature is, spatially, in the data (image, point cloud, etc), and the
descriptor assigns descriptive information to the feature used for matching
features from different pieces of data. The descriptors can come in many forms.

Location
- Feature's location in the network output
-- layer name, and location within layer (i->width, j->height, k->channel)
- Feature's location in the image space
-- (i->width, j->height)

Descriptor
- data (any sort of desciptor)
- type (enum describing the type of descriptor used)

Feature
- Location
- Descriptor


Feature Detector
input: data
output: list of locations

Feature Describer
input: data, list of locations
output: list of features (locationa and description)

Image goes into network - get network output (each layer).
network output goes into feature detector - list of feature locations comes out
list of feature locations and network output goes into feature descriptor - list of features (location and description) comes out
'''

# ImageLoc(ation) is the (x,y) location of the feature in image space
ImageLocation = namedtuple('ImageLocation', ['x','y'])
# NetworkLoc(ation) is the neuron in the network where the feature was found
NetworkLocation = namedtuple('NetworkLocation', ['layer_name', 'height', 'width', 'channel', 'layer_scale'])
# PixelValue stores the value of the pixel as well as any subpixel value calculated
PixelValue = namedtuple('PixelValue', ['actual_pixel_value', 'sub_pixel_value'])
# FeatureLoc(cation) is the combined set of image and network locations
FeatureLocation = namedtuple('FeatureLocation', ['image_loc', 'network_loc', 'pixel_value'])

# Contained in each feature will be a descriptor and a descriptor type. The
# descriptor will depend on the descriptor type and can be all sorts of data
# types (binary string, list of floats, etc). The descriptor type is of data
# type Enum and can be used in case statements to determine which algorithm was
# used to create the descriptor and which to use to match features.
class DescriptorType(Enum):
  RAW = 1

Descriptor = namedtuple('Descriptor', ['data', 'descriptor_type'])

# Feature is made up of a location, a descriptor, and an enum of the
# descriptor type, which can be used to determine which algorithm was used to
# create the descriptor and which to use to match features.
Feature = namedtuple('Feature', ['location', 'descriptor'])


if __name__ == "__main__":
  ''' Quick example of a Feature in use '''
  location = FeatureLocation(image_loc=ImageLocation(x=5, y=7),
                             network_loc=NetworkLocation(layer_name='Conv1d_1',
                                                         height=56,
                                                         width=38,
                                                         channel=94))
  descriptor = Descriptor(data=99, descriptor_type=DescriptorType.RAW)
  feature = Feature(location=location, descriptor=descriptor)

  print(feature)
