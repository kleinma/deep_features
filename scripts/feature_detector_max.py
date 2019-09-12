from feature_detector_base import FeatureDetectorBase
from layer_output import LayerOutput
from feature import NetworkLocation, FeatureLocation, DescriptorType, Descriptor, FeatureLocation, Feature
import numpy as np

class FeatureDetectorMax(FeatureDetectorBase):
  '''
  Finds all maximal points in each feature map in each layer of a network output
  '''
  def detect(self, network_output):
    all_feature_locations = []
    all_features = []
    for name, out in network_output:
      for i in range(out.shape[-1]): #Iterate over the channels
        feature_locations, features = self.find_max_subpixels_in_feature_map(np.squeeze(out[...,i],0), name, i)
        # features.sort(key=lambda tup: tup.descriptor, reverse=True)
        all_feature_locations.extend(feature_locations)
        all_features.extend(features)

    return all_feature_locations, all_features

  def find_max_subpixels_in_feature_map(self, fmap, layer_name=None, channel=None):
    y_max = fmap.shape[0]
    x_max = fmap.shape[1]

    feature_locations = []
    features = []
    descriptor_type = DescriptorType.MAX
    max_points = []
    max_vals = []
    for q in range(1,y_max-1):
      for p in range(1,x_max-1):
        is_max, x, y, max_val = self.subpixel_max(fmap, q, p)
        if is_max:
          max_points.append((x,y))
          max_vals.append(max_val)
          network_loc = NetworkLocation(layer_name, y, x, channel)
          feature_loc = FeatureLocation(image_loc=None, network_loc=network_loc)
          descriptor = Descriptor(data=max_val, descriptor_type=descriptor_type)

          feature_locations.append(feature_loc)
          features.append(Feature(location=feature_loc, descriptor=descriptor))
    return feature_locations, features

  def subpixel_max(self, fmap, q, p):
    '''
    Algorithm taken from:
    Sun, C. (2002). Fast Optical Flow Using 3D Shortest Path Techniques. Image
    and Vision Computing, 20(13/14), 981–991.
    Retrieved from http://vision-cdc.csiro.au/changs/doc/sun02ivc.pdf

    pixels are described in the following way
    b** (*: m(inus) = -1, p(lus) = +1, s(ame) = +0)
    ex: bmp = feature_map[q+1,p-1] (reversed due to S[y,x] indexing)
    '''
    is_max = False
    x = p
    y = q
    max_val = fmap[q,p]
    if fmap[q-1,p-1] < fmap[q,p]:
      bmm = fmap[q-1,p-1]
      if fmap[q-1,p] < fmap[q,p]:
        bsm = fmap[q-1,p]
        if fmap[q-1,p+1] < fmap[q,p]:
          bpm = fmap[q-1,p+1]
          if fmap[q,p-1] < fmap[q,p]:
            bms = fmap[q,p-1]
            if fmap[q,p] <= fmap[q,p]:
              bss = fmap[q,p]
              if fmap[q,p+1] < fmap[q,p]:
                bps = fmap[q,p+1]
                if fmap[q+1,p-1] < fmap[q,p]:
                  bmp = fmap[q+1,p-1]
                  if fmap[q+1,p] < fmap[q,p]:
                    bsp = fmap[q+1,p]
                    if fmap[q+1,p+1] < fmap[q,p]:
                      bpp = fmap[q-1,p+1]
                      # fmap[q,p] is a maxpoint. Calculate the subpixel max
                      A, B, C, D, E, F = self.pixels_to_quad_coeffs(bmm, bsm, bpm, bms, bss, bps, bmp, bsp, bpp)
                      is_max = True
                      x = (B*E - 2*C*D)/(4*A*C - B*B)
                      y = (B*D - 2*A*E)/(4*A*C - B*B)
                      max_val = A*x*x + B*x*y + C*y*y + D*x + E*y + F
                      x = x + p
                      y = y + q
    return is_max, x, y, max_val

  def pixels_to_quad_coeffs(self, bmm, bsm, bpm, bms, bss, bps, bmp, bsp, bpp):
    A = (bmm - 2*bsm + bpm + bms - 2*bss + bps + bmp - 2*bsp + bpp) / 6.0
    B = (bmm - bpm - bmp + bpp) / 4.0
    C = (bmm + bsm + bpm - 2*bms - 2*bss - 2*bps + bmp + bsp + bpp) / 6.0
    D = (-bmm + bpm - bms + bps - bmp + bpp) / 6.0
    E = (-bmm - bsm - bpm + bmp + bsp + bpp) / 6.0
    F = (-bmm + 2*bsm - bpm + 2*bms + 5*bss + 2*bps - bmp + 2*bsp - bpp) / 9.0
    return A, B, C, D, E, F

if __name__ == "__main__":
  # Try with mnist model
  from model_mnist import ModelMnist
  model = ModelMnist()
  image = np.random.rand(1,50,50,1)
  network_output = model.fcn_pass(image)
  fdm = FeatureDetectorMax()
  feature_locations, features = fdm.detect(network_output)
  data = [f.descriptor.data for f in features]
  import matplotlib.pyplot as plt

  plt.hist(data, int(len(data)/10.))
  plt.show()

  # Try with vgg16 model
  from model_vgg16 import ModelVGG16
  model = ModelVGG16()

  from tensorflow.keras.applications.vgg16 import preprocess_input
  image = np.random.randint(0,256,(224,224,3))
  image = np.expand_dims(image, axis=0)
  image = preprocess_input(image)
  network_output = model.fcn_pass(image)
  fdm = FeatureDetectorMax()
  feature_locations, features = fdm.detect(network_output)
  data = [f.descriptor.data for f in features]
  plt.hist(data, int(len(data)/10.))
  plt.show()
