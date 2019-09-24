import numpy as np

class AffineTransform():
  """
  Helper class to build up affine tranformations

  Attributes
  ----------
  tf_list : List[np.ndarray]
    List of transforms in order of application
  tf_names : List[str]
    Description of each transform in order of application
  """
  def __init__(self, tf=None, frame=None):
    """
    Initialize the transforms and their names

    Parameters
    ----------
    tf : AffineTransform or None
      Initialize this transform using another as the start
    frame : str
      Apply transforms with respect to a mobile (default) or fixed frame
    """
    self.tf_list = []
    self.tf_names = []

    if tf:
      # Check and see if the copied transform's frame matches the frame param
      if frame:
        if tf.frame != frame.lower():
          # If not, raise an exception
          raise RuntimeException('frame parameter does not match copied tf frame attribute.')
      self.frame = tf.frame
      self.tf_list.extend(tf.tf_list)
      self.tf_names.extend(tf.tf_names)

    elif not frame and not tf:
      self.set_frame_mobile()
      print('frame parameter was not set and no existing transform was copied. Defaulting to apply transforms about mobile coordinate frame.')

    else:
      self.frame = frame.lower()
      if self.frame != 'mobile' and self.frame != 'fixed':
        self.set_frame_mobile()
        raise RuntimeError('frame parameter must be \'mobile\' or \'fixed\'. Setting to \'mobile\'.')

  def set_frame_mobile(self):
    """
    Apply the transforms about a mobile coordinate frame.
    """
    self.frame = 'mobile'

  def set_frame_fixed(self):
    """
    Apply the transforms about a fixed coordinate frame.
    """
    self.frame = 'fixed'

  def frame_is_mobile(self):
    """
    Test if the coordinate frame is mobile.
    """
    return True if self.frame == 'mobile' else False

  def frame_is_fixed(self):
    """
    Test if the coordinate frame is fixed.
    """
    return True if self.frame == 'fixed' else False

  def add_identity(self):
    """
    Add an identity transform
    """
    self.tf_list.append(np.identity(3))
    self.tf_names.append('Identity')

  def add_reflection(self, axis):
    """
    Add a reflection transform about an axis (0=x, 1=y)

    Parameters
    ----------
    axis : int
      0 to reflect about the x axis and 1 to reflect about the y axis
    """
    tf = np.identity(3)
    if axis == 0:
      tf[0,0] = -1
    elif axis == 1:
      tf[1,1] = -1
    else:
      raise RuntimeError('axis must be 0 (x) or 1 (y). axis = {}'.format(axis))
    self.tf_list.append(tf)
    self.tf_names.append('Relection about {} axis'.format('x' if axis == 0 else 'y'))

  def add_scale(self, cxx=1, cyy=1):
    """
    Add a scaling transform

    Parameters
    ----------
    cxx, cyy : float
      The amount to scale in the x and y directions
    """
    tf = np.identity(3)
    tf[0,0] = cxx
    tf[1,1] = cyy
    self.tf_list.append(tf)
    self.tf_names.append('Scale with cxx = {} and cyy = {}'.format(cxx, cyy))

  def add_rotation(self, theta=0):
    """
    Add a rotation transform

    Parameters
    ----------
    theta : float
      The angle that the image is rotated about the point (0,0)
    """
    s = np.sin(theta)
    c = np.cos(theta)
    tf = np.identity(3)
    tf[0,0] = c
    tf[1,1] = c
    tf[0,1] = -s
    tf[1,0] = s
    self.tf_list.append(tf)
    self.tf_names.append('Rotation with theta = {}'.format(theta))

  def add_shear(self, cxy=0, cyx=0):
    """
    Add a shearing transform

    Parameters
    ----------
    cxy, cyx : float
      The amount to shear in the xy and yx directions
    """
    tf = np.identity(3)
    tf[0,1] = cxy
    tf[1,0] = cyx
    self.tf_list.append(tf)
    self.tf_names.append('Shear with cxy = {} and cyx = {}'.format(cxy, cyx))

  def add_translation(self, bx=0, by=0):
    """
    Add a transloation transform

    Parameters
    ----------
    bx, by : float
      The two elements of the translation vector
    """
    tf = np.identity(3)
    tf[0,2] = bx
    tf[1,2] = by
    self.tf_list.append(tf)
    self.tf_names.append('Translation with bx = {} and by = {}'.format(bx, by))

  def add_arbitrary(self, cxx=1, cxy=0, bx=0, cyx=0, cyy=1, by=0):
    """
    Add an arbitrary affine transform

    Parameters
    ----------
    cxx, cxy, cyx, cyy : float
      The four elements of the rotation matrix
    bx, by : float
      The two elements of the translation vector
    """
    tf = np.array(((cxx, cxy, bx), (cyx, cyy, by), (0, 0, 1)))
    self.tf_list.append(tf)
    self.tf_names.append('Arbitrary with cxx = {}, cxy = {}, bx = {}, cyx = {}, cyy = {}, and by = {}'.format(cxx, cxy, bx, cyx, cyy, by))

  def add_tf(self, tf):
    """
    Add an existing AffineTransform class to this AffineTransform

    Parameters
    ----------
    tf : AffineTransform or None
      Add the transform list and names from another AffineTransform
    """
    if tf.frame != self.frame:
      raise RuntimeError('Added transform\'s frame doesn\'t match current frame.')
    self.tf_list.extend(tf.tf_list)
    self.tf_names.extend(tf.tf_names)


  def as_ndarray(self):
    """
    Return transform as a 3x3 numpy array.

    Returns
    -------
    total_tf : np.ndarray
      All the arrays multiplied together
    """
    total_tf = np.identity(3)

    if self.frame_is_mobile():
      for tf in reversed(self.tf_list):
        total_tf = np.matmul(tf, total_tf)
    elif self.frame_is_fixed():
      for tf in self.tf_list:
        total_tf = np.matmul(tf, total_tf)
    return total_tf

  def print_tfs(self):
    """
    Print all the transforms and their names
    """
    for tf, name in zip(self.tf_list, self.tf_names):
      print(name)
      print(tf)

if __name__ == "__main__":
  tf = AffineTransform()
  tf.add_identity()
  tf.add_reflection(axis=1)
  tf.add_scale(cxx=1, cyy=2)
  tf.add_rotation(theta=np.pi/4.0)
  tf.add_translation(1,1)
  tf.add_shear(cxy=0.4)
  tf.add_arbitrary(cyx=0.3, by=2)

  new_tf = AffineTransform()
  new_tf.add_identity()

  tf.add_tf(new_tf)
  print(tf.as_ndarray())
  tf.print_tfs()
