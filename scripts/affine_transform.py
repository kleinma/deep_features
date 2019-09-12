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
  def __init__(self):
    self.tf_list = []
    self.tf_names = []

  def add_identity(self):
    self.tf_list.append(np.identity(3))
    self.tf_names.append('Identity')

  def add_reflection(self, axis):
    """
    Add reflection about axis (0=x, 1=y)
    """
    tf = np.identity(3)
    if axis == 0:
      tf[0,0] = -1
    elif axis == 1:
      tf[1,1] = -1
    else:
      raise Exception('axis must be 0 (x) or 1 (y). axis = {}'.format(axis))
    self.tf_list.append(tf)
    self.tf_names.append('Relection about {} axis'.format('x' if axis == 0 else 'y'))

  def add_scale(self, cxx=1, cyy=1):
    tf = np.identity(3)
    tf[0,0] = cxx
    tf[1,1] = cyy
    self.tf_list.append(tf)
    self.tf_names.append('Scale with cxx = {} and cyy = {}'.format(cxx, cyy))

  def add_rotation(self, theta=0):
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
    tf = np.identity(3)
    tf[0,1] = cxy
    tf[1,0] = cyx
    self.tf_list.append(tf)
    self.tf_names.append('Shear with cxy = {} and cyx = {}'.format(cxy, cyx))

  def add_translation(self, bx=0, by=0):
    tf = np.identity(3)
    tf[0,2] = bx
    tf[1,2] = by
    self.tf_list.append(tf)
    self.tf_names.append('Translation with bx = {} and by = {}'.format(bx, by))

  def add_arbitrary(self, cxx=1, cxy=0, bx=0, cyx=0, cyy=1, by=0):
    tf = np.array(((cxx, cxy, bx), (cyx, cyy, by), (0, 0, 1)))
    self.tf_list.append(tf)
    self.tf_names.append('Arbitrary with cxx = {}, cxy = {}, bx = {}, cyx = {}, cyy = {}, and by = {}'.format(cxx, cxy, bx, cyx, cyy, by))

  def as_ndarray(self):
    """
    Return transform as a 3x3 numpy array.
    """
    total_tf = np.identity(3)
    for tf in self.tf_list:
      total_tf = np.matmul(tf, total_tf)
    return total_tf

  def print_tfs(self):
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
  print(tf.as_ndarray())
  tf.print_tfs()
