"""
A Python program to check if a given point lies inside a given polygon
Refer https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
for explanation of functions on_segment(), orientation() and do_intersect()

Transcribed into Python from the C++ code at the following website
https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
"""

from collections import namedtuple
import numpy as np
import sys

INF = 1e50

Point = namedtuple('Point', ['x', 'y'])

def on_segment(p, q, r):
  """
  Given three colinear points p, q, r, the function checks if point q lies on
  line segment 'pr'

  Parameters
  ----------
  p, r : Point
    Endpoints of the line segment
  q : Point
    Point, colinear to p and r, that may or may not be on the line segment 'pr'

  Returns
  -------
  bool
    True if q lies on line segment 'pr'
  """
  p = convert_to_point(p)
  q = convert_to_point(q)
  r = convert_to_point(r)

  if (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y)):
    return True
  else:
    return False

def orientation(p, q, r):
  """
  Find orientation of ordered triplet (p, q, r).

  Parameters
  ----------
  p, q, r : Point
    Any three points

  Returns
  -------
  int
    0 --> p, q and r are colinear
    1 --> Clockwise
    2 --> Counterclockwise
  """
  p = convert_to_point(p)
  q = convert_to_point(q)
  r = convert_to_point(r)

  val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
  if val == 0:
    return 0 # colinear
  return 1 if val > 0 else 2 # clock or counterclock wise

def do_intersect(p1, q1, p2, q2):
  """
  Test if two line segments intersect

  Parameters
  ----------
  p1, q1 : Point
    Endpoints of the first line segment
  p2, q2 : Point
    Endpoints of the first second segment

  Returns
  -------
  bool
    True if the line segments intersect
  """

  p1 = convert_to_point(p1)
  q1 = convert_to_point(q1)
  p2 = convert_to_point(p2)
  q2 = convert_to_point(q2)

  o1 = orientation(p1, q1, p2)
  o2 = orientation(p1, q1, q2)
  o3 = orientation(p2, q2, p1)
  o4 = orientation(p2, q2, q1)

  # General case
  if o1 != o2 and o3 != o4:
    return True

  # Special cases
  # p1, q1, and p2 are colinear and p2 lies on segment p1q1
  if o1 == 0 and on_segment(p1, p2, q1): return True
  # p1, q1, and p2 are colinear and q2 lies on segment p1q1
  if o2 == 0 and on_segment(p1, q2, q1): return True
  # p2, q2, and p1 are colinear and p1 lies on segment p2q2
  if o3 == 0 and on_segment(p2, p1, q2): return True
  # p2, q2 and, q1 are colinear and q1 lies on segment p2q2
  if o4 == 0 and on_segment(p2, q1, q2): return True

  # Doesn't fall in any of the above cases
  return False

def is_inside(polygon, p):
  """
  Returns true if the point p lies inside the polygon

  Parameters
  ----------
  polygon : Seq[Point]
    List of points that make up the vertices of the polygon
  p : Point
    Point that lies either inside or outside the polygon

  Returns
  -------
  bool
    True if the point lies inside the polygon
  """
  polygon = convert_to_polygon(polygon)
  p = convert_to_point(p)

  # There must be at least 3 vertices in polygon
  N = len(polygon)
  if N < 3:
    raise RuntimeError('Polygon must have at least 3 vertices.')

  # Create a point for line segment from p to infinite
  extreme = Point(INF, p.y)

  # Count intersections of the above line with sides of polygon
  count = 0
  for vert in range(N):
    next_vert = (vert+1)%N

    # Check if the line segment from 'p' to 'extreme' intersects
    # with the line segment from 'polygon[vert]' to 'polygon[next_vert]'
    if (do_intersect(polygon[vert], polygon[next_vert], p, extreme)):
      # If the point 'p' is colinear with line segment 'vert-next_vert',
      # then check if it lies on segment. If it lies, return true,
      # otherwise false
      if orientation(polygon[vert], p, polygon[next_vert]) == 0:
        return on_segment(polygon[vert], p, polygon[next_vert])

      count = count + 1

  # Return true if count is odd, false otherwise
  return True if count%2 == 1 else False


def convert_to_point(p):
  """
  Helper function to convert p to Point if a numpy array
  """
  if hasattr(p, 'x') and hasattr(p, 'y'):
    return p
  else:
    p_new = Point(p[0],p[1])
    return p_new

def convert_to_polygon(polygon):
  """
  Helper function to convert polygon to List[Point] if a numpy array
  """
  if len(polygon) == 0:
    return polygon
  elif hasattr(polygon[0], 'x') and hasattr(polygon[0], 'y'):
    return polygon
  else:
    polygon_new = []
    for i in range(polygon.shape[1]):
      polygon_new.append(convert_to_point(polygon[:,i]))
    return polygon_new

if __name__ == "__main__":

  print('Point version')
  polygon1 = [Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)]
  p = Point(0, 20)
  print("Yes" if is_inside(polygon1, p) else  "No")

  p = Point(5, 5)
  print("Yes" if is_inside(polygon1, p) else  "No")

  polygon2 = [Point(0, 0), Point(5, 5), Point(5, 0)]
  p = Point(3, 3)
  print("Yes" if is_inside(polygon2, p) else  "No")

  p = Point(5, 1)
  print("Yes" if is_inside(polygon2, p) else  "No")

  p = Point(8, 1)
  print("Yes" if is_inside(polygon2, p) else  "No")

  polygon3 = [Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)]
  p = Point(-1,10)
  print("Yes" if is_inside(polygon3, p) else  "No")

  print('\nnumpy version')
  polygon1 = np.array(((0, 10, 10, 0),(0, 0, 10, 10)))
  p = np.array((0, 20))
  print("Yes" if is_inside(polygon1, p) else  "No")

  p = np.array((5, 5))
  print("Yes" if is_inside(polygon1, p) else  "No")

  polygon2 = np.array(((0, 5, 5),(0, 5, 0)))
  p = np.array((3, 3))
  print("Yes" if is_inside(polygon2, p) else  "No")

  p = np.array((5, 1))
  print("Yes" if is_inside(polygon2, p) else  "No")

  p = np.array((8, 1))
  print("Yes" if is_inside(polygon2, p) else  "No")

  polygon3 = np.array(((0, 10, 10, 0),(0, 0, 10, 10)))
  p = np.array((-1,10))
  print("Yes" if is_inside(polygon3, p) else  "No")
