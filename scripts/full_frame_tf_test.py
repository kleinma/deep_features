from affine_transform import AffineTransform
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from transform_utils import transform_image, transform_feature_locations, is_affine_transform, transform_points, make_full_frame_transformed_images
from plotting_utils import get_feature_map_and_feature_locations, plot_feature_map_with_feature_locations, plot_two_feature_maps_with_feature_locations, plot_test_result
import math

orig_img = misc.face()
tf_0 = AffineTransform()
tf_0.add_identity()
tf_0.add_rotation(np.pi/12.0)
tf_0.add_translation(-100, 200)
tf_0 = tf_0.as_ndarray()

tf_1 = AffineTransform()
tf_1.add_rotation(-np.pi/12.0)
tf_1.add_translation(100, -200)
tf_1.add_shear(.1, .2)
tf_1 = tf_1.as_ndarray()

make_full_frame_transformed_images(orig_img, tf_0, tf_1, plot=True)

"""
orig_img = misc.face()
orig_img_height = orig_img.shape[0]
orig_img_width = orig_img.shape[1]

print('orig_img_height = {}, orig_img_width = {}'.format(orig_img_height, orig_img_width))

orig_box = np.array(((0.0, orig_img_width, orig_img_width, 0.0),(0.0, 0.0, orig_img_height, orig_img_height)))

print(orig_box)

tf_0 = AffineTransform()
tf_0.add_identity()
tf_0.add_rotation(np.pi/12.0)
tf_0.add_translation(-100, 200)
tf_0 = tf_0.as_ndarray()

tf_1 = AffineTransform()
tf_1.add_rotation(-np.pi/12.0)
tf_1.add_translation(100, -200)
tf_1.add_shear(.1, .2)
tf_1 = tf_1.as_ndarray()

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

print(full_box)

# tf_img_0 = transform_image(orig_img, tf_0, (math.ceil(full_box_max_y)-math.floor(full_box_min_y), math.ceil(full_box_max_x)-math.floor(full_box_min_x)))
# tf_img_1 = transform_image(orig_img, tf_1, (math.ceil(full_box_max_y)-math.floor(full_box_min_y), math.ceil(full_box_max_x)-math.floor(full_box_min_x)))
tf_img_0 = transform_image(orig_img, tf_0)
tf_img_1 = transform_image(orig_img, tf_1)


plt.subplot(231)
plt.imshow(orig_img)
plt.scatter(orig_box[0,:],orig_box[1,:])
plt.scatter(full_box[0,:],full_box[1,:])

plt.subplot(232)
plt.imshow(tf_img_0)
plt.scatter(tf_box_0[0,:],tf_box_0[1,:])
plt.scatter(full_box[0,:],full_box[1,:])

plt.subplot(233)
plt.imshow(tf_img_1)
plt.scatter(tf_box_1[0,:],tf_box_1[1,:])
plt.scatter(full_box[0,:],full_box[1,:])

orig_img_exp = 0*np.ones((math.ceil(full_box_max_y)-math.floor(full_box_min_y), math.ceil(full_box_max_x)-math.floor(full_box_min_x),3), dtype=tf_img_0.dtype)
orig_img_exp_at_0 = orig_img_exp.copy()

orig_img_exp_at_0[0:orig_img_height, 0:orig_img_width] = orig_img

offset_tf = AffineTransform()
offset_tf.add_translation(-math.floor(full_box_min_x), -math.floor(full_box_min_y))
offset_tf = offset_tf.as_ndarray()

orig_img_exp = transform_image(orig_img_exp_at_0, offset_tf)
tf_img_0_exp = transform_image(orig_img_exp_at_0, np.matmul(offset_tf, tf_0))
tf_img_1_exp = transform_image(orig_img_exp_at_0, np.matmul(offset_tf, tf_1))

orig_box_exp = transform_points(orig_box, offset_tf)
tf_box_0_exp = transform_points(orig_box, np.matmul(offset_tf, tf_0))
tf_box_1_exp = transform_points(orig_box, np.matmul(offset_tf, tf_1))

tf0_2_tf1 = np.matmul(offset_tf, tf_1) @ np.linalg.inv(np.matmul(offset_tf, tf_0))

tf_box_0_exp_tf_1 = transform_points(tf_box_0_exp, tf0_2_tf1)

full_box_exp = transform_points(full_box, offset_tf)

plt.subplot(234)
plt.imshow(orig_img_exp)
plt.scatter(orig_box_exp[0,:],orig_box_exp[1,:])
plt.scatter(full_box_exp[0,:],full_box_exp[1,:])

plt.subplot(235)
plt.imshow(tf_img_0_exp)
plt.scatter(tf_box_0_exp[0,:],tf_box_0_exp[1,:])
plt.scatter(full_box_exp[0,:],full_box_exp[1,:])

plt.subplot(236)
plt.imshow(tf_img_1_exp)
plt.scatter(tf_box_1_exp[0,:],tf_box_1_exp[1,:])
plt.scatter(full_box_exp[0,:],full_box_exp[1,:])
plt.scatter(tf_box_0_exp_tf_1[0,:], tf_box_0_exp_tf_1[1,:],color='red', marker='x')
plt.show()
"""
