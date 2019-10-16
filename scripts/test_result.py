from collections import namedtuple

TestResult = namedtuple('TestResult', ['original_image', 'transform_0', 'transform_1','offset_transform','transformed_image_0', 'transformed_image_1', 'network_output_0', 'network_output_1', 'feature_locations_0', 'feature_locations_1', 'repeatability'])
