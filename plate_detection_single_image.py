import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageDraw

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches

import cv2
import reader


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[9]:

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

image_array = []


def find_plate(image):
    gray_image = rgb2gray(image)

    # HEIGHT AND WIDTH OF IMAGE
    image_height, image_width = gray_image.shape

    # OTSU METHOD FINDS BEST THRESHOLD VALUE
    threshold = threshold_otsu(gray_image)
    binary_image = gray_image > threshold

    # LABELING CONNECTED COMPONENTS
    components = measure.label(binary_image)

    # CREATE A WINDOW TO SHOW IMAGE AND RESULT; LEFT IS FOR ORIGINAL IMAGE RIGHT IS FOR RESULT
    # window, (left, right) = plt.subplots(1, 2)

    # AN ARRAY TO KEEP PLATE LABEL's COORDINATES etc.
    labels = []

    for component in regionprops(components):

        # COORDINATES OF COMPONENT
        x_min, y_min, x_max, y_max = component.bbox

        # FIND HEIGHT AND WIDTH OF COMPONENT
        component_height = x_max - x_min
        component_width = y_max - y_min

        if component.area < 50:
            # SKIP TOO SMALL COMPONENTS
            continue
        # IF SIZE OF COMPONENT IS LIKE PLATE ADD IT TO LABELS ARRAY
        if image_height * 0.2 >= component_height >= image_height * 0.1 and \
                image_width * 0.4 >= component_width >= image_width * 0.1 and \
                5 > float(component_width) / float(component_height) > 3:
            x_min, y_min, x_max, y_max = component.bbox
            label_borders = patches.Rectangle((y_min, x_min), y_max - y_min, x_max - x_min, edgecolor="red",
                                              linewidth=2, fill=False)
            borders = [int(x_min), int(y_min), int(x_max), int(y_max)]
            # SHOW LABEL IN RIGHT WINDOW
            # right.add_patch(label_borders)
            labels.append(borders)


    # left.imshow(image, cmap="gray")
    # right.imshow(binary_image, cmap="gray")
    # plt.show()

    return labels


def proc_img(filepath):
	image = cv2.imread(filepath)

	# This is needed since the notebook is stored in the object_detection folder.
	sys.path.append("/home/eko/.local/lib/python3.5/site-packages/tensorflow/models/research")


	# ## Object detection imports
	# Here are the imports from the object detection module.

	# In[3]:

	from object_detection.utils import label_map_util

	from object_detection.utils import visualization_utils as vis_util
	# Path to frozen detection graph. This is the actual model that is used for the object detection.
	PATH_TO_CKPT = 'frozen_inference_graph_stan.pb'

	# List of the strings that is used to add correct label for each box.
	PATH_TO_LABELS = 'labelmap.pbtxt'

	NUM_CLASSES = 1

	# ## Load a (frozen) Tensorflow model into memory.

	# In[6]:

	detection_graph = tf.Graph()
	with detection_graph.as_default():
	  od_graph_def = tf.GraphDef()
	  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
	    serialized_graph = fid.read()
	    od_graph_def.ParseFromString(serialized_graph)
	    tf.import_graph_def(od_graph_def, name='')


	# ## Loading label map
	# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

	# In[7]:

	label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
	categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)


	# ## Helper code

	# In[8]:



	# In[10]:

	start_time = time.time()
	img_cnt = 1

	font = cv2.FONT_HERSHEY_SIMPLEX


	try:

		with detection_graph.as_default():
			with tf.Session(graph=detection_graph) as sess:
				image_np = np.copy(image)
				# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
				image_np_expanded = np.expand_dims(image_np, axis=0)
				image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
				# Each box represents a part of the image where a particular object was detected.
				boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
				# Each score represent how level of confidence for each of the objects.
				# Score is shown on the result image, together with the class label.
				scores = detection_graph.get_tensor_by_name('detection_scores:0')
				classes = detection_graph.get_tensor_by_name('detection_classes:0')
				num_detections = detection_graph.get_tensor_by_name('num_detections:0')
				# Actual detection.
				(boxes, scores, classes, num_detections) = sess.run(
					[boxes, scores, classes, num_detections],
					feed_dict={image_tensor: image_np_expanded})

				vehicles = []
				for i in range(len(scores[0])):
					if scores[0][i] > 0.8:

						x1 = int(boxes[0][i][0] * image_np.shape[0])
						y1 = int(boxes[0][i][1] * image_np.shape[1])
						x2 = int(boxes[0][i][2] * image_np.shape[0])
						y2 = int(boxes[0][i][3] * image_np.shape[1])

						if x2-x1 > image_np.shape[0] / 6 and y2-y1 > image_np.shape[1] / 6:
							edges = [int(x1 + ((x2 - x1) / 4)), y1, int(x2 - ((x2 - x1) / 4)), y2]
							vehicles.append(edges)

				plates = []

				for b in vehicles:
					car_np = image_np[b[0]:b[2], b[1]:b[3]]
					labels = find_plate(car_np)

					for l in labels:
					    plates.append([b[0] + l[0], b[1] + l[1], b[0] + l[2], b[1] + l[3]])

				for p in plates:
					# print 'RATIO : ', float(p[3] - p[1]) / float(p[2] - p[0])
					xmin = p[0]
					ymin = p[1]
					xmax = p[2]
					ymax = p[3]
					if p[0] - 5 >= 0:
						xmin =  p[0] - 5
					if p[1] - 5 >= 0:
						ymin = p[1] - 5
					if p[2] + 5 <= image_np.shape[0]:
						xmax =  p[2] + 5
					if p[3] + 5 <= image_np.shape[1]:
						ymax = p[3] + 5
					plt_img = image_np[xmin: xmax, ymin:ymax]
					plate_reader = reader.reader(plt_img)
					plate_txt = plate_reader.read()
					if len(plate_txt) > 0:
						print 'THE PLATE IS : ' + str(plate_txt)
						image_np = cv2.rectangle(image_np,(p[1], p[0]),(p[3], p[2]),(0,255,255),3)
						image_np = cv2.putText(image_np, plate_txt, (p[1], p[0]), font, 0.8 ,(80,127,255),2,cv2.LINE_AA)

				for i in range(len(scores[0])):
					if scores[0][i] > 0.8:

						x1 = int(boxes[0][i][0] * image_np.shape[0])
						y1 = int(boxes[0][i][1] * image_np.shape[1])
						x2 = int(boxes[0][i][2] * image_np.shape[0])
						y2 = int(boxes[0][i][3] * image_np.shape[1])

						image_np = cv2.rectangle(image_np,(y1,x1),(y2, x2),(0,255,0),3)


	except Exception as e:
		print(e)

	finally:
		cv2.imwrite('dtcimg.png', image_np)
		return plate_txt
