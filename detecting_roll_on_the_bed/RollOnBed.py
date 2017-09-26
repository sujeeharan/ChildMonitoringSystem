

import numpy as np
import os
import urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2


from detecting_roll_on_the_bed.utils import label_map_util

from detecting_roll_on_the_bed.utils import visualization_utils as vis_util

import detecting_roll_on_the_bed.childdetect as ch

def Rolling_On_Bed():
    print('Entering Roll On The Bed')
    cap=cv2.VideoCapture('detecting_roll_on_the_bed/naga.mp4')



    sys.path.append("..")


   
    MODEL_NAME = 'detecting_roll_on_the_bed/ssd_mobilenet_v1_coco_11_06_2017'
    MODEL_FILE = MODEL_NAME + '.tar.gz'



    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

 
    PATH_TO_LABELS = os.path.join('detecting_roll_on_the_bed/data', 'mscoco_label_map.pbtxt')

    NUM_CLASSES = 90


    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


    # ## Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine



    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)


    # ## Helper code



    def load_image_into_numpy_array(image):
      (im_width, im_height) = image.size
      return np.array(image.getdata()).reshape(
          (im_height, im_width, 3)).astype(np.uint8)


    # # Detection



    # For the sake of simplicity we will use only 2 images:
    # image1.jpg
    # image2.jpg
    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    PATH_TO_TEST_IMAGES_DIR = 'test_images'
    TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

    # Size, in inches, of the output images.
    IMAGE_SIZE = (12, 8)


    coordinatedic_dic={}
    coordinatedic_dic_param={}
    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        #for image_path in TEST_IMAGE_PATHS:
        while True :
          ret,image_np=cap.read()

          #image = Image.open(image_path)
          ## the array based representation of the image will be used later in order to prepare the
          ## result image with boxes and labels on it.
          #image_np = load_image_into_numpy_array(image)
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
          # Visualization of the results of a detection.
          coordinatedic_dic_param=vis_util.visualize_boxes_and_labels_on_image_array(
                              image_np,
                              np.squeeze(boxes),
                              np.squeeze(classes).astype(np.int32),
                              np.squeeze(scores),
                              category_index,
                              use_normalized_coordinates=True,
                              line_thickness=8)
          print (coordinatedic_dic_param)
          if coordinatedic_dic_param is not None:
            coordinatedic_dic=coordinatedic_dic_param
            print(coordinatedic_dic['top'])
          cv2.imshow('object',cv2.resize(image_np,(800,600)))
          if cv2.waitKey(25) & 0xFF == ord('q'):
              cv2.destroyAllWindows()
              break
    if coordinatedic_dic is not None:
       ch.chiddetectmethod(cap,coordinatedic_dic)