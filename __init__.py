# Copyright 2016 Mycroft AI, Inc.
#
# This file is part of Mycroft Core.
#
# Mycroft Core is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Mycroft Core is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Mycroft Core.  If not, see <http://www.gnu.org/licenses/>.

from adapt.intent import IntentBuilder

from mycroft.skills.core import MycroftSkill
from mycroft.util.log import getLogger

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from os.path import dirname

sys.path.append(dirname(__file__))

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

__author__ = 'eClarity' , 'jarbas'

LOGGER = getLogger(__name__)

CWD_PATH = os.path.dirname(__file__)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def detect_objects(image_np, sess, detection_graph):
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
    #vis_util.visualize_boxes_and_labels_on_image_array(
    #    image_np,
    #    np.squeeze(boxes),
    #    np.squeeze(classes).astype(np.int32),
    #    np.squeeze(scores),
    #    category_index,
    #    use_normalized_coordinates=True,
    #    line_thickness=8)
    return image_np, boxes, scores, classes, num_detections


class ObjectRecogSkill(MycroftSkill):
    def __init__(self):
        super(ObjectRecogSkill, self).__init__(name="ObjectRecogSkill")

    def initialize(self):
        view_objects_intent = IntentBuilder("TestObjectRecogIntent"). \
            require("TestViewObjectsKeyword").build()
        self.register_intent(view_objects_intent, self.handle_view_objects_intent)

    def handle_view_objects_intent(self, message):
        self.speak('Testing object recognition')
        # Load a (frozen) Tensorflow model into memory.
        self.log.info("Loading tensorflow model into memory")
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)
        self.log.info("Loading test image")
        frame = cv2.imread(dirname(__file__) + "/test.jpg")
        self.log.info("Detecting objects")
        image_np, boxes, scores, classes, num_detections = detect_objects(frame, sess, detection_graph)
        objects = []
        self.log.info("Processing labels")
        for i in classes:
            for c in i:
                obj = {}
                c = int(c)
                if c in category_index.keys():
                    class_name = category_index[c]['name']
                else:
                    class_name = 'N/A'
                obj["label"] = class_name
                objects.append(obj)

        self.log.info("Processing scores")
        for i in scores:
            o = 0
            for c in i:
                c = int(c * 100)
                objects[o]["score"] = c
                o += 1

        self.log.info("Processing bounding boxes")
        for i in boxes:
            o = 0
            for c in i:
                # TODO process into x,y coords rects
                objects[o]["box"] = c
                o += 1

        self.log.info("Counting objects and removing low scores")
        labels = {}
        for obj in objects:
            # bypass low scores
            if obj["score"] < 30:
                continue
            if obj["label"] not in labels.keys():
                labels[obj["label"]] = 1
            else:
                labels[obj["label"]] += 1

        self.log.info("detected : " + str(labels))
        ut = ""
        for object in labels:
            ut += str(labels[object]) + " " + object + " \n"
        self.speak(ut)

    def stop(self):
        pass


def create_skill():
    return ObjectRecogSkill()


