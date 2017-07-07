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
from mycroft.messagebus.message import Message
import urllib
from time import time, sleep
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from os.path import dirname

sys.path.append(dirname(__file__))

from object_detection.utils import label_map_util

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

        self.emitter.on("object.recognition.request", self.handle_recognition_request)

    def handle_view_objects_intent(self, message):
        self.speak('Testing object recognition')
        objrecog = ObjectRecogService(self.emitter, timeout=30)
        result = objrecog.recognize_objects(dirname(__file__) + "/test.jpg", server=False)
        labels = result.get("labels", {})
        ut = ""
        for object in labels:
            count = labels[object]
            ut += str(count) + " " + object + " \n"
        self.speak(ut)

    def handle_recognition_request(self, message):
        if message.context is not None:
            self.context.update(message.context)
        file = message.data.get("file", dirname(__file__) + "/test.jpg")
        self.log.info("Loading tensorflow model into memory")
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)
        self.log.info("Loading image")
        frame = cv2.imread(file)
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
                #objects[o]["box"] = c
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

        self.log.info("detected : " + str(objects))
        self.emitter.emit(Message("object.recognition.result", {"labels": labels, "objects": objects}, self.context))
        # to source socket
        if ":" in self.context.get("source", ""):
            if self.context["destinatary"].split(":")[1].isdigit():
                self.emitter.emit(Message("message_request",
                                          {"context": self.context, "data": {"labels": labels, "objects": objects},
                                           "type": "object.recognition.result"}, self.context))

    def stop(self):
        pass


def create_skill():
    return ObjectRecogSkill()


def url_to_pic(url):
    saved_url = dirname(__file__) + "/temp.jpg"
    f = open(saved_url, 'wb')
    f.write(urllib.urlopen(url).read())
    f.close()
    return saved_url


class ServiceBackend(object):
    """
        Base class for all service implementations.

        Args:
            name: name of service (str)
            emitter: eventemitter or websocket object
            timeout: time in seconds to wait for response (int)
            waiting_messages: end wait on any of these messages (list)

    """

    def __init__(self, name, emitter=None, timeout=5, waiting_messages=None, logger=None):
        self.initialize(name, emitter, timeout, waiting_messages, logger)

    def initialize(self, name, emitter, timeout, waiting_messages, logger):
        """
           initialize emitter, register events, initialize internal variables
        """
        self.name = name
        self.emitter = emitter
        self.timeout = timeout
        self.result = None
        self.waiting = False
        self.waiting_for = "any"
        if logger is None:
            self.logger = getLogger(self.name)
        else:
            self.logger = logger
        if waiting_messages is None:
            waiting_messages = []
        self.waiting_messages = waiting_messages
        for msg in waiting_messages:
            self.emitter.on(msg, self.end_wait)
        self.context = {"source": self.name, "waiting_for": self.waiting_messages}

    def send_request(self, message_type, message_data=None, message_context=None, server=False):
        """
          send message
        """
        if message_data is None:
            message_data = {}
        if message_context is None:
            message_context = {"source": self.name, "waiting_for": self.waiting_messages}
        if not server:
            self.emitter.emit(Message(message_type, message_data, message_context))
        else:
            type = "bus"
            if "file" in message_data.keys():
                type = "file"
            self.emitter.emit(Message("server_request",
                                      {"server_msg_type": type, "requester": self.name,
                                       "message_type": message_type,
                                       "message_data": message_data}, message_context))

    def wait(self, waiting_for="any"):
        """
            wait until result response or time_out
            waiting_for: message that ends wait, by default use any of waiting_messages list
            returns True if result received, False on timeout
        """
        self.waiting_for = waiting_for
        if self.waiting_for != "any" and self.waiting_for not in self.waiting_messages:
            self.emitter.on(waiting_for, self.end_wait)
            self.waiting_messages.append(waiting_for)
        self.waiting = True
        start = time()
        elapsed = 0
        while self.waiting and elapsed < self.timeout:
            elapsed = time() - start
            sleep(0.3)
        self.process_result()
        return not self.waiting

    def end_wait(self, message):
        """
            Check if this is the message we were waiting for and save result
        """
        if self.waiting_for == "any" or message.type == self.waiting_for:
            self.result = message.data
            if message.context is None:
                message.context = {}
            self.context.update(message.context)
            self.waiting = False

    def get_result(self):
        """
            return last processed result
        """
        return self.result

    def process_result(self):
        """
         process and return only desired data
         """
        if self.result is None:
            self.result = {}
        return self.result


class ObjectRecogService(ServiceBackend):
    def __init__(self, emitter=None, timeout=25, waiting_messages=None, logger=None):
        super(ObjectRecogService, self).__init__(name="ObjectRecogService", emitter=emitter, timeout=timeout, waiting_messages=waiting_messages, logger=logger)

    def recognize_objects(self, picture_path, context=None, server=False):
        self.send_request("object.recognition.request", {"file": picture_path}, context, server=server)
        self.wait("object.recognition.result")
        return self.result

    def recognize_objects_from_url(self, picture_url, context=None, server=False):
        self.send_request("object.recognition.request", {"url": url_to_pic(picture_url)}, context, server=server)
        self.wait("object.recognition.result")
        return self.result
