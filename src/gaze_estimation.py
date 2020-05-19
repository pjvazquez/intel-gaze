'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
Since you will be using four models to build this project, you will need to replicate this file
for each of the models.

This has been provided just to give you an idea of how to structure your model class.
'''

import cv2
import os
import sys
import math
import json
import logging as log
from input_feeder import InputFeeder
from openvino.inference_engine import IENetwork, IECore

FORMATTER = log.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
console_handler = log.StreamHandler(sys.stdout)
console_handler.setFormatter(FORMATTER)
logger = log.getLogger(__name__)
logger.setLevel(log.DEBUG)
logger.addHandler(console_handler)

# retrieve and parse configuration
with open("conf/application.conf", "r") as confFile:
    conf = json.loads(confFile.read())

CPU_EXTENSION = conf['CPU_extension']
MODEL_NAME = conf['gaze_model']

class gaze:
    '''
    Class for a Model.
    '''
    def __init__(self, model_name, device="CPU", extensions=CPU_EXTENSION):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.cpu_extension = extensions
        self.device = device
        self.model_name = model_name
        self.input_shape = None

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        logger.info("Loading model")
        self.plugin = IECore()

        if self.cpu_extension and "CPU" in self.device:
            self.plugin.add_extension(self.cpu_extension, "CPU")
        
        model_xml_file = self.model_name
        model_weights_file = os.path.splitext(model_xml_file)[0]+".bin"

        self.net = IENetwork(model=model_xml_file, weights = model_weights_file)

        self.net_plugin = self.plugin.load_network(self.net, self.device)
        self.input_blob = next(iter(self.net.inputs))
        self.output_blob = next(iter(self.net.outputs))

        self.check_plugin()

        self.input_shape = (self.net.inputs[self.input_blob].shape)
        logger.debug("net shape: {}".format(self.input_shape))
        
        return self.plugin

    def predict(self, head_pose, eyes):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        logger.info("making prediction")
        left_eye, right_eye, angles = self.preprocess_input(eyes, head_pose)
        prediction =  self.net_plugin.infer(inputs={
            "head_pose_angles": angles,
            "left_eye_image": left_eye,
            "right_eye_image": right_eye})
        processed_prediction = self.preprocess_output(prediction)
        return processed_prediction

    def check_plugin(self):
        '''
        TODO: You will need to complete this method as a part of the 
        standout suggestions

        This method checks whether the model(along with the plugin) is supported
        on the CPU device or not. If not, then this raises and Exception
        '''
        network_supported_layers = self.plugin.query_network(network=self.net, device_name="CPU")
        
        not_supported_layers = []
        for layer in self.net.layers.keys():
            if layer not in network_supported_layers:
                not_supported_layers.append(layer)
        if len(not_supported_layers)>0:
            logger.debug("Not supported layers in model: {} ".format(not_supported_layers))
            exit(1)

    def preprocess_input(self, eyes, head_pose):
        '''
        TODO: You will need to complete this method.
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        left_eye = cv2.resize(eyes[0], (60, 60))
        left_eye = left_eye.transpose((2, 0, 1))
        left_eye = left_eye.reshape((1, 3, 60, 60))

        right_eye = cv2.resize(eyes[1], (60, 60))
        right_eye = right_eye.transpose((2, 0, 1))
        right_eye = right_eye.reshape((1, 3, 60, 60))

        logger.debug("SHAPES  LE:{} RE:{}".format(left_eye.shape, right_eye.shape))

                # Pose, yaw and roll.
        p_fc = head_pose["angle_p_fc"][0]
        y_fc = head_pose["angle_y_fc"][0]
        r_fc = head_pose["angle_r_fc"][0]
        
        angles = [[y_fc, p_fc, r_fc]]

        return left_eye,right_eye, angles
        

    def preprocess_output(self, outputs):
        '''
        TODO: You will need to complete this method.
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        outputs=outputs[self.output_blob]
        logger.debug("To process: {}".format(outputs))
        gaze = outputs[0]
        r = gaze[2]
        
        x = math.cos(r * math.pi / 180.0)
        y = math.sin(r * math.pi / 180.0)

        X = gaze[0] * x + gaze[1] * y
        Y = -gaze[0] * y + gaze[1] * x

        return (X, -Y), gaze


def get_draw_boxes(boxes, image):
    '''
        Function that returns the boundinng boxes detected for class "person" 
        with a confidence greater than 0, paint the bounding boxes on image
        and counts them
    '''
    image_h, image_w, _ = image.shape
    logger.debug("Image shape {}".format(image.shape))
    image_h=384
    image_w = 672
    num_detections = 0
    max_conf = 0
    processed_image = cv2.resize(image,(image_w, image_h))
    list_classes = []
    for i, box in enumerate(boxes):
        list_classes.append(box[2])
        if box[2] > max_conf:
            max_conf = box[2]
        if box[2] > 0.071:
            if box[1] == 1:
                cv2.rectangle(processed_image,(int(box[3]*image_w), int(box[4]*image_h)), (int(box[5]*image_w), int(box[6]*image_h)), (0,0,255), 2)
            num_detections +=1

    logger.debug("MAX THR: {}, NUM CLASSES {}".format(max_conf, set(list_classes)))
    return processed_image, num_detections

