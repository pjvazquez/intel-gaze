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
import logging as log
from openvino.inference_engine import IENetwork, IECore

FORMATTER = log.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
console_handler = log.StreamHandler(sys.stdout)
console_handler.setFormatter(FORMATTER)
logger = log.getLogger(__name__)
logger.setLevel(log.DEBUG)
logger.addHandler(console_handler)

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
CPU_EXTENSION = "bin/libcpu_extension_sse4.so"

MODEL_NAME = "/mnt/DATA/Python_Projects/intel-gaze/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml"

class facedetection:
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

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        logger.info("making prediction")
        processed_image = self.preprocess_input(image)
        logger.debug("To predict, image shape {}".format(processed_image.shape))
        prediction =  self.net_plugin.infer(inputs={self.input_blob: processed_image})
        return prediction

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

    def preprocess_input(self, image):
        '''
        TODO: You will need to complete this method.
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        self.shape = self.net.inputs[self.input_blob].shape
        processed_image = cv2.resize(image,(self.input_shape[3], self.input_shape[2]))
        transposed_image = processed_image.transpose((2,0,1))
        reshaped_image = processed_image.reshape(self.shape)
        logger.debug("SHAPE: {}  {}   {}    {}".format(self.shape, processed_image.shape, transposed_image.shape, reshaped_image.shape))
        return reshaped_image

    def preprocess_output(self, outputs):
        '''
        TODO: You will need to complete this method.
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        logger.info("Processing output blobs")
        outputs = outputs[self.output_blob]
        logger.debug("model predictions {}".format(outputs))
        return outputs[0][0]


def get_draw_boxes(boxes, image, shape=None):
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
        if box[2] > 0.039:
            if box[1] == 1:
                cv2.rectangle(processed_image,(int(box[3]*image_w), int(box[4]*image_h)), (int(box[5]*image_w), int(box[6]*image_h)), (0,0,255), 2)
            num_detections +=1

    logger.debug("MAX THR: {}, NUM CLASSES {}".format(max_conf, set(list_classes)))
    return processed_image, num_detections

def main():
    image = cv2.imread("/home/pjvazquez/Imágenes/captura-1-3.jpg")
    logger.debug("image frame {}".format(image.shape))
    model = facedetection(MODEL_NAME)
    model.load_model()
    prediction = model.predict(image)
    logger.debug("Prediction: {}".format(prediction['detection_out'].shape))
    logger.debug(prediction['detection_out'][0,0,1,:])
    result = model.preprocess_output(outputs=prediction)
    logger.debug("result shape: {} model shape {}".format(result.shape, model.shape))
    image2, num = get_draw_boxes(result, image)
    logger.info("Obtained {} Result: {}".format(num, result))
    cv2.imwrite("image2.jpg", image2)

if __name__ == '__main__':
    main()