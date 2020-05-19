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
MODEL_NAME = conf['facial_landmarks_model']

class facial_landmarks:
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
        # processed_image = self.preprocess_input(image)
        logger.debug("To predict, image shape {}".format(image.shape))
        prediction =  self.net_plugin.infer(inputs={self.input_blob: image})
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
        logger.debug("Preprocess landmarks for {}".format(image))
        self.shape = self.net.inputs[self.input_blob].shape
        processed_image = cv2.resize(image,(self.input_shape[3], self.input_shape[2]))
        transposed_image = processed_image.transpose((2,0,1))
        reshaped_image = transposed_image.reshape(self.shape)
        logger.debug("SHAPE: {}  {}   {}    {}".format(self.input_shape, processed_image.shape, transposed_image.shape, reshaped_image.shape))
        return reshaped_image

    def preprocess_output(self, outputs, image):
        '''
        TODO: You will need to complete this method.
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        shape = image.shape
        image_h = shape[0]
        image_w = shape[1]
        processed_image = cv2.resize(image,(image_w, image_h))
        logger.info("Processing output of len: {}, on image shape: {}".format(len(outputs), image.shape))
        outputs = outputs[self.output_blob].reshape(10)
        positions = []
        size=30
        eyes = [0]*2
        for i in range(0,len(outputs),2):
            position = [outputs[i], outputs[i+1]]
            if i==0:
                positions.append(position)
                logger.debug("pos num: {}, position: {}".format(i, position))
                xmin = 0 if int(position[0]*image_w-size)<0 else int(position[0]*image_w-size)
                ymin = 0 if int(position[1]*image_h-size)<0 else int(position[1]*image_h-size)
                xmax = image_w if int(position[0]*image_w+size)>image_w else int(position[0]*image_w+size)
                ymax = image_h if int(position[1]*image_h+size)>image_h else int(position[1]*image_h+size)
                logger.debug("11111111 {},{},{},{}".format(xmin, ymin, xmax, ymax))
                eyes[0] = processed_image[xmin:xmax, ymin:ymax ]
            if i==2:
                positions.append(position)
                logger.debug("pos num: {}, position: {}".format(i, position))
                xmin = 0 if int(position[0]*image_w-size)<0 else int(position[0]*image_w-size)
                ymin = 0 if int(position[1]*image_h-size)<0 else int(position[1]*image_h-size)
                xmax = image_w if int(position[0]*image_w+size)>image_w else int(position[0]*image_w+size)
                ymax = image_h if int(position[1]*image_h+size)>image_h else int(position[1]*image_h+size)
                logger.debug("222222222 {},{},{},{}".format(xmin, ymin, xmax, ymax))
                eyes[1] = processed_image[xmin:xmax, ymin:ymax ]

        logger.debug("model predictions {}".format(positions))
        return positions, eyes


def get_draw_points(positions, image):
    '''
        Function that returns the boundinng boxes detected for class "person" 
        with a confidence greater than 0, paint the bounding boxes on image
        and counts them
    '''
    shape = image.shape
    image_h = shape[0]
    image_w = shape[1]
    logger.debug("shape for landmarks {}".format(shape))
    processed_image = cv2.resize(image,(image_w, image_h))
    for point in positions:
        cv2.circle(processed_image,(int(point[0]*image_w), int(point[1]*image_h)), 4, (0,0,255))
        logger.debug("position: {}".format(int(point[0]*image_h), int(point[1]*image_w)))
    return processed_image

def main():
    image_file = "/home/pjvazquez/Imágenes/captura-1-7.jpg"

    model = facial_landmarks(MODEL_NAME)
    model.load_model()

    feed=InputFeeder(input_type='image', input_file=image_file)
    feed.load_data()
    for batch in feed.next_batch():
        processed_image = model.preprocess_input(batch)
        prediction = model.predict(processed_image)
        # logger.debug("Prediction: {}".format(prediction))
        # TODO: finish this
        logger.debug(prediction['95'])
        result, eyes = model.preprocess_output(outputs=prediction, image=batch)
        logger.debug("result shape: {} model shape {}".format(result, model.shape))
        image2 = get_draw_points(result, batch)
        logger.info("Obtained Result: {}".format(result))
        cv2.imshow("image", image2)
        logger.debug("eye shape: {}".format(eyes[0].shape))
        cv2.imshow("le", eyes[0])
        cv2.imshow("re", eyes[1])

        if cv2.waitKey(500) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()