'''
This is a sample class that you can use to control the mouse pointer.
It uses the pyautogui library. You can set the precision for mouse movement
(how much the mouse moves) and the speed (how fast it moves) by changing 
precision_dict and speed_dict.
Calling the move function with the x and y output of the gaze estimation model
will move the pointer.
This class is provided to help get you started; you can choose whether you want to use it or create your own from scratch.
'''
import cv2
import sys
import json
import pyautogui
import logging as log


from input_feeder import InputFeeder
from face_detection import facedetection, get_draw_boxes
from facial_landmarks_detection import facial_landmarks, get_draw_points
from gaze_estimation import gaze
from head_pose_estimation import head_pose

FORMATTER = log.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
console_handler = log.StreamHandler(sys.stdout)
console_handler.setFormatter(FORMATTER)
logger = log.getLogger(__name__)
logger.setLevel(log.DEBUG)
logger.addHandler(console_handler)

# retrieve and parse configuration
with open("conf/application.conf", "r") as confFile:
    conf = json.loads(confFile.read())

class MouseController:
    def __init__(self, precision, speed):
        precision_dict={'high':100, 'low':1000, 'medium':500}
        speed_dict={'fast':1, 'slow':10, 'medium':5}

        self.precision=precision_dict[precision]
        self.speed=speed_dict[speed]

    def move(self, x, y):
        pyautogui.moveRel(x*self.precision, -1*y*self.precision, duration=self.speed)


def main():
    image = cv2.imread("/home/pjvazquez/Imágenes/captura-1-1.jpg")
    logger.debug("image frame {}".format(image.shape))
    fd_model = facedetection(conf['face_detection_model'])
    fd_model.load_model()
    fl_model = facial_landmarks(conf['facial_landmarks_model'])
    fl_model.load_model()
    hp_model = head_pose(conf['head_pose_model'])
    hp_model.load_model()

    feed=InputFeeder(input_type="realsense", input_file="bin/demo.mp4")
    feed.load_data()
    for batch in feed.next_batch():
        try:
            # FACE DETECTION
            processed_image = fd_model.preprocess_input(batch)
            prediction = fd_model.predict(processed_image)
            logger.debug("Have Prediction")
            fd_result = fd_model.preprocess_output(outputs=prediction)
            logger.debug("result shape: {} model shape {}".format(fd_result.shape, fd_model.shape))
            image2, num, face = get_draw_boxes(fd_result, batch)
            logger.info("Obtained {} Result: {}, face shape: {}".format(num, fd_result.shape, face.shape))
            
            # FACIAL LANDMARK
            fl_preprocesed_face = fl_model.preprocess_input(face)
            fl_result = fl_model.predict(fl_preprocesed_face)
            logger.debug("Faciual Landmarks {}".format(fl_result['95']))
            result, eyes = fl_model.preprocess_output(outputs=fl_result)
            painted_face = get_draw_points(result, face)
            
            # HEAD POSE
            hp_processed_face = hp_model.preprocess_input(face)
            hp_result = hp_model.predict(hp_processed_face)
            logger.debug("Result of head pose is {}".format(hp_result))

            # GAZE ESTIMATION
            

            cv2.imshow("image", image2)
            painted_face = cv2.resize(painted_face,(180,180))
            cv2.imshow("face", painted_face)
            if cv2.waitKey(500) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        finally:
            pass
    

if __name__ == '__main__':
    main()