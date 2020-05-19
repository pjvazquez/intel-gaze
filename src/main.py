import cv2
import sys
import json
import pyautogui
import logging as log

from argparse import ArgumentParser

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

def args_parser():
    """
    Parse args.
    """
    ap = ArgumentParser()
    ap.add_argument("-i", "--input", 
                    help="Can be 'cam', 'realsense', 'video' or 'image'" ,
                    default="realsense")
    ap.add_argument("-if", "--input_file", 
                    default=None)
    ap.add_argument("-d", "--device", 
                    default="CPU")
    ap.add_argument("-t", "--threshold", 
                    default=0.5)
    ap.add_argument("-o", "--output_dir", 
                    help = "Path to output directory", 
                    type = str, 
                    default = None)

    return ap


def main():
    # load arguments
    args = args_parser().parse_args()

    # loading models 
    fd_model = facedetection(model_name=conf['face_detection_model'],
                            device=args.device )
    fd_model.load_model()
    fl_model = facial_landmarks(model_name=conf['facial_landmarks_model'],
                                device=args.device)
    fl_model.load_model()
    hp_model = head_pose(conf['head_pose_model'],
                        device=args.device)
    hp_model.load_model()
    gz_model = gaze(conf['gaze_model'],
                    device=args.device)
    gz_model.load_model()

    feed=InputFeeder(input_type="realsense", input_file="bin/demo.mp4")
    # feed=InputFeeder(input_type=args.input, input_file="bin/demo.mp4")
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
            if face is not None:
                logger.info("Obtained {} Result: {}, face shape: {}".format(num, fd_result.shape, face.shape))
                
                # FACIAL LANDMARK
                fl_preprocesed_face = fl_model.preprocess_input(face)
                fl_result = fl_model.predict(fl_preprocesed_face)
                logger.debug("Faciual Landmarks {}".format(fl_result['95']))
                result, eyes = fl_model.preprocess_output(outputs=fl_result, image=face)
                logger.debug("Eye centers {}".format(result))
                painted_face = get_draw_points(result, face)
                
                # HEAD POSE
                hp_processed_face = hp_model.preprocess_input(face)
                hp_result = hp_model.predict(hp_processed_face)
                logger.debug("Result of head pose is {}".format(hp_result))

                # GAZE ESTIMATION
                gz_predict = gz_model.predict(hp_result, eyes)
                logger.debug("GAZE predict: {}".format(gz_predict))

                xd, yd = gz_predict[0]
                # Gaze directions.
                h, w, z = painted_face.shape

                xl = result[0][0]*w
                yl = result[0][1]*h
                xr = result[1][0]*w
                yr = result[1][1]*h
                cv2.line(painted_face, (int(xl), int(yl)), (int(xl + xd * 150), int(yl + yd * 150)), (0, 55, 255), 2)
                cv2.line(painted_face, (int(xr), int(yr)), (int(xr + xd * 150), int(yr + yd * 150)), (0, 55, 255), 2)

                painted_face = cv2.resize(painted_face,(180,180))
                cv2.imshow("face", painted_face)
            cv2.imshow("image", image2)
            if cv2.waitKey(500) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        finally:
            pass
    

if __name__ == '__main__':
    main()