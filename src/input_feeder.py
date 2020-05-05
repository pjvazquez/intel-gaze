'''
This class can be used to feed input from an image, webcam, or video to your model.
Sample usage:
    feed=InputFeeder(input_type='video', input_file='video.mp4')
    feed.load_data()
    for batch in feed.next_batch():
        do_something(batch)
    feed.close()
'''
import cv2
import numpy as np
from numpy import ndarray
import pyrealsense2 as rs

class InputFeeder:
    def __init__(self, input_type, input_file=None):
        '''
        input_type: str, The type of input. Can be 'video' for video file, 'image' for image file,
                    or 'cam' to use webcam feed.
        input_file: str, The file that contains the input image or video file. Leave empty for cam input_type.
        '''
        self.input_type=input_type
        self.cap = None
        self.pipeline = None
        if input_type=='video' or input_type=='image':
            self.input_file=input_file
    
    def load_data(self):
        if self.input_type=='video':
            self.cap=cv2.VideoCapture(self.input_file)
        elif self.input_type=='cam':
            self.cap=cv2.VideoCapture(0)
        elif self.input_type=="realsense":
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            # self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipeline.start(self.config)
         
        else:
            self.cap=cv2.imread(self.input_file)

    def next_batch(self):
        '''
        Returns the next image from either a video file or webcam.
        If input_type is 'image', then it returns the same image.
        '''
        if isinstance(self.cap, ndarray):
            while True:
                yield self.cap
        elif isinstance(self.pipeline, rs.pipeline):
            while True:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                color_image = np.asanyarray(color_frame.get_data())
                cv2.imshow("image", color_image)
                yield color_image
        else:
            while True:
                #self.cap.set(3, 672)
                #self.cap.set(4, 384)
                _, frame=self.cap.read()
                # self.cap.release()
                yield frame
                # self.cap=cv2.VideoCapture(0)

    def close(self):
        '''
        Closes the VideoCapture.
        '''
        if not self.input_type=='image':
            self.cap.release()

