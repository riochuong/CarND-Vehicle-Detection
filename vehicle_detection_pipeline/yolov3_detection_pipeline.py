from vehicle_detection_utils import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2
import pickle
from scipy.ndimage.measurements import label
from queue import Queue
CAR_CLASS_ID = 3
CONFIDENCE_THRESHOLD = 0.5
class YoloV3VehicleDetectionPipeline(object):
    def __init__(self, 
                 model_file, 
                 weights_file,
                 yolo_score_threshold=0.5,
                 heat_map_threshold=0.5
                 ):
        assert(os.path.exists(model_file))
        assert(os.path.exists(weights_file))
        self.model = cv2.dnn.readNetFromDarknet(model_file, weights_file)
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.yolo_score_threshold = yolo_score_threshold
        self.heat_map_threshold = heat_map_threshold

    def run(self, video_file, save_video=False, debug=True):
        assert os.path.exists(video_file)
        print ("Start Video Processing")
        # open the video and feed the frame here
        cap = cv2.VideoCapture(video_file)
        if save_video:
            codec = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter('project_output_yolo.avi', codec, 20.0, (1280, 720))
        # using for quick debug to skip frames
        frame_count = 0
        skip = 0
        while cap.isOpened():
            frame_count += 1
            ret, orig_frame = cap.read()
            if frame_count < skip:
                continue
            # if frame is valid then run it through the pipe line
            if not ret:
                break
            labels = None
            heat_map = np.zeros((orig_frame.shape[0], orig_frame.shape[1]))
            # process each frame with YOLO
            heat_map = process_each_frame_yolov3(orig_frame, 
                                                 self.model, 
                                                 300, 620, self.yolo_score_threshold, heat_map)
            # add heatmap to queue 
            if heat_map is not None:
                print ("heat map val: ",np.max(heat_map))
                if debug:
                    cv2.imshow('heatmap', heat_map)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                # filter out heatmap threshold here 
                labels = label(heat_map)
                draw_bounding_boxes_from_labels(orig_frame, labels)
      
            if save_video:
                out.write(orig_frame)
            # show image
            if debug:
                cv2.imshow('Frame', orig_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        if save_video:
            out.release()
        print ("Clean up and Close")
        cap.release()

if __name__ == "__main__":
    pipeline = YoloV3VehicleDetectionPipeline(model_file='../yolov3_models/yolov3_320.cfg', 
                                        weights_file='../yolov3_models/yolov3_320.weights')
    pipeline.run('../project_video.mp4', save_video=True, debug=False)



