from vehicle_detection_pipeline import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2
import pickle
from scipy.ndimage.measurements import label
from queue import Queue
class VehicleDetectionPipeline(object):
    def __init__(self, model_file, scaler_file,
                 orient=9, pix_per_cell=8,
                 cell_per_block=2,
                 hist_bins=32,
                 cell_per_step=1,
                 threshold=7,
                 spatial_size=(32,32)):
        self.svc_model = None
        self.scaler = None
        assert(os.path.exists(model_file))
        assert(os.path.exists(scaler_file))
        with open(model_file, 'rb') as f:
            self.svc_model = pickle.load(f)
        with open(scaler_file, 'rb') as f:
            self.scaler = pickle.load(f)
        self.heat_map = None
        self.heat_map_queue = []
        self.scales = [2, 3, 4]
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hist_bins = hist_bins
        self.cell_per_step = cell_per_step
        self.threshold = threshold
        self.spatial_size = spatial_size
        self.queue_max_size = 7

    def get_average_heat_map_queue(self):
        if len(self.heat_map_queue) > 0:
            return np.average(np.array(self.heat_map_queue), axis=0)

    def add_heat_map_to_queue(self, heat_map):
        if len(self.heat_map_queue) == self.queue_max_size:
            self.heat_map_queue.pop(0)
        self.heat_map_queue.append(heat_map)

    def get_heat_map_avg(self, new_heatmap, count):
        self.heat_map = self.heat_map + (new_heatmap - self.heat_map) / (count + 1)

    def run(self, video_file, save_video=False, debug=True):
        assert os.path.exists(video_file)
        print ("Start Video Processing")
        # open the video and feed the frame here
        cap = cv2.VideoCapture(video_file)
        if save_video:
            codec = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter('output.avi', codec, 20.0, (1280, 720))
        count = 0
        while cap.isOpened():
            ret, orig_frame = cap.read()
            # if frame is valid then run it through the pipe line
            if not ret:
                break
            # run sliding windows on the interest area of each frame
            if self.heat_map is None:
                self.heat_map = np.zeros((orig_frame.shape[0], orig_frame.shape[1]))
            print(orig_frame.shape)
            ystart = orig_frame.shape[0] // 2
            ystop = orig_frame.shape[0] - 200
            print(ystart, ystop)
            labels = None
            heat_map = np.zeros((orig_frame.shape[0], orig_frame.shape[1]))
            for scale in self.scales:
                heat_map = slide_windows_and_update_heat_map(orig_frame,
                                                      ystart,
                                                      ystop,
                                                      scale,
                                                      self.svc_model,
                                                      self.scaler,
                                                      self.orient,
                                                      self.pix_per_cell,
                                                      self.cell_per_block,
                                                      self.spatial_size,
                                                      self.hist_bins,
                                                      window_size=64,
                                                      cells_per_step=self.cell_per_step,
                                                      threshold=self.threshold,
                                                      heat_map=heat_map)
            #assert(labels is not None)
            #self.get_heat_map_avg(heat_map, count)
            self.add_heat_map_to_queue(heat_map)
            self.heat_map = self.get_average_heat_map_queue()
            self.heat_map[heat_map <= self.threshold] = 0
            labels = label(self.heat_map)
            draw_bounding_boxes_from_labels(orig_frame, labels)
            if save_video:
                out.write(orig_frame)
            # show image
            if debug:
                cv2.imshow('Frame', orig_frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                cv2.imshow('heatmap', self.heat_map)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            count += 1
        if save_video:
            out.release()
        cap.release()

if __name__ == "__main__":
    pipeline = VehicleDetectionPipeline('./svc_model.pkl', './svc_scaler.pkl')
    pipeline.run('../project_video.mp4', save_video=True, debug=False)

