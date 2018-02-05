import cv2
import numpy as np
import os
import sys
import time
import glob
import pickle
import vision as vis
import vehicles as veh
from moviepy.editor import VideoFileClip

def calibration_progress(msg):
    print(msg)

camera_cal = vis.CameraCal()
camera_cal.calibrate_from_chessboard(
    chessboard_files=glob.glob('./camera_cal/calibration*.jpg'), 
    chessboard_size=(9,6), 
    save_diags=False,
    progress=calibration_progress)

print('loading trained classifier...')
clf_data_path = sys.argv[1]
spatial_params, hist_params, hog_params, ftr_scaler, veh_clf = veh.load_trained_classifier(clf_data_path)

ftr_extractor = veh.VehicleFeatureExtractor(spatial_params, hist_params, hog_params)
veh_detector = veh.VehicleDetector(ftr_extractor, ftr_scaler, veh_clf)
detect_vehicles = veh.VehicleDetectionPipeline(camera_cal, veh_detector)
    
clip = VideoFileClip('./project_video.mp4')
veh_clip = clip.fl_image(detect_vehicles)
veh_clip.write_videofile('./output_videos/project_video_out.mp4', audio=False)
