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

# constants & script arguments
in_file = sys.argv[1]
out_file = sys.argv[2]
clf_file = sys.argv[3]
detector_history_depth = int(sys.argv[4])
detector_heat_thresh = np.float(sys.argv[5])

# calibrate camera from checkerboard images
print("calibrating camera")
camera_cal = vis.CameraCal()
camera_cal.calibrate_from_chessboard(
    chessboard_files=glob.glob('./camera_cal/calibration*.jpg'), 
    chessboard_size=(9,6), 
    save_diags=False,
    progress=None)

# load trained classifier, create vehicle detector
print('loading trained classifier')
spatial_params, hist_params, hog_params, ftr_scaler, veh_clf = veh.load_classifier(clf_file)

ftr_extractor = veh.VehicleFeatureExtractor(
    spatial_params, 
    hist_params, 
    hog_params)
    
match_vehicles = veh.VehicleRecognizer(
    ftr_extractor, 
    ftr_scaler, 
    veh_clf,
    tile_sizes=[64,80,96])
    
vis_out_dir = './output_images/vis_detect/'
    
def vis_recog(recog_img, img_num):
    recog_file_path = os.path.join(vis_out_dir, 'recog-%d.png' % img_num)
    bgr = cv2.cvtColor(recog_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(recog_file_path, bgr)
    
def vis_heat(heat_img, img_num):
    heat_file_path = os.path.join(vis_out_dir, 'heat-%d.png' % img_num)
    cv2.imwrite(heat_file_path, heat_img)
    
def vis_labels(labels_img, img_num):
    labels_file_path = os.path.join(vis_out_dir, 'labels-%d.png' % img_num)
    cv2.imwrite(labels_file_path, labels_img)
    
def vis_detect(detect_img, img_num):
    detect_file_path = os.path.join(vis_out_dir, 'detect-%d.png' % img_num)
    bgr = cv2.cvtColor(detect_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(detect_file_path, bgr)
    
detect_vehicles = veh.VehicleDetector(
    match_vehicles, 
    history_depth=detector_history_depth, 
    heat_thresh=detector_heat_thresh,
    vis_recog=vis_recog,
    vis_heat=vis_heat,
    vis_labels=vis_labels,
    vis_detect=vis_detect)
    
def process_image(raw_rgb):
    '''
    Vehicle detection pipeline. Input raw RGB images from camera.
    '''
    # undistort raw image from camera
    rgb = camera_cal.undistort_image(raw_rgb)
    
    # draw vehicle bounding boxes on the image
    for box in detect_vehicles(rgb):
        cv2.rectangle(rgb, box[0], box[1], (0,0,255), 5)
        
    return rgb
    
#in_file_path = os.path.join('./', in_file)
out_file_path = os.path.join('./output_videos/', out_file)

clip = VideoFileClip(in_file)
veh_clip = clip.fl_image(process_image)
veh_clip.write_videofile(out_file_path, audio=False)
