import cv2
import numpy as np
import os
import sys
import time
import glob
import pickle
import vision as vis
import vehicles as veh

def calibration_progress(msg):
    print(msg)

camera_cal = vis.CameraCal()
camera_cal.calibrate_from_chessboard(
    chessboard_files=glob.glob('./camera_cal/calibration*.jpg'), 
    chessboard_size=(9,6), 
    save_diags=False,
    progress=calibration_progress)

print('loading saved classifier...')
clf_data_path = sys.argv[1]

spatial_params, hist_params, hog_params, ftr_scaler, veh_clf = veh.load_trained_classifier(clf_data_path)

ftr_extractor = veh.VehicleFeatureExtractor(spatial_params, hist_params, hog_params)
veh_detector = veh.VehicleDetector(ftr_extractor, ftr_scaler, veh_clf)

def process_test_image(src_path, dst_path):
    print('processing %s' % src_path)
    bgr_in = cv2.imread(src_path)
    rgb_in = cv2.cvtColor(bgr_in, cv2.COLOR_BGR2RGB)
    rgb_in = camera_cal.undistort_image(rgb_in)

    start_time = time.time()
    veh_boxes = veh_detector.match_vehicles(rgb_in)
    elapsed_time = time.time() - start_time
    print('processing time = %.3f' % elapsed_time)

    rgb_out = np.copy(rgb_in)
    box_color = (0,0,255)
    box_thick = 1
    for box in veh_boxes:
        cv2.rectangle(rgb_out, box[0], box[1], box_color, box_thick)

    bgr_out = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(dst_path, bgr_out)

process_test_image('./test_images/test1.jpg', './output_images/test1.jpg')
process_test_image('./test_images/test2.jpg', './output_images/test2.jpg')
process_test_image('./test_images/test3.jpg', './output_images/test3.jpg')
process_test_image('./test_images/test4.jpg', './output_images/test4.jpg')
process_test_image('./test_images/test5.jpg', './output_images/test5.jpg')
process_test_image('./test_images/test6.jpg', './output_images/test6.jpg')
