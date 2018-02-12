import cv2
import numpy as np
import os
import sys
import time
import glob
import pickle
import vision as vis
import vehicles as veh

images_path = sys.argv[1]
dst_dir = sys.argv[2]
clf_path = sys.argv[3]

# calibrate camera from checkerboard images
print('calibrating camera')
camera_cal = vis.CameraCal()
camera_cal.calibrate_from_chessboard(
    chessboard_files=glob.glob('./camera_cal/calibration*.jpg'), 
    chessboard_size=(9,6), 
    save_diags=False,
    progress=None)

# load trained classifier, create vehicle recognizer
print('loading trained classifier')
spatial_params, hist_params, hog_params, ftr_scaler, veh_clf = veh.load_classifier(clf_path)
ftr_extractor = veh.VehicleFeatureExtractor(spatial_params, hist_params, hog_params)
match_vehicles = veh.VehicleRecognizer(ftr_extractor, ftr_scaler, veh_clf, tile_sizes=[64,80,96])

def process_image(src_path, dst_dir):
    print('processing %s' % src_path)
    rgb = cv2.cvtColor(cv2.imread(src_path), cv2.COLOR_BGR2RGB)
    rgb = camera_cal.undistort_image(rgb)

    start_time = time.time()
    for match in match_vehicles(rgb):
        veh.draw_vehicle_match(rgb, match)
    elapsed_time = time.time() - start_time
    print('processing time = %.3f' % elapsed_time)

    (src_dir,fname) = os.path.split(src_path)
    dst_path = os.path.join(dst_dir, fname)
    print('saving file %s' % dst_path)
    cv2.imwrite(dst_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

for src_file in glob.iglob(images_path):
    process_image(src_file, dst_dir)
