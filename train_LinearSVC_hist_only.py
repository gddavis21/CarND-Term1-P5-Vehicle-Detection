import numpy as np
import cv2
import sys
import glob
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import svm
import vehicles as veh

spatial_params = None
hist_params = None
hog_params = None

# # configure color spatial features
# spatial_params = veh.ColorSpatialParams(color_space="HSV", image_size=16)
    
# configure color (hue/saturation) histogram features
hist_params = veh.ColorHistParams(
    hue_bins=(0, 15, 45, 75, 105, 135, 165, 180),
    sat_bins=(0, 32, 64, 96, 128, 160, 192, 224, 256))

# # configure HOG descriptor features    
# hog_params = veh.HogFeatureParams(
    # color_space='HSV',
    # cell_size=8,
    # block_size=2,
    # num_orient=9)

# create feature extractor    
extractor = veh.VehicleFeatureExtractor(spatial_params, hist_params, hog_params)
    
# extract features & labels from training image sets
ftr_scaler, X_train, X_test, y_train, y_test = veh.load_clf_training_data(
    extractor,
    test_size=0.2)

print('training classifier...')

# train classifier
clf = svm.LinearSVC(C=0.001)
clf.fit(X_train, y_train)

# output metrics
y_pred = clf.predict(X_test)
print('Precision: %.4f' % precision_score(y_test, y_pred))
print('Recall: %.4f' % recall_score(y_test, y_pred))

clf_file_path = sys.argv[1]

print('saving classifier data to file...')
veh.save_trained_classifier(
    clf_file_path, 
    spatial_params, 
    hist_params, 
    hog_params, 
    ftr_scaler, 
    clf)
