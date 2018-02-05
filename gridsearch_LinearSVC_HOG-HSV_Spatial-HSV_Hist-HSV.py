import numpy as np
import cv2
import sys
import glob
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import svm
import vehicles as veh

spatial_params = None
hist_params = None
hog_params = None
    
# configure color spatial features
spatial_params = veh.ColorSpatialParams(color_space="HSV", image_size=16)
    
# configure color (hue/saturation) histogram features
hist_params = veh.ColorHistParams(
    color_space='HSV',
    bins_ch1=(0, 15, 45, 75, 105, 135, 165, 180),
    range_ch1=(0,179),
    bins_ch2=(0, 32, 64, 96, 128, 160, 192, 224, 256),
    range_ch2=(0,255),
    bins_ch3=(0, 32, 64, 96, 128, 160, 192, 224, 256),
    range_ch3=(0,255))
    
# configure HOG descriptor features    
hog_params = veh.HogFeatureParams(
    color_space='HSV',
    cell_size=8,
    block_size=2,
    num_orient=9)

# create feature extractor    
extractor = veh.VehicleFeatureExtractor(spatial_params, hist_params, hog_params)
    
# extract features & labels from training image sets
ftr_scaler, X_train, X_test, y_train, y_test = veh.load_clf_training_data(
    extractor, 
    test_size=0.5)

# Set the parameters by cross-validation
param_grid = {'C': np.logspace(-5, 3, 9)}

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        svm.LinearSVC(), 
        param_grid, 
        scoring='%s_macro' % score, 
        verbose=3)
        
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

