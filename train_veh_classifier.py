import numpy as np
import pickle
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import vehicles as veh

# configure color spatial features
spatial_params = veh.ColorSpatialParams(color_space="HSV", image_size=16)
    
# configure color (hue/saturation) histogram features
hist_params = veh.ColorHistParams(
    'hue_bins'=(0, 15, 45, 75, 105, 135, 165, 180),
    'sat_bins'=(0, 32, 64, 96, 128, 160, 192, 224, 256))

# configure HOG descriptor features    
hog_params = veh.HogFeatureParams(
    color_space='RGB',
    cell_size=8,
    block_size=2,
    num_orient=12)

# create feature extractor    
extractor = veh.VehicleFeatureExtractor(
    spatial_params=spatial_params,
    hist_params=hist_params,
    hog_params=hog_params)

# extract features from training images
car_features = []
notcar_features = []

car_paths = [
    '../veh-det-training-data/vehicles/KITTI_extracted/*.png',
    '../veh-det-training-data/vehicles/GTI_Far/*.png',
    '../veh-det-training-data/vehicles/GTI_Left/*.png',
    '../veh-det-training-data/vehicles/GTI_MiddleClose/*.png',
    '../veh-det-training-data/vehicles/GTI_Right/*.png'
]

notcar_paths = [
    '../veh-det-training-data/non-vehicles/Extras/*.png',
    '../veh-det-training-data/non-vehicles/GTI/*.png'
]

for path in car_paths:
    for img_file in glob.iglob(path):
        rgb = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
        car_features.append(extractor.extract_features(rgb))

for path in notcar_paths:
    for img_file in glob.iglob(path):
        rgb = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
        notcar_features.append(extractor.extract_features(rgb))

# build features matrix
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# build labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# fit per-column scaler
feature_scaler = StandardScaler().fit(X)

# scale features
scaled_X = feature_scaler.transform(X)

# split data into training & test sets
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, 
    test_size=0.2, 
    random_state=np.random.randint(0, 100))

# train classifier
clf = svm.LinearSVC()
clf.fit(X_train, y_train)

# measure accuracy on test set
print('Test Accuracy: ', round(clf.score(X_test, y_test), 4))

# TODO: pickle classifier, scaler & feature extractor for re-use
# TODO: extend this script to a grid search parameter optimization