import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from collections import namedtuple

class VehicleDetectionPipeline:
    '''
    '''
    def __init__(self, camera_cal, veh_clf):
        self._camera_cal = camera_cal
        self._veh_clf = veh_clf
        
    def __call__(self, raw_rgb):
        '''
        Vehicle detection pipeline. Input raw RGB image from camera.
        '''
        # undistort raw camera image
        rgb = self._camera_cal.undistort_image(raw_rgb)
        
        # pre-compute HOG descriptors over entire ROI
        
        # compute search windows
        
        # detect sliding window matches
        
        # build heat map from matches
        
        # threshold heat map
        
        # find detected vehicles & bounding boxes on heat map
        
        # draw vehicle boxes on image
        result = np.copy(rgb)
        self._draw_vehicle_boxes(result, vboxes)
        
        return result
        
    def _compute_search_windows(x1, y1, x2, y2, size, overlap):
        # Compute the number of pixels per step in x/y
        dx = int(size[0] * (1.0 - overlap[0]))
        dy = int(size[1] * (1.0 - overlap[1]))
        # Compute the number of windows in x/y
        nx = ((x2-x1) - size[0]) // dx + 1
        ny = ((y2-y1) - size[1]) // dy + 1
        # Compute & return list of search windows
        windows = []
        for row in range(ny):
            T = y1 + row*dy 
            B = T + size[1]
            for col in range(nx):
                # Calculate each window position
                L = x1 + col*dx
                R = L + size[0]
                # Append window position to list
                windows.append(((L,T), (R,B)))
        return windows
        
    def _draw_vehicle_boxes(self, img, bboxes):
        color = (0,0,255)
        thickness = 5
        for box in bboxes:
            cv2.rectangle(img, box[0], box[1], color, thickness)
            
            
def extract_image_features(src_rgb):
    features = []
    rgb = src_rgb
    if src_rgb.shape != (64,64):
        rgb = cv2.resize(src_rgb, (64,64))
    # color conversions
    # extract color spatial features
    # extract color histogram features
    # extract HOG descriptor features
    # return concatenation of features

def convert_image(img_rgb, color_space):
    color_space = color_space.upper()
    if color_space == 'RGB':
        return img_rgb
    elif color_space == 'HSV':
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    elif color_space == 'LUV':
        return cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    elif color_space == 'HLS':
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    elif color_space == 'YUV':
        return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    elif color_space == 'YCRCB':
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    elif color_space == 'GRAY':
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        return None
        
# configuration of color distribution features    
ColorFeatureParams = namedtuple('ColorFeatureParams', [
    'color_space',      # 'RGB', 'HSV', 'HLS', 'LUV', 'YUV', 'YCrCb'
    'enable_spatial',   # enable spatial features: True/False
    'spatial_size',     # spatial features size (8,8), (32,32), etc.
    'enable_hist',      # enable histogram features: True/False
    'hist_bins'         # histogram bins
])

# configuration of HOG descriptor features
HogFeatureParams = namedtuple('HogFeatureParams', [
    'enable_hog',
    'color_space',
    'cell_size',
    'block_size'
    'num_orient'
])

class VehicleFeatureExtractor:
    def __init__(self, color_params, hog_params):
        self._color_params = color_params
        self._hog_params = hog_params
        
    def extract_features(self, img_rgb):
        patch_rgb = img_rgb
        if img_rgb.shape != (64,64,3):
            patch_rgb = cv2.resize(img_rgb, (64,64,3))
        features = []
        clr_params = self._color_params
        clr_patch = None
        if clr_params.enable_spatial or clr_params.enable_hist:
            clr_patch = convert_image(patch_rgb, clr_params.color_space)
            if clr_params.enable_spatial:
                features.append(bin_spatial(clr_patch, size=clr_params.spatial_size)
            if clr_params.enable_hist:
                features.append(color_hist(clr_patch, nbins=clr_params.hist_bins)
        hog_params = self._hog_params
        if hog_params.enable_hog:
            # reuse converted patch if we can
            hog_patch = clr_patch
            if not clr_patch or clr_params.color_space != hog_params.color_space:
                hog_patch = convert_image(patch_rgb, hog_params.color_space)
            features.append(self._get_hog_features(hog_patch))
        return np.concatenate(features)
        
        
# compute binned color features
def bin_spatial(img, size=(32,32)):
    return cv2.resize(img, size).ravel()
    
# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    hist1 = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    hist2 = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    hist3 = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    return np.concatenate((hist1[0], hist2[0], hist3[0]))

    