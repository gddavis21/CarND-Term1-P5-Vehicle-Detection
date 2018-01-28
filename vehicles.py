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
    def __init__(self, camera_cal, extract_veh_ftrs, veh_ftr_scaler, veh_clf):
        self._camera_cal = camera_cal
        self._extract_veh_ftrs = extract_veh_ftrs
        self._veh_ftr_scaler = veh_ftr_scaler
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
        
    def _match_vehicles(self, rgb_img, windows):
        matches = []
        for window in windows:
            L,T = window[0][0], window[0][1]
            R,B = window[1][0], window[1][1]
            features = self._extract_veh_ftrs(rgb_img[T:B, L:R])
            features = self._veh_ftr_scaler.transform(np.array(features).reshape(1,-1))
            if self._veh_clf.predict(features) == 1:
                matches.append(window)
        return matches
        
    def _draw_vehicle_boxes(self, img, bboxes):
        color = (0,0,255)
        thickness = 5
        for box in bboxes:
            cv2.rectangle(img, box[0], box[1], color, thickness)
            
            
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
        
# # configuration of color distribution features    
# ColorFeatureParams = namedtuple('ColorFeatureParams', [
    # 'color_space',      # 'RGB', 'HSV', 'HLS', 'LUV', 'YUV', 'YCrCb'
    # 'enable_spatial',   # enable spatial features: True/False
    # 'spatial_size',     # spatial features size (8,8), (32,32), etc.
    # 'enable_hist',      # enable histogram features: True/False
    # 'hist_bins',        # histogram bins
    # 'hue_bins',
    # 'sat_bins',
    # 'hist_range'        # histogram range
# ])

ColorSpatialParams = namedtuple('ColorSpatialParams', [
    'color_space',
    'image_size'
])

# hue/saturation histograms
ColorHistParams = namedtuple('ColorHistParams', [
    'hue_bins',
    'sat_bins'
])

# configuration of HOG descriptor features
HogFeatureParams = namedtuple('HogFeatureParams', [
    'color_space',
    'cell_size',
    'block_size'
    'num_orient'
])

class VehicleFeatureExtractor:
    def __init__(self, spatial_params=None, hist_params=None, hog_params=None):
        self._spatial_params = spatial_params
        self._hist_params = hist_params
        self._hog_params = hog_params
        self._patch_cache = {}
        
    def __call__(self, rgb):
        self._reset_patches(rgb)
        features = []

        if self._spatial_params:
            image_size = self._spatial_params.image_size
            features.append(cv2.resize(
                self._get_patch(self._spatial_params.color_space), 
                (image_size, image_size)).ravel())

        if self._hist_params:
            # compute separate hue & saturation histograms
            hsv = self._get_patch('HSV')
            hue_hist = np.histogram(
                hsv[:,:,0], 
                bins=self._hist_params.hue_bins,
                range=(0,179))
            sat_hist = np.histogram(
                hsv[:,:,1],
                bins=self._hist_params.sat_bins,
                range=(0,255))
            # concatenate histograms into a single feature vector
            features.append(np.concatenate((hue_hist[0], sat_hist[0]))
        
        if self._hog_params:
            cell_size = self._hog_params.cell_size
            block_size = self._hog_params.block_size
            features.append(hog(
                self._get_patch(self._hog_params.color_space),
                orientations=self._hog_params.num_orient,
                pixels_per_cell=(cell_size, cell_size),
                cells_per_block=(block_size, block_size),
                block_norm='L2',
                visualize=False,
                transform_sqrt=True,
                feature_vector=True))
        
        return np.concatenate(features)
        
    def _reset_patches(self, img_rgb):
        self._patch_cache.clear()
        if img_rgb.shape == (64,64,3):
            self._patch_cache['RGB'] = img_rgb
        else:
            self._patch_cache['RGB'] = cv2.resize(img_rgb, (64,64,3))
            
    def _get_patch(self, color_space):
        if color_space not in self._patch_cache:
            self._patch_cache[color_space] = convert_image(
                self._patch_cache['RGB'], 
                color_space)
        return self._patch_cache[color_space]
                
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


