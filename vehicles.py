import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import skimage
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from collections import namedtuple

ScaledROI = namedtuple('ScaledROI', [
    'scale',    # integer 2-tuple, scale image by scale[0]/scale[1]
    'offs',
    'steps'
])


class VehicleDetectionPipeline:
    '''
    '''
    def __init__(self, ROI_rect, camera_cal, feature_extractor, feature_scaler, veh_clf):
        self._ROI_rect = ROI_rect
        self._camera_cal = camera_cal
        self._feature_extractor = feature_extractor
        self._feature_scaler = feature_scaler
        self._veh_clf = veh_clf
        
        # self._search_windows = self._compute_search_windows()
        self._scales = _compute_scales(ROI_rect)
        
    def __call__(self, raw_rgb):
        '''
        Vehicle detection pipeline. Input raw RGB image from camera.
        '''
        # undistort raw camera image
        rgb = self._camera_cal.undistort_image(raw_rgb)
        T,L = self._ROI_rect[0][1], self._ROI_rect[0][0]
        B,R = self._ROI_rect[1][1], self._ROI_rect[1][0]
        image = rgb[T:B,L:R]
        W = R-L
        H = B-T
        
        matches = []
        
        for scale in self._scales:
            a = scale.scale[0]
            b = scale.scale[1]
            scaled_W = (R-L) * a // b
            scaled_H = (B-T) * a // b
            scaled_image = cv2.resize(image, (scaled_W, scaled_H))
            self._feature_extractor.set_source_image(scaled_image)
            for i in range(scale.steps[0]):
                for j in range(scale.steps[1]):
                    x = scale.offs[0] + 8*i
                    y = scale.offs[1] + 8*j
                    ftrs = self._feature_extractor.extract_patch_features((x,y))
                    ftrs = self._feature_scaler.transform(np.array(ftrs).reshape(1,-1))
                    if self._veh_clf.predict(ftrs) == 1:
                        match_L, match_R = x * b // a, (x+64) * b // a
                        match_T, match_B = y * b // a, (y+64) * b // a
                        matches.append(((match_L, match_T), (match_R, match_B)))
        
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
        
    # def _compute_search_windows(self):
        # return [
            # _compute_search_windows(self._ROI_rect, 192, 24),
            # _compute_search_windows(self._ROI_rect, 176, 22),
            # _compute_search_windows(self._ROI_rect, 160, 20),
            # _compute_search_windows(self._ROI_rect, 144, 18),
            # _compute_search_windows(self._ROI_rect, 128, 16),
            # _compute_search_windows(self._ROI_rect, 112, 14),
            # _compute_search_windows(self._ROI_rect, 96, 12),
            # _compute_search_windows(self._ROI_rect, 80, 10),
            # _compute_search_windows(self._ROI_rect, 64, 8),
            # _compute_search_windows(self._ROI_rect, 48, 6),
            # _compute_search_windows(self._ROI_rect, 32, 4)
        # ]
        
    @staticmethod
    def _compute_scales(orig_ROI):
        '''
        '''
        return [
            _compute_scaled_ROI(orig_ROI, (1,3)),   # tile = 192, delta = 24
            _compute_scaled_ROI(orig_ROI, (4,11)),  # tile = 176, delta = 22
            _compute_scaled_ROI(orig_ROI, (2,5)),   # tile = 160, delta = 20
            _compute_scaled_ROI(orig_ROI, (4,9)),   # tile = 144, delta = 18
            _compute_scaled_ROI(orig_ROI, (1,2)),   # tile = 128, delta = 16
            _compute_scaled_ROI(orig_ROI, (4,7)),   # tile = 112, delta = 14
            _compute_scaled_ROI(orig_ROI, (2,3)),   # tile =  96, delta = 12
            _compute_scaled_ROI(orig_ROI, (4,5)),   # tile =  80, delta = 10
            _compute_scaled_ROI(orig_ROI, (1,1)),   # tile =  64, delta =  8
            _compute_scaled_ROI(orig_ROI, (4,3)),   # tile =  48, delta =  6
            _compute_scaled_ROI(orig_ROI, (2,1))    # tile =  32, delta =  4
        ]
    
    @staticmethod
    def _compute_scaled_ROI(orig_ROI, scale):
        block = 64
        delta = 8
        L = orig_ROI[0][0] * scale[0] // scale[1]
        T = orig_ROI[0][1] * scale[0] // scale[1]
        R = orig_ROI[1][0] * scale[0] // scale[1] 
        B = orig_ROI[1][1] * scale[0] // scale[1] 
        w = R - L
        h = B - T
        offs = ((w-block) % delta // 2, (h-block) % delta // 2)
        steps = ((w-block) // delta + 1, (h-block) // delta + 1)
        return ScaledROI(scale=scale, offs=offs, steps=steps)
    
    # @staticmethod
    # def _compute_search_windows(ROI_rect, size, overlap):
        # x1 = ROI_rect[0][0]
        # y1 = ROI_rect[0][1]
        # x2 = ROI_rect[1][0]
        # y2 = ROI_rect[1][1]
        # # Compute the number of pixels per step in x/y
        # dx = int(size[0] * (1.0 - overlap[0]))
        # dy = int(size[1] * (1.0 - overlap[1]))
        # # Compute the number of windows in x/y
        # nx = ((x2-x1) - size[0]) // dx + 1
        # ny = ((y2-y1) - size[1]) // dy + 1
        # # Compute & return list of search windows
        # windows = []
        # for row in range(ny):
            # T = y1 + row*dy 
            # B = T + size[1]
            # for col in range(nx):
                # # Calculate each window position
                # L = x1 + col*dx
                # R = L + size[0]
                # # Append window position to list
                # windows.append(((L,T), (R,B)))
        # return windows
        
    def _match_vehicles(self, rgb_img, windows):
        matches = []
        for window in windows:
            L,T = window[0][0], window[0][1]
            R,B = window[1][0], window[1][1]
            features = self._feature_extractor.extract_features(rgb_img[T:B, L:R])
            features = self._feature_scaler.transform(np.array(features).reshape(1,-1))
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
        # self._patch_cache = {}
        # self._src_rgb = None
        self._src_spat = None
        self._src_hist = None
        self._src_hog = None
        
    def set_source_image(self, rgb):
        '''
        '''
        self._src_spat = None
        self._src_hist = None
        self._src_hog = None
        
        if self._spatial_params:
            self._src_spat = convert_image(rgb, self._spatial_params.color_space)
            
        if self._hist_params:
            self._src_hist = convert_image(rgb, 'HSV')
            
        if self._hog_params:
            self._src_hog = self._compute_hog_array(rgb)
    
    def extract_patch_features(self, patch_corner):
        '''
        '''
        features = []

        if self._spatial_params:
            features.append(self._bin_spatial(patch_corner))

        if self._hist_params:
            features.append(self._hsv_histogram(patch_corner))
        
        if self._hog_params:
            features.append(self._hog_features(patch_corner))
        
        return np.concatenate(features)
        
    def extract_image_features(self, rgb):
        '''
        '''
        if rgb.shape == (64,64,3):
            self.set_source_image(rgb)
        else:
            self.set_source_image(cv2.resize(rgb, (64,64,3)))
        
        return self.extract_patch_features((0,0))

    def _bin_spatial(self, patch_corner):
        T,L = patch_corner[1], patch_corner[0]
        B,R = T+64, L+64
        patch = self._src_spat[T:B,L:R]
        size = self._spatial_params.image_size
        return cv2.resize(patch, (size, size)).ravel()
        
    def _hsv_histogram(self, patch_corner):
        # compute separate hue & saturation histograms
        T,L = patch_corner[1], patch_corner[0]
        B,R = T+64, L+64
        hsv_patch = self._src_hist[T:B,L:R]
        hue_hist = np.histogram(
            hsv_patch[:,:,0], 
            bins=self._hist_params.hue_bins,
            range=(0,179))
        sat_hist = np.histogram(
            hsv_patch[:,:,1],
            bins=self._hist_params.sat_bins,
            range=(0,255))
        # concatenate histograms into a single feature vector
        return np.concatenate((hue_hist[0], sat_hist[0]))
        
    def _hog_features(self, patch_corner):
        # HOG descriptor features
        cell_size = self._hog_params.cell_size
        block_size = self._hog_params.block_size
        blocks_per_patch = (64 // cell_size) - block_size + 1
        T,L = (patch_corner[1] // 8), (patch_corner[0] // 8)
        B,R = T+blocks_per_patch, L+blocks_per_patch
        hog_feat1 = self._src_hog[0][T:B,L:R].ravel()
        hog_feat2 = self._src_hog[1][T:B,L:R].ravel()
        hog_feat3 = self._src_hog[2][T:B,L:R].ravel()
        return np.hstack((hog_feat1, hog_feat2, hog_feat3))
            
    # def _make_patch_cache(self, patch_corner):
        # T,L = patch_corner[1], patch_corner[0]
        # B,R = T+64, L+64
        # cache = {}
        # cache['RGB'] = self._src_rgb[T:B,L:R]
        # return cache
        
    # @staticmethod
    # def _get_patch(color_space, patch_cache):
        # if color_space not in patch_cache:
            # rgb_patch = patch_cache['RGB']
            # patch_cache[color_space] = convert_image(rgb_patch, color_space)
        # return patch_cache[color_space]
        
    @staticmethod
    def _compute_hog_channel(img, num_orient, cell_size, block_size):
        return skimage.feature.hog(
            img,
            orientations=num_orient,
            pixels_per_cell=(cell_size, cell_size),
            cells_per_block=(block_size, block_size),
            block_norm='L2',
            visualize=False,
            transform_sqrt=True,
            feature_vector=False)
    
    def _compute_hog_array(self, rgb):
        hog_img = convert_image(rgb, self._hog_params.color_space)
        num_orient = self._hog_params.num_orient
        cell_size = self._hog_params.cell_size
        block_size = self._hog_params.block_size
        hog1 = _compute_hog_channel(hog_img[:,:,0], num_orient, cell_size, block_size)
        hog2 = _compute_hog_channel(hog_img[:,:,1], num_orient, cell_size, block_size)
        hog3 = _compute_hog_channel(hog_img[:,:,2], num_orient, cell_size, block_size)
        return (hog1, hog2, hog3)
                
    # def _reset_patches(self, rgb):
        # self._patch_cache.clear()
        # if rgb.shape == (64,64,3):
            # self._patch_cache['RGB'] = rgb
        # else:
            # self._patch_cache['RGB'] = cv2.resize(rgb, (64,64,3))
            
    # def _get_patch(self, color_space):
        # if color_space not in self._patch_cache:
            # self._patch_cache[color_space] = convert_image(
                # self._patch_cache['RGB'], 
                # color_space)
        # return self._patch_cache[color_space]
                
