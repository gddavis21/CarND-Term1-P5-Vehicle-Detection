import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from skimage.feature import hog
import sklearn.preprocessing
import sklearn.svm
from collections import namedtuple
from collections import deque

ROI_Scaler = namedtuple('ROI_Scaler', [
    'scale',    # integer 2-tuple, scale image by scale[0]/scale[1]
    'ROI',
    'offs',
    'steps'
])


class VehicleDetectionPipeline:
    '''
    '''
    def __init__(self, camera_cal, veh_detector):
        '''
        '''
        self._camera_cal = camera_cal
        self._veh_detector = veh_detector
        self._match_deque = deque(maxlen=4)
        
    def __call__(self, raw_rgb):
        '''
        Vehicle detection pipeline. Input raw RGB image from camera.
        '''
        # undistort raw camera image
        rgb = self._camera_cal.undistort_image(raw_rgb)

        # find vehicle match tiles
        veh_matches = self._veh_detector.match_vehicles(rgb)
        self._match_deque.append(veh_matches)
        
        if len(self._match_deque) < self._match_deque.maxlen:
            return rgb
            
        veh_boxes = self._locate_vehicles(self._match_deque, rgb.shape, threshold=3)
            
        # pre-compute HOG descriptors over entire ROI
        
        # compute search windows
        
        # detect sliding window matches
        
        # build heat map from matches
        
        # threshold heat map
        
        # find detected vehicles & bounding boxes on heat map
        
        # draw vehicle boxes on image
        result = np.copy(rgb)
        self._draw_vehicle_boxes(result, veh_boxes)
        return result
        
    def _locate_vehicles(self, matches_list, img_shape, threshold):
        '''
        '''
        # make heat-map
        heatmap = np.zeros(img_shape)
        for matches in matches_list:
            for match in matches:
                heatmap[match[0][1]:match[1][1], match[0][0]:match[1][0]] += 1
                
        # threshold heat-map
        heatmap[heatmap <= threshold] = 0
        
        # label objects in heat-map
        from scipy.ndimage.measurements import label
        labels, obj_count = label(heatmap)
        
        # compute object bounding boxes
        obj_boxes = []
        for obj_number in range(1, obj_count+1):
            # Find pixels with each car_number label value
            nonzero = (labels == obj_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            L,T = np.min(nonzerox), np.min(nonzeroy)
            R,B = np.max(nonzerox), np.max(nonzeroy)
            obj_boxes.append(((L,T), (R,B)))
            
        return obj_boxes
        
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
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LUV)
    elif color_space == 'HLS':
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
    elif color_space == 'YUV':
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
    elif color_space == 'YCRCB':
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    elif color_space == 'GRAY':
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        return None
        

class VehicleDetector:
    '''
    '''
    def __init__(self, feature_extractor, feature_scaler, veh_clf):
        '''
        '''
        self._feature_extractor = feature_extractor
        self._feature_scaler = feature_scaler
        self._veh_clf = veh_clf
        self._ROI_scalers = self._make_ROI_scalers()
        
    def match_vehicles(self, rgb):
        '''
        '''
        matches = []
        for ROI_scaler in self._ROI_scalers:
            ROI = ROI_scaler.ROI
            L,T = ROI_scaler.ROI[0][0], ROI_scaler.ROI[0][1]
            R,B = ROI_scaler.ROI[1][0], ROI_scaler.ROI[1][1]
            a,b = ROI_scaler.scale[0], ROI_scaler.scale[1]
            sL,sT = L*a//b, T*a//b
            sR,sB = R*a//b, B*a//b
            scaled_image = cv2.resize(rgb[T:B,L:R], (sR-sL, sB-sT))
            self._feature_extractor.set_source_image(scaled_image)
            for i in range(ROI_scaler.steps[0]):
                for j in range(ROI_scaler.steps[1]):
                    x = ROI_scaler.offs[0] + 16*i
                    y = ROI_scaler.offs[1] + 16*j
                    ftrs = self._feature_extractor.extract_patch_features((x,y))
                    ftrs = self._feature_scaler.transform(np.array(ftrs).reshape(1,-1))
                    if self._veh_clf.predict(ftrs) == 1:
                        mL, mR = (x*b//a)+L, ((x+64)*b//a)+L
                        mT, mB = (y*b//a)+T, ((y+64)*b//a)+T
                        matches.append(((mL, mT), (mR, mB)))
        return matches
    
    def _make_ROI_scalers(self):
        '''
        '''
        return [
            # self._make_ROI_scaler((4,1),  ((0,400), (1280,420))),
            self._make_ROI_scaler((8,3), ((340,416), (940,440))), # tile = 24
            self._make_ROI_scaler((2,1), ((300,412), (980,444))),  # tile =  32, step =  4
            self._make_ROI_scaler((4,3), ((260,408), (1020,456))),  # tile =  48, step =  6
            self._make_ROI_scaler((1,1), ((220,404), (1060,468))),  # tile =  64, step =  8
            self._make_ROI_scaler((4,5), ((0,400), (1280,480))),  # tile =  80, step = 10
            self._make_ROI_scaler((2,3), ((0,396), (1280,492))),  # tile =  96, step = 12
            # self._make_ROI_scaler((4,7),  ((0,400), (1280,540))),  # tile = 112, step = 14
            self._make_ROI_scaler((1,2), ((0,388), (1280,516))),  # tile = 128, step = 16
            self._make_ROI_scaler((2,5), ((0,380), (1280,540))),   # tile = 160
            self._make_ROI_scaler((1,3), ((0,372), (1280,564))),   # tile = 192
            self._make_ROI_scaler((2,7), ((0,364), (1280,588))),   # tile = 224
            self._make_ROI_scaler((1,4), ((0,356), (1280,612))),   # tile = 256
            self._make_ROI_scaler((2,9), ((0,348), (1280,636))),    # tile = 288
            self._make_ROI_scaler((1,5), ((0,340), (1280,660)))    # tile = 320
        ]
    
    def _make_ROI_scaler(self, scale, ROI):
        '''
        '''
        block = 64
        delta = 16
        a,b = scale[0], scale[1]
        sL,sT = ROI[0][0]*a//b, ROI[0][1]*a//b
        sR,sB = ROI[1][0]*a//b, ROI[1][1]*a//b
        sW = sR - sL
        sH = sB - sT
        offs = ((sW-block) % delta // 2, (sH-block) % delta // 2)
        steps = ((sW-block) // delta + 1, (sH-block) // delta + 1)
        return ROI_Scaler(scale=scale, ROI=ROI, offs=offs, steps=steps)
        
        
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
    'block_size',
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
        L,T = patch_corner[0], patch_corner[1]
        R,B = L+64, T+64
        patch = self._src_spat[T:B,L:R]
        size = self._spatial_params.image_size
        return cv2.resize(patch, (size, size)).ravel()
        
    def _hsv_histogram(self, patch_corner):
        # compute separate hue & saturation histograms
        L,T = patch_corner[0], patch_corner[1]
        R,B = L+64, T+64
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
        L,T = (patch_corner[0] // 8), (patch_corner[1] // 8)
        R,B = L+blocks_per_patch, T+blocks_per_patch
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
        
    def _compute_hog_channel(self, img, num_orient, cell_size, block_size):
        return hog(
            img,
            orientations=num_orient,
            pixels_per_cell=(cell_size, cell_size),
            cells_per_block=(block_size, block_size),
            block_norm='L2',
            visualise=False,
            transform_sqrt=True,
            feature_vector=False)
    
    def _compute_hog_array(self, rgb):
        hog_img = convert_image(rgb, self._hog_params.color_space)
        num_orient = self._hog_params.num_orient
        cell_size = self._hog_params.cell_size
        block_size = self._hog_params.block_size
        hog1 = self._compute_hog_channel(hog_img[:,:,0], num_orient, cell_size, block_size)
        hog2 = self._compute_hog_channel(hog_img[:,:,1], num_orient, cell_size, block_size)
        hog3 = self._compute_hog_channel(hog_img[:,:,2], num_orient, cell_size, block_size)
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
                
