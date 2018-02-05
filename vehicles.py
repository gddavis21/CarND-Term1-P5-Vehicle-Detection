import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import glob
import pickle
from skimage.feature import hog
import sklearn.preprocessing
import sklearn.svm
from collections import namedtuple
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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
        
        # add image results (matching tiles) to history
        self._match_deque.append(veh_matches)
        
        # make copy of image for drawing overlay graphics & annotations
        result = np.copy(rgb)
        
        # if we have enough history to start tracking...
        if len(self._match_deque) == self._match_deque.maxlen:
            # locate vehicle bounding boxes from past few images
            veh_boxes = self._locate_vehicles(self._match_deque, rgb.shape, threshold=3)
            # draw bounding boxes on the image
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
    elif color_space == 'GRAY':
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
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
            # unpack ROI scaler into ROI corners and scale factors
            L,T = ROI_scaler.ROI[0][0], ROI_scaler.ROI[0][1]
            R,B = ROI_scaler.ROI[1][0], ROI_scaler.ROI[1][1]
            a,b = ROI_scaler.scale[0], ROI_scaler.scale[1]
            # start with unscaled ROI...
            scaled_ROI = rgb[T:B,L:R]
            # ...and resize only if we have to (unequal scale factors)
            if a != b:
                # scale ROI corners
                sL,sT = L*a//b, T*a//b
                sR,sB = R*a//b, B*a//b
                # resize image
                interp = cv2.INTER_AREA if a < b else cv2.INTER_LINEAR
                scaled_ROI = cv2.resize(scaled_ROI, (sR-sL, sB-sT), interpolation=interp)
            # initialize extractor with full scaled ROI...
            self._feature_extractor.set_full_image(scaled_ROI)
            for i in range(ROI_scaler.steps[0]):
                for j in range(ROI_scaler.steps[1]):
                    # ...extract features for each 64x64 sliding tile...
                    x = ROI_scaler.offs[0] + 16*i
                    y = ROI_scaler.offs[1] + 16*j
                    ftrs = self._feature_extractor.extract_tile_features((x,y))
                    # normalize features
                    ftrs = self._feature_scaler.transform(np.array(ftrs).reshape(1,-1))
                    # predict whether or not the tile contains a vehicle
                    if self._veh_clf.predict(ftrs) == 1:
                        # vehicle --> record unscaled tile rectangle
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
    'color_space',
    'bins_ch1',
    'range_ch1',
    'bins_ch2',
    'range_ch2',
    'bins_ch3',
    'range_ch3'
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
        self._src_spat = None
        self._src_hist = None
        self._src_hog = None
        
    def set_full_image(self, rgb):
        '''
        Prepare extractor for multiple tile feature extractions over an image.
        
          extractor.set_full_image(image)
          for tile in image:
            features = extractor.extract_tile_features(tile)
            
        '''
        self._src_spat = None
        self._src_hist = None
        self._src_hog = None
        
        if self._spatial_params:
            self._src_spat = convert_image(rgb, self._spatial_params.color_space)
            
        if self._hist_params:
            self._src_hist = convert_image(rgb, self._hist_params.color_space)
            
        if self._hog_params:
            self._src_hog = self._compute_hog_array(rgb)
    
    def extract_tile_features(self, tile_corner):
        '''
        Extract features for a single image tile. 

          extractor.set_full_image(image)
          for tile in image:
            features = extractor.extract_tile_features(tile)
            
        '''
        features = []

        if self._spatial_params:
            features.append(self._bin_spatial(tile_corner))

        if self._hist_params:
            features.append(self._color_histogram(tile_corner))
        
        if self._hog_params:
            features.append(self._hog_features(tile_corner))
        
        return np.concatenate(features)
        
    def extract_image_features(self, rgb):
        '''
        Extract features for an entire image. 
        
        If you need to extract features for multiple tiles within an image, 
        call set_full_image() once (with the full image) and then call
        extract_tile_features() once for each tile (much better performance).
        '''
        src_img = rgb

        # resize to 64x64 if necessary
        if rgb.shape != (64,64,3):
            interp = cv2.INTER_AREA if rgb.shape[0] > 64 else cv2.INTER_LINEAR
            src_img = cv2.resize(rgb, (64,64), interpolation=interp)
        
        self.set_full_image(src_img)
        return self.extract_tile_features((0,0))

    def _bin_spatial(self, tile_corner):
        L,T = tile_corner[0], tile_corner[1]
        R,B = L+64, T+64
        tile = self._src_spat[T:B,L:R]
        size = self._spatial_params.image_size
        if size == 64:
            return tile
        interp = cv2.INTER_AREA if size < 64 else cv2.INTER_LINEAR
        return cv2.resize(tile, (size, size), interpolation=interp).ravel()
        
    def _color_histogram(self, tile_corner):
        # compute separate histograms for each channel
        L,T = tile_corner[0], tile_corner[1]
        R,B = L+64, T+64
        tile = self._src_hist[T:B,L:R]
        bins1, range1 = self._hist_params.bins_ch1, self._hist_params.range_ch1
        bins2, range2 = self._hist_params.bins_ch2, self._hist_params.range_ch2
        bins3, range3 = self._hist_params.bins_ch3, self._hist_params.range_ch3
        hist_ch1 = np.histogram(tile[:,:,0], bins=bins1, range=range1)
        hist_ch2 = np.histogram(tile[:,:,1], bins=bins2, range=range2)
        hist_ch3 = np.histogram(tile[:,:,2], bins=bins3, range=range3)
        # concatenate histograms into a single feature vector
        return np.concatenate((hist_ch1[0], hist_ch2[0], hist_ch3[0]))
        
    def _hog_features(self, tile_corner):
        # HOG descriptor features
        cell_size = self._hog_params.cell_size
        block_size = self._hog_params.block_size
        blocks_per_tile = (64 // cell_size) - block_size + 1
        L,T = (tile_corner[0] // 8), (tile_corner[1] // 8)
        R,B = L+blocks_per_tile, T+blocks_per_tile
        if len(self._src_hog) == 1:
            # HOG features from grayscale image
            return self._src_hog[0][T:B,L:R].ravel()
        else:
            # HOG features from color image
            hog_feat1 = self._src_hog[0][T:B,L:R].ravel()
            hog_feat2 = self._src_hog[1][T:B,L:R].ravel()
            hog_feat3 = self._src_hog[2][T:B,L:R].ravel()
            return np.hstack((hog_feat1, hog_feat2, hog_feat3))
            
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
        if hog_img.ndim == 2:
            # grayscale image (just 1 channel)
            hog = self._compute_hog_channel(hog_img, num_orient, cell_size, block_size)
            return (hog,)
        else:
            # color image --> compute HOG features separately for each channel
            hog1 = self._compute_hog_channel(hog_img[:,:,0], num_orient, cell_size, block_size)
            hog2 = self._compute_hog_channel(hog_img[:,:,1], num_orient, cell_size, block_size)
            hog3 = self._compute_hog_channel(hog_img[:,:,2], num_orient, cell_size, block_size)
            return (hog1, hog2, hog3)
                

def load_clf_training_data(ftr_extractor, test_size=0.3):
    '''
    '''
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

    # extract features from training images
    car_features = []
    notcar_features = []

    print('extracting vehicle image features...')
    for path in car_paths:
        for img_file in glob.iglob(path):
            rgb = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
            car_features.append(ftr_extractor.extract_image_features(rgb))

    print('extracting non-vehicle image features...')
    for path in notcar_paths:
        for img_file in glob.iglob(path):
            rgb = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
            notcar_features.append(ftr_extractor.extract_image_features(rgb))

    # build features matrix
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        

    # build labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # fit per-column scaler
    ftr_scaler = StandardScaler().fit(X)

    # scale features
    scaled_X = ftr_scaler.transform(X)

    # split data into training & test sets
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, 
        test_size=test_size, 
        random_state=np.random.randint(0, 100))
        
    return ftr_scaler, X_train, X_test, y_train, y_test
    
def save_trained_classifier(
    file_path, 
    spatial_params, hist_params, hog_params, 
    ftr_scaler, veh_clf):
    '''
    '''
    clf_data = {}
    clf_data['spatial_params'] = pickle.dumps(spatial_params)
    clf_data['hist_params'] = pickle.dumps(hist_params)
    clf_data['hog_params'] = pickle.dumps(hog_params)
    clf_data['feature_scaler'] = pickle.dumps(ftr_scaler)
    clf_data['classifier'] = pickle.dumps(veh_clf)

    with open(file_path, 'wb') as f:
        pickle.dump(clf_data, f)
    
    
def load_trained_classifier(file_path):
    '''
    '''
    clf_data = {}
    with open(file_path, 'rb') as f:
        clf_data = pickle.load(f)
        
    spatial_params = pickle.loads(clf_data['spatial_params'])
    hist_params = pickle.loads(clf_data['hist_params'])
    hog_params = pickle.loads(clf_data['hog_params'])
    ftr_scaler = pickle.loads(clf_data['feature_scaler'])
    veh_clf = pickle.loads(clf_data['classifier'])

    return spatial_params, hist_params, hog_params, ftr_scaler, veh_clf
