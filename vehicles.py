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



# class VehicleDetectionPipeline:
    # '''
    # '''
    # def __init__(self, camera_cal, detect_vehicles):
        # '''
        # camera_cal: instance of vision.CameraCal
        # detect_vehicles: function(image) --> vehicle bounding boxes
        # '''
        # self._camera_cal = camera_cal
        # self._detect_vehicles = detect_vehicles
        
    # def __call__(self, raw_rgb):
        # '''
        # Vehicle detection pipeline. Input raw RGB images from camera.
        # '''
        # # undistort raw image from camera
        # rgb = self._camera_cal.undistort_image(raw_rgb)
        
        # # locate vehicles (bounding boxes) on image
        # veh_boxes = detect_vehicles(rgb)
        
        # if veh_boxes:
            # # draw vehicle bounding boxes on the image
            # self._draw_vehicle_boxes(rgb, veh_boxes)
            
        # return rgb
        
    # def _draw_vehicle_boxes(self, img, bboxes):
        # color = (0,0,255)
        # thickness = 5
        # for box in bboxes:
            # cv2.rectangle(img, box[0], box[1], color, thickness)
            
            
def draw_vehicle_match(rgb, match):

    # unpack match data
    box, score = match[0], match[1]

    # draw color-coded match box
    if score < 0.2:
        box_color = (255,0,0)   # red
    elif score < 0.4:
        box_color = (255,255,0) # yellow
    elif score < 0.6:
        box_color = (0,255,0)   # green
    else:
        box_color = (0,0,255)   # blue

    cv2.rectangle(rgb, box[0], box[1], box_color, thickness=2)

    # # draw match score centered in box
    # text = '%.2f' % score
    # font_face = cv2.FONT_HERSHEY_SIMPLEX
    # font_scale = 0.5
    # text_color = (255,0,255)
    # text_thickness = 1
    # text_size, baseline = cv2.getTextSize(
        # text, 
        # font_face, 
        # font_scale, 
        # text_thickness)
        
    # text_L = ((box[0][0] + box[1][0]) // 2) - (text_size[0] // 2)
    # text_T = ((box[0][1] + box[1][1]) // 2) - (text_size[1] // 2)
        
    # cv2.putText(
        # rgb,
        # text,
        # (text_L, text_T),
        # font_face,
        # font_scale,
        # text_color,
        # text_thickness,
        # lineType=cv2.LINE_AA)

            

class VehicleDetector:
    '''
    '''
    def __init__(self, match_vehicles, history_depth, heat_thresh, diagnostics=False):
        '''
        match_vehicles: function(image) --> vehicle candidate boxes
        history_depth: number of consecutive frames to accumulate
        heat_thresh: heatmap threshold for object detection
        '''
        self._match_vehicles = match_vehicles
        self._heat_history = deque(maxlen=history_depth)
        self._heat_thresh = heat_thresh
        self._diagnostics = diagnostics
        self._img_count = 0
        
    def __call__(self, rgb):
        '''
        function(image) --> bounding boxes of detected vehicles
        '''
        self._img_count += 1

        # find vehicle candidates and add to history
        self._recognize_vehicles_in_frame(rgb)
        
        # combine info from past few frames (history) to detect vehicle objects
        return self._locate_vehicles(rgb)
        
    def _recognize_vehicles_in_frame(self, rgb):
        '''
        '''
        # find vehicle candidates and add to history
        veh_matches = self._match_vehicles(rgb)
        heatmap = np.zeros_like(rgb[:,:,0]).astype(np.float)
        for box,score in veh_matches:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += score
        self._heat_history.append(heatmap)
        
        if self._diagnostics == True:
            # save recognition image
            recog_img = np.copy(rgb)
            for match in veh_matches:
                draw_vehicle_match(recog_img, match)
            recog_path = './diagnostics/recog%d.png' % self._img_count
            cv2.imwrite(recog_path, cv2.cvtColor(recog_img, cv2.COLOR_RGB2BGR))
            # save heatmap image
            heat_img = np.clip(heatmap, 0, 255).astype(np.uint8)
            heat_img = cv2.merge((heat_img, heat_img, heat_img))
            heat_path = './diagnostics/heat%d.png' % self._img_count
            cv2.imwrite(heat_path, heat_img)
            
    def _locate_vehicles(self, rgb):
        '''
        '''
        veh_boxes = []
        
        # detect nothing until we have enough history
        if len(self._heat_history) == self._heat_history.maxlen:
            
            # make combined heat-map from history of candidates
            heatmap = np.sum(self._heat_history, axis=0)
                    
            # threshold heat-map
            heatmap[heatmap <= self._heat_thresh] = 0
            
            # label vehicle objects in heat-map
            from scipy.ndimage.measurements import label
            labels, veh_count = label(heatmap)
            
            # compute vehicle bounding boxes
            for veh_number in range(1, veh_count+1):
                # Find pixels with each car_number label value
                nonzero = (labels == veh_number).nonzero()
                # Identify x and y values of those pixels
                nonzeroy = np.array(nonzero[0])
                nonzerox = np.array(nonzero[1])
                # Define a bounding box based on min/max x and y
                L,T = np.min(nonzerox), np.min(nonzeroy)
                R,B = np.max(nonzerox), np.max(nonzeroy)
                veh_boxes.append(((L,T), (R,B)))
                
            if self._diagnostics == True:
                # save labels image
                labels_path = './diagnostics/labels%d.png' % self._img_count
                cv2.imwrite(labels_path, labels)
                # save detection image
                detect_img = np.copy(rgb)
                for box in veh_boxes:
                    cv2.rectangle(detect_img, box[0], box[1], (0,0,255), 5)
                detect_path = './diagnostics/detect%d.png' % self._img_count
                cv2.imwrite(detect_path, cv2.cvtColor(detect_img, cv2.COLOR_RGB2BGR))
                
        return veh_boxes
            

ROI_Scaler = namedtuple('ROI_Scaler', [
    'scale',    # integer 2-tuple, scale image by scale[0]/scale[1]
    'ROI',
    'offs',
    'steps'
])

class VehicleRecognizer:
    '''
    callable(image) --> list of (box,score) tuples
    '''
    def __init__(self, ftr_extractor, ftr_scaler, veh_clf):
        '''
        ftr_extractor: instance of VehicleFeatureExtractor
        ftr_scaler: trained sklearn feature scaler (i.e. StandardScaler)
        veh_clf: trained sklearn classifier
        '''
        self._ftr_extractor = ftr_extractor
        self._ftr_scaler = ftr_scaler
        self._veh_clf = veh_clf
        self._ROI_scalers = self._make_ROI_scalers()
        
    def __call__(self, rgb):
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
            self._ftr_extractor.set_full_image(scaled_ROI)
            for i in range(ROI_scaler.steps[0]):
                for j in range(ROI_scaler.steps[1]):
                    # ...extract features for each 64x64 sliding tile...
                    x = ROI_scaler.offs[0] + 16*i
                    y = ROI_scaler.offs[1] + 16*j
                    ftrs = self._ftr_extractor.extract_tile_features((x,y))
                    # normalize features
                    ftrs = self._ftr_scaler.transform(np.array(ftrs).reshape(1,-1))
                    # predict whether or not the tile contains a vehicle
                    if self._veh_clf.predict(ftrs) == 1:
                        # vehicle --> record unscaled tile rectangle
                        mL, mR = (x*b//a)+L, ((x+64)*b//a)+L
                        mT, mB = (y*b//a)+T, ((y+64)*b//a)+T
                        score = self._veh_clf.decision_function(ftrs)
                        matches.append((((mL, mT), (mR, mB)), score))
        return matches
    
    def _make_ROI_scalers(self):
        '''
        '''
        return [
            # self._make_ROI_scaler((4,1),  ((416,404), (864,436))),  # tile = 16
            # self._make_ROI_scaler((8,3), ((340,416), (940,440))), # tile = 24
            # self._make_ROI_scaler((2,1), ((288,402), (992,450))),  # tile =  32, step =  4
            # self._make_ROI_scaler((4,3), ((232,397), (1048,469))),  # tile =  48, step =  6
            self._make_ROI_scaler((1,1), ((160,392), (1120,488))),  # tile =  64, step =  8
            # self._make_ROI_scaler((4,5), ((120,387), (1160,507))),  # tile =  80, step = 10
            self._make_ROI_scaler((2,3), ((80,382), (1200,526))),  # tile =  96, step = 12
            # self._make_ROI_scaler((4,7),  ((0,400), (1280,540))),  # tile = 112, step = 14
            self._make_ROI_scaler((1,2), ((0,372), (1280,564)))  # tile = 128, step = 16
            # self._make_ROI_scaler((2,5), ((0,362), (1280,602))),   # tile = 160
            # self._make_ROI_scaler((1,3), ((0,352), (1280,640)))   # tile = 192
            # self._make_ROI_scaler((2,7), ((0,342), (1280,678))),   # tile = 224
            # self._make_ROI_scaler((1,4), ((0,332), (1280,716))),   # tile = 256
            # self._make_ROI_scaler((2,9), ((0,322), (1280,754))),    # tile = 288
            # self._make_ROI_scaler((1,5), ((0,312), (1280,792)))    # tile = 320
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
    '''
    parameterized image-based feature extractor for vehicle recognition
    '''
    def __init__(self, spatial_params=None, hist_params=None, hog_params=None):
        '''
        spatial_params: instance of ColorSpatialParams
        hist_params: instance of ColorHistParams
        hog_params: instance of HogFeatureParams
        '''
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
            self._src_spat = self._convert_image(rgb, self._spatial_params.color_space)
            
        if self._hist_params:
            self._src_hist = self._convert_image(rgb, self._hist_params.color_space)
            
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
        hog_img = self._convert_image(rgb, self._hog_params.color_space)
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
                
    def _convert_image(self, img_rgb, color_space):
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
        

def load_clf_training_data(ftr_extractor, test_size=0.3):
    '''
    Setup for training a vehicle recognition classifier:
     1. load training-set images
     2. extract features from images (via ftr_extractor), create labels
     3. train feature scaler and scale features
     4. split features & labels into training & test sets
     5. return feature scaler & training/test sets
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
        '../veh-det-training-data/non-vehicles/GTI/*.png',
        '../veh-det-training-data/hard-negs/hardneg*.png'
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
    Save to file everything needed to instantiate & use VehicleRecognizer:
     -feature extraction parameters
     -trained feature scaler
     -trained classifier
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
    Load from file everything needed to instantiate & use VehicleRecognizer:
     -feature extraction parameters
     -trained feature scaler
     -trained classifier
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
