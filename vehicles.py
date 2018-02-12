from collections import namedtuple
from collections import deque
import glob
import pickle
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.feature import hog
import sklearn.preprocessing
import sklearn.svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


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
    image-based feature extractor for vehicle recognition
    '''
    def __init__(self, spatial_params=None, hist_params=None, hog_params=None):
        '''
        spatial_params: instance of ColorSpatialParams (None --> no spatial features)
        hist_params: instance of ColorHistParams (None --> no histogram features)
        hog_params: instance of HogFeatureParams (None --> no HOG features)
        '''
        self._spatial_params = spatial_params
        self._hist_params = hist_params
        self._hog_params = hog_params
        self._src_rgb = None
        self._spatial_img = None
        self._hist_img = None
        self._hog_ftrs = None
        self._hog_imgs = None
        
    def set_full_image(self, rgb, visualise=False):
        '''
        Prepare extractor for multiple tile feature extractions over an image.
        
          extractor.set_full_image(image)
          for tile in image:
            features = extractor.extract_tile_features(tile)
            
        '''
        self._src_rgb = rgb
        self._spatial_img = None
        self._hist_img = None
        self._hog_ftrs = None
        self._hog_imgs = None
        
        if self._spatial_params:
            self._spatial_img = self._convert_image(rgb, self._spatial_params.color_space)
            
        if self._hist_params:
            self._hist_img = self._convert_image(rgb, self._hist_params.color_space)
            
        if self._hog_params:
            self._hog_ftrs, self._hog_imgs = self._compute_hog_array(rgb, visualise)
    
    def extract_tile_features(self, tile_corner, vis_hog=None, vis_spatial=None, vis_hist=None):
        '''
        Extract features for a single image tile. 

          extractor.set_full_image(image)
          for tile in image:
            features = extractor.extract_tile_features(tile)
            
        '''
        features = []
        vis_imgs = []

        if self._spatial_params:
            features.append(self._bin_spatial(tile_corner, vis_spatial))

        if self._hist_params:
            features.append(self._color_histogram(tile_corner, vis_hist))
        
        if self._hog_params:
            features.append(self._hog_features(tile_corner, vis_hog))
        
        return np.concatenate(features)
        
    def extract_image_features(self, rgb, vis_hog=None, vis_spatial=None, vis_hist=None):
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
        
        self.set_full_image(src_img, vis_hog != None)
        return self.extract_tile_features((0,0), vis_hog, vis_spatial, vis_hist)

    def _bin_spatial(self, tile_corner, vis_spatial=None):
        ''' compute binned spatial feature vector '''
        # sub-sample cached spatial image 
        L,T = tile_corner[0], tile_corner[1]
        R,B = L+64, T+64
        tile = self._spatial_img[T:B,L:R]
        # resize to desired size, bin each channel separately 
        size = self._spatial_params.image_size
        interp = cv2.INTER_AREA if size < 64 else cv2.INTER_LINEAR
        binned = cv2.resize(tile, (size,size), interpolation=interp)
        if vis_spatial:
            # user-requested visualization:
            #  - 2x2 plot
            #  - original tile/image
            #  - binned color map for each channel (3)
            fig, axes = plt.subplots(2,2)
            axes[0][0].imshow(self._src_rgb[T:B,L:R])
            axes[0][0].set_title('Image')
            axes[0][1].imshow(binned[:,:,0], cmap='gray')
            axes[0][1].set_title('Ch1 Spatial')
            axes[1][0].imshow(binned[:,:,1], cmap='gray')
            axes[1][0].set_title('Ch2 Spatial')
            axes[1][1].imshow(binned[:,:,2], cmap='gray')
            axes[1][1].set_title('Ch3 Spatial')
            fig.tight_layout()
            vis_spatial(fig)
        return binned.ravel()
        
    def _color_histogram(self, tile_corner, vis_hist=None):
        ''' compute color histogram feature vector '''
        # sub-sample cached histogram image 
        L,T = tile_corner[0], tile_corner[1]
        R,B = L+64, T+64
        tile = self._hist_img[T:B,L:R]
        # compute separate histograms for each channel 
        hp = self._hist_params
        bins1, range1 = hp.bins_ch1, hp.range_ch1
        bins2, range2 = hp.bins_ch2, hp.range_ch2
        bins3, range3 = hp.bins_ch3, hp.range_ch3
        hist_ch1 = np.histogram(tile[:,:,0], bins=bins1, range=range1)
        hist_ch2 = np.histogram(tile[:,:,1], bins=bins2, range=range2)
        hist_ch3 = np.histogram(tile[:,:,2], bins=bins3, range=range3)
        if vis_hist:
            # user-requested visualization:
            #  - 2x2 plot
            #  - original tile/image
            #  - histogram for each channel (3)
            fig, axes = plt.subplots(2,2)
            axes[0][0].imshow(self._src_rgb[T:B,L:R])
            axes[0][0].set_title('Image')
            bin_edges = hist_ch1[1]
            bin_centers = (bin_edges[0:-1] + bin_edges[1:]) / 2
            axes[0][1].bar(bin_centers, hist_ch1[0])
            axes[0][1].set_xlim(range1[0], range1[1]+1)
            axes[0][1].set_title('Ch1 Histogram')
            bin_edges = hist_ch2[1]
            bin_centers = (bin_edges[0:-1] + bin_edges[1:]) / 2
            axes[1][0].bar(bin_centers, hist_ch2[0])
            axes[1][0].set_xlim(range2[0], range2[1]+1)
            axes[1][0].set_title('Ch2 Histogram')
            bin_edges = hist_ch3[1]
            bin_centers = (bin_edges[0:-1] + bin_edges[1:]) / 2
            axes[1][1].bar(bin_centers, hist_ch3[0])
            axes[1][1].set_xlim(range3[0], range3[1]+1)
            axes[1][1].set_title('Ch3 Histogram')
            fig.tight_layout()
            vis_hist(fig)
        # concatenate histograms into a single feature vector
        return np.concatenate((hist_ch1[0], hist_ch2[0], hist_ch3[0]))
        
    def _hog_features(self, tile_corner, vis_hog=None):
        ''' compute HOG descriptor feature vector '''
        if vis_hog:
            # user-requested visualization
            L,T = tile_corner[0], tile_corner[1]
            R,B = L+64, T+64
            if len(self._hog_ftrs) == 1:
                # HOG features on grayscale image
                # 1x2 plot: original tile & single-channel HOG visualization image
                fig, axes = plt.subplots(2,2)
                axes[0][0].imshow(self._src_rgb[T:B,L:R])
                axes[0][0].set_title('Image')
                axes[0][1].imshow(self._hog_imgs[0][T:B,L:R], cmap='gray')
                axes[0][1].set_title('HOG')
                fig.tight_layout()
                vis_hog(fig)
            else:
                # HOG features on color image
                # 2x2 plot: original tile & HOG visualization image for each channel
                fig, axes = plt.subplots(2,2)
                axes[0][0].imshow(self._src_rgb[T:B,L:R])
                axes[0][0].set_title('Image')
                axes[0][1].imshow(self._hog_imgs[0][T:B,L:R], cmap='gray')
                axes[0][1].set_title('Ch1 HOG')
                axes[1][0].imshow(self._hog_imgs[1][T:B,L:R], cmap='gray')
                axes[1][0].set_title('Ch2 HOG')
                axes[1][1].imshow(self._hog_imgs[2][T:B,L:R], cmap='gray')
                axes[1][1].set_title('Ch3 HOG')
                fig.tight_layout()
                vis_hog(fig)
        # sub-sample cached HOG feature map
        cell_size = self._hog_params.cell_size
        block_size = self._hog_params.block_size
        blocks_per_tile = (64 // cell_size) - block_size + 1
        L,T = (tile_corner[0] // 8), (tile_corner[1] // 8)
        R,B = L+blocks_per_tile, T+blocks_per_tile
        if len(self._hog_ftrs) == 1:
            # HOG features on grayscale image
            return self._hog_ftrs[0][T:B,L:R].ravel()
        else:
            # HOG features on color image
            hog_feat1 = self._hog_ftrs[0][T:B,L:R].ravel()
            hog_feat2 = self._hog_ftrs[1][T:B,L:R].ravel()
            hog_feat3 = self._hog_ftrs[2][T:B,L:R].ravel()
            return np.hstack((hog_feat1, hog_feat2, hog_feat3))
            
    def _compute_hog_channel(self, img, num_orient, cell_size, block_size, visualise=False):
        ''' extract HOG descriptor map for a single-channel image '''
        return hog(
            img,
            orientations=num_orient,
            pixels_per_cell=(cell_size, cell_size),
            cells_per_block=(block_size, block_size),
            block_norm='L2',
            visualise=visualise,
            transform_sqrt=True,
            feature_vector=False)
    
    def _compute_hog_array(self, rgb, visualise=False):
        ''' 
        compute HOG descriptor map(s) and HOG visualization image(s) for each channel
        of the given source image
        '''
        src_img = self._convert_image(rgb, self._hog_params.color_space)
        num_orient = self._hog_params.num_orient
        cell_size = self._hog_params.cell_size
        block_size = self._hog_params.block_size
        if src_img.ndim == 2:
            # grayscale image (just 1 channel)
            hog = self._compute_hog_channel(src_img, num_orient, cell_size, block_size, visualise)
            if visualise:
                hog_ftrs, hog_imgs = (hog[0],), (hog[1],)
            else:
                hog_ftrs, hog_imgs = (hog,), None
        else:
            # color image --> compute HOG features separately for each channel
            hog1 = self._compute_hog_channel(src_img[:,:,0], num_orient, cell_size, block_size, visualise)
            hog2 = self._compute_hog_channel(src_img[:,:,1], num_orient, cell_size, block_size, visualise)
            hog3 = self._compute_hog_channel(src_img[:,:,2], num_orient, cell_size, block_size, visualise)
            if visualise:
                hog_ftrs, hog_imgs = (hog1[0], hog2[0], hog3[0]), (hog1[1], hog2[1], hog3[1])
            else:
                hog_ftrs, hog_imgs = (hog1, hog2, hog3), None
        return hog_ftrs, hog_imgs
                
    def _convert_image(self, img_rgb, color_space):
        ''' color conversion utility '''
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
        

def load_training_data(ftr_extractor, test_size=0.3):
    '''
    Load training data for training a vehicle recognition classifier:
      1. load training-set images
      2. extract features from images (via ftr_extractor) 
      3. create corresponding labels
      4. return features & labels
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
    car_ftrs = []
    notcar_ftrs = []

    print('extracting vehicle image features...')
    for path in car_paths:
        for img_file in glob.iglob(path):
            rgb = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
            car_ftrs.append(ftr_extractor.extract_image_features(rgb))

    print('extracting non-vehicle image features...')
    for path in notcar_paths:
        for img_file in glob.iglob(path):
            rgb = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
            notcar_ftrs.append(ftr_extractor.extract_image_features(rgb))

    # build features matrix
    features = np.vstack((car_ftrs, notcar_ftrs)).astype(np.float64)                        

    # build labels vector
    labels = np.hstack((np.ones(len(car_ftrs)), np.zeros(len(notcar_ftrs))))
    
    return features, labels
    
def train_LinearSVC(ftr_extractor, test_size, svc_C):
    '''
    Train LinearSVC classifier for vehicle recognition:
      1. extract features & labels from training data
      2. train feature normalization scaler
      3. split normalized training/test sets
      4. train LinearSVC classifier
      5. print precision & recall metrics
      6. return trained scaler & classifier
    '''
    # load training images & extract features/labels
    features, labels = load_training_data(ftr_extractor)

    # compute feature normalization transform
    ftr_scaler = StandardScaler().fit(features)

    # split data into training & test sets
    X_train, X_test, y_train, y_test = train_test_split(
        ftr_scaler.transform(features), 
        labels, 
        test_size=test_size, 
        random_state=np.random.randint(0, 100))

    clf = svm.LinearSVC(C=svc_C)
    clf.fit(X_train, y_train)

    # output metrics
    y_pred = clf.predict(X_test)
    print('Precision: %.4f' % precision_score(y_test, y_pred))
    print('Recall: %.4f' % recall_score(y_test, y_pred))
    
    # client must use fitted scaler with trained classifier
    return ftr_scaler, clf
    
def optimize_LinearSVC(ftr_extractor, svc_C_values):
    '''
    Optimize the LinearSVC regularization parameter (C) for vehicle recognition:
      1. extract features & labels from training data
      2. train feature normalization scaler
      3. split normalized training/test sets
      4. cross-validate over user-defined range of C parameter values
      5. print precision & recall metrics for each parameter value
    '''
    # load training images & extract features/labels
    features, labels = load_training_data(ftr_extractor)

    # compute feature normalization transform
    ftr_scaler = StandardScaler().fit(features)

    # split data into training & test sets
    X_train, X_test, y_train, y_test = train_test_split(
        ftr_scaler.transform(features), 
        labels, 
        test_size=0.5, 
        random_state=np.random.randint(0, 100))
    
    # optimize SVC regularization parameter via cross-validation
    param_grid = {'C': svc_C_values}

    for score in ['precision', 'recall']:
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
        print(classification_report(y_test, clf.predict(X_test)))
        print()
        
    
def save_classifier(
    file_path, 
    spatial_params, hist_params, hog_params, 
    ftr_scaler, veh_clf):
    '''
    Save to pickle file everything needed to instantiate VehicleRecognizer:
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
    
    
def load_classifier(file_path):
    '''
    Load from pickle file everything needed to instantiate VehicleRecognizer:
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

TileScaler = namedtuple('TileScaler', [
    'scale',    # integer 2-tuple, scale ROI by scale[0]/scale[1]
    'ROI',
    'offs',
    'steps'
])


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
    

class VehicleRecognizer:
    '''
    callable(image) --> list of (box,score) tuples
    '''
    def __init__(self, ftr_extractor, ftr_scaler, veh_clf, tile_sizes):
        '''
        ftr_extractor: instance of VehicleFeatureExtractor
        ftr_scaler: trained sklearn feature scaler (i.e. StandardScaler)
        veh_clf: trained sklearn classifier
        '''
        self._ftr_extractor = ftr_extractor
        self._ftr_scaler = ftr_scaler
        self._veh_clf = veh_clf
        self._tile_scalers = VehicleRecognizer._make_TileScalers(tile_sizes)
        
    def __call__(self, rgb):
        '''
        '''
        matches = []
        for tile_scaler in self._tile_scalers:
            # unpack ROI scaler into ROI corners and scale factors
            L,T = tile_scaler.ROI[0][0], tile_scaler.ROI[0][1]
            R,B = tile_scaler.ROI[1][0], tile_scaler.ROI[1][1]
            a,b = tile_scaler.scale[0], tile_scaler.scale[1]
            # start with unscaled ROI...
            ROI = rgb[T:B,L:R]
            scaled_ROI = ROI
            # ...and resize only if we have to (unequal scale factors)
            if a != b:
                # scale ROI corners
                sL,sT = L*a//b, T*a//b
                sR,sB = R*a//b, B*a//b
                # resize image
                interp = cv2.INTER_AREA if a < b else cv2.INTER_LINEAR
                scaled_ROI = cv2.resize(scaled_ROI, (sR-sL, sB-sT), interpolation=interp)
            # initialize extractor with full-size scaled ROI
            self._ftr_extractor.set_full_image(scaled_ROI)
            # for each 64x64 sliding window...
            for i in range(tile_scaler.steps[0]):
                for j in range(tile_scaler.steps[1]):
                    x = tile_scaler.offs[0] + 16*i
                    y = tile_scaler.offs[1] + 16*j
                    # ...extract normalized features for the tile...
                    ftrs = self._ftr_extractor.extract_tile_features((x,y))
                    ftrs = self._ftr_scaler.transform(np.array(ftrs).reshape(1,-1))
                    # ...and predict whether or not the tile contains a vehicle
                    if self._veh_clf.predict(ftrs) == 1:
                        # vehicle --> record unscaled tile rectangle and match strength
                        mL, mR = (x*b//a)+L, ((x+64)*b//a)+L
                        mT, mB = (y*b//a)+T, ((y+64)*b//a)+T
                        score = self._veh_clf.decision_function(ftrs)
                        matches.append((((mL, mT), (mR, mB)), score))
        return matches
    
    @staticmethod
    def _make_TileScalers(tile_sizes):
        '''
        '''
        scalers = {
            16:  VehicleRecognizer._make_TileScaler((4,1), ((416,404), ( 864,436))),
            24:  VehicleRecognizer._make_TileScaler((8,3), ((340,416), ( 940,440))),
            32:  VehicleRecognizer._make_TileScaler((2,1), ((288,402), ( 992,450))),
            48:  VehicleRecognizer._make_TileScaler((4,3), ((232,397), (1048,469))),
            64:  VehicleRecognizer._make_TileScaler((1,1), ((  0,392), (1280,488))),
            80:  VehicleRecognizer._make_TileScaler((4,5), ((  0,387), (1280,507))),
            96:  VehicleRecognizer._make_TileScaler((2,3), ((  0,382), (1280,526))),
            128: VehicleRecognizer._make_TileScaler((1,2), ((  0,372), (1280,564))),
            160: VehicleRecognizer._make_TileScaler((2,5), ((  0,362), (1280,602))),
            192: VehicleRecognizer._make_TileScaler((1,3), ((  0,352), (1280,640)))
        }
        return [scalers[size] for size in tile_sizes]
    
    @staticmethod
    def _make_TileScaler(scale, ROI):
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
        return TileScaler(scale=scale, ROI=ROI, offs=offs, steps=steps)
        
        
class VehicleDetector:
    '''
    '''
    def __init__(self, match_vehicles, history_depth, heat_thresh, 
        vis_recog=None, vis_heat=None, vis_labels=None, vis_detect=None):
        '''
        match_vehicles: function(image) --> vehicle candidate boxes
        history_depth: number of consecutive frames to accumulate
        heat_thresh: heatmap threshold for object detection
        '''
        self._match_vehicles = match_vehicles
        self._heat_history = deque(maxlen=history_depth)
        self._heat_thresh = heat_thresh
        self._vis_recog = vis_recog
        self._vis_heat = vis_heat
        self._vis_labels = vis_labels
        self._vis_detect = vis_detect
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
        
        if self._vis_recog:
            # report recognition visualization image
            recog_img = np.copy(rgb)
            for match in veh_matches:
                draw_vehicle_match(recog_img, match)
            self._vis_recog(recog_img, self._img_count)
            
        if self._vis_heat:
            # report heatmap visualization image
            heat_img = np.clip(heatmap*20, 0, 255).astype(np.uint8)
            self._vis_heat(heat_img, self._img_count)
            
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
                
            if self._vis_labels:
                # report object labels visualization
                self._vis_labels(labels.astype(np.uint8), self._img_count)
            
            if self._vis_detect:
                # report detected vehicles visualization
                detect_img = np.copy(rgb)
                for box in veh_boxes:
                    cv2.rectangle(detect_img, box[0], box[1], (0,0,255), 5)
                self._vis_detect(detect_img, self._img_count)
                
        return veh_boxes

        
