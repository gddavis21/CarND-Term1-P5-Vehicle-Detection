import sys
import numpy as np
import vehicles as veh

# configure color spatial features
# spatial_params = veh.ColorSpatialParams(color_space="YUV", image_size=16)
spatial_params = None
    
# configure color (hue/saturation) histogram features
hist_params = veh.ColorHistParams(
    color_space='YUV',
    bins_ch1=16, range_ch1=(0,255),
    bins_ch2=16, range_ch2=(0,255),
    bins_ch3=16, range_ch3=(0,255))
    
# configure HOG descriptor features    
hog_params = veh.HogFeatureParams(
    color_space='YUV',
    cell_size=8,
    block_size=2,
    num_orient=11)

# create feature extractor    
ftr_extractor = veh.VehicleFeatureExtractor(spatial_params, hist_params, hog_params)

# train feature scaler & LinearSVC classifier
print('training classifier...')
ftr_scaler, clf = veh.train_LinearSVC(ftr_extractor, test_size=0.2, svc_C=0.01)
    
# save classifier data
print('saving classifier data to file...')
file_path = sys.argv[1]
veh.save_classifier(file_path, spatial_params, hist_params, hog_params, ftr_scaler, clf)
