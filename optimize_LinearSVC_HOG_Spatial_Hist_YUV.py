
import numpy as np
import vehicles as veh

# configure color spatial features
spatial_params = veh.ColorSpatialParams(color_space="YUV", image_size=16)
    
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

# use grid-search to find (near) optimal regularization parameter
veh.optimize_LinearSVC(ftr_extractor, svc_C_values=np.logspace(-5, 3, 9))
