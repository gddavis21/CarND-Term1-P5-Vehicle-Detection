import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import vehicles as veh
import sys
import os

# car_img = mpimg.imread('../Veh-Det-Training-Data/vehicles/GTI_Right/image0227.png')
# notcar_img = mpimg.imread('../Veh-Det-Training-Data/non-vehicles/GTI/image995.png')

# configure color spatial features
spatial_params = veh.ColorSpatialParams(color_space="YUV", image_size=16)
    
# configure color (hue/saturation) histogram features
hist_params = veh.ColorHistParams(
    color_space='YUV',
    bins_ch1=16, range_ch1=(0,255),
    bins_ch2=16, range_ch2=(0,255),
    bins_ch3=16, range_ch3=(0,255))
# hist_params = veh.ColorHistParams(
    # color_space='HSV',
    # bins_ch1=(0, 15, 45, 75, 105, 135, 165, 180),
    # range_ch1=(0,179),
    # bins_ch2=(0, 32, 64, 96, 128, 160, 192, 224, 256),
    # range_ch2=(0,255),
    # bins_ch3=(0, 32, 64, 96, 128, 160, 192, 224, 256),
    # range_ch3=(0,255))
    
# configure HOG descriptor features    
hog_params = veh.HogFeatureParams(
    color_space='YUV',
    cell_size=8,
    block_size=2,
    num_orient=11)

# create feature extractor    
ftr_extractor = veh.VehicleFeatureExtractor(spatial_params, hist_params, hog_params)

in_file_path = sys.argv[1]
print(in_file_path)
bgr = cv2.imread(in_file_path)
print(bgr.shape)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

in_dir, fname = os.path.split(in_file_path)
file_name, file_ext = os.path.splitext(fname)

def vis_hog(fig):
    fig.savefig('./output_images/%s-vis-HOG.png' % file_name, bbox_inches='tight')
    
def vis_spatial(fig):
    fig.savefig('./output_images/%s-vis-spatial.png' % file_name, bbox_inches='tight')
    
def vis_hist(fig):
    fig.savefig('./output_images/%s-vis-histogram.png' % file_name, bbox_inches='tight')
    
ftrs = ftr_extractor.extract_image_features(rgb, vis_hog, vis_spatial, vis_hist)

