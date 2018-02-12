## Writeup Report

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier.
* Optionally, you can also apply a color transform and append binned color features and/or histograms of color to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[img_car_notcar]: ./output_images/car_notcar.png
[img_car_vis_hog]: ./output_images/image0227-vis-HOG.png
[img_vis_hog]: ./output_images/vis-HOG.png
[img_vis_spatial]: ./output_images/vis-spatial.png
[img_vis_hist]: ./output_images/vis-hist.png
[img_search_bands]: ./output_images/search_bands_plot.png
[img_sliding_windows]: ./output_images/vis-sliding-windows.png
[img_recog_examples]: ./output_images/vis-recog-examples.png
[img_recog_heat]: ./output_images/vis-recog-heat.png
[img_vis_labels]: ./output_images/vis-labels.png
[img_vis_detect]: ./output_images/vis-detect.png
[proj_video]: ./output_videos/project_video_out.mp4

## Code Overview

My implementation for this project consists of a Python module (vehicles.py) of classes and utility functions that make up the core functionality, and several scripts that leverage this module to implement the various tasks required for the project.

* Module `vehicles.py`
  + class `VehicleFeatureExtractor` (vehicles.py TODO)
    - implements image-based feature extraction for use with image classifier
    - supports any combination of Histogram of Oriented Gradients (HOG) features, binned color features, and color histogram features
  + function `load_training_data()` (vehicles.py TODO)
    - loads vehicle & non-vehicle training images from disk
    - extracts image features & labels (using `VehicleFeatureExtractor`)
  + function `train_LinearSVC()` (vehicles.py TODO)
    - fits feature normalization "scaler" to extracted training features
    - splits training data into training/test sets
    - trains linear SVM classifier
  + function `optimize_LinearSVC()` (vehicles.py TODO)
    - fits feature normalization "scaler" to extracted training features
    - splits training data into training/test sets
    - cross-validates precision/recall metrics over range of SVM parameter values
  + function `save_classifier()` (vehicles.py TODO)
    - saves feature-extraction parameters, trained feature scaler & trained classifier to pickle file
    - enables efficient re-use of trained classifier
  + function `load_classifier()` (vehicles.py TODO)
    - loads previously saved feature-extraction parameters, trained feature scaler & trained classifier from pickle file
    - enables efficient re-use of trained classifier
  + class `VehicleRecognizer` (vehicles.py TODO)
    - uses supplied `VehicleFeatureExtractor`, trained feature scaler and trained image classifier
    - performs sliding-window search on given image(s) for vehicle "matches"
    - for each image, returns vehicle-match bounding boxes & corresponding match-strength scores
  + class `VehicleDetector` (vehicles.py TODO)
    - track vehicles in a time series of images (video stream) 
    - uses supplied `VehicleRecognizer` to identify candidate vehicle matches
    - implements additional false positive filtering
    - for each image, returns bounding boxes of tracked vehicles
* Module `vision.py`
  + class `CameraCal` (vision.py TODO)
    - computes camera calibration from checkerboard images
    - applies calibration to undistort given image(s)
    - this class was integrated from the previous project
* Scripts
  + `train_LinearSVC_HOG_Spatial_Hist_YUV.py`
    - train & save LinearSVC classifier & feature scaler, along with corresponding feature extraction parameters
    - encodes feature extraction parameters selected for best performance
  + `optimize_LinearSVC_HOG_Spatial_Hist_YUV.py`
    - estimate LinearSVC classifier performance across a range of regularization parameter values
    - uses feature extraction parameters selected for best performance
  + `recognition_images.py`
    - applies vehicle recognition algorithm (VehicleRecognizer) to 1 or more given images
    - draws recognition result overlay graphics and saves annotated images
  + `recognition_video.py`
    - applies vehicle recognition algorithm (VehicleRecognizer) to each frame in video stream
    - draws recognition overlay graphics on each frame
    - saves result video with vehicle recognition graphics
  + `detection_video.py`
    - applies vehicle detection/tracking algorithm (VehicleDetector) to each frame in video stream
    - draws detection overlay graphics on each frame
    - saves result video with vehicle detection graphics

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I used a combination of HOG, binned color and color histogram features extracted from the training images. Feature extraction is implemented in class vehicles.VehicleFeatureExtractor (vehicles.py TODO).
  * VehicleFeatureExtractor instance is constructed with user-defined combination of HOG, spatial and histogram extraction parameters.
  * User extracts features from each training image by calling method VehicleFeatureExtractor.extract_image_features() (vehicles.py TODO).
  * extract_image_features() calls method VehicleFeatureExtractor._hog_features() (vehicles.py TODO) to extract HOG features.
  * extract_image_features() calls method VehicleFeatureExtractor._bin_spatial() (vehicles.py TODO) to extract binned spatial features.
  * extract_image_features() calls method VehicleFeatureExtractor._color_histogram() (vehicles.py TODO) to extract color histogram features.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Vehicle & Non-Vehicle][img_car_notcar]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example of `vehicle` and `non-vehicle` HOG features using `YUV` color and HOG parameters of `orientations=11`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![Visualize HOG features][img_vis_hog]

Here is an example of `vehicle` and `non-vehicle` binned spatial features using `YUV` color and binned to 16x16 array:

![Visualize binned spatial features][img_vis_spatial]

Here is an example of `vehicle` and `non-vehicle` color histogram features using `YUV` color and 16 bins per channel:

![Visualize color histogram features][img_vis_hist]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of color-space and `skimage.hog()` parameters for extracting HOG features:
  * color spaces: RGB, HSV and YUV
  * orientations: 8, 9, 11, 12
  * cell size 8x8
  * block sizes 2x2, 3x3
  
For each of these combinations I trained a classifier and computed precision & recall metrics against the test set. I also tested the trained classifiers on a series of images extracted from the project video, annotating the images with vehicle match boxes. This was a good demonstration of classifier performance, both in terms of identifying vehicles as well as avoiding false positives.

The main HOG parameter that affected classifier performance was the choice of color space. I found that `RGB` performed poorly, `HSV` a little better, and `YUV` best. I noticed minor performance variations from varying cell-size, block-size and number of orientations. In the end I chose `YUV` color with cell-size = 8x3, block-size = 2x2 and orientations = 11.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

My code for training a linear SVM classifier consists of the following:
  * script `train_LinearSVC_HOG_Spatial_Hist_YUV.py`
    - configures HOG, spatial and histogram feature extraction parameters
    - creates `VehicleFeatureExtractor` instance
    - calls `vehicles.train_LinearSVC()` to train the classifier (see below)
    - calls `vehicles.save_classifier()` to save the trained classifier to a file (see below)
  * function `vehicles.train_LinearSVC()` (vehicles.py TODO)
    - calls `vehicles.load_training_data()` (vehicles.py TODO) to load training images and extract features & labels
    - fits an `sklearn.StandardScaler` to the feature data and normalizes the features
    - uses `sklearn.train_test_split()` to randomly split training data into training & test sets
    - fits an `sklearn.LinearSVC` classifier to the training set
  * function `vehicles.save_classifier()` (vehicles.py TODO)
    - saves feature extraction parameters, feature scaler and trained classifier to pickle file
    - enables efficient re-use of trained classifier
    
NOTE: Factoring the classifier training implementation into the module functions made it easy to train, save and test classifiers with a variety of feature extraction parameters (by creating other similar top-level scripts with alternative parameter selections).

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Class `vehicles.VehicleRecognizer` (vehicles.py TODO) performs a sliding window search for vehicle matches on user-supplied images.
  * `VehicleRecognizer` is constructed with the following user-defined arguments:
    - `VehicleFeatureExtractor` instance
    - trained feature scaler (`sklearn StandardScaler`)
    - trained image classifier (`sklearn LinearSVC`) 
    - list of 1 or more window sizes to use in sliding window search
  * The class defines a `search band` for each supported window size--this is the region where the sliding window search is performed for that window size. These regions were carefully chosen to minimize search area without sacrificing recognition accuracy. This design is critical to run-time performance.
  * The sliding window search algorithm is found in `VehicleRecognizer.__call__()` (vehicles.py TODO).
  * Algorithm outline:
    - for each selected window size
      + resize the window-specific search band by the scale required to resize the window to 64x64
      + call `VehicleFeatureExtractor.set_full_image()` to pre-load extractor with resized search band
      + step a 64x64 window across and down the resized search band (step-size is defined as 16 pixels = 75% overlap)
      + at each step, extract 64x64 tile (same size as training images)
        * call `VehicleFeatureExtractor.extract_tile_features()` to extract tile image features
        * use `clf.predict()` method to classify the tile as `vehicle` or `not-vehicle`
        * vehicle --> record tile box & match-strength score (calculated by `clf.decision_function()` method)
    - report list of vehicle-match boxes and corresponding match-strength scores
  * VehicleFeatureExtractor `set_full_image()` and `extract_tile_features()` methods implement a caching scheme that minimizes tile resizing and HOG feature extraction (by extracting HOG features once for each scaled ROI and sub-sampling on demand). This design is also critical to run-time performance.
      
Here's the procedure I used to calibrated the window-specific search bands: 
  * Extracted several images from the project video.
  * Recorded tight bounding-box sizes and positions for the vehicles in these images.
  * Plotted bounding-box size vs. bounding-box top & bottom positions (see below).
  * The plot shows a simple linear relationship between window size and vertical image position where vehicles of that size would be located.
  * From the plot curves I estimated search band top/bottom positions for a set of pre-defined window sizes, also enlarging the search bands enough to allow for the window to slide up and down 1 step from the center position.
  
![Window-specific search bands][img_search_bands]

Here is a video frame with search bands and example tiles for 32x32, 64x64, 96x96 and 128x128 sliding windows:

![Search bands & sliding windows][img_sliding_windows]

For the final project video I chose sliding window sizes of 64x64, 80x80 and 96x96, with 75% overlap. These sizes did an acceptable job of consistently matching vehicles in the video without excessive false positives. I found that tiles smaller than 64x64 and larger than 96x96 generated significant false positives. 

I tried 25% and 50% overlap for faster performance, but 75% was the least overlap I tried that performed well enough at consistently matching vehicles. Less overlap resulted in significant gaps where vehicles would not be matched.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 3 scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Below are some example images.

NOTE: I color-coded the vehicle match boxes to indicate a few levels of match-strength score (0 < red < yellow < green < blue). This technique proved useful, both for confirming classifier performance, and for choosing an appropriate threshold for the false positives filter (see next section).

![Vehicle recognition examples][img_recog_examples]

The `sklearn.LinearSVC` classifier exposes a regularization/penalty parameter `C` which is generally critical to classifier performance. I executed a `grid search` to estimate classifier performance via 3-fold cross validation at different `C` values.

The code for the grid search is in script `optimize_LinearSVC_HOG_Spatial_Hist_YUV.py`:
  * configures HOG, spatial binning & color histogram feaure extraction parameters
  * calls function `vehicles.optimize_LinearSVC()` to perform the grid search
    - loads training data & splits into training/test sets
    - performs grid search (using `sklearn GridSearchCV()`) to estimate precision & recall performance on range of `C` values
    
|     C      | Accuracy  | Precision |  Recall   |
|:----------:|:---------:|:---------:|:---------:|
|    0.00001 | 0.988     | 0.993     | 0.983     |
|    0.0001  | 0.993     | 0.996     | 0.990     |
|    0.001   | 0.993     | 0.996     | 0.990     |
|    0.01    | 0.993     | 0.996     | 0.990     |
|    0.1     | 0.993     | 0.996     | 0.990     |
|    1.0     | 0.993     | 0.996     | 0.990     |
|   10.0     | 0.993     | 0.996     | 0.990     |
|  100.0     | 0.993     | 0.996     | 0.990     |
| 1000.0     | 0.993     | 0.996     | 0.990     |

It's clear from these results that the classifier performs very similarly for a wide range of `C` values. I chose C=0.01 for training my project classifier.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_videos/project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Class `vehicles.VehicleDetector` (vehicles.py TODO) applies frame-by-frame vehicle matching (via `VehicleRecognizer`) and then applies an algorithm for filtering false positives out of the match results.
  * record positions and match-strength scores of vehicle matches in each video frame
  * create a frame-by-frame `heatmap`
    - start with an empty (zero) heatmap same size as frame
    - within each vehicle match box, add the match-strength score
    - keep heatmaps for the past N frames (N is a user-defined parameter--the `history depth`)
  * combine heatmaps from last N frames, and threshold the combined heatmap (threshold T is a user-defined parameter)
  * use `scipy.measurements.label()` to identify objects in the combined heatmap
  * assuming each labeled object corresponds to a vehicle, compute bounding box of each object
  * report list of object/vehicle bounding boxes for each frame
  
Using match-strength score to weight the heatmap was critical to getting good performance out of the false positives filter. 
  * Originally I weighted all matches equally
  * This scheme made it very difficult to eliminate false positives without eliminating true matches.
  * It turned out that nearly all false positives have a relative low match score, while the true vehicle matches tend to be 'covered' by multiple medium/high-score boxes.
  * Adding up match-strength scores led to consistently good separation between false positives and true matches--critical for any threshold-based technique.
  
Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![Vehicle matches & heat-maps][img_recog_heat]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![Labeled vehicle objects][img_vis_labels]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![Vehicle detection][img_vis_detect]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

