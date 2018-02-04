import os
import cv2
import numpy as np

NOMINAL_LANE_WIDTH = 3.7  # meters
NOMINAL_DASH_LENGTH = 3.0 # meters
RECT_LANE_WIDTH = 614     # pixels in rectified image
RECT_DASH_LENGTH = 75     # pixels in rectified image
PIXEL_SIZE_X = NOMINAL_LANE_WIDTH / RECT_LANE_WIDTH
PIXEL_SIZE_Y = NOMINAL_DASH_LENGTH / RECT_DASH_LENGTH

LANE_WIDTH_TOL = 1.5      # meters
PARALLEL_LINES_TOL = 1.0  # meters


class LaneDetectionPipeline:
    '''
    Callable class for lane detection pipeline:
      - removes camera distortion from input image
      - detects left & right lane-lines in image
      - overlays image with lane-line graphics
      - annotates image with measured radius-of-curvature
      - annotates image with vehicle offset from lane center
    '''
    def __init__(self, 
        camera_cal,     # vision.CameraCal instance
        rect_prsp):     # vision.PerspectiveTransform instance
        '''
        '''
        self._camera_cal = camera_cal
        self._rect_prsp = rect_prsp
        self._last_lane = None
        # self._fail_counter = 0
        
    def __call__(self, raw_rgb):
        '''
        Pipeline function (callable class). Input image must be RGB.
        '''
        # undistort image (remove sensor/lens distortion)
        rgb = self._camera_cal.undistort_image(raw_rgb)

        # create pixels-of-interest mask
        poi_mask = thresh_lane_lines(rgb)
        
        # detect lane lines
        lane = None
        
        if self._last_lane:
            # fast-fit next lane given prior lane
            lane = self._detect_lane(poi_mask, prior=self._last_lane)
                
        if not lane:
            # no prior (or fast-fit failed) -> use reliable/slow method
            lane = self._detect_lane(poi_mask, prior=None)
                
        if not lane:
            # lane detection failed
            if self._last_lane:
                # re-use last lane fit
                lane = self._last_lane
            else:
                # nothing to do but bail
                return LaneDetectionPipeline._failed_frame(rgb)
                
        # remember this lane to help with the next image
        self._last_lane = lane

        # annotate image with lane overlay & metrics
        result = self._draw_lane_overlay(rgb, lane)
        result = self._annotate_metrics(result, lane)
        return result
        
    def _detect_lane(self, poi_mask, prior=None):
        '''
        Make new Lane instance from given pixels-of-interest mask.
        Returns None if new lane is invalid.
        '''
        pix_size = (PIXEL_SIZE_X, PIXEL_SIZE_Y)
        lane = Lane(poi_mask, self._rect_prsp, pix_size, prior=prior)
        
        is_valid = lane.is_valid(
            NOMINAL_LANE_WIDTH, 
            LANE_WIDTH_TOL, 
            PARALLEL_LINES_TOL)
        
        return lane if is_valid else None
        
    def _draw_lane_overlay(self, image, lane):
        # make overlay image of lane bounded by left/right contours
        overlay = lane.overlay_image(lane_color=(0,255,0))

        # alpha-blend source image with overlay
        return cv2.addWeighted(image, 1, overlay, 0.3, 0)
        
    def _annotate_metrics(self, image, lane):
        # annotate radius of curvature
        cv2.putText(
            image,
            'Radius of Curvature = %d m' % int(lane.radius_of_curvature()),
            org=(20,50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255,255,255),
            thickness=1,
            lineType=cv2.LINE_AA)

        # annotate vehicle offset from lane center
        image_size = (image.shape[1], image.shape[0])
        veh_offs = self._vehicle_offset(lane, image_size)
        
        if veh_offs < 0.0:
            offs_side = 'left'
        else:
            offs_side = 'right'
        
        cv2.putText(
            image,
            'Vehicle is %.2fm %s of center' % (abs(veh_offs), offs_side),
            org=(20,100),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255,255,255),
            thickness=1,
            lineType=cv2.LINE_AA)

        return image
        
    def _vehicle_offset(self, lane, image_size):
        '''
        Compute offset of vehicle from center of lane
        '''
        # vehicle position = bottom-center of image, warped to top-down view
        x_vehicle = image_size[0] / 2
        y_bottom = image_size[1]
        x_vehicle, y_bottom = self._rect_prsp.warp_point(x_vehicle, y_bottom)
        return (x_vehicle * PIXEL_SIZE_X) - lane.position()
    
    @staticmethod
    def _failed_frame(frame):
        '''
        '''
        cv2.putText(
            frame,
            'FAILED',
            org=(20,100),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(255,0,0),
            thickness=2,
            lineType=cv2.LINE_AA)
            
        return frame
        
        

class Lane:
    def __init__(self, poi_mask, rect_prsp, rect_pixel_size, prior=None):
        self._poi_mask = poi_mask
        self._image_size = (poi_mask.shape[1], poi_mask.shape[0])
        self._pixel_size = rect_pixel_size
        self._prsp = rect_prsp
        self._prior = prior
        self._rect_poi_mask = rect_prsp.warp_image(poi_mask, self._image_size)
        self._rect_poi_mask[self._rect_poi_mask > 192] = 255
        self._left_line = None
        self._right_line = None
        if prior:
            self._detect_lane_lines_with_prior(prior)
        else:
            self._detect_lane_lines_from_scratch()
        
    def _detect_lane_lines_from_scratch(self):
        '''
        '''
        # start with rectified binary
        rect_binary = self._rect_poi_mask // 255
        # rect_binary = np.zeros_like(self._rect_poi_mask)
        # rect_binary[self._rect_poi_mask > 0] = 1
        # Take a histogram of the bottom half of the image
        histogram = np.sum(rect_binary[rect_binary.shape[0]//2:,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(rect_binary.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = rect_binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 60
        # Set minimum number of pixels found to recenter window
        minpix = 30
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = rect_binary.shape[0] - (window+1)*window_height
            win_y_high = rect_binary.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 
        
        self._left_line = LaneLine(
            leftx, lefty, 
            self._image_size, 
            self._prsp, 
            self._pixel_size)
            
        self._right_line = LaneLine(
            rightx, righty, 
            self._image_size, 
            self._prsp, 
            self._pixel_size) 
        
    def _detect_lane_lines_with_prior(self, prior):
        '''
        '''
        # start with rectified binary
        rect_binary = self._rect_poi_mask // 255
        # rect_binary = np.zeros_like(self._rect_poi_mask)
        # rect_binary[self._rect_poi_mask > 0] = 1

        # compute left-line and right-line boundary regions
        nonzero = rect_binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 60

        prior_left = prior._left_line.contour(nonzeroy)
        left_bound = prior_left - margin
        right_bound = prior_left + margin
        left_lane_inds = ((nonzerox > left_bound) & (nonzerox < right_bound))

        prior_right = prior._right_line.contour(nonzeroy)
        left_bound = prior_right - margin
        right_bound = prior_right + margin
        right_lane_inds = ((nonzerox > left_bound) & (nonzerox < right_bound))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        self._left_line = LaneLine(
            leftx, lefty, 
            self._image_size, 
            self._prsp, 
            self._pixel_size)
            
        self._right_line = LaneLine(
            rightx, righty, 
            self._image_size, 
            self._prsp, 
            self._pixel_size) 

    def is_valid(self, nom_lane_width, lane_width_tol, parallel_lines_tol):
        '''
        '''
        # compute lane-line contours & differences
        img_height = self._image_size[1]
        yvals = np.linspace(0, img_height-1, img_height)
        xfit_L = self._left_line.contour(yvals)
        xfit_R = self._right_line.contour(yvals)
        diff = np.absolute(xfit_R - xfit_L) * self._pixel_size[0]
        diff_bar = np.mean(diff)
        
        # check lane for correct width
        if np.absolute(diff_bar - nom_lane_width) > lane_width_tol:
            return False
            
        # check lane-lines for approximate parallelism
        if np.any(np.absolute(diff - diff_bar) > parallel_lines_tol):
            return False
        
        return True
        
    def position(self):
        pos_L = self._left_line.position()
        pos_R = self._right_line.position()
        return (pos_L + pos_R) / 2
        
    def radius_of_curvature(self):
        rad_L = self._left_line.radius_of_curvature()
        rad_R = self._right_line.radius_of_curvature()
        return (rad_L + rad_R) / 2
        
    def lane_lines_diagnostic(self):
        out_img = np.dstack((self._rect_poi_mask, self._rect_poi_mask, self._rect_poi_mask))
        xpix_L, ypix_L = self._left_line.lane_pixels()
        xpix_R, ypix_R = self._right_line.lane_pixels()
        out_img[ypix_L, xpix_L] = [255,0,0]
        out_img[ypix_R, xpix_R] = [0,0,255]
        img_height = self._image_size[1]
        yvals = np.linspace(0, img_height-1, img_height)
        xfit_L = self._left_line.contour(yvals)
        xfit_R = self._right_line.contour(yvals)
        out_img[np.int_(yvals), np.int_(xfit_L)] = [255,255,0]
        out_img[np.int_(yvals), np.int_(xfit_R)] = [255,255,0]
        return out_img
        
    def overlay_image(self, lane_color):
        '''
        '''
        # Create empty image to draw the overlay on
        img_width = self._image_size[0]
        img_height = self._image_size[1]
        empty_channel = np.zeros((img_height, img_width), dtype=np.uint8)
        overlay = np.dstack((empty_channel, empty_channel, empty_channel))
        
        # compute left/right lane contours
        yvals = np.linspace(0, img_height-1, img_height)
        xfit_L = self._left_line.contour(yvals)
        xfit_R = self._right_line.contour(yvals)

        # Recast x and y points into usable format for cv2.fillPoly()
        pts_L = np.array([np.transpose(np.vstack([xfit_L, yvals]))])
        pts_R = np.array([np.flipud(np.transpose(np.vstack([xfit_R, yvals])))])
        pts = np.hstack((pts_L, pts_R))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(overlay, np.int_([pts]), lane_color)
        
        # return unwarped overlay image
        return self._prsp.unwarp_image(overlay, self._image_size)

        
class LaneLine:
    def __init__(self, xpix, ypix, image_size, rect_prsp, pixel_size):
        self._xpix = xpix
        self._ypix = ypix
        self._image_size = image_size
        self._rect_prsp = rect_prsp
        self._pixel_size = pixel_size
        self._coeffs_pix = np.polyfit(ypix, xpix, 2)
        self._coeffs_world = np.polyfit(ypix*pixel_size[1], xpix*pixel_size[0], 2)
        
    def lane_pixels(self):
        return self._xpix, self._ypix
        
    def contour(self, yvals):
        return np.polyval(self._coeffs_pix, yvals)
    
    def position(self):
        # compute x position at bottom of rectified image
        y_bottom = self._image_size[1]
        x_line = np.polyval(self._coeffs_pix, y_bottom)
        # scale to meters
        return x_line*self._pixel_size[0]
        
        
    def radius_of_curvature(self):
        # evaluate radius (in meters) at bottom of image
        A = self._coeffs_world[0]
        B = self._coeffs_world[1]
        y = self._image_size[1] * self._pixel_size[1]
        return ((1 + (2*A*y + B)**2)**1.5) / np.absolute(2*A)


def sobel_xy(gray, kernel_size=3):
    scale = 1.0 / 2.0**(kernel_size*2 - 4)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size, scale=scale)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size, scale=scale)
    return sobel_x, sobel_y
    
def scharr_xy(gray):
    scale = 0.0625  # 1/16
    scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0, scale=scale)
    scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1, scale=scale)
    return scharr_x, scharr_y
    
def polar_gradient(grad_x, grad_y):
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_dir = np.arctan2(np.absolute(grad_x), np.absolute(grad_y))
    return grad_mag, grad_dir
    
def image_gradient(gray, kernel_size=3):
    scale = 1.0 / 2.0**(kernel_size*2 - 4)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size, scale=scale)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size, scale=scale)
    return cv2.cartToPolar(sobel_x, sobel_y, angleInDegrees=True)
    
def thresh_yellow(hsv):
    HSV_YELLOW_LOWER = np.array([15, 70, 130])
    HSV_YELLOW_UPPER = np.array([25, 255, 255])
    return cv2.inRange(hsv, HSV_YELLOW_LOWER, HSV_YELLOW_UPPER)
    
# def thresh_white(gray):
    # return cv2.adaptiveThreshold(
        # gray, 
        # maxValue=255, 
        # adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, 
        # thresholdType=cv2.THRESH_BINARY,
        # blockSize=33,
        # C=-60)
        
# def thresh_white_proj(gray_proj):
    # return cv2.adaptiveThreshold(
        # gray_proj, 
        # maxValue=255, 
        # adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, 
        # thresholdType=cv2.THRESH_BINARY,
        # blockSize=53,
        # C=-20)

def thresh_white(rgb):
    RGB_WHITE_LOWER = np.array([190, 190, 190])
    RGB_WHITE_UPPER = np.array([255, 255, 255])
    return cv2.inRange(rgb, RGB_WHITE_LOWER, RGB_WHITE_UPPER)
    
def thresh_lane_edges(gray):
    grad_mag, grad_dir = image_gradient(gray, kernel_size=13)
    strong_edges = cv2.inRange(grad_mag, 25, 255)
    pos_horz_edges = cv2.inRange(grad_dir, 80, 100)
    neg_horz_edges = cv2.inRange(grad_dir, 260, 280)
    horz_edges = cv2.bitwise_or(pos_horz_edges, neg_horz_edges)
    return cv2.bitwise_and(strong_edges, cv2.bitwise_not(horz_edges)) 

# def thresh_lane_lines(rgb, prsp):
    # hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # image_size = (gray.shape[1], gray.shape[0])
    # gray_proj = prsp.warp_image(gray, image_size)
    # yellow = thresh_yellow(hsv)
    # white_proj = thresh_white_proj(gray_proj)
    # white = prsp.unwarp_image(white_proj, image_size)
    # result = cv2.bitwise_or(yellow, white)
    # # edges = thresh_lane_edges(gray)
    # # result = cv2.bitwise_or(result, edges)
    # return result
    
def thresh_lane_lines(rgb):
    white = thresh_white(rgb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    yellow = thresh_yellow(hsv)
    result = cv2.bitwise_or(yellow, white)
    # edges = thresh_lane_edges(gray)
    # result = cv2.bitwise_or(result, edges)
    return result
    
# def detect_lanes(binary_warped):
    # # Assuming you have created a warped binary image called "binary_warped"
    # # Take a histogram of the bottom half of the image
    # histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # # # Create an output image to draw on and  visualize the result
    # # out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # # Find the peak of the left and right halves of the histogram
    # # These will be the starting point for the left and right lines
    # midpoint = np.int(histogram.shape[0]/2)
    # leftx_base = np.argmax(histogram[:midpoint])
    # rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # # Choose the number of sliding windows
    # nwindows = 9
    # # Set height of windows
    # window_height = np.int(binary_warped.shape[0]/nwindows)
    # # Identify the x and y positions of all nonzero pixels in the image
    # nonzero = binary_warped.nonzero()
    # nonzeroy = np.array(nonzero[0])
    # nonzerox = np.array(nonzero[1])
    # # Current positions to be updated for each window
    # leftx_current = leftx_base
    # rightx_current = rightx_base
    # # Set the width of the windows +/- margin
    # margin = 100
    # # Set minimum number of pixels found to recenter window
    # minpix = 50
    # # Create empty lists to receive left and right lane pixel indices
    # left_lane_inds = []
    # right_lane_inds = []

    # # Step through the windows one by one
    # for window in range(nwindows):
        # # Identify window boundaries in x and y (and right and left)
        # win_y_low = binary_warped.shape[0] - (window+1)*window_height
        # win_y_high = binary_warped.shape[0] - window*window_height
        # win_xleft_low = leftx_current - margin
        # win_xleft_high = leftx_current + margin
        # win_xright_low = rightx_current - margin
        # win_xright_high = rightx_current + margin
        # # # Draw the windows on the visualization image
        # # cv2.rectangle(
            # # out_img,
            # # (win_xleft_low,win_y_low),
            # # (win_xleft_high,win_y_high),
            # # color=(0,255,0), 
            # # thickness=2) 
        # # cv2.rectangle(
            # # out_img,
            # # (win_xright_low,win_y_low),
            # # (win_xright_high,win_y_high),
            # # color=(0,255,0), 
            # # thickness=2) 
        # # Identify the nonzero pixels in x and y within the window
        # good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        # (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        # good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        # (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # # Append these indices to the lists
        # left_lane_inds.append(good_left_inds)
        # right_lane_inds.append(good_right_inds)
        # # If you found > minpix pixels, recenter next window on their mean position
        # if len(good_left_inds) > minpix:
            # leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        # if len(good_right_inds) > minpix:        
            # rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # # Concatenate the arrays of indices
    # left_lane_inds = np.concatenate(left_lane_inds)
    # right_lane_inds = np.concatenate(right_lane_inds)

    # # Extract left and right line pixel positions
    # leftx = nonzerox[left_lane_inds]
    # lefty = nonzeroy[left_lane_inds] 
    # rightx = nonzerox[right_lane_inds]
    # righty = nonzeroy[right_lane_inds] 
    
    # return leftx, lefty, rightx, righty

    # # # Fit a second order polynomial to each
    # # left_fit = np.polyfit(lefty, leftx, 2)
    # # right_fit = np.polyfit(righty, rightx, 2)

    # # # Generate x and y values for plotting
    # # ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    # # left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    # # right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # # out_img[np.round(ploty), np.round(left_fitx)] = [255, 255, 0]
    # # out_img[np.round(ploty), np.round(right_fitx)] = [255, 255, 0]
    # # return out_img
    
