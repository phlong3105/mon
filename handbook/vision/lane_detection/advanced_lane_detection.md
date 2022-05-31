<div align="center">
<br><br>
<div>
	<a href="https://github.com/phlong3105/one/blob/master/handbook/README.md"><img src="../../data/badge/handbook_home.svg"></a>
</div>

Advanced Lane Detection
=============================
<div>
	<p>Oana Gaskey</p>
	<p>GitHub 2020</p>
</div>

<div align="center">
    <a href="https://github.com/OanaGaskey/Advanced-Lane-Detection"><img src="../../data/badge/paper_code.svg"></a>
</div>
</div>


## Highlight
- Computer Vision algorithm to compute road curvature and lane vehicle offset using OpenCV Image Processing, Camera Calibration, Perspective Transform, Color Masks, Sobels and Polynomial Fit. 
- Using a video recording of highway driving, this project's goal is to compute the radius of the curvature of the road. Curved roads are a more challenging task than straight ones. To correctly compute the curvature, the lane lines need to be identified but on top of that, the images needs to be undistorted. Image transformation is necessary for camera calibration and for perspective transform to obtain a bird's eye view of the road.

<div align="center">
	<img width="400" src="data/lane_detection.gif">
</div>


## Method

### 1. Camera Calibration
- Optic distortion is a physical phenomenon that occurs in image recording, in which straight lines are projected as slightly curved ones when perceived through camera lenses. The highway driving video is recorded using the front facing camera on the car and the images are distorted. 
- The distortion coefficients are specific to each camera and can be calculated using known geometrical forms.
- Chessboard images captured with the embedded camera are provided in camera_cal folder. The advantage of these images is that they have high contrast and known geometry. The images provided present 9 * 6 corners to work with.
- OpenCV `undistort` function is used to transform the images using the camera matrix and distortion coefficients.

<div align="center">
	<img width="700" src="data/undistorted_chessboard.png">
</div>

- The result of the camera calibration technique is visible when comparing these pictures. While on the chessboard picture the distortion is more obvious, on the road picture it's more subtle. Nevertheless, an undistorted picture would lead to an incorrect road curvature calculation.

<div align="center">
	<img width="700" src="data/undistorted_road.png">
</div>

### 2. Perspective Transform from Camera Angle to Bird's Eye View
- To calucluate curvature, the ideal perspective is a bird's eye view. This means that the road is perceived from above, instead of at an angle through the vehicle's windshield.
- This perspective transform is computed using a straight lane scenario and prior common knowledge that the lane lines are in fact parallel. Source and destination points are identified directly from the image for the perspective transform.

<div align="center">
	<img width="700" src="data/ending_points.png">
</div>

- OpenCV provides perspective transform functions to calculate the transformation matrix for the images given the source and destination points. Using `warpPerspective` function, the bird's eye view perspective transform is performed.

<div align="center">
	<img width="700" src="data/warp_perspective.png">
</div>

### 3. Process Binary Thresholded Images
- The objective is to process the image in such a way that the lane line pixels are preserved and easily differentiated from the road. Four transformations are applied and then combined.
- The first transformation takes the x `sobel` on the gray-scaled image. This represents the derivative in the x direction and helps detect lines that tend to be vertical. Only the values above a minimum threshold are kept.
- The second transformation selects the white pixels in the gray scaled image. White is defined by values between 200 and 255 which were picked using trial and error on the given pictures.
- The third transformation is on the saturation component using the HLS colorspace. This is particularly important to detect yellow lines on light concrete road.
- The fourth transformation is on the hue component with values from 10 to 25, which were identified as corresponding to yellow.

<div align="center">
	<img width="800" src="data/binary_threshold.png">
</div>

### 4. Lane Line Detection Using Histogram
- The lane line detection is performed on binary thresholded images that have already been undistorted and warped. Initially a histogram is computed on the image. This means that the pixel values are summed on each column to detect the most probable x position of left and right lane lines.
- Starting with these base positions on the bottom of the image, the sliding window method is applied going upwards searching for line pixels. Lane pixels are considered when the x and y coordinates are within the area defined by the window. When enough pixels are detected to be confident they are part of a line, their average position is computed and kept as starting point for the next upward window.
- All these pixels are put together in a list of their x and y coordinates. This is done symmetrically on both lane lines. `leftx`, `lefty`, `rightx`, `righty` pixel positions are returned from the function and afterwards, a second-degree polynomial is fitted on each left and right side to find the best line fit of the selected pixels.
- Here, the identified left and right line pixels are marked in red and blue respectively. The second degree polynomial is traced on the resulting image.

<div align="center">
	<img width="800" src="data/fit_poly.png">
</div>

### 5. Detection of Lane Lines Based on Previous Cycle
- To speed up the lane line search from one video frame to the next, information from the previous cycle is used. It is more likely that the next image will have lane lines in proximity to the previous lane lines. This is where the polynomial fit for the left line and right line of the previous image are used to define the searching area. 
- The sliding window method is still used, but instead of starting with the histogram’s peak points, the search is conducted along the previous lines with a given margin for the window’s width.
- The search returns `leftx`, `lefty`, `rightx`, `righty` pixel coordinates that are fitted with a second degree polynomial function for each left and right side.

<div align="center">
	<img width="800" src="data/prev_poly.png">
</div>

### 6. Calculate Vehicle Position and Curve Radius
- To calculate the radius and the vehicle's position on the road in meters, scaling factors are needed to convert from pixels. The corresponding scaling values are 30 meters to 720 pixels in the y direction and 3.7 meters to 700 pixels in the x dimension.
- A polynomial fit is used to make the conversion. Using the x coordinates of the aligned pixels from the fitted line of each right and left lane line, the conversion factors are applied and polynomial fit is performed on each.
- The radius of the curvature is calculated using the y point at the bottom of the image. To calculate the vehicle’s position, the polynomial fit in pixels is used to determine the x position of the left and right lane corresponding to the y at the bottom of the image.
- The average of these two values gives the position of the center of the lane in the image. If the lane’s center is shifted to the right by `nbp` amount of pixels that means that the car is shifted to the left by `nbp * xm_per_pix meters`. This is based on the assumption that the camera is mounted on the central axis of the vehicle.


## Results


## Citation
