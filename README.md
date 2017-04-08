# Advanced Lane Finding

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[chessboards]: ./doc_imgs/chessboards.png "Chessboards"
[pipeline_diagram]: ./doc_imgs/pipeline_diagram.PNG "Pipeline"
[undistort]: ./doc_imgs/undistort.png "Distortion-correction"
[warped_process]: ./doc_imgs/warped_process.png "Perspective Transformation - process"
[warped]: ./doc_imgs/warped.png "Perspective Transformation"
[thresholds]: ./doc_imgs/thresholds.png "After Thresholds applied"
[histogram]: ./doc_imgs/histogram.png "Histogram"
[sliding_windows]: ./doc_imgs/sliding_windows.png "Sliding Windows"
[rcurve]: ./doc_imgs/rcurve.png "R Curve"
[output]: ./doc_imgs/output.png "Output"


## Camera Calibration

The camera was calibrated by using 20 chessboard photos taken from various directions.

Here is the result of undistorting an image:

![chessboards][chessboards]

## Pipeline

I used the following pipeline to detect lane lines from an image:

![pipeline_diagram][pipeline_diagram]

I created a Tensorflow-like framework which allowed me to try out several threshold arrangements. This framework also allowed me to obtain various stages of the processed image.

Take a look at "Find Threshold Parameters" notebook documents for some examples of its use.

*Note: In the following sections, for each "location in code", look at file* `image_pipeline.py`.

### 1. Example of a distortion-corrected image.

![Distortion-correction][undistort]

The undistorted image correctly shows the horizontal signboard.

**Location in code:** `ImagePipeline` class, `calibrate()` function and then the correction is applied in the `process()` function.

### 2. Perspective Transformation

Perspective transformation converts the image to a birdview perspective.

In order to find the right source point for this transformation, I mirrored the image and manually selected the four points correlated with the lane lines:

![Perspective Transformation - Process][warped_process]

Assuming the road the car was driven on follows the [US highway standard](https://en.wikipedia.org/wiki/Lane#Lane_width), the width of the lane in the picture above is 3.7 meters. I adjusted the transform destination matrix so that the distance between left and right lanes was 700 pixels (calculated from the point closest to the car).

This is how the final warped image looks like in comparison to the original undistorted image:

![Perspective Transformation][warped]

**Location in code:** `ImagePipeline` class, `process()` function.

### 3. Thresholds

Intuitively, we want to get both yellow and white colored elements from the image. InRange yellow and white thresholds were applied for that purpose. I then combined these lines with a Magnified Sobel-Y threshold to get only the lane lines.

![After Thresholds applied][thresholds]

**Location in code:** `Threshold` class, which can group together multiple objects of `ThresholdOperation` class.

### 4. Lane Lines Identification

To decide on where to draw the lane lines, the system creates a histogram of all activated pixels of each y-axis section.

![Histogram][histogram]

It then stacks windows based on the next histograms until the entire y sections are covered with 9 sliding windows.

![Sliding Windows][sliding_windows]

**Location in code:** `FindLinesSlidingWindows` class, `_calculate_fits()` function.

For subsequent frames, we first try using the polynomial fit from previous frame to find pixels in a frame. If none found, or the resulting lines were not "plausible", recalculate using the sliding windows technique above.

**Location in code:** `FindLinesSlidingWindows` class, `_reuse_fits()` function. For plausibility testing, look into function `_check_lines`.

Another interesting case here is if even after recalculation, we were not able to find plausible lines. In that case, use the exact same lines from the previous frame.

### 5. Radius of Curvature and Position to Center

Curvature to the circle's center is calculated with the following formula:

![R Curve][rcurve]

And distance to center of lane is also calculated and presented in the output. Here is an example of the final output:

![Output][output]

**Location in code:** `calculate_curvature_radius()` function. Final presentation is done by an object of `Annotate` class.

## Video

For video outputs, look at the root directory of this project, `[video_name]_processed.mp4` files.

## Discussion

The main problem I faced in my implementation of this project was the exploration to find a good set of thresholds. The file `0. Initial Experiments.ipynb` file kept some traces of initial explorations that were done with quick copy-pasted code pieces which did not result in a decent result.

I then created a framework to help me with finding good parameters for my thresholds. My hope is to be able to expand this framework later in the future after completing this program, perhaps using it in the previous Behavioral Cloning project?

The final pipeline was able to correctly annotate the lane lines in the `project_video.mp4` and `challenge_video.mp4`, but it did not perform too well in the `harder_challenge_video.mp4`, especially when dealing with sharp turns and when a motorcycle rider cut through the path. Maybe some techniques from the next project "Vehicle Detection and Tracking" can be used to deal with this problem, but I am not quite sure how just yet.