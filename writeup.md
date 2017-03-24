# **Finding Lane Lines on the Road** 

## Goal

Using the provided template ipython notebook, a line finding and smoothing
algorithm had to be developed. The input consisted of the
videos provided under `test_videos` and the images from the folder `test_images`.
The images and videos show a typical US highway
with mostly good lane markings, sparse traffic and only minor curvature.

## Reflection

### 1. Lane Finding Pipeline

My line finding algorithm builds upon some of the pieces tought in the introduction
but takes more advanced steps for color filtering, line fitting and smoothing.

First the input image is transformed into the HSV color space.
In this color space it is easier to specify color ranges describing
the typical color of white and yellow lanes.
The function `generate_color_mask` takes the HSV input image and
a set of color selectors and produces a binary mask.

![maskedImage]

Then the canny edge detection is applied to the mask, followed by a
Gaussian blur operation.

![edges]

Finally, a region of interest is cut out from the image.
The parameters describing the polygon vary based on the resolution
of the images and videos.

![roi]

This image is then passed on to the Hough lines transformation.
The single line segments detected by the function are converted
into a custom `Line` objects. The Line class pre-calculates
the slope and intersect of the line, as well as the length.
It provides convenience methods to calculate a given point
on the line in x-y and y-x coordinates. Additionally, it contains a
method `is_valid` which checks if the slope of the line is too small,
i.e. if the line is mostly horizontal.

![detectedLines]

The resulting array of lines is divided into two groups,
one for the left, and one for the right lane marking.
Here, the slope and the lower x-coordinate are taken into account.
This filters out some lines that have, for instance, a positive slope,
but lie on the left-hand side of the image.

The line segments in each group are then combined by fitting a first-order polynomial,
i.e. a  line, through the points. To give points nearer to the ego-vehicle
a higher influence than far-away points I made use of the weight parameter
of the fitting function. The weights are the product of the length
of the single line segment and the y coordinate nearest to the camera.
By this long lines and near lines have a higher influence on the resulting
slope and intersect.

To smooth the lanes throughout consecutive frames, I implemented a simple
linear Kalman filter where the state and measurement space are identical.
By adjusting the process noise of the filter and the measurement uncertainty
it is possible to get a smooth output, without having to hold buffers of
previous information.

Finally, the filtered results are drawn onto the original image.

![result]

### 2. Shortcomings of the current pipeline

The approach presented in the previous section has several shortcomings.
It relies on masking parts of the image by their color.
For the given data, this worked very well, but may fail for other lighting conditions
or scenarios.
Further it is only possible to track at most two lane markings, one to each side of the car.
For some driving functions it may be necessary to have information about
the second next lane markings.
Finally, only a linear lane model has been implemented.
For steering on a highway, assuming a straight road may be a good approximation,
but going to a second or third order polynomial
(such as professional systems like MobilEye do),
would improve the quality of the output and thus
improve systems using the lane information.


### 3. Possible improvements

As the input only consisted of videos it was not possible to implement
an ego-motion compensation.
In a scenario where the vehicle departs the lane, the current implementation
would have a lag in the filtered lane slopes and positions.
Integrating the ego-motion between two consecutive frames allows to compensate for the motion of the vehicle and could improve the prediction step of the Kalman filter.
By this the possible lag would be reduced as well as the uncertainty estimated by the Kalman filter.

Further it would be beneficial to remove the explicit coding of left and right lanes
and generalize it to an n-lane detection and tracking system.
Based on such a system, it would be possible to extract features of the lanes,
such as dashed-ness, color and quality.

[maskedImage]: doc/maskedImage.png "Masked Image"
[edges]: doc/edges.png "Edges after Canny and Gaussian blur"
[roi]: doc/roi.png "Region of interest mask applied"
[detectedLines]: doc/detected_lines.png "Hough lines detected in the image"
[result]: doc/result.png "Final result"