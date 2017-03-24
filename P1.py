
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# 
# ## Project: **Finding Lane Lines on the Road** 
# ***
# In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 
# 
# Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.
# 
# In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.
# 
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.
# 
# **Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**
# 
# ---

# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**
# 
# ---
# 
# <figure>
#  <img src="examples/line-segments-example.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
#  </figcaption>
# </figure>
#  <p></p> 
# <figure>
#  <img src="examples/laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
#  </figcaption>
# </figure>

# **Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, see [this forum post](https://carnd-forums.udacity.com/cq/viewquestion.action?spaceKey=CAR&id=29496372&questionTitle=finding-lanes---import-cv2-fails-even-though-python-in-the-terminal-window-has-no-problem-with-import-cv2) for more troubleshooting tips.**  

# ## Import Packages

# In[1]:

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from scipy.linalg import inv
#get_ipython().magic('matplotlib inline')

# In[24]:

import math

class Line(object):
    SLOPE_THRESHOLD = 0.4
    def __init__(self, x1, y1, x2, y2):
        if x1 > x2: (x1, y1), (x2, y2) = (x2, y2), (x1, y1)
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.m = (y2 - y1) / (x2 - x1)
        self.n = y1 - self.m * x1
        self.length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    
    def __repr__(self):
        return 'Line: x1={}, y1={}, x2={}, y2={}, m={}, n={}, length={}'.format(
            self.x1, self.y1, self.x2, self.y2, round(self.m, 2), round(self.n, 2), round(self.length, 2))

    def get_x_coord(self, y):
        return int((y - self.n) / self.m)
    
    def get_y_coord(self, x):
        return int(self.m * x + self.n)
    
    def is_valid(self):
        if abs(self.m) < self.SLOPE_THRESHOLD:
            return False
        return True

    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    image = img.copy()
    for line in lines:
        cv2.line(image, (line.x1, line.y1), (line.x2, line.y2), color, thickness)
    return image

def scatter_on_image(img, lines, color=[255, 0, 0], radius=2):
    image = img.copy()
    for line in lines:
        cv2.circle(image, (line.x1, line.y1), radius, color)
        cv2.circle(image, (line.x2, line.y2), radius, color)
    return image

def to_line(houghLine):
    """
    `houghLine` should be a single line output by the HoughLinesP function.

    Returns a Line.
    """
    x1,y1,x2,y2 = houghLine[0]
    return Line(x1, y1, x2, y2)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns a list of hough lines.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    slopes = [to_line(line) for line in lines]
    return slopes

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def generate_color_mask(image, colors):
    masks = []
    for color in colors:
        mask = cv2.inRange(image, color["lower"], color["upper"])
        masks.append(mask)
    return cv2.add(*masks)

def apply_mask(image, mask):
    masked_image = np.zeros_like(image)
    for i in range(3): 
        masked_image[:,:,i] = mask.copy()
    return masked_image

def fit_line(lines):
    x = np.array([[line.x1, line.x2] for line in lines]).flatten()
    y = np.array([[line.y1, line.y2] for line in lines]).flatten()
    w = np.array([[line.y2 *line.length, line.y2 * line.length] for line in lines]).flatten()
    return np.polyfit(x, y, 1, w=w)

# ## Test Images
# 
# Build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# In[5]:

testImages = os.listdir("test_images/")


# ## Build a Lane Finding Pipeline
# 
# 

# Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.
# 
# Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.

# In[106]:

# hough transform settings
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 5     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 15 #minimum number of pixels making up a line
max_line_gap = 8    # maximum gap in pixels between connectable line segments


# ## Test on Videos
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# 
# We can test our solution on two provided videos:
# 
# `solidWhiteRight.mp4`
# 
# `solidYellowLeft.mp4`
# 
# **Note: if you get an `import error` when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt. Also, check out [this forum post](https://carnd-forums.udacity.com/questions/22677062/answers/22677109) for more troubleshooting tips.**
# 
# **If you get an error that looks like this:**
# ```
# NeedDownloadError: Need ffmpeg exe. 
# You can download it by calling: 
# imageio.plugins.ffmpeg.download()
# ```
# **Follow the instructions in the error message and check out [this forum post](https://carnd-forums.udacity.com/display/CAR/questions/26218840/import-videofileclip-error) for more troubleshooting tips across operating systems.**

# In[ ]:


class KalmanFilter(object):
    def __init__(self, m, n):
        self.x = np.array([m, n])
        self.P = np.array([[0.03, 0.0],[0.0, 1000]]) # initial covariance
        self.F = np.array([[1.0, 0.0],[0.0, 1.0]])   # state transition matrix
        self.H = np.diag([1.0, 1.0])                 # measurement function, in this case unit matrix as measurement space equals state space
        self.R = np.diag([0.02, 1000])               # measurement uncertainty
        self.Q = np.array([[0.001, 0],[0, 10]])      # process noise

    def predict_and_update(self, z):
        # predict
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, self.P).dot(self.F.T) + self.Q
        
        #update
        S = np.dot(self.H, self.P).dot(self.H.T) + self.R
        K = np.dot(self.P, self.H.T).dot(inv(S))
        y = z - np.dot(self.H, self.x)
        self.x += np.dot(K, y)
        self.P = self.P - np.dot(K, self.H).dot(self.P)

leftLaneFilter = None
rightLaneFilter = None

def get_vertices_white_and_yellow(shape):
    return np.array([[(75, shape[0]),(427, 326), (520, 326), (shape[1] - 30, shape[0])]], dtype=np.int32)

def get_vertices_challange(shape):
    return np.array([[(190, shape[0]),(590, 450), (736, 450), (shape[1] - 30, shape[0])]], dtype=np.int32)

get_vertices = get_vertices_white_and_yellow
first = True
def process_image(orig):
    global leftLaneFilter, rightLaneFilter, first
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # you should return the final output (image where lines are drawn on lanes)
    
    image = cv2.cvtColor(orig, cv2.COLOR_RGB2HSV)
    shape = image.shape

    whiteLanes = {
        "lower": np.array([0, 0, 204]),
        "upper": np.array([180, 26, 255])
    }
    yellowLanes = {
        "lower": np.array([18, 51, 77]),
        "upper": np.array([33, 255, 255])
    }
    mask = generate_color_mask(image, [whiteLanes, yellowLanes])
    
    maskedImage = apply_mask(image, mask)
    
    edges = gaussian_blur(canny(maskedImage, 280, 360),5)

    vertices = get_vertices(shape)

    roi = region_of_interest(edges, vertices)

    lines = hough_lines(roi, rho, theta, threshold, min_line_length, max_line_gap)
    
    linesWithNegativeSlope = [line for line in lines if line.is_valid() and line.m < 0.0 and line.x2 < shape[1] / 2.0]
    linesWithPositiveSlope = [line for line in lines if line.is_valid() and line.m >= 0.0 and line.x2 >= shape[1] / 2.0]

    leftLaneM, leftLaneN = fit_line(linesWithNegativeSlope)
    rightLaneM, rightLaneN = fit_line(linesWithPositiveSlope)

    if (leftLaneFilter is None):
        leftLaneFilter = KalmanFilter(leftLaneM, leftLaneN)
        rightLaneFilter = KalmanFilter(rightLaneM, rightLaneN)
    else:
        leftLaneFilter.predict_and_update(np.array([leftLaneM, leftLaneN]))
        rightLaneFilter.predict_and_update(np.array([rightLaneM, rightLaneN]))

    leftLaneM, leftLaneN = leftLaneFilter.x
    rightLaneM, rightLaneN = rightLaneFilter.x

    line_img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    maximumDistance = vertices[0,1,1]
    cv2.line(line_img, (int((maximumDistance - leftLaneN)/leftLaneM), maximumDistance), (int((shape[0] - leftLaneN)/leftLaneM), shape[0]), [255,0,0], 5)
    cv2.line(line_img, (int((maximumDistance - rightLaneN)/rightLaneM), maximumDistance), (int((shape[0] - rightLaneN)/rightLaneM), shape[0]), [255,0,0], 5)

    result = weighted_img(line_img, orig)

    #if first:
    #    first = False
    #    plt.imshow(maskedImage)
    #    plt.savefig("maskedImage.png")
    #    plt.imshow(edges, cmap='Greys_r')
    #    plt.savefig("edges.png")
    #    plt.imshow(roi, cmap='Greys_r')
    #    plt.savefig("roi.png")
    #    plt.imshow(draw_lines(orig, lines))
    #    plt.savefig("detected_lines.png")
    #    plt.imshow(result)
    #    plt.savefig("result.png")

    #f, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2,2)
    #ax1.imshow(maskedImage)
    #ax2.imshow(edges)
    #ax3.imshow(roi)
    #ax4.imshow(draw_lines(orig, lines))
    ##ax4.imshow(scatter_on_image(orig, linesWithNegativeSlope))
    #plt.show()
    return result

white_output = 'test_videos_output/solidWhiteRight.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)

yellow_output = 'test_videos_output/solidYellowLeft.mp4'
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)

get_vertices = get_vertices_challange

challenge_output = 'test_videos_output/challenge.mp4'
clip2 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip2.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)

