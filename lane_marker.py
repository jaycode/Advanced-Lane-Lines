"""
(Unused) Lane marker from project 1 of Self Driving Car.

Following code lines calculate and draw hough lines, and apply them
into a given image:

```
gray_image = grayscale(raw_image)

image = gaussian_blur(gray_image, kernel_size=3)

image = canny(image, low_threshold=50, high_threshold=150)

imshape = image.shape
vertices = np.array([[(0.1*imshape[1], 1*imshape[0]),
                      (0.45*imshape[1], 0.60*imshape[0]),
                      (0.57*imshape[1], 0.60*imshape[0]),
                      (1*imshape[1], 1*imshape[0])]], dtype=np.int32)
image = region_of_interest(image, vertices)

l, r = hough_lines(image,
                   rho=2,
                   theta=np.pi/180,
                   threshold=50,
                   min_line_len=100,
                   max_line_gap=150,
                   extrapolate=True)

lcolor = [255, 0, 0]
lcolor = [0, 255, 0]
thickness = 1
cv2.line(img, l[0], l[1], lcolor, thickness)
cv2.line(img, r[0], r[1], rcolor, thickness)

result = weighted_img(hough, raw_image)
```
"""

import cv2
import math
import numpy as np

class LaneMarker(object):
    def __init__(self):
        # Stores image dimensions after running region_of_interest
        self.img_dim = []
        
    def extrapolate(self, lines):
        """ Combine several lines and extrapolate by y.
        """
        y = max(self.img_dim[:, 1])
        x1 = 0
        y1 = 0
        x2 = 0
        y2 = 0
        if len(lines) > 0:
            avg_x1 = int(sum(map(lambda p: p[0], lines)) / len(lines))
            avg_y1 = int(sum(map(lambda p: p[1], lines)) / len(lines))
            avg_x2 = int(sum(map(lambda p: p[2], lines)) / len(lines))
            avg_y2 = int(sum(map(lambda p: p[3], lines)) / len(lines))
            if y:
                slope = (avg_x2-avg_x1)/(avg_y2-avg_y1)
                # Decide whether to replace point 1 or point 2, based on which one closer to bottom
                if avg_y1 > avg_y2:
                    x1 = avg_x2
                    y1 = avg_y2
                else:
                    x1 = avg_x1
                    y1 = avg_y1
                x2 = int(slope*(y-y1) + x1)
                y2 = y
            else:
                x1 = avg_x1
                y1 = avg_y1
                x2 = avg_x2
                y2 = avg_y2

        return (x1, y1, x2, y2)

    def group_and_extrapolate(self, img, lines, slope_threshold=0.5):
        """ Group and extrapolate lines

        First, we group all left and right lines, then calculate the middle lines for each.
        Args:
            y: Extrapolate x based on this y value.
            slope_threshold: We don't want to include lines with slope lower than this.
        """
        left_lines = []
        right_lines = []
        for line in lines:
            for x1,y1,x2,y2 in line:
                slope = (y2-y1)/(x2-x1)
                # Use square instead of abs for slightly faster calculation
                if slope**2 > slope_threshold**2:
                    if slope < 0:
                        # left lane
                        left_lines.append((x1, y1, x2, y2))
                    else:
                        # right lane
                        right_lines.append((x1, y1, x2, y2))

        lx1, ly1, lx2, ly2 = self.extrapolate(left_lines)
        rx1, ry1, rx2, ry2 = self.extrapolate(right_lines)

        # Use the furthest point
        if ly1 != 0 and ry1 != 0:
            if ly1 < ry1:
                ry1 = ly1
            else:
                ly1 = ry1

        return [[(lx1, ly1), (lx2, ly2)],
                [(rx1, ry1), (rx2, ry2)]]

    def grayscale(self, img):
        """Applies the Grayscale transform
        This will return an image with only one color channel
        but NOTE: to see the returned image as grayscale
        you should call plt.imshow(gray, cmap='gray')"""
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def canny(self, img, low_threshold, high_threshold):
        """Applies the Canny transform"""
        return cv2.Canny(img, low_threshold, high_threshold)

    def gaussian_blur(self, img, kernel_size):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def region_of_interest(self, img, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        self.img_dim = vertices[0]
        
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


    def hough_lines(self, img, rho, theta, threshold, min_line_len, max_line_gap, slope_threshold=0.55):
        """
        `img` should be the output of a Canny transform.

        Returns an image with hough lines drawn.
        """
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        line_img = np.zeros((*img.shape, 3), dtype=np.uint8)

        return self.group_and_extrapolate(line_img, lines, slope_threshold=slope_threshold)

    # Python 3 has support for cool math symbols.

    def weighted_img(self, img, initial_img, α=0.8, β=1., λ=0.):
        """
        `img` is the output of the hough_lines(), An image with lines drawn on it.
        Should be a blank image (all black) with lines drawn on it.

        `initial_img` should be the image before any processing.

        The result image is computed as follows:

        initial_img * α + img * β + λ
        NOTE: initial_img and img must be the same shape!
        """
        return cv2.addWeighted(initial_img, α, img, β, λ)