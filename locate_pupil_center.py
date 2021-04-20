import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from pylab import *

img = io.imread('data/frame/frame1.jpg')

# Left eye
print(img[1]/2)
img = np.around(img[0:img.shape[0], 0:int(img.shape[1] / 2)])

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_image = np.float32(gray_image)

# Note: at higher scales the isophotes curvature might degenerate with the inherent
# effect of losing important structures in the image.
gray_image = cv2.GaussianBlur(gray_image, (9, 9), 0)

# Calculate Isophotes curvature k where Lx and Ly are the first order derivatives
# of the luminance function L(x,y) in the x and y dimension, respectively.
# The sign of the isophote depends on the intensity of the outer side of the curve,
# so its vector direction will always point towards the highest change in the luminance.
Ly, Lx = np.gradient(gray_image)
Lxy, Lxx = np.gradient(Lx)
Lyy, Lyx = np.gradient(Ly)
Lvv = Ly ** 2 * Lxx - 2 * Lx * Lxy * Ly + Lx ** 2 * Lyy
Lw = Lx ** 2 + Ly ** 2
k = - Lvv / (Lw ** 1.5)

# The displacement vectors Dx and Dy, point to the estimated position of the centers.
Dx = -Lx * (Lw / Lvv)
Dy = -Ly * (Lw / Lvv)
magnitude_displacement = np.sqrt(Dx ** 2 + Dy ** 2)

# The curvedness indicates how curved a shape is.
curvedness = np.absolute(np.sqrt(Lxx ** 2 + 2 * Lxy ** 2 + Lyy ** 2))

# Sometimes the isophote curvature could assume extremely small or big values.
# This indicates that we are dealing with a “straight line” or a “single dot”
# isophote. Since the estimated radius to the isophote center would be too high
# to fall into the centermap or too little to move away from the originating pixel,
# the calculation of the displacement vectors in these extreme cases can simply
# be avoided.

minrad = 2
maxrad = 20

# A negative sign indicates a change in the direction of the gradient
# (i.e. from brighter to darker areas). Therefore, it is possible to
# discriminate between dark and bright centers by analyzing the sign of the
# curvature

center_map = np.zeros(gray_image.shape, gray_image.dtype)
(height, width) = center_map.shape
for y in range(height):
    for x in range(width):
        if Dx[y][x] == 0 and Dy[y][x] == 0:
            continue
        if (y + Dx[y][x]) > 0 and (x + Dy[y][x]) > 0:
            if (x + Dx[y][x]) < center_map.shape[1] and (y + Dy[y][x]) < center_map.shape[0] and k[y][x] < 0:
                if magnitude_displacement[y][x] >= minrad and magnitude_displacement[y][x] <= maxrad:
                    center_map[int(y + Dy[y][x])][int(x + Dx[y][x])] += curvedness[y][x]

# Since every vector gives a rough estimate of the center, the accumulator
# is convolved with a Gaussian kernel so that each cluster of
# votes will form a single center estimate.
center_map = cv2.GaussianBlur(center_map, (5, 5), 0)

# Maximum isocenter (MIC) in the centermap will be used as the most probable
# estimate for the soughtafter location of the center of the eye.
mic = np.unravel_index(np.argmax(center_map), center_map.shape)

# Draw it
fig = plt.figure()
plt.subplot(121), plt.imshow(gray_image, cmap='gray'), plt.title('Input')
fig.gca().add_artist(plt.Circle((mic[1], mic[0]), 2, color='r'))
plt.subplot(122), plt.imshow(center_map, cmap='gray'), plt.title('Output')
plt.show()

savefig('result/face/pupil_center.jpg')
