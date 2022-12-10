import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import os

img = cv2.imread('gilette1.jpg', cv2.IMREAD_GRAYSCALE)

thresh = 200
img_bw = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]

kernel = np.ones((5,5),np.uint8)

erosion = cv2.erode(img_bw,kernel,iterations = 1)
dilation = cv2.dilate(img_bw,kernel,iterations = 1)
opening = cv2.morphologyEx(img_bw, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(img_bw, cv2.MORPH_GRADIENT, kernel)
tophat = cv2.morphologyEx(img_bw, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(img_bw, cv2.MORPH_BLACKHAT, kernel)

width = 384
height = 384
dim = (width, height)

empty_img = np.ones(dim)

def cross(c1,c2):
    x1, y1 = -25, 0
    x2, y2 = 25, 0
    x3, y3 = 0, -25
    x4, y4 = 0, 25
    line_thickness = 2
    cv2.line(img, (x1 + c1, y1 + c2), (x2 + c1, y2 + c2), (0, 0, 0), thickness=line_thickness)
    cv2.line(img, (x3 + c1, y3 + c2), (x4 + c1, y4 + c2), (0, 0, 0), thickness=line_thickness)

cross(20,20)

resized1 = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
BLACK = (127,127,127)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1.1
font_color = BLACK
font_thickness = 2
text = 'Original Image'
x,y = 50,50
resized1 = cv2.putText(resized1, text, (x,y), font, font_size, font_color, font_thickness, cv2.LINE_AA)

resized2 = cv2.resize(img_bw, dim, interpolation = cv2.INTER_AREA)
BLACK = (127,127,127)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1.1
font_color = BLACK
font_thickness = 2
text = 'Binary Image'
x,y = 50,50
resized2 = cv2.putText(resized2, text, (x,y), font, font_size, font_color, font_thickness, cv2.LINE_AA)

resized3 = cv2.resize(erosion, dim, interpolation = cv2.INTER_AREA)
resized4 = cv2.resize(dilation, dim, interpolation = cv2.INTER_AREA)
resized5 = cv2.resize(opening, dim, interpolation = cv2.INTER_AREA)
resized6 = cv2.resize(closing, dim, interpolation = cv2.INTER_AREA)

numpy_horizontal1 = np.hstack((resized1, resized2, resized3, resized4, resized5))
numpy_horizontal2 = np.hstack((resized6, empty_img, empty_img, empty_img, empty_img))
numpy_horizontal3 = np.hstack((empty_img, empty_img, empty_img, empty_img, empty_img))
numpy_vertical = np.concatenate((numpy_horizontal1, numpy_horizontal2, numpy_horizontal3))

# cv2.namedWindow("ImageProcess", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("ImageProcess", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("ImageProcess", numpy_horizontal1)
cv2.waitKey(0)
cv2.destroyAllWindows()

os.system("cls")

# plt.imshow(im_bw)

# # Rectangular Kernel
# >>> cv.getStructuringElement(cv.MORPH_RECT,(5,5))
# array([[1, 1, 1, 1, 1],
#        [1, 1, 1, 1, 1],
#        [1, 1, 1, 1, 1],
#        [1, 1, 1, 1, 1],
#        [1, 1, 1, 1, 1]], dtype=uint8)
# # Elliptical Kernel
# >>> cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
# array([[0, 0, 1, 0, 0],
#        [1, 1, 1, 1, 1],
#        [1, 1, 1, 1, 1],
#        [1, 1, 1, 1, 1],
#        [0, 0, 1, 0, 0]], dtype=uint8)
# # Cross-shaped Kernel
# >>> cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
# array([[0, 0, 1, 0, 0],
#        [0, 0, 1, 0, 0],
#        [1, 1, 1, 1, 1],
#        [0, 0, 1, 0, 0],
#        [0, 0, 1, 0, 0]], dtype=uint8)