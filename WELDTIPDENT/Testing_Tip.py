import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import os

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

# get_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# ret, img = cap.read()

# cap.release()

D50_A30 = "D-50_A-30"
Tip_Laser_Img = "tip-laser-1.jpg"

img = cv2.imread("C:\\Users\\hsynk\\Desktop\\BVISIO\POC\\ImageProcess\\Data Sets\\" + D50_A30 + "\\" + Tip_Laser_Img)

rotated_img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
cropped_img = rotated_img[1000:2500, 500:2000]

cropped_img = cv2.GaussianBlur(cropped_img, (15, 15), 100)

hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
# Threshold of blue in HSV space
lower_red = np.array([160,100,50])
upper_red = np.array([180,255,255])
# preparing the mask to overlay
mask = cv2.inRange(hsv, lower_red, upper_red)
# The black region in the mask has the value of 0,
# so when multiplied with original image removes all non-blue regions
filter_img = cv2.bitwise_and(cropped_img, cropped_img, mask = mask)

gray_img = cv2.cvtColor(filter_img, cv2.COLOR_BGR2GRAY)

width = 384
height = 384
dim = (width, height)

# overlay = np.zeros([gray_img.shape[0], gray_img.shape[1], 3], dtype = np.uint8)
empty_img = np.zeros(dim)

thresh = 13
bw_img = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))

opening_img = cv2.morphologyEx(bw_img, cv2.MORPH_OPEN, kernel)
closing_img = cv2.morphologyEx(opening_img, cv2.MORPH_CLOSE, kernel)
dilate_img = cv2.dilate(closing_img,kernel,iterations = 3)
erode_img = cv2.erode(dilate_img,kernel,iterations = 3)


resized1 = cv2.resize(bw_img, dim, interpolation = cv2.INTER_AREA)
BLACK = (127,127,127)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1.1
font_color = BLACK
font_thickness = 2
text = 'BW Image'
x,y = 50,50
resized1 = cv2.putText(resized1, text, (x,y), font, font_size, font_color, font_thickness, cv2.LINE_AA)

resized2 = cv2.resize(opening_img, dim, interpolation = cv2.INTER_AREA)
BLACK = (127,127,127)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1.1
font_color = BLACK
font_thickness = 2
text = 'Opening Image'
x,y = 50,50
resized2 = cv2.putText(resized2, text, (x,y), font, font_size, font_color, font_thickness, cv2.LINE_AA)

resized3 = cv2.resize(closing_img, dim, interpolation = cv2.INTER_AREA)
BLACK = (127,127,127)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1.1
font_color = BLACK
font_thickness = 2
text = 'Closing Image'
x,y = 50,50
resized3 = cv2.putText(resized3, text, (x,y), font, font_size, font_color, font_thickness, cv2.LINE_AA)

resized4 = cv2.resize(dilate_img, dim, interpolation = cv2.INTER_AREA)
BLACK = (127,127,127)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1.1
font_color = BLACK
font_thickness = 2
text = 'Dilate Image'
x,y = 50,50
resized4 = cv2.putText(resized4, text, (x,y), font, font_size, font_color, font_thickness, cv2.LINE_AA)

resized5 = cv2.resize(erode_img, dim, interpolation = cv2.INTER_AREA)
BLACK = (127,127,127)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1.1
font_color = BLACK
font_thickness = 2
text = 'Erode Image'
x,y = 50,50
resized5 = cv2.putText(resized5, text, (x,y), font, font_size, font_color, font_thickness, cv2.LINE_AA)

resized6 = cv2.resize(empty_img, dim, interpolation = cv2.INTER_AREA)
BLACK = (127,127,127)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1.1
font_color = BLACK
font_thickness = 2
text = "Result Image"
x,y = 50,50
resized6 = cv2.putText(resized6, text, (x,y), font, font_size, font_color, font_thickness, cv2.LINE_AA)

numpy_horizontal1 = np.hstack((resized1, resized2, resized3))
numpy_horizontal2 = np.hstack((resized4, resized5, resized6))
numpy_vertical1 = np.vstack((numpy_horizontal1, numpy_horizontal2))

#######################################################################################################################

resized7 = cv2.resize(cropped_img, dim, interpolation = cv2.INTER_AREA)
BLACK = (127,127,127)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1.1
font_color = BLACK
font_thickness = 2
text = 'Crop Image'
x,y = 50,50
resized7 = cv2.putText(resized7, text, (x,y), font, font_size, font_color, font_thickness, cv2.LINE_AA)

resized8 = cv2.resize(filter_img, dim, interpolation = cv2.INTER_AREA)
BLACK = (127,127,127)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1.1
font_color = BLACK
font_thickness = 2
text = 'Filter Image'
x,y = 50,50
resized8 = cv2.putText(resized8, text, (x,y), font, font_size, font_color, font_thickness, cv2.LINE_AA)

resized9 = cv2.resize(cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB) * 255, dim, interpolation = cv2.INTER_AREA)
BLACK = (127,127,127)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1.1
font_color = BLACK
font_thickness = 2
text = 'Gray Image'
x,y = 50,50
resized9 = cv2.putText(resized9, text, (x,y), font, font_size, font_color, font_thickness, cv2.LINE_AA)

numpy_horizontal3 = np.hstack((resized7, resized8, resized9))

cv2.imshow("ImageProcess", numpy_vertical1)
cv2.imshow("Image", numpy_horizontal3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imwrite('opencv.png', numpy_horizontal1)
# cv2.cvtColor(erode_img, cv2.COLOR_GRAY2RGB) * 255