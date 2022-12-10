import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import os
from scipy.interpolate import interp1d

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

blurred_img = cv2.blur(cropped_img, (30, 30))

hsv = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)
# Threshold of blue in HSV space
lower_red = np.array([160,100,64])
upper_red = np.array([180,255,255])
# preparing the mask to overlay
mask = cv2.inRange(hsv, lower_red, upper_red)
# The black region in the mask has the value of 0,
# so when multiplied with original image removes all non-blue regions
filter_img = cv2.bitwise_and(blurred_img, blurred_img, mask = mask)

gray_img = cv2.cvtColor(filter_img, cv2.COLOR_BGR2GRAY)

width = 384
height = 384
dim = (width, height)

overlay = np.zeros([gray_img.shape[0], gray_img.shape[1], 3], dtype = np.uint8)
empty_img = np.zeros(dim)

thresh = 13
bw_img = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY)[1]

kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))

closing_img = cv2.morphologyEx(bw_img, cv2.MORPH_CLOSE, kernel2)
opening_img = cv2.morphologyEx(closing_img, cv2.MORPH_OPEN, kernel1)
dilate_img = cv2.dilate(opening_img,kernel1,iterations = 3)
erode_img = cv2.erode(dilate_img,kernel1,iterations = 2)
final_img = erode_img;

img_x = 1500
center_y = 750

red_line_top_x = []
red_line_top_y = []

for j in range(center_y,0,-1):
    for i in range(img_x - 1,0,-1):
        if final_img[j][i] > 127:
            if i >= 500:
                red_line_top_x.append(i)
                red_line_top_y.append(j)
                i = -1
                j = -1
            
cv2.circle(overlay, (max(red_line_top_x),red_line_top_y[np.argmax(red_line_top_x)]), 10, (0, 0, 255), -1)
top_tip_y = red_line_top_y[np.argmax(red_line_top_x)]

red_line_bottom_x = []
red_line_bottom_y = []

for j in range(center_y,1499,1):
    for i in range(img_x - 1,0,-1):
        if final_img[j][i] > 127:
            if i >= 500:
                red_line_bottom_x.append(i)
                red_line_bottom_y.append(j)
                i = -1
                j = 1499
            
cv2.circle(overlay, (max(red_line_bottom_x),red_line_bottom_y[np.argmax(red_line_bottom_x)]), 10, (0, 0, 255), -1)
bottom_tip_y = red_line_bottom_y[np.argmax(red_line_bottom_x)]

tip_length_px = bottom_tip_y - top_tip_y
tip_lenght_mm = tip_length_px * 0.015

cropped_red_line_top_x = red_line_top_x[:np.argmax(red_line_top_x)]
cropped_red_line_bottom_x = red_line_bottom_x[:np.argmax(red_line_bottom_x)]
cropped_red_line_bottom_x.reverse()
tip_arc = cropped_red_line_bottom_x + cropped_red_line_top_x

tip_depth_px = max(tip_arc) - min(tip_arc)
tip_depth_mm = tip_depth_px * 0.015


backtorgbbw = cv2.cvtColor(bw_img,cv2.COLOR_GRAY2RGB)

backtorgb = cv2.cvtColor(final_img,cv2.COLOR_GRAY2RGB)
backtorgb = cv2.resize(backtorgb, dim, interpolation = cv2.INTER_AREA)
overlay_resized = cv2.resize(overlay, dim, interpolation = cv2.INTER_AREA)

alpha = 0.5
result_img = cv2.addWeighted(overlay_resized, alpha, backtorgb, 1 - alpha, 0)

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
font_size = 0.8
font_color = BLACK
font_thickness = 2
text = "Tip L,D: " + str("{:.2f}".format(round(tip_lenght_mm, 2))) + " , " + str("{:.2f}".format(round(tip_depth_mm, 2)))
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

resized9 = cv2.resize(result_img, dim, interpolation = cv2.INTER_AREA)
BLACK = (127,127,127)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1.1
font_color = BLACK
font_thickness = 2
text = 'Result Image'
x,y = 50,50
resized9 = cv2.putText(resized9, text, (x,y), font, font_size, font_color, font_thickness, cv2.LINE_AA)

resized10 = cv2.resize(blurred_img, dim, interpolation = cv2.INTER_AREA)
BLACK = (127,127,127)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1.1
font_color = BLACK
font_thickness = 2
text = 'Blurred Image'
x,y = 50,50
resized10 = cv2.putText(resized10, text, (x,y), font, font_size, font_color, font_thickness, cv2.LINE_AA)

numpy_horizontal3 = np.hstack((resized7, resized10, resized8, resized9))

#######################################################################################################################

# tip_arc.append(0)
plot_x = np.linspace(0, len(tip_arc)-1, num=len(tip_arc), endpoint=True)
plot_x_new = np.linspace(0, len(tip_arc)-1, num=2*len(tip_arc), endpoint=True)
f1 = interp1d(plot_x, tip_arc, kind='cubic')
weights = np.polyfit(plot_x, tip_arc, 2)
f2 = np.poly1d(weights)
test = f1(plot_x_new)
plt.plot(plot_x, tip_arc, 'o', plot_x_new, f2(plot_x_new), '-')
plt.show()

#######################################################################################################################

# cv2.namedWindow("ImageProcess", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("ImageProcess", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.imshow("ImageProcess", numpy_vertical1)
cv2.imshow("Image", numpy_horizontal3)
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

# erosion = cv2.erode(img_bw,kernel,iterations = 1)
# dilation = cv2.dilate(img_bw,kernel,iterations = 1)
# opening = cv2.morphologyEx(img_bw, cv2.MORPH_OPEN, kernel)
# closing = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, kernel)
# gradient = cv2.morphologyEx(img_bw, cv2.MORPH_GRADIENT, kernel)
# tophat = cv2.morphologyEx(img_bw, cv2.MORPH_TOPHAT, kernel)
# blackhat = cv2.morphologyEx(img_bw, cv2.MORPH_BLACKHAT, kernel)