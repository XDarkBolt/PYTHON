import cv2
import numpy as np
from time import process_time
import math

def ref_line_function(x0,x1,c,f_x0,f_y0,f_x1,f_y1):
    m = (f_y1 - f_y0)/(f_x1 - f_x0);
    y0 = round(m * x0 + c + 4);
    y1 = round(m * x1 + c + 4);
    print("Angle of Tip:", -math.atan(m) * 180);
    return y0,y1;
    
def cross(img,c1,c2):
    print("Centers:",c1,c2);
    scale = 10;
    line_thickness = 2;
    x1, y1 = -scale, 0;
    x2, y2 = scale, 0;
    x3, y3 = 0, -scale;
    x4, y4 = 0, scale;
    cv2.line(img, (x1 + c1, y1 + c2), (x2 + c1, y2 + c2), (62, 255, 10), thickness=line_thickness);
    cv2.line(img, (x3 + c1, y3 + c2), (x4 + c1, y4 + c2), (62, 255, 10), thickness=line_thickness);
    
def ref_line(img,x0,x1,f_x0,f_y0,f_x1,f_y1):
    c = (f_y0 + f_y1) / 2;
    y0, y1 = ref_line_function(x0,x1,c,f_x0,f_y0,f_x1,f_y1);
    line_thickness = 1;
    cv2.line(img, (x0, y0), (x1, y1), (62, 255, 10), thickness=line_thickness);
    
t1_start = process_time()

img = cv2.imread('gilette1.jpg')
img = cv2.bitwise_not(img)
overlay = np.zeros([img.shape[0], img.shape[1], 3], dtype = np.uint8)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
thresh = 127
img_bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
contours, hierarchy = cv2.findContours(img_bw,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
idx =0
for cnt in contours:
    idx += 1
    x,y,w,h = cv2.boundingRect(cnt)
    roi=img[y:y+h,x:x+w]
    cv2.imwrite(str(idx) + '.jpg', roi)
    # cv2.rectangle(overlay,(x,y),(x+w,y+h),(127,127,127),2)
    
    center_x = x + int(w / 2);
    center_y = y + int(h / 2);
    cross(overlay,center_x,center_y)
    
    done = False
    if idx == 1:
        inter = False
        for i in range(center_y,0,-1):
            if img_bw[i][center_x] > 127 & inter == False:
                inter = True
            
            if img_bw[i][center_x] < 127 & inter == True:
                if done == False:
                    first_hole_x = center_x
                    first_hole_y = i
                    done = True
    done = False
    if idx == 3:
        inter = False
        for i in range(center_y,0,-1):
            if img_bw[i][center_x] > 127 & inter == False:
                inter = True
            
            if img_bw[i][center_x] < 127 & inter == True:
                if done == False:
                    second_hole_x = center_x
                    second_hole_y = i
                    done = True
                    
ref_line(overlay,0,img.shape[0],first_hole_x,first_hole_y,second_hole_x,second_hole_y);
print("FirstHole:", first_hole_y)
print("SecondHole:", second_hole_y)     

alpha = 0.3
backtorgb = cv2.cvtColor(img_bw,cv2.COLOR_GRAY2RGB)
result = cv2.addWeighted(overlay, alpha, backtorgb, 1 - alpha, 0)

t1_stop = process_time()
print("Elapsed time during the whole program in seconds:", t1_stop-t1_start)

cv2.imshow('ImageTools',result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# for i in range(0, img.shape[0], 1):
#     for j in range(0, img.shape[1], 1):
#         if img[i][j] <= 127:
#             img_bin[i][j] = 255
#             if is_top == False:
#                 cross(draw_layer,j,i);
#                 top = i,j
#                 is_top = True