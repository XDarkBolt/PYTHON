import numpy as np
import cv2
import time
import threading
import keyboard

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

get_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

i = 1

def save_image(img, name, i, img_format):
    cv2.imwrite("C:\\Users\\hsynk\\Desktop\\BVISIO\POC\\ImageProcess\\Test Data Sets\\" + name + str(i) + "." + img_format, img)

while(cap.isOpened()):
    while(True):
        ret, img = cap.read()
        show_img = cv2.resize(img, (1280,720), interpolation = cv2.INTER_AREA)
        cv2.imshow('img', show_img)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
        if cv2.waitKey(1) == ord('s'):
            t1 = threading.Thread(target=save_image, args=(img, "opencv", i, "png",))
            t1.start()
            t1.join()
            i += 1
            
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Alert ! Camera disconnected")