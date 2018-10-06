import cv2
import numpy as np
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    cv2.imshow("frame1", frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                       
    lower_red = np.array([110,50,50])
    upper_red = np.array([130,255, 255])
    lower_blue = np.array([78, 158, 124])
    upper_blue = np.array([140, 255, 255])
    lower_yellow = np.array([0, 50, 50])
    upper_yellow = np.array([20, 255, 255])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
    mask2 = cv2.inRange(hsv, lower_yellow, upper_yellow)
  
    frame1 = mask1+mask2+mask
    kernelOpen = np.ones((5,5))
    kernelClose = np.ones((20,20))
    kernel = np.ones((5,5), np.uint8)
    
    maskOpen = cv2.morphologyEx(frame1, cv2.MORPH_OPEN, kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)
    res_2 = cv2.medianBlur(maskClose, 15)
    res_2 = cv2.GaussianBlur(res_2,(15,15), 0)
    
   
    kernel = np.ones((5,5), np.uint8)
    smoothed = cv2.filter2D(res_2, -1, kernel)
    erosion = cv2.erode(smoothed, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((3,3), np.uint8)
    smoothed = cv2.filter2D(closing, -1, kernel)
    erosion = cv2.erode(smoothed, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    maskFinal = closing
    im2, conts, h = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame, conts, -1, (255,0,0), 3)
    cv2.imshow("skincolor", mask2)
    
    cv2.imshow("final", frame)
   
    
   
 
   
  
    k = cv2.waitKey(20) & 0xff
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()
    
