

#import numpy as np
#import cv2

#cap = cv2.VideoCapture('naga.mp4')

##kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
##fgbg = cv2.createBackgroundSubtractorMOG2(40,30,1)

#while(1):
#    ret, frame1 = cap.read()

#    frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
#    ret,thresh = cv2.threshold(frame,127,255,0)
#    #frameDelta = cv2.absdiff(thresh, gray)
#    #thresh = cv2.threshold(frameDelta, 70, 255, cv2.THRESH_BINARY)[1]
#    _,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#    cnt = contours[0]
#    x,y,w,h = cv2.boundingRect(cnt)
#    img = cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
#    #fgmask = fgbg.apply(frame)
#   # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

#    cv2.imshow('frame',cv2.resize(img,(800,600)))
#    k = cv2.waitKey(30) & 0xff
#    if k == 27:
#        break

#cap.release()
#cv2.destroyAllWindows()

import numpy as np
import cv2

cap = cv2.VideoCapture('naga.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2(0,180,0)
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
kernel=np.ones((4,4),np.uint8)
while(1):
    ret, frame = cap.read()
   
    fgmask = fgbg.apply(frame)
    #thresh = cv2.threshold(fgmask,127,255,0)
    opening=cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('fgmask',fgmask)
    #cv2.imshow('frame',fgmask)

    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    

cap.release()
cv2.destroyAllWindows()