import numpy as np
import cv2
import imutils
from math import atan2, pi
from numpy.random import rand
from math import sin, cos, pi
import math

def play_Mode():
    print('Entering Play Mode')
    vid= cv2.VideoCapture('Sample_Videos/sleepMode.mp4')

    firstframe= None

    fullBody=cv2.CascadeClassifier('HaarCascade/haarcascade_fullbody.xml')

    while True:
        ret, frame= vid.read()

    
        #frame=imutils.resize(frame,width=480)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        if firstframe is None:
            firstframe=gray
            continue

        grayFull=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        body=fullBody.detectMultiScale(grayFull)
    
        for (x,y,w,h) in body:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            b_gray=gray[y:y+h,x:x+w]
            b_color=frame[y:y+h,x:x+w]
        
            #Draw a line
            angel=cv2.line(frame, (int(round((x+(w/2)))),int(round(y))), (int(round(x+(w/2))),int(round((y+h)))), (0,0,255),4)
        
            # Take the angle of the movement line

            # point 1
            x1 =(round(x+(w/2))) # 0.5
            y1 =y #0.0

            # point 2
            x2 =(round((x+(w/2)))) #-0.5
            y2 =(round((y+h)))# -1.0

            deltax = x2 - x1
            deltay = y2 - y1

            angle_rad = atan2(deltay,deltax)
            angle_deg = angle_rad*180.0/pi

            print ("The angle is %.5f radians (%.5f degrees)." % (angle_rad,angle_deg))

            if (angle_deg<90.000):
                safe=False
                print("Danger")
            else :
                safe=True
                print("Safe")
            
        cv2.imshow('Body',grayFull)
        cv2.imshow('Test',frame)

        if cv2.waitKey(5) == 27:
            break
        #if cv2.waitKey==ord("q"):
        #    break
    vid.relase()
    cv2.destroyAllWindows()