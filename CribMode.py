import cv2 
from crib_mode import detect_crib as dc
import winsound
import numpy as np

from utils import visualization_utils as vis_util
from crib_mode import utilSujee

def crib_Mode():
    print('Entering Crib Mode')
    cap = cv2.VideoCapture('Sample_Videos/2.mp4')
    firstframe = None
    ret,frame = cap.read()
    x=frame.size
    #print (y)
    print (x)
    (left,right,top,bottom) =(0,0,0,0)
    face_cascade = cv2.CascadeClassifier('crib_mode/haarcascade/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('crib_mode/haarcascade/haarcascade_eye.xml')
    upper_body = cv2.CascadeClassifier('crib_mode/haarcascade/haarcascade_upperbody.xml')
    status = False

    while True:
        ret,frame = cap.read()
        
        while status == False:
            ret, frame = cap.read()
            cv2.imshow('testingImage',frame)
            cv2.imwrite('test_images/cribtes.jpg',frame)
            (left,right,top,bottom,status)=dc.run_detection(frame)
            print(status)
            firstframe = frame
        #    firstframe = frame
        cv2.rectangle(frame,(round(left),round(top)),(round(right),round(bottom)),(0,255,0),2)
        cv2.imshow('fdg',frame)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces= face_cascade.detectMultiScale(gray)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            #cv2.rectangle(frame,(cx,cy),(200,100),(0,255,0),3)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h,x:x+w]
            print (y)
            if( y<top ):
                safe = False
                print ("Child in Danger Area")
                winsound.Beep(250,10)
            else:
                safe = True
                print ("Child in Safe Area")

            eyes = eye_cascade.detectMultiScale(gray)
            for(ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh), (0,255,0),1)

            upperbody = upper_body.detectMultiScale(gray)
            for (ub_x,ub_y,ub_w,ub_h) in upperbody:
                cv2.rectangle(frame,(ub_x,ub_y),(ub_x+ub_w,ub_y+ub_h),(255,0,0),2)

        cv2.imshow('Image',frame)
        if cv2.waitKey(5) == 27:
            break
