import cv2 
from crib_mode import detect_crib as dc
import winsound
import numpy as np

from utils import visualization_utils as vis_util
from crib_mode import utilSujee
import imutils

def crib_Mode():
    print('Entering Crib Mode')
    cap = cv2.VideoCapture('Sample_Videos/5.mp4')
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
        #frame = imutils.resize(frame,width=480)
        while status == False:
            ret, frame = cap.read()
            cv2.imwrite('test_images/cribtes.jpg',frame) 
            (left,right,top,bottom,status)=dc.run_detection(frame)
            print(status)
            firstframe = frame
        #    firstframe = frame

        #Defining Boundries
        (h, w) = frame.shape[:2]
        print(h)
        print(w)
        cv2.rectangle(frame,(round(left),round(top)),(round(right),round(bottom)),(0,0,255),2)

        #Boundry Calculation
        cribmidpoint = round((bottom - top)/2)+round(top)

        #Drawing Boundries
        cv2.rectangle(frame,(round(left),round(top)),(round(right),round(bottom)),(0,0,255),2)
        cv2.rectangle(frame,(0,cribmidpoint),(w,h),(123,122,122),3)

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces= face_cascade.detectMultiScale(gray)

        safecount= 0
        dangercount = 0
        
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            #cv2.rectangle(frame,(cx,cy),(200,100),(0,255,0),3)

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h,x:x+w]
            print (y)
            

            upperbody = upper_body.detectMultiScale(gray)
            for (ub_x,ub_y,ub_w,ub_h) in upperbody:
                cv2.rectangle(frame,(ub_x,ub_y),(ub_x+ub_w,ub_y+ub_h),(255,0,0),2)

            if( x<top ):
                safe = False
                print ("Child in Safe Area")
                
                safecount = safecount+1
                dangercount=0
            else:
                safe = True
                
                print ("Child in Danger Area")
                dangercount = dangercount +1
                safecount=0
            
            if dangercount ==2:
                winsound.Beep(5000,2000)

            if (x < cribmidpoint):
                print("is Child inside Crib?") 
            
        cv2.imshow('Image',frame)
        cv2.imwrite('test2.jpg',frame)
        if cv2.waitKey(5) == 27:
            break
