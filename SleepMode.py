import argparse
import datetime
import imutils
import time
import cv2
import winsound
from PIL import Image
from tkinter import Image

def sleep_Mode():
    print('Entering Sleeping Mode')

    camera= cv2.VideoCapture(0)
  


    firstFrame = None
    frameCount = 0
    alarm=0

    Freq = 2500 # Set Frequency To 2500 Hertz
    Dur = 1000 # Set Duration To 1000 ms == 1 second


    # loop over the frames of the video
    #while(camera.isOpened()):

    while True:
   


        (grabbed, frame) = camera.read()
        text = "Baby is sleeping"
                      ###########
        img = cv2.imread("Sample_Videos/sleepMode.mp4")
        ret, frame = camera.read()
        frame = imutils.resize(frame,width=1366)
        cropped = frame[384:900, 512:1000]
        cv2.imshow("cropped", cropped)
        cv2.imshow("Resize",frame)
        cv2.imshow("frame",frame)
       # cv2.waitKey(0)

    ############### 

        ###########
        # img = cv2.imread("test.mp4")
        #ret, frame = camera.read()
        #frame = imutils.resize(frame,width=480)
        #cropped = frame[140:340, 880:1080]
     #   cv2.imshow("cropped", cropped)
        #cv2.imshow("Baby Monitoring System",frame)
        #cv2.waitKey(0)

    ###############

  
        if not grabbed: #end vid
            break

    
        frameCount = frameCount + 1

   
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

   
        if firstFrame is None or frameCount%15 is 0:
            firstFrame = gray
            continue

        # diff
        DifFrame = cv2.absdiff(firstFrame, gray)
        thresholdframe = cv2.threshold(DifFrame, 80, 255, cv2.THRESH_BINARY)[1]

    
        #thresholdframe = cv2.erode(thresholdframe, None, iterations=3)
        thresholdframe = cv2.erode(thresholdframe, None, iterations=3)
        # There are differences in openCV2.x and 3.0 for findContour calls
        if imutils.is_cv2():
            (cnts, _) = cv2.findContours(thresholdframe.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        elif imutils.is_cv3():
            (_, cnts, _) = cv2.findContours(thresholdframe.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours

        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < args["min_area"]:
                continue

            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
     
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Baby is awake"
       
        
        # draw the text and timestamp on the frame
            if(SleepMode=='Y' or SleepMode=='y'):
                  
                       #  frame = cv2.rectangle(frame,(100,640),(440,128),(0,0,255),3)
                         cv2.putText(frame, "Baby Monitoring Ongoing: {}".format(text), (10, 20),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (224, 255, 255), 2)              

                         alarm=1
                     
                        #winsound.Beep(2000,1000)

                         print("alarm ring and alarm val =",alarm)
                             

           
             


   

        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        frame=cv2.resize(frame,(1020,720))
                        
        # show the frame and record if the user presses a key
        cv2.imshow("Baby Monitoring System", frame)
        cv2.imshow("Thresh", thresholdframe)
       # cv2.imshow("Frame Delta", DifFrame)
        key = cv2.waitKey(1) & 0xFF
 

        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break
  



    camera.release()
    cv2.destroyAllWindows()