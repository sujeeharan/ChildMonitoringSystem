import argparse
import datetime
import imutils
import time
import cv2
import winsound
from PIL import Image
from tkinter import Image
import time
import numpy as np

#R.M.U.A.Rathnayaka IT14092848
#Project- Child monitoring system using gesture recognition
#this function- Child wake up notifier

def sleep_Mode():

    SleepMode ='y'

    #ap = argparse.ArgumentParser()
    #ap.add_argument("-v", "--video", help="path to the video file")
    video="sample_videos/test.mp4"
    #ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
    min_area=500
    #args = vars(ap.parse_args())



    #img = cv2.imread('sachin.jpg')
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    """faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)"""


    # chk vid
    if (video is None): 
        camera = cv2.VideoCapture(0)

    ###########
    ###########
        #camera = imutils.resize(frame, width = 1000)  not working yet withit tkr
        time.sleep(0.25)


    else:
        camera = cv2.VideoCapture(video)
      #  rect = cv2.rectangle(camera,(384,0),(510,128),(0,255,0),3)   
  


    firstFrame = None
    frameCount = 0
    alarm=0

    Freq = 2500 # Set Frequency To 2500 Hertz
    Dur = 1000 # Set Duration To 1000 ms == 1 second
    timer=0 # to take motion time


    # loop over the frames of the video
    #while(camera.isOpened()):

    start_time = time.time()
    #print(start_time)

    while True:

        (grabbed, frame) = camera.read()
        text = "Baby is sleeping"
                      ###########
        img = cv2.imread("test.mp4")
      #  im=Image.open(img)
        ret, frame = camera.read()
        frame = imutils.resize(frame,width=1366)
        cropped = frame[384:900, 512:1000]
     
        #histo = np.int32(cv2.calcHist([frame],[0],None,[256],[0,256]))
        #print("histo val is "+str(histo))

        ##50% dark check
        histo=np.mean(frame)
        print(np.mean(frame))

        ##%50 dark check end

        ##CHECK MODE

        if(histo<60):
            mode="Low light"

        elif (60<=histo<=100):
            mode="Normal light"
            mode_value=0.0004

        elif(100<histo<=130):
            mode="Bright Light"
            mode_value=0.0003
    
        elif(histo>130):
            mode="High Intensity"
            mode_value=0.0005

        else:
            continue


        ##CHECK MODE END

    
        cv2.imshow("cropped", cropped)
        #cv2.imshow("Resize",frame)
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

   
        frame = imutils.resize(cropped, width=500)
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

   
        if firstFrame is None or frameCount%15 is 0:
            firstFrame = gray
            continue

        # diff
        DifFrame = cv2.absdiff(firstFrame, gray)
        thresholdframe = cv2.threshold(DifFrame, 80, 255, cv2.THRESH_BINARY)[1]

         ##CHECK MODE

        if(histo<60):
            mode="Low light"
            thresholdframe = cv2.dilate(thresholdframe, None, iterations=3)
            mode_value=0.0001

        elif (60<=histo<=100):
            mode="Normal light"
            mode_value=0.0004
            thresholdframe = cv2.erode(thresholdframe, None, iterations=1)

        elif(100<histo<=130):
            mode="Bright Light"
            mode_value=0.0003
            thresholdframe = cv2.erode(thresholdframe, None, iterations=1)
    
        elif(histo>130):
            mode="High Intensity"
            mode_value=0.0005
            thresholdframe = cv2.erode(thresholdframe, None, iterations=1)

        else:
            continue


        ##CHECK MODE END

    
   
        #thresholdframe = cv2.dilate(thresholdframe, None, iterations=2)
        # There are differences in openCV2.x and 3.0 for findContour calls
        if imutils.is_cv2():
            (cnts, _) = cv2.findContours(thresholdframe.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        elif imutils.is_cv3():
            (_, cnts, _) = cv2.findContours(thresholdframe.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
  
        for c in cnts:
            start_time = time.time()
           # print(start_time)
        
            # if the contour is too small, ignore it
            if cv2.contourArea(c) <= min_area:
                continue

            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
     
            (x, y, w, h) = cv2.boundingRect(c)
         #   print("got it")
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Baby is awake"
         #   print("got it")
       
        
        # draw the text and timestamp on the frame
            if(SleepMode=='Y' or SleepMode=='y'):
                  
                       #  frame = cv2.rectangle(frame,(100,640),(440,128),(0,0,255),3)
                         cv2.putText(frame, "Baby Monitoring Ongoing: {}".format(text), (10, 20),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (224, 255, 255), 2)              
                     
                     
                         end_time = time.time()
                       #  print(end_time)

                         time_def=end_time-start_time
                         print("time def is "+str(time_def))
                         timer=timer+1

                         if(time_def>=mode_value): #minimum time to make alert
                               if(timer>2):
                                   alarm=1
                                   timer=0
                                   winsound.Beep(2000,1000)
                               else:
                                        continue
                         else:
                            alarm=0
                        

                     
                     
                    

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