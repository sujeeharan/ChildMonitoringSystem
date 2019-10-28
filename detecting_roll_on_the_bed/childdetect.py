import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
import winsound

def chiddetectmethod(cap,coordinatedic_dic):
    
    cap.release()
    #time.sleep(10.25)
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",default='detecting_roll_on_the_bed/naga.mp4', help="path to the video file")
    ap.add_argument("-a", "--min-area", type=int, default=1000, help="minimum area size")
    args = vars(ap.parse_args())

    if args.get("video", None) is None:
        camera = cv2.VideoCapture(1)
        time.sleep(0.25)
    else:
       
        camera = cv2.VideoCapture(args["video"])
       
       
    
    firstFrame = None

    while True:

        (grabbed, frame) = camera.read()
        text = "Unoccupied"

        if not grabbed:
            break

       
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if firstFrame is None:
            firstFrame = gray
            continue
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 70, 255, cv2.THRESH_BINARY)[1]
        thresh1=thresh[coordinatedic_dic['top']:coordinatedic_dic['bottom'],coordinatedic_dic['left']:coordinatedic_dic['right']].copy()
        cv2.imshow("trss",thresh1)
        (_,cnts, _) = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL,
		    cv2.CHAIN_APPROX_SIMPLE)

     
        ss=frame[coordinatedic_dic['top']:coordinatedic_dic['bottom'],coordinatedic_dic['left']:coordinatedic_dic['right']].copy()
        box=DetectActualBedSerface(coordinatedic_dic,ss)
    
        for c in cnts:
		    # if the contour is too small, ignore it
            if cv2.contourArea(c) < args["min_area"]:
                continue

            frame = cv2.drawContours(frame,[box],0,(0,255,255),2)
            cv2.rectangle(frame, (coordinatedic_dic['left'], coordinatedic_dic['top']), (coordinatedic_dic['right'], coordinatedic_dic['bottom']), (0, 255, 0), 2)



          
            (x, y, w, h) = cv2.boundingRect(c)

          

            if (regoinvalidation(x, y, w, h,coordinatedic_dic)) :
               
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                #text = "Occupied"
                if(firealarm(x,x+w,box[0][0],box[0][0]+(box[3][0] -box[0][0]))):
                    cv2.line(frame,(x,y),(x,y+h),(255,255,255),3)
                    cv2.line(frame,(box[0][0],box[0][1]),(box[0][0],box[0][1]+(box[1][1] -box[0][1])),(255,255,255),3)
                    text = "ALARM"
                    winsound.Beep(2500,1000)

        cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
		    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
		    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        cv2.imshow("Security Feed", cv2.resize(frame,(800,600)))
        #cv2.imshow("Thresh", thresh)
       # cv2.imshow("Frame Delta", frameDelta)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


    camera.release()
    cv2.destroyAllWindows()

def firealarm(x,xw,bx,bxw):
    
    #if x>bx:
    #    return True
    if xw > (bxw-200):
        return True 
    print(xw)
    print(" ")
    print(bxw)
    return False

def reDrawBox(xb,yb,xbr,ybb,xc,yc,xcr,ycb):


    if xc<xb:
        xb=xc
    if yc<yb:
        yb=yc

    if xcr<xbr:
        xbr=xcr
    if ycb<ybb:
        ybb=ycb

   

    return [xb,yb,xbr,ybb]

def regoinvalidation(x, y, w, h,coordinatedic_dic):
    check=False
    if x > coordinatedic_dic['left']:
        check=True
    else:
        check=False
    if (x+w)<coordinatedic_dic['right']:
        check=True
    else:
        check=False
    if y > coordinatedic_dic['top']:
        check=True
    else:
        check=False
    if(y+h) < coordinatedic_dic['bottom']:
        check=True
    else:
        check=False
                    

    return check


def DetectActualBedSerface(coordinatedic_dic,frame):
    
    firstfram=None
    fx=0
    fy=0
    fw=0
    fh=0
    areas=[]
    box=None

    imgray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #  fram=cv2.resize(fram,(1080,800))
    lower_red=np.array([127,127,127])
    upper_red=np.array([255,255,255])
    kernal=np.ones((5,5),np.uint8)
    kernal2=np.ones((2,2),np.uint8)
    mask=cv2.inRange(frame,lower_red,upper_red)
    opening=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernal)


    ret,thresh = cv2.threshold(imgray,127,255,0)
    image, contours, hierarchy = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)



    areas = [cv2.contourArea(c) for c in contours]
  
    max_index1 = np.argmax(areas)
    cnt1=contours[max_index1]
    rect = cv2.minAreaRect(cnt1)
    box1 = cv2.boxPoints(rect)
    box1 = np.int0(box1)
    im = cv2.drawContours(frame,[box1],0,(0,255,255),2)

    for c in contours:
      
        cnttm=c
        rect = cv2.minAreaRect(cnttm)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        if regoinvalidation(box[0][0], box[0][1],(box[3][0] -box[0][0]),(box[1][1] -box[0][1]),coordinatedic_dic):
            #areas=cnttm
            areas.append(cv2.contourArea(c))



    max_index = np.argmax(areas)
    cnt=contours[max_index]



    rect = cv2.minAreaRect(cnt)
    box1 = cv2.boxPoints(rect)
    box1 = np.int0(box1)
   
    return box1



    
                    