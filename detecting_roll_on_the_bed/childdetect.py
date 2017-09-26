import argparse
import datetime
import imutils
import time
import cv2


def chiddetectmethod(cap,coordinatedic_dic):
    
    cap.release()
    #time.sleep(10.25)
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",default='detecting_roll_on_the_bed/naga.mp4', help="path to the video file")
    ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
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

        #frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if firstFrame is None:
            firstFrame = gray
            continue
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 70, 255, cv2.THRESH_BINARY)[1]

        (_,cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		    cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
		    # if the contour is too small, ignore it
            if cv2.contourArea(c) < args["min_area"]:
                continue
            cv2.rectangle(frame, (coordinatedic_dic['left'], coordinatedic_dic['top']), (coordinatedic_dic['right'], coordinatedic_dic['bottom']), (0, 255, 0), 2)
            cv2.rectangle(frame, (coordinatedic_dic['left']*8, coordinatedic_dic['top']), (round(coordinatedic_dic['right']*0.8), coordinatedic_dic['bottom']), (255, 0, 0), 2)
            (x, y, w, h) = cv2.boundingRect(c)

            

            if (regoinvalidation(x, y, w, h,coordinatedic_dic)) :
                #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                text = "Occupied"
    
        cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
		    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
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


def regoinvalidation(x, y, w, h,coordinatedic_dic):

    if x > coordinatedic_dic['left']:
        if (x+w)<coordinatedic_dic['right']:
            if y > coordinatedic_dic['top']:
                if(y+h) < coordinatedic_dic['bottom']:
                    return True

    return False
                    