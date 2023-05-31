import cv2
import time
import numpy as np
import Hand_Tracking_Module as htm
import math
from subprocess import call

################################
width_cam,height_cam = 1280,720
################################

cap = cv2.VideoCapture(1)
# cap.set(3,width_cam)
# cap.set(4,height_cam)

win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

curr_time = 0
prev_time = 0

detector = htm.HandDetector(min_detection_confidence=0.7)



    
while cv2.waitKey(1) != 27:
    has_frame, frame = cap.read()
    detector.find_hands(frame,draw=True)
    lm_list = detector.find_position(frame,draw=False)
    if len(lm_list)!=0:
        # print(lm_list[4],lm_list[8])
        x1,y1 = lm_list[4][1],lm_list[4][2] #[id,x,y]
        x2,y2 = lm_list[8][1],lm_list[8][2]
        cx,cy = (x1+x2)//2,(y1+y2)//2
        cv2.circle(frame,(x1,y1),15,(255,0,255),cv2.FILLED)
        cv2.circle(frame,(x2,y2),15,(255,0,255),cv2.FILLED)
        cv2.line(frame,(x1,y1),(x2,y2),(255,0,255),3)
        cv2.circle(frame,(cx,cy),15,(255,0,255),cv2.FILLED)
        
        length  = math.hypot(x2-x1,y2-y1)
        # print(length)
        
        #Hand range 50-300
        #Volume range 0 - 100
        
        vol = np.interp(length,[50,300],[0,100])
        print(vol)
        
        if length<50:
            cv2.circle(frame,(cx,cy),15,(0,255,0),cv2.FILLED)
            
        cv2.rectangle(frame,(50,150),(85,400),(0,255,0),3)
        cv2.rectangle(frame,(50,400-int(vol*(400-150)/100)),(85,400),(0,255,0),cv2.FILLED)
        cv2.putText(frame,f'{int(vol)}%',(40,450),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
        
        call(["amixer","-D","pulse","sset","Master",str(vol)+"%"])
    
    curr_time = time.time()
    fps = 1/(curr_time-prev_time)
    prev_time = curr_time
    # print(fps)
    
    cv2.putText(frame,f'FPS:{str(int(fps))}',(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    
    if not has_frame:
        break
    cv2.imshow("Camera Preview",frame)
    

cap.release()
cap.destroyWindow(win_name)
