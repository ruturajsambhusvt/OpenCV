import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)

win_name = "Camera Preview"
cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
#arguments for Hands() are: here -> static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
#explain the parameters -> means that we are not using static image, we are using video, max_num_hands=2 means that we are detecting 2 hands, min_detection_confidence=0.5 means that we are detecting hands with confidence of 50%, min_tracking_confidence=0.5 means that we are tracking hands with confidence of 50%
#explained in https://google.github.io/mediapipe/solutions/hands.html
mpDraw = mp.solutions.drawing_utils
#mpDraw is used to draw the landmarks on the hand

prev_time = 0
curr_time = 0

while cv2.waitKey(1) != 27:
    has_frame,image = cap.read()
    image_RGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #the hands object requires RGB image
    results = hands.process(image_RGB)
    #USE THE RESULTS
    # print(results.multi_hand_landmarks)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            #get all information about the hand landmarks
            for id,landmark in enumerate(hand_landmarks.landmark):
                # print(id,landmark)
                h,w,c = image.shape #height, width, channel
                cx,cy = int(landmark.x*w),int(landmark.y*h) #this cx, cy is the pixel value of the landmark and the most important thing we need for post processing
                print(id,cx,cy)
                
                """ if id==4:#thumb tip
                    cv2.circle(image,(cx,cy),15,(255,0,255),cv2.FILLED) """
            
            mpDraw.draw_landmarks(image,hand_landmarks,mpHands.HAND_CONNECTIONS)
            
    
    if not has_frame:
        break
    curr_time = time.time()
    fps = 1/(curr_time-prev_time)
    prev_time = curr_time
    
    cv2.putText(image,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    
    cv2.imshow(win_name,image)
   

cap.release()
cap.destroyWindow(win_name)