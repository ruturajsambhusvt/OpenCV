import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) -> None:
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.static_image_mode, max_num_hands=self.max_num_hands,
                                        min_detection_confidence=self.min_detection_confidence, min_tracking_confidence=self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        
    def find_hands(self, image,draw=True):
        self.image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(self.image_RGB)
        
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                # get all information about the hand landmarks
                if draw:
                    self.mpDraw.draw_landmarks(image, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
                    
                

                """ if id==4:#thumb tip
                    cv2.circle(image,(cx,cy),15,(255,0,255),cv2.FILLED) """
        return image
    
    def find_position(self,image,hand_num=0,draw=True):
        lm_list = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hand_num]
            for id, landmark in enumerate(myHand.landmark):
                # print(id,landmark)
                h, w, c = image.shape  # height, width, channel
                # this cx, cy is the pixel value of the landmark and the most important thing we need for post processing
                cx, cy = int(landmark.x*w), int(landmark.y*h)
                # print(id, cx, cy)
                lm_list.append([id,cx,cy])
                
                if draw:
                    cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            
        return lm_list
        

                


def main():

    cap = cv2.VideoCapture(1)

    win_name = "Camera Preview"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    prev_time = 0
    curr_time = 0
    
    detector = HandDetector()

    while cv2.waitKey(1) != 27:
        has_frame, image = cap.read()
        # the hands object requires RGB image
        image = detector.find_hands(image)
        lm_list = detector.find_position(image,draw=False)
        # lm_list_other = detector.find_position(image,1)
        
        if len(lm_list)!=0:
            print(lm_list[4])
            
        if not has_frame:
            break
        curr_time = time.time()
        fps = 1/(curr_time-prev_time)
        prev_time = curr_time

        cv2.putText(image, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow(win_name, image)

    cap.release()
    cap.destroyWindow(win_name)


if __name__ == '__main__':
    main()
