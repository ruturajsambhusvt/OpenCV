import cv2
import sys

device = 1 #Since we use USB camera, the device is 1

if len(sys.argv)>1:
    device = sys.argv[1]
    
cap = cv2.VideoCapture(device)

win_name = "Camera Preview"
cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)

while cv2.waitKey(1) != 27: #esc
    has_frame, frame = cap.read()
    if not has_frame:
        break
    cv2.imshow(win_name,frame)
    
cap.release()
cv2.destroyWindow(win_name)