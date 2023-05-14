import cv2
import sys
import numpy as np

PREVIEW = 0  # Preview mode
BLUR = 1  # Blurring Filter
FEATURES = 2  # Corner Feature Detection
CANNY = 3  # Canny Edge Detection

feature_params = dict(maxCorners=500, qualityLevel=0.2,
                      minDistance=15, blockSize=9)
# maxCorners: Maximum number of corners to return. If there are more corners than are found, the strongest of them is returned.
# qualityLevel: Parameter characterizing the minimal accepted quality of image corners. The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue or the Harris function response (see cornerHarris() ). The corners with the quality measure less than the product are rejected. For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected.
# minDistance: Minimum possible Euclidean distance between the returned corners.
# blockSize: Size of an average block for computing a derivative covariation matrix over each pixel neighborhood. See cornerEigenValsAndVecs() .

device = 1  # Since we use USB camera, the device is 1

if len(sys.argv) > 1:
    device = sys.argv[1]

image_filter = PREVIEW
alive = True

win_name = "Camera Filters"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
result = None

cap = cv2.VideoCapture(device)

while alive:
    has_frame, frame = cap.read()
    if not has_frame:
        break

    frame = cv2.flip(frame, 1)  # Why do we need to flip the frame?
    # Because the camera is mirrored, so we need to flip it to make it look normal

    if image_filter == PREVIEW:
        result = frame
    elif image_filter == BLUR:
        result = cv2.blur(frame, (13, 13)) 
        # cv2.blur() is used to blur an image using the normalized box filter.
        # The function smooths an image using the kernel: K = 1/(ksize.width*ksize.height)*[[1, 1, ..., 1], [1, 1, ..., 1], ..., [1, 1, ..., 1]]
        #explanation of the parameters: https://docs.opencv.org/3.4/d4/d13/tutorial_py_filtering.html
    elif image_filter == CANNY:
        result = cv2.Canny(frame, 80, 150)
        # cv2.Canny() is used to detect edges in an image
        # The function finds edges in the input image image and marks them in the output map edges using the Canny algorithm. The smallest value between threshold1 and threshold2 is used for edge linking. The largest value is used to find initial segments of strong edges. See http://en.wikipedia.org/wiki/Canny_edge_detector
        #param1 – first threshold for the hysteresis procedure. param2 – second threshold for the hysteresis procedure.
        #Explaination of the parameters: https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html
    elif image_filter == FEATURES:
        result = frame
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
        # cv2.goodFeaturesToTrack() finds strong corners on an image.
        # The function finds the most prominent corners in the image or in the specified image region, as described in Shi94
        #explaination of the parameters: https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541 
        if corners is not None:
            for x,y in np.float32(corners).reshape(-1,2):
                #why do we need to reshape the corners? because the corners are in a 2D array, but we need to draw circles on the image, so we need to convert it to a 1D array
                cv2.circle(result,(int(x),int(y)),10,(0,255,0),1)
                #explanation of the parameters: https://docs.opencv.org/3.4/dc/da5/tutorial_py_drawing_functions.html 
            
    cv2.imshow(win_name, result)

    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q') or key ==27:
        alive = False
    elif key == ord('C') or key == ord('c'):
        image_filter = CANNY
    elif key == ord('B') or key == ord('b'):
        image_filter = BLUR
    elif key == ord('F') or key == ord('f'):
        image_filter = FEATURES
    elif key == ord('P') or key == ord('p'):
        image_filter = PREVIEW
        
cap.release()
cv2.destroyWindow(win_name) 