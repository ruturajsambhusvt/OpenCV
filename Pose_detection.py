
import cv2
import sys
import os
import matplotlib.pyplot as plt
import time


device = 1

protoFile = "pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = os.path.join("model","pose_iter_160000.caffemodel")

#This changes with what we want to detect, this is for human pose estimation, for hand pose estimation, we need to change the prototxt and caffemodel files
nPoints = 15
POSE_PAIRS = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,14],[14,8],[8,9],[9,10],[14,11],[11,12],[12,13]]


def detect_joints(img,net,nPoints=15,threshold=0.1):
    dim = 368
    blob = cv2.dnn.blobFromImage(img,1.0,(dim,dim),mean=(0,0,0),swapRB=True,crop=False)
    net.setInput(blob)
    start_time  = time.time()
    output = net.forward()
        
    end_time  = time.time()
    print("Time taken to detect joints: {}".format(end_time-start_time))
    
    width = img.shape[1]
    height = img.shape[0]
    
    scalex = width/output.shape[3]
    scaley = height/output.shape[2]
    print("Scalex: {}, Scaley: {}".format(scalex,scaley))
    points = []

    
    for i in range(nPoints):
        probMap = output[0,i,:,:]
        minVal,prob,minLoc,point = cv2.minMaxLoc(probMap)
        
        x = point[0]*scalex
        y = point[1]*scaley
    
        if prob > threshold:
            points.append((int(x),int(y)))
        else:
            points.append(None)
                
    imPoints = img.copy()
    imSkeleton = img.copy()
    print(points)
    #Draw the detected points
    for i,p in enumerate(points):
        cv2.circle(imPoints,p,8,(255,255,0),thickness=-1,lineType=cv2.FILLED)
        cv2.putText(imPoints,"{}".format(i),p,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        
    # #Draw Skeleton
    # for pair in POSE_PAIRS:
    #     partA  = pair[0]
    #     partB = pair[1]
        
    #     if points[partA] and points[partB]:
    #         cv2.line(imSkeleton,points[partA],points[partB],(255,255,0),2)
    #         cv2.circle(imSkeleton,points[partA],8,(255,0,0),thickness=-1,lineType=cv2.FILLED)

    return imPoints


    


while __name__ == '__main__':

    if len(sys.argv)>1:
        device = sys.argv[1]
        
    cap = cv2.VideoCapture(device)

    window_name = "Camera Preview"

    cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)


    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)      

    while cv2.waitKey(1)!=27:
        has_frame, frame = cap.read()
        # cv2.flip(frame,1,frame)
        if not has_frame:
            break
        img = cv2.imread("Tiger_Woods.png")
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        impoints= detect_joints(img,net)

        cv2.imshow(window_name,impoints)

    cap.release()
    cv2.destroyWindow(window_name)
    sys.exit()