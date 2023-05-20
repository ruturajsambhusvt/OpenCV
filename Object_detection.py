import cv2
import sys
import os
import matplotlib.pyplot as plt


device = 1
modelFile = os.path.join("models","ssd_mobilenet_v2_coco_2018_03_29","frozen_inference_graph.pb")
configFile = os.path.join("models","ssd_mobilenet_v2_coco_2018_03_29.pbtxt")
classFile = "coco_class_labels.txt"

with open(classFile) as fp:
    labels = fp.read().split("\n")

def detect_object(img,net):
    dim = 300
    blob = cv2.dnn.blobFromImage(img,1.0,(dim,dim),mean=(0,0,0),swapRB=True,crop=False)
    net.setInput(blob)
    output= net.forward()
    
    return output

def display_text(img,text,x,y):
    text_size = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
    dim = text_size[0]
    baseline = text_size[1]
    
    cv2.rectangle(img,(x,y-baseline-dim[1]),(x+dim[0],y+baseline),(0,0,0),cv2.FILLED)
    cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),thickness=1)
    
def display_object(img,objects,labels,threshold=0.25):
    img_height = img.shape[0]
    img_width = img.shape[1]
    
    for i in range(objects.shape[2]):
        classId = objects[0,0,i,1]
        score= objects[0,0,i,2]
        #coordinates of the bounding box
        x = objects[0,0,i,3]*img_width
        y = objects[0,0,i,4]*img_height
        w = objects[0,0,i,5]*img_width-x
        h = objects[0,0,i,6]*img_height-y
        
        if score>threshold:
            display_text(img,"{}:{:.4f}".format(labels[int(classId)],score),int(x),int(y)-5)
            cv2.rectangle(img,(int(x),int(y)),(int(x+w),int(y+h)),(0,255,0),thickness=1)
             
    return img
    


while __name__ == '__main__':

    if len(sys.argv)>1:
        device = sys.argv[1]

    #Read the Tensorflow model
    net = cv2.dnn.readNetFromTensorflow(modelFile,configFile)
        
    cap = cv2.VideoCapture(device)

    window_name = "Camera Preview"

    cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)



    while cv2.waitKey(1)!=27:
        has_frame, frame = cap.read()
        # cv2.flip(frame,1,frame)
        if not has_frame:
            break
        objects = detect_object(frame,net)
        img = display_object(frame,objects,labels)
        
        cv2.imshow(window_name,frame)

    cap.release()
    cv2.destroyWindow(window_name)
    sys.exit()