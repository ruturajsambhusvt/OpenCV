import numpy as np  
import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torchvision
from torchvision import transforms,datasets
import torch.onnx as onnx
import os
import sys  
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import argparse
from Training import preprocess




def main():
    model_path = "/home/trec/WorkRaj/Coding_Tutorials/OpenCV/TextDetection/saved_models/1686874684_epochs_100_model.onnx"
    opencv_net = cv2.dnn.readNetFromONNX(model_path)
    imagenet_labels = ["0","1","2","3","4","5","6","7","8","9"]
    
    
    #############################
    width = 640
    height = 480
    #############################
    device = 0
    cap = cv2.VideoCapture(device)
    cap.set(3,width) #id number for width is 3
    cap.set(4,height) #id number for height is 4
    
    while True:
        success, img_raw = cap.read()
        img = np.asarray(img_raw)
        img = cv2.resize(img,(32,32))
        img = preprocess(img)
        # cv2.imshow("Processed Image",img)
        # cv2.imshow("Raw Image",img_raw)
        img = np.expand_dims(img,axis=[0,1]) ##cleaner way to do this? Yes, use np.expand_dims(img, axis=0)
        # set OpenCV DNN input
        opencv_net.setInput(img)
        # OpenCV DNN inference
        out = opencv_net.forward()
        # print("OpenCV DNN prediction: \n")
        # print("* shape: ", out.shape)
        # get the predicted class ID
        imagenet_class_id = np.argmax(out)
        probs = F.softmax(torch.from_numpy(out), dim=1)
        # get confidence
        confidence = out[0][imagenet_class_id]
        confidence_prob = probs[0][imagenet_class_id].detach().cpu().numpy()
        # print("* class ID: {}, label: {}".format(imagenet_class_id, imagenet_labels[imagenet_class_id]))
        # print("* confidence: {:.4f}".format(confidence))
        cv2.putText(img_raw, "predicted class: {},confidence: {:.4f} ".format(imagenet_labels[imagenet_class_id],confidence_prob), (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Raw Image",img_raw)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
    
    
    
    
    


if __name__ == '__main__':
    main()