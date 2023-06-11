import numpy as np  
import cv2
import time
# import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torchvision
from torchvision import transforms,datasets
import os
import sys  
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super().__init__(in_dim,out_dim,kernel_size,stride,padding)
        self.conv1 = nn.Conv2d()
        


def preprocess(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image) #makes the lighting distribution uniform
    image = image/255 #normalizing the image
    
    return image

def main():

    ######config
    path =  'new_dataset'
    test_ratio = 0.2
    validation_ratio  = 0.2
    ###########

    myList = os.listdir(path)

    myList.sort()
    if '.DS_Store' in myList:
        myList.pop(0) #remove .DS_Store
    print(f"Total number of classes detected {len(myList)}")
    num_classes = len(myList)

    print("Importing images...")
    #put all images in a list
    images  = []
    class_label = []
    for num in range(num_classes):
        image_list = os.listdir(os.path.join(path,myList[num]))
        image_list.sort()
        image_list.pop(0) #remove .DS_Store
        # print(image_list)
        for img in image_list:
            curr_image = cv2.imread(os.path.join(path,myList[num],img))
            curr_image = cv2.resize(curr_image,(32,32))
            
            images.append(curr_image)
            class_label.append(num)    
        print(num)
            
    print(f"The number of images are {len(images)} and the number of labels are {len(class_label)}")

    images = np.asarray(images)
    class_label = np.asarray(class_label)

    print(f"The shape of images is {images.shape} and the shape of labels is {class_label.shape}")

    ##Splitting the data into train and test
    X_train,X_test,Y_train,Y_test = train_test_split(images,class_label,test_size=test_ratio)
    X_train,X_validation,Y_train,Y_validation = train_test_split(X_train,Y_train,test_size=validation_ratio)

    print(f"The shape of X_train is {X_train.shape} and the shape of X_test is {X_test.shape} and the shape of X_validation is {X_validation.shape}")

    num_of_samples = []
    ## Checking if the data is balanced
    for x in range(num_classes):
        # print(len(np.where(Y_train==x)[0]))
        num_of_samples.append(len(np.where(Y_train==x)[0]))
    print(num_of_samples)

    plt.figure(figsize=(10,5))
    plt.bar(range(0,num_classes),num_of_samples)
    plt.title("Number of images for each class")
    plt.xlabel("Class ID")
    plt.ylabel("Number of images")
    plt.show(block=False)
    plt.pause(5)
  #Testing the preprocess function  
    """ img = preprocess(X_train[30])
    print(img.shape)
    img = cv2.resize(img,(300,300))
    cv2.imshow("Preprocessed image",img)
    cv2.waitKey(0) """
    
     ##map to the preprocess function to all images
    # print(X_train[30].shape)
    X_train = np.asarray(list(map(preprocess, X_train)))
    # print(X_train[30].shape)
    X_test = np.asarray(list(map(preprocess, X_test)))
    X_validation = np.asarray(list(map(preprocess, X_validation)))

    ##Depth needed for CNN
    X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
    X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))
    X_validation = X_validation.reshape((X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1))

    
    
    
if __name__ == '__main__':
    main()

