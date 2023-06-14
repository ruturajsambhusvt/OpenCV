import numpy as np  
import cv2
import time
import torch
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
    def __init__(self, in_channels=1, out_channels=4, kernel_size=(3,3),num_classes=10):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=(1,1), padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None) #out_dim = 30
        self.pool = nn.MaxPool2d(kernel_size=kernel_size,stride=(1,1)) #out_dim = 28
        self.conv2  = nn.Conv2d(out_channels, out_channels*2, kernel_size, stride=(1,1), padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None) #out_dim = 26, pool = 24
        self.conv3 = nn.Conv2d(out_channels*2, out_channels*4, kernel_size, stride=(1,1), padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None) #out_dim = 22, pool = 20
        self.fc1 = nn.Linear(out_channels*4*20*20,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,num_classes)
        
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        print(f"First conv and max pool {x.shape}")
        x = self.pool(F.relu(self.conv2(x)))
        print(f"Second conv and max pool {x.shape}")
        x = self.pool(F.relu(self.conv3(x))) 
        print(f"Third conv and max pool {x.shape}")
        x = torch.flatten(x,1)
        # x = x.reshape(1,6400)
        print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
        

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
    
    print(f"Before reshaping: {X_train.shape}")

    ##Depth needed for CNN
    X_train = X_train.reshape((X_train.shape[0],1,X_train.shape[1],X_train.shape[2]))
    X_test = X_test.reshape((X_test.shape[0],1,X_test.shape[1],X_test.shape[2]))
    X_validation = X_validation.reshape((X_validation.shape[0],1,X_validation.shape[1],X_validation.shape[2]))
    
    X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
    X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
    X_validation = torch.from_numpy(X_validation).type(torch.FloatTensor)
    
    Y_train = torch.from_numpy(Y_train).type(torch.FloatTensor)
    Y_test = torch.from_numpy(Y_test).type(torch.FloatTensor)
    Y_validation = torch.from_numpy(Y_validation).type(torch.FloatTensor)
    
    print(f"After reshaping X_train: {X_train.shape}")
    print(f"After reshaping Y_train: {Y_train.shape}")
    ##NN instance
    net = Net()
    x = torch.randn(1,1,32,32)
    print(f"Verifying:{net(x).shape}")
    #Loss and optimizer
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr=0.001)
    
    ##Training the model
    for epoch in range(100):
        running_loss = 0.0
        for i in range(len(X_train)):
            #zero the parameter gradients'
            print(X_train[i].shape)
            optimizer.zero_grad()
            #forward + backward + optimize
            train_val = X_train[i].reshape(1,1,32,32)
            outputs = net(train_val)
            print(f"Shape of outputs {outputs.shape}")
            print(f"Shape of Y_train {Y_train[i].shape}")
            loss = criterion(outputs,Y_train[i])
            loss.backward()
            optimizer.step()
            
            
            #print statistics
            running_loss += loss.item()
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        

print('Finished Training')

    
    
    
if __name__ == '__main__':
    main()

