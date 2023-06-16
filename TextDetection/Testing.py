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
import time
from Training import preprocess
from Training import Net
from Training import get_data
import argparse

parser = argparse.ArgumentParser(description='Testing the model')
parser.add_argument('-path', type=str, default='new_dataset',  help='path to the dataset')


def main():
    
    args = parser.parse_args()

    path =  'larger_dataset'
    myList = os.listdir(path)
    test_ratio = 0.2
    validation_ratio  = 0.2
    save_model_path = args.path
    print(save_model_path)

    X_train,X_test,X_validation,Y_train,Y_test,Y_validation = get_data(path,test_ratio,validation_ratio)
    
    net = Net()
    net.load_state_dict(torch.load(save_model_path))
    
    # X_train = X_train.to(net.device)
    X_test = X_test.to(net.device)
    X_validation = X_validation.to(net.device)
    
    # Y_train = Y_train.to(net.device)
    Y_test  = Y_test.to(net.device)
    Y_validation = Y_validation.to(net.device)
    
    
    criterion = nn.CrossEntropyLoss()
    
    transform = transforms.Compose([
    #  transforms.RandomCrop((8, 8)),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomRotation(degrees=(-90, 90)),
     transforms.Normalize((0.5), (0.5)),
    #  transforms.ToTensor(),
    #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     ])
    
    batch_size = 1

    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(int(len(X_test)/batch_size)):
            test_val = X_test[i*batch_size:(i+1)*batch_size].reshape(batch_size,X_test[0].shape[0],X_test[0].shape[1],X_test[0].shape[2]).clone().detach()
            test_val = transform(test_val)
            out_test = net(test_val)
            # print(out_test.shape)
            some, predicted = torch.max(out_test.data, 1)
            label = Y_test[i*batch_size:(i+1)*batch_size].long().clone().detach() # from warnings, just cloning the tensor alreadt on GPU and detaching
            total+=label.size(0)
            correct+=(predicted==label).sum().item()
            
            
    
    print(f'Accuracy of the network on the test images: {100*correct/total}%')    
    
    correct_val = 0
    total_val = 0
    
    
    with torch.no_grad():
        for i in range(int(len(X_validation)/batch_size)):
            validation_val = X_validation[i*batch_size:(i+1)*batch_size].reshape(batch_size,X_validation[0].shape[0],X_validation[0].shape[1],X_validation[0].shape[2]).clone().detach()
            validation_val = transform(validation_val)
            out_validation = net(validation_val)
            # print(out_test.shape)
            some, predicted = torch.max(out_validation.data, 1)
            label = Y_validation[i*batch_size:(i+1)*batch_size].long().clone().detach() # from warnings, just cloning the tensor alreadt on GPU and detaching
            total_val+=label.size(0)
            correct_val+=(predicted==label).sum().item()
            
            
    
    print(f'Accuracy of the network on the validation images: {100*correct_val/total_val}%') 
    
    
    
    
if __name__ == '__main__':
    main()