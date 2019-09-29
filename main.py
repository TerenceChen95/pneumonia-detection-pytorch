# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 10:31:37 2019

@author: tians
"""

from torchvision import transforms,models
import torch
import torch.optim as optim
import torch.nn as nn
from collections import OrderedDict
import os
from train import train_function, save_checkpoint
from test import test_function
from pneumonia import Pneumonia
import pandas as pd

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda:0') 
root = os.path.abspath(os.path.join(os.getcwd(), '../..'))
#train_path = root+'/dataset/stage_2_train_images'
train_path = root+'/dataset/stage_2_train_images'
test_path = root+'/dataset/stage_2_test_images'
train_txt = root+'/dataset/train_labels.txt'
balanced_txt = root+'/dataset/balanced_label.txt'

### _: class to index
class_to_idx = {'No Lung Opacity / Not Normal': 0, 'Normal': 1, 'Lung Opacity': 2}
cat_to_name = {class_to_idx[i]: i for i in list(class_to_idx.keys())}

checkpoint = None
batch_size = 40 
start_epoch = 0  
epochs = 1200  
epochs_since_improvement = 0  
best_loss = 1

data_transforms = {
            'train': transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.CenterCrop(224),
                        transforms.RandomHorizontalFlip(), # randomly flip and rotate
                        transforms.RandomRotation(10),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ]),
    
            'test': transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        ]),
    
            'valid': transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
            }

def main():
    global epochs_since_improvement, start_epoch, best_loss, epoch, checkpoint
   
    #balance three labels sample in ../code/original/fix.py
    train_data = Pneumonia(txt=balanced_txt, mode='train', class_to_idx=class_to_idx, transforms=data_transforms['train'])
    train_data, valid_data, test_data = torch.utils.data.random_split(train_data, [7000 , 2000, 3000])
    print('train_data size: ', len(train_data))
    print('valid_data_size: ', len(valid_data))
    print('test_data_size: ', len(test_data))
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=0, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=0, shuffle=True)
    
    # we will use a pretrained model and we are going to change only the last layer
    model = models.densenet121(pretrained=True) 
    #model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = True
        
    model.classifier = nn.Sequential(OrderedDict([
        ('fcl1', nn.Linear(1024,256)),
        ('dp1', nn.Dropout(0.3)),
        ('r1', nn.ReLU()),
        ('fcl2', nn.Linear(256,32)),
        ('dp2', nn.Dropout(0.3)),
        ('r2', nn.ReLU()),
        ('fcl3', nn.Linear(32,3)),
        #('out', nn.Softmax(dim=1)),
    ]))
    
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        print('GPU is  available :)   Training on GPU ...')
    else:
        print('GPU is not available :(  Training on CPU ...')
    
    #need to remove comment after first trainning
    checkpoint = torch.load('/home/tianshu/pneumonia/code/checkpoint.pth.tar', map_location={'cuda:2': 'cuda:0'}) 
    if checkpoint is None:
        optimizer = optim.Adadelta(model.parameters())
    else:
        #load checkpoint
        #checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epoch_since_improvement']
        best_loss = checkpoint['best_loss']
        print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
    
    
    criterion = nn.CrossEntropyLoss()
    
    #train the model
    for epoch in range(start_epoch, epochs):
        val_loss = train_function(model,
                                  train_loader,
                                  valid_loader,
                                  criterion=criterion,
                                  optimizer=optimizer,
                                  train_on_gpu=train_on_gpu,
                                  epoch=epoch,
                                  device=device,
                                                      scheduler=None
                                  )
        
        # Did validation loss improve?
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
    
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
    
        else:
            epochs_since_improvement = 0
    
        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, val_loss, best_loss, is_best)
        
            
            
    test_function(model, test_loader, device, criterion, cat_to_name)

if __name__ == '__main__':
    main()
