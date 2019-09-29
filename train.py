# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 18:41:02 2019

@author: tians
"""
import torch
import numpy as np

def train_function(model, train_loader, valid_loader, criterion, optimizer, epoch, device=None, scheduler=None, train_on_gpu=True):
    
    #valid_loss_min = 0.218098#np.Inf
    valid_loss_min = 1.0
    if train_on_gpu:
        model.to(device)
    train_loss = 0.0
    valid_loss = 0.0
    if scheduler != None:
        scheduler.step()
    model.train()
    for data, target in train_loader:
        if train_on_gpu:
            data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()   
        train_loss += loss.item() * data.size(0)
    ######################    
    # validate the model #
    ######################
    model.eval()
    number_correct, number_data = 0, 0
    for data, target in valid_loader:
        if train_on_gpu:
            data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item() * data.size(0)
        ############# calculate the accurecy
        _, pred = torch.max(output, 1) 
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu \
                                else np.squeeze(correct_tensor.cpu().numpy())
        number_correct += sum(correct)
        number_data += correct.shape[0]
    ###################################
    train_loss = train_loss / len(train_loader.dataset)
    valid_loss = valid_loss / len(valid_loader.dataset)
    accuracy = (100 * number_correct / number_data)
    print('Epoch: {} \n-----------------\n \tTraining Loss: {:.6f} \t Validation Loss: {:.6f} \t accuracy : {:.4f}% '.format(epoch, train_loss, valid_loss,accuracy))
    model.to(device)
    return valid_loss
    
def save_checkpoint(epoch, epoch_since_improvement, model, optimizer, loss, best_loss, is_best):
    state = {
                'epoch' : epoch,
                'epoch_since_improvement' : epoch_since_improvement,
                'loss' : loss,
                'best_loss' : best_loss,
                'model' : model,
                'optimizer' : optimizer
            }
    path = '/home/tianshu/pneumonia/code/'
    filename = 'checkpoint.pth.tar'
    torch.save(state, path + filename)
    if is_best:
        torch.save(state, path + 'BEST_' + filename)
        
