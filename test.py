# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 18:43:32 2019

@author: tians
"""
import torch
import numpy as np

train_on_gpu = True
def test_function(model, test_loader, device, criterion,classes):
    test_loss = 0
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    model.to(device)
    model.eval()
    for data,target in test_loader:
        if train_on_gpu:
            #data,target = data.cuda(),target.cuda()
            data,target = data.to(device),target.to(device)
        output = model(data)
        loss = criterion(output, target)
        # update test loss 
        test_loss += loss.item()*data.size(0)
        
        _, pred = torch.max(output, 1) 
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu \
                                else np.squeeze(correct_tensor.cpu().numpy())
        for i in range(data.shape[0]):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}'.format(test_loss))
    print('Test Accuracy (Overall): %2d%% (%2d/%2d) \n ----------------------' % (100. * np.sum(class_correct) / np.sum(class_total),np.sum(class_correct), np.sum(class_total)))
    for i in range(len(classes)):
        if class_total[i] > 0:
            print('Test Accuracy of %s : %d%% (%2d/%2d)' % (classes[i], 100 * class_correct[i] / class_total[i],np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %s: N/A (no training examples)' % (classes[str(i+1)]))
