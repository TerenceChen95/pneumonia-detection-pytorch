# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:27:30 2019

@author: tians
"""
import seaborn as sns
#import matplotlib as mpl
#mpl.use("agg")
import matplotlib.pyplot as plt
from predict import predict, imshow, process_image, plot
import torch
import os

device = torch.device('cuda:0')
#load saved model
root = os.path.join(os.getcwd(), '../..')
model = torch.load('../BEST_checkpoint.pth.tar')['model']
#predict image class and plot     
model.to(device)
img = "00b4ac1b-fa09-4dbe-b93f-7d9e52992a68"
img_path = os.path.abspath(root+'/dataset/stage_2_test_images/'+img+'.dcm')
print(img_path)
plot(img_path, model, device, 3)
#proba, flowers = predict(img_path, model, device, 3)
#plt.figure(figsize=(6,10))
'''
plt.subplot(2,1,1)
title = img_path.split('/')[6]
img_transpose =  process_image(img_path)
img_real = img_transpose.transpose((1, 2, 0))
plt.plot(img_real, title=title)

plt.subplot(2,1,2)
sns.barplot(x=proba, y=flowers, color=sns.color_palette()[0]);
plt.savefig('predict_img.png')
for i in range(3):
    print("The probalility of %s is %.30f" %(flowers[i], proba[i]))
'''
