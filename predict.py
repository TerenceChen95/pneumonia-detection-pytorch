# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 18:44:58 2019

@author: tians
"""
from PIL import Image
import numpy as np
import torch
import seaborn as sns
import matplotlib as plt
#plt.use('agg')
import pydicom

class_to_idx = {'No Lung Opacity / Not Normal': 0, 'Normal': 1, 'Lung Opacity':2}
cat_to_name = {class_to_idx[i]: i for i in list(class_to_idx.keys())}

def process_image(image):
    #transfer dcm to Image
    dcm_file = pydicom.read_file(image)
    img_arr = dcm_file.pixel_array
    img = Image.fromarray(img_arr).convert('RGB')
    #Scales, crops, and normalizes a PIL image for a PyTorch model,returns an Numpy array
    ##########Scales 
    if img.size[0] > img.size[1]:
        img.thumbnail((1000000, 256))
    else:
        img.thumbnail((256 ,1000000))
    #######Crops: to crop the image we have to specifiy the left,Right,button and the top pixels because the crop function take a rectongle ot pixels
    Left = (img.width - 224) / 2
    Right = Left + 224
    Top = (img.height - 244) / 2
    Buttom = Top + 224
    img = img.crop((Left, Top, Right, Buttom))
    #img = np.stack((img,)*3, axis=-1)# to repeate the the one chanel of a gray image to be RGB image 
    #img = np.repeat(image[..., np.newaxis], 3, -1)

    #normalization (divide the image by 255 so the value of the channels will be between 0 and 1 and substract the mean and divide the result by the standtared deviation)
    img = ((np.array(img) / 255) - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img = img.transpose((2, 0, 1))
    return img

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    if title:
        plt.pyplot.title(title)
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    
    image = image.transpose((1, 2, 0))
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    #image=np.transpose(image)
    ax.imshow(image)
    #return image

def predict(image_path, model, device, topk=3):
    #Predict the class (or classes) of an image using a trained deep learning model.
    img = process_image(image_path)
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    model_input = image_tensor.unsqueeze(0)
    model_input = model_input.to(device)
    probs = torch.exp(model.forward(model_input))
    
    # Top probs
    top_probs, top_labs = probs.topk(topk)
    top_probs = top_probs.cpu().detach().numpy().tolist()[0] 
    top_labs = top_labs.cpu().detach().numpy().tolist()[0]
    
    # Convert indices to classes
    #top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[lab] for lab in top_labs]

    return top_probs, top_flowers
    # TODO: Implement the code to predict the class from an image file
    
def plot(image_path,model,device, top_k=3):
    proba, flowers = predict(image_path, model, device, top_k)
    plt.pyplot.figure(figsize=(6,10))
    ax = plt.pyplot.subplot(2,1,1)
    
    title = image_path.split('/')[6]
    imshow(process_image(image_path), ax, title=title)
    
    plt.pyplot.subplot(2,1,2)
    sns.barplot(x=proba, y=flowers, color=sns.color_palette()[0]);
    plt.pyplot.savefig('predict_img.png')
    #plt.pyplot.show()
