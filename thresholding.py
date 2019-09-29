#This is a code for thresholding the CAM image and output a mask
import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
root = "./result/"
img_path = root+"00436515-870c-4b36-a041-de91049b9ab4-densenet121-cam.jpg"
img = mpimg.imread(img_path)
img_name = (img_path.split("/")[2]).split(".")[0]
img_id = "00436515-870c-4b36-a041-de91049b9ab4"
csv_file = "/home/tianshu/pneumonia/dataset/stage_2_train_labels/stage_2_train_labels.csv"

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def thresholding(img):
    plt.figure()
    gray = rgb2gray(img).astype("uint8")
    arr = np.asarray(gray, dtype="uint8")
    for j in range(arr.shape[1]):
        for i in range(arr.shape[0]):
            if(arr[i][j]>=60 and arr[i][j]<=180):
                arr[i][j] = 255
            else:
                arr[i][j] = 0
    
    im2, contours, hierarchy = cv2.findContours(arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    C = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area<10000 and area>1000:
            C.append(contour)
    #cv2.drawContours(img, C, -1, (0,255,0), 3)
    
    #assume only 1 bbox detected
    location = []
    for i in range(len(C)):
        location = cv2.boundingRect(C[i])
    x, y, w, h = location
    print(location)
    #resize mask to original size
    fractor = 1024.0/224.0
    for i in range(len(location)):
        location[i] = int(location[i]*fractor)
    print(location)
    #plt.figure()
    #cv2.rectangle(img,(x,y),(x+w,y+h),(255, 0, 0),2)
    
    #original_size = (1024, 1024)
    #img = cv2.resize(img, original_size, interpolation=cv2.INTER_AREA)
    #plt.imshow(img)
    #plt.savefig("BBox-%s.png" %(img_name))
    
    #draw ground truth
    import pandas as pd
    df = pd.read_csv(csv_file)
    index = 0
    for i in range(df.shape[1]):
        if(df.loc[i]['patientId']==img_id):
            index = i
            break
    
    x, y, w, h = df.iloc[index][1:-1].astype("int")
    #plt.figure()
    #cv2.rectangle(img,(x,y), (x+w, y+h), (0,255,0),2)
    #plt.imshow(img)
    #plt.savefig("IoU-%s.png" %(img_name))

thresholding(img)
