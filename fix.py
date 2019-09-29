import os
import pandas as pd
import numpy as np

TRAIN_SAMPLES = 12000
df = pd.read_csv('/home/tianshu/pneumonia/dataset/stage_2_detailed_class_info/stage_2_detailed_class_info.csv')
print(len(df))
df = df.groupby('class').apply(lambda x: x.sample(TRAIN_SAMPLES//3).reset_index(drop=True))
print(len(df))

#np.savetxt(r'/home/tianshu/pneumonia/dataset/balanced_label.txt', df.values, fmt='%s')
f = open('/home/tianshu/pneumonia/dataset/balanced_label.txt', 'w')
arr = df.values
new_line = ''
for index in range(arr.shape[0]):
    line = arr[index]
    img = line[0]
    label = line[1]
    if label=='No Lung Opacity / Not Normal':
        label = '0'
    elif label=='Normal':
        label = '1'
    elif label=='Lung Opacity':
        label= '2'

    new_line = img + ',' + label + '\n'
    f.write(new_line)

f.close()

