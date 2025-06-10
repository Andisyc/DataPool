# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 16:04:21 2022

@author: Cheng Yuxuan Original

Explain: Eliminate images of certain scale range in dataset
"""

import os
import cv2
import shutil
import pandas as pd


img_dir = 'D:/AICV-YoloXReDST-ADP/datasets_results/images/'
lab_dir = 'D:/AICV-YoloXReDST-ADP/datasets_results/labels/'
mov_img = 'D:/AICV-YoloXReDST-ADP/datasets_results/extra_img_256_1024/'
mov_lab = 'D:/AICV-YoloXReDST-ADP/datasets_results/extra_lab_256_1024/'

label_list = []

for label_name in os.listdir(lab_dir):
    list1 = []
    img = cv2.imread(img_dir + label_name[0:-4] + '.jpg') # [h, w, c]
    df = pd.read_table(lab_dir + label_name, header=None)
    list1 = df.values.tolist() # 列表化标签,有多个目标标签值的列表
    
    for i in range(len(list1)): # 拆分每个目标的标签值
        list2 = list1[i][0].split()
        # list2[0] = int(list2[0]) # list2[0]=cls
        # list2[1] = float(list2[1]) # list2[1]=xcen
        # list2[2] = float(list2[2]) # list2[2]=ycen
        list2[3] = float(list2[3]) # list2[3]=w
        list2[4] = float(list2[4]) # list2[4]=h
        
        if 256 <= list2[3] * img.shape[1] * list2[4] * img.shape[0] <=1024:
            if label_name in label_list:
                pass
            else:
                label_list.append(label_name)

for i in label_list:
    shutil.move(lab_dir + i, mov_lab + i)
    shutil.move(img_dir + i[0:-4] + '.jpg', mov_img + i[0:-4] + '.jpg')
