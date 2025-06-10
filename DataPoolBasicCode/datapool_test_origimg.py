# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 16:53:36 2022

@author: DELL
"""
import cv2
import torch
import numpy as np
import pandas as pd


def xywh_to_xyxy(tensor): # [x1, y1, x2, y2, cls]
    for i in range(tensor.shape[0]):
        x1 = tensor[i][0] - tensor[i][2] / 2
        y1 = tensor[i][1] - tensor[i][3] / 2
        x2 = tensor[i][0] + tensor[i][2] / 2
        y2 = tensor[i][1] + tensor[i][3] / 2
        tensor[i][0] = x1
        tensor[i][1] = y1
        tensor[i][2] = x2
        tensor[i][3] = y2
        
    return tensor


def txt_target(txt_path, height, width): # COCO txt = xywh
    res = np.empty((0, 5))
    df = pd.read_table(txt_path, header = None)
    list1 = df.values.tolist()
    for i in range(len(list1)):
        list2 = list1[i][0].split()
        list2[0] = int(list2[0])# list2[0]=cls
        list2[1] = float(list2[1])# list2[1]=Xcen
        list2[2] = float(list2[2])# list2[2]=Ycen
        list2[3] = float(list2[3])# list2[3]=W
        list2[4] = float(list2[4])# list2[4]=H
        res = np.vstack((res, [list2[1]*width, list2[2]*height, list2[3]*width, list2[4]*height, list2[0]]))
        # [0xcen, 1ycen, 2w, 3h, 4label_ind]
    
    return res


# 将科学计数法转换为数字
np.set_printoptions(suppress=True)

img = cv2.imread('D:/AICV-YoloXReDST-SGD/datasets_sigimg/000000003220.jpg')
height, width = img.shape[0], img.shape[1]

# 设定txt文件路径
label_path = 'D:/AICV-YoloXReDST-SGD/datasets_sigann/000000003220.txt'

# 读取txt文件中的目标坐标
lab1 = txt_target(label_path, height, width)

print(lab1)

lab1 = xywh_to_xyxy(lab1)

label_tensor = torch.from_numpy(lab1)
print(label_tensor)
_COLORS = np.array([0.000, 0.447, 0.741,
                    0.850, 0.325, 0.098,
                    0.929, 0.694, 0.125,
                    0.494, 0.184, 0.556,
                    0.466, 0.674, 0.188,
                    0.301, 0.745, 0.933,
                    0.635, 0.078, 0.184,
                    0.300, 0.300, 0.300,
                    0.600, 0.600, 0.600,
                    1.000, 0.000, 0.000,
                    1.000, 0.500, 0.000,
                    0.749, 0.749, 0.000,
                    0.000, 1.000, 0.000,
                    0.000, 0.000, 1.000,
                    0.667, 0.000, 1.000,
                    0.333, 0.333, 0.000,
                    0.333, 0.667, 0.000,
                    0.333, 1.000, 0.000,]).astype(np.float32).reshape(-1, 3)
for i in range(len(label_tensor)):
    box = label_tensor[i]
    x0 = int(box[0])
    y0 = int(box[1])
    x1 = int(box[2])
    y1 = int(box[3])
    color = (_COLORS[i] * 255).astype(np.uint8).tolist()
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
cv2.imwrite('D:/AICV-YoloXReDST-SGD/datasets_visual/img_test.jpg', img)