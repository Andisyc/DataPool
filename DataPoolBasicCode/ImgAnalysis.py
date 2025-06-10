# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 14:30:07 2023

@author: Pilot Crysi
"""
import os
import cv2
import csv
import copy
import torch
# import torchvision
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


# 设定颜色参数向量
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

# 设定并封装类别 classes
VOC_CLASSES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", 
               "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", 
               "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")
class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))


def place_labels(object_size, _COLORS, img_temp, x0, y0, x1, y1): # 放置标签
    text = object_size
    txt_color = (0, 0, 0) if np.mean(_COLORS[0]) > 0.5 else (255, 255, 255)
    txt_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    txt_bk_color = (_COLORS[0] * 255 * 0.7).astype(np.uint8).tolist()
    if y0 - int(1.5 * txt_size[1]) - 2 <= 0:
        cv2.rectangle(img_temp, (x0 - 1, y1), (x0 + txt_size[0] + 1, y1 + int(1.5*txt_size[1])), txt_bk_color, -1)
        cv2.putText(img_temp, text, (x0, y1 + txt_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, txt_color, thickness=1)
    else:
        cv2.rectangle(img_temp, (x0 - 1, y0 - int(1.5*txt_size[1]) - 2), (x0 + txt_size[0] + 1, y0 - 1), txt_bk_color, -1)
        cv2.putText(img_temp, text, (x0, y0 - txt_size[1] + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, txt_color, thickness=1)


def save_img(img, i, res, save_path, label_name, object_size):
    img_temp = copy.deepcopy(img)
    color = (_COLORS[0] * 255).astype(np.uint8).tolist()
    res[i][0], res[i][1], res[i][2], res[i][3]
    cv2.rectangle(img_temp, (int(res[i][0]), int(res[i][1])), (int(res[i][2]), int(res[i][3])), color, 2)
    place_labels(object_size, _COLORS, img_temp, int(res[i][0]), int(res[i][1]), int(res[i][2]), int(res[i][3]))
    cv2.imwrite(save_path + str(label_name[0:-4]) + '_' + str(i) + '.jpg', img_temp)


def area_analysis(img, res, label_name, save_path, visual): # 原图, 坐标, 图片名, 保存路径, 可视化flag, 保存flag
    area, obj_size = [], [] # 初始化面积列表, 初始化目标尺寸判定列表
    for i in range(res.shape[0]): # res: 二维数组, 0 xmin, 1 ymin, 2 xmax, 3 ymax, 4 cls
        area.append((res[i][2] - res[i][0]) * (res[i][3] - res[i][1]))
    
    for i in range(len(res)):
        if (res[i][2] - res[i][0]) <= 32 and (res[i][3] - res[i][1]) <= 32:
            if visual:
                save_img(img, i, res, save_path, label_name, "Tiny")
            else:
                obj_size.append(0.0)
            continue
        
        if 0.15 >= (area[i] / (img.shape[0] * img.shape[1]))**0.5:
            if (res[i][2] - res[i][0]) <= 32 or (res[i][3] - res[i][1]) <= 32:
                if visual:
                    save_img(img, i, res, save_path, label_name, "Small")
                else:
                    obj_size.append(1.0)
                continue
            else:
                if visual:
                    save_img(img, i, res, save_path, label_name, "LowerM")
                else:
                    obj_size.append(2.0)
                continue
        
        if 0.3 >= (area[i] / (img.shape[0] * img.shape[1]))**0.5 > 0.15:
            if visual:
                save_img(img, i, res, save_path, label_name, "LowerM")
            else:
                obj_size.append(2.0)
            continue
        
        if 0.59 > (area[i] / (img.shape[0] * img.shape[1]))**0.5 > 0.3:
            if visual:
                save_img(img, i, res, save_path, label_name, "UpperM")
            else:
                obj_size.append(3.0)
            continue
        
        if (area[i] / (img.shape[0] * img.shape[1]))**0.5 >= 0.59:
            if img.shape[0] > img.shape[1]: # height > width
                if (res[i][3] - res[i][1]) > (img.shape[0] / 2): # 目标高超过一半的图片高
                    if visual:
                        save_img(img, i, res, save_path, label_name, "Large")
                    else:
                        obj_size.append(4.0) # 
                    continue
                else:
                    if visual:
                        save_img(img, i, res, save_path, label_name, "UpperM")
                    else:
                        obj_size.append(3.0)
                    continue
            else: # width > height
                if (res[i][2] - res[i][0]) > (img.shape[1] / 2): # 目标宽超过一半的图片宽
                    if visual:
                        save_img(img, i, res, save_path, label_name, "Large")
                    else:
                        obj_size.append(4.0)
                    continue
                else:
                    if visual:
                        save_img(img, i, res, save_path, label_name, "UpperM")
                    else:
                        obj_size.append(3.0)
                    continue


def xml_target(xml_path, class_to_ind):
    # 读取xml文件中所有目标
    target = ET.parse(xml_path).getroot()
    res = np.empty((0, 5))
    for obj in target.iter("object"):
        name = obj.find("name").text.strip()
        bbox = obj.find("bndbox")
        pts = ["xmin", "ymin", "xmax", "ymax"]
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(float(bbox.find(pt).text)) - 1
            bndbox.append(cur_pt)
        label_idx = class_to_ind[name]
        bndbox.append(label_idx)
        res = np.vstack((res, bndbox)) # [0xmin, 1ymin, 2xmax, 3ymax, 4label_ind]
    
    # 取得图片宽高
    width = int(target.find("size").find("width").text)
    height = int(target.find("size").find("height").text)
    
    return res, height, width


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


if __name__ == "__main__":
    # 将科学计数法转换为数字
    np.set_printoptions(suppress=True)
    
    # 设定单张图片与标签的DeBug测试文件夹路径 Single image test
    img_path = 'D:/AICV-DSTRethink/Code-DataPoolTest&Results/singleimg/'
    label_path = 'D:/AICV-DSTRethink/Code-DataPoolTest&Results/singlelab/'
    test_path = 'D:/AICV-DSTRethink/Code-DataPoolTest&Results/singleres/'

    # 可视化标签确认分割正确还是写入csv文件
    visual, wrcsv = True, False
    
    # 测试路径与csv文件保存路径不能同时存在
    if 'test_path' in dir():
        save_path = locals()['test_path']
    if 'csv_path' in dir():
        save_path = locals()['csv_path']
    
    global threshold
    threshold = 0.2
    
    for label_name in os.listdir(label_path):
        # 设定图片路径并读取图片宽高, [xcen, ycen, w, h] to [xmin, ymin, xmax, ymax]
        img = cv2.imread(img_path + label_name[0:-4] + '.jpg')
        height, width = img.shape[0], img.shape[1]
        
        # 读取txt文件或xml文件中的目标坐标
        if label_name[-4:] == '.txt':
            res = xywh_to_xyxy(txt_target(label_path + label_name, height, width))
        elif label_name[-4:] == '.xml':
            res, height, width = xml_target(label_path + label_name, class_to_ind)
        
        print(len(res))
        
        # 判断目标尺寸并拼接进坐标矩阵中
        area_analysis(img, res, label_name, save_path, visual)