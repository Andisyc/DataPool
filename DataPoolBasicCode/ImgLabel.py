# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:25:35 2022
@author: Cheng Yuxuan
Explain: Visualize Labels
"""
import os
import cv2
import copy
import torch
import shutil
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from datapool_analysis_overlap import overlap


#%% 函数部分

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
        if name == 'face_mask':
            name = 'mask'
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

def xywh_to_xyxy(tensor): # [x1, y1, x2, y2, cls]
    for i in range(tensor.shape[0]):
        x1 = copy.deepcopy(tensor[i][0] - tensor[i][2] / 2)
        y1 = copy.deepcopy(tensor[i][1] - tensor[i][3] / 2)
        x2 = copy.deepcopy(tensor[i][0] + tensor[i][2] / 2)
        y2 = copy.deepcopy(tensor[i][1] + tensor[i][3] / 2)
        tensor[i][0] = x1
        tensor[i][1] = y1
        tensor[i][2] = x2
        tensor[i][3] = y2
        
    return tensor

def place_labels(VOC_CLASSES, _COLORS, img_temp, x0, y0, x1, y1):
    # 放置标签
    text = '{}'.format(VOC_CLASSES[int(box[4])])
    txt_color = (0, 0, 0) if np.mean(_COLORS[int(box[4])]) > 0.5 else (255, 255, 255)
    txt_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    txt_bk_color = (_COLORS[int(box[4])] * 255 * 0.7).astype(np.uint8).tolist()
    if y0 - int(1.5 * txt_size[1]) - 2 <= 0:
        cv2.rectangle(img_temp, (x0 - 1, y1), (x0 + txt_size[0] + 1, y1 + int(1.5*txt_size[1])), txt_bk_color, -1)
        cv2.putText(img_temp, text, (x0, y1 + txt_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, txt_color, thickness=1)
    else:
        cv2.rectangle(img_temp, (x0 - 1, y0 - int(1.5*txt_size[1]) - 2), (x0 + txt_size[0] + 1, y0 - 1), txt_bk_color, -1)
        cv2.putText(img_temp, text, (x0, y0 - txt_size[1] + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, txt_color, thickness=1)

def outline_objects(res, label_tensor, _COLORS):
    for i in range(len(res)):
        for k in range(len(label_tensor)):
            img_temp = copy.deepcopy(img)
            
            # 读取坐标
            box = label_tensor[k]
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
        
            # 框出目标
            color = (_COLORS[3] * 255).astype(np.uint8).tolist()
            cv2.rectangle(img_temp, (x0, y0), (x1, y1), color, 2) # cv2 reqire [h, w, c]
            
            # 保存图片
            cv2.imwrite(test_path + str(img_name[:-4]) + '_' + str(k) + '.jpg', img_temp)

#%% 主体部分
if __name__ == "__main__":
    # 将科学计数法转换为数字
    np.set_printoptions(suppress=True)

    # 设定并封装类别
    VOC_CLASSES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
                   "car", "cat", "chair", "cow", "diningtable", "dog", "horse", 
                   "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")
    # VOC_CLASSES = ("face", "mask", "person")
    class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
    
    img_path = 'D:/AICV-DSTRethink/Code-DataPoolTest&Results/singleimg/' # 图片路径
    label_path = 'D:/AICV-DSTRethink/Code-DataPoolTest&Results/singlelab/' # 标签路径
    test_path = 'D:/AICV-DSTRethink/Code-DataPoolTest&Results/labeledimg/' # 保存路径
    # shutil_image = 'D:/AICV-DSTRethink/Code-DataPoolTest&Results/overlapimages/' # 转存路径
    # shutil_label = 'D:/AICV-DSTRethink/Code-DataPoolTest&Results/overlaplabels/' # 转存路径

#%% 读取图片
    for label_name in os.listdir(label_path):
        img = cv2.imread(img_path + label_name[0:-4] + '.jpg')
        height, width = img.shape[0], img.shape[1]
        
        # 读取txt文件或xml文件中的目标坐标
        if label_name[-4:] == '.txt':
            res = xywh_to_xyxy(txt_target(label_path + label_name, height, width))
        elif label_name[-4:] == '.xml':
            res, height, width = xml_target(label_path + label_name, class_to_ind)
        
        # 读取xml文件中的目标坐标
        # res, height, width = xml_target(label_path + img_name[:-4] + '.xml', class_to_ind)
        
        # 读取txt文件中的目标坐标, [xcen, ycen, w, h] to [xmin, ymin, xmax, ymax]
        # res = xywh_to_xyxy(txt_target(label_path + img_name[:-4]+'.txt', height, width))
        
        # 逐个目标地可视化图块计算是否正确
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
                            0.333, 1.000, 0.000,
                            0.300, 0.300, 0.300,
                            0.600, 0.600, 0.600,]).astype(np.float32).reshape(-1, 3)

#%% 筛选重叠目标
        """
        # 初始化label_flag
        label_flag = 0
        
        # 如果图片只有一个目标, 则跳过该图片
        if len(res) == 1:
            continue
        else:
            # 检查是否存在相交区域,如果存在相交区域则无法使用本算法
            for i in range(res.shape[0]):
                for j in range(res.shape[0]):
                    if res[i].tolist() == res[j].tolist():
                        continue
                    else:
                        if overlap(res[i][:4], res[j][:4]) > 0:
                            break
                else: # for-else逻辑为执行完for就会执行else, 当for无法正常执行完毕时不会执行else
                    continue
                label_flag = 1 # label_flag为1则该图片存在重叠目标
                break
        
        # 检测该图片是否存在重叠
        if label_flag == 1:
            pass
        else:
            continue
        
        # 复制重叠目标图片至新路径
        # shutil.copy(img_path + label_name[0:-4] + '.jpg', shutil_image + label_name[0:-4] + '.jpg')
        # shutil.copy(label_path + label_name[0:-4] + '.xml', shutil_label + label_name[0:-4] + '.xml')
        """

#%% 标记目标
        
        # 取得坐标
        label_tensor = torch.from_numpy(res[:, :5])
        
        # 针对一张图片的多个目标循环打上方框
        for i in range(len(res)):
            img_temp = copy.deepcopy(img)
            for k in range(len(label_tensor)):
                # 读取坐标
                box = label_tensor[k]
                x0 = int(box[0])
                y0 = int(box[1])
                x1 = int(box[2])
                y1 = int(box[3])
                
                # 框出目标
                color = (_COLORS[k] * 255).astype(np.uint8).tolist()
                cv2.rectangle(img_temp, (x0, y0), (x1, y1), color, 2) # cv2 reqire [h, w, c]
                
                # 为每个目标框放置标签, 并保证标签不会超出图片边界
                # place_labels(VOC_CLASSES, _COLORS, img_temp, x0, y0, x1, y1)
                
            # 保存图片
            cv2.imwrite(test_path + str(label_name[0:-4]) + '.jpg', img_temp)
            
            # 按照标签内目标顺序在图上框出目标, 每个目标单独保存一张图
            # outline_objects(res, label_tensor, _COLORS)