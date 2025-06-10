# -*- coding: utf-8 -*-
"""
Created on Fri May 20 17:26:37 2022

@author: Cheng Yuxuan

Explain: 将xml标签中的目标信息读取到csv中
"""
import os
import csv
import numpy as np
import xml.etree.ElementTree as ET


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
        bndbox.insert(0, label_idx) # 第1个位置为index, 第2个位置为插入参数
        res = np.vstack((res, bndbox)) # [0label_ind, 1xmin, 2ymin, 3xmax, 4ymax]
    
    return res

# 设定xml标签的文件夹与csv文件的保存路径
file_dir = 'D:/AICV-YoloXReDST-ADP/datasets_results/labels/'
csv_path = 'D:/AICV-YoloXReDST-ADP/dataanalysis/COCO_ObjectAnalysis.csv'

# 设定csv文件的抬头
headers=['cls','xmin','ymin','xmax','ymax','name']

"""
VOC_CLASSES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", 
               "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", 
               "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")
"""
VOC_CLASSES = ('person')

class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))

with open(csv_path,'a',newline="") as csvfile0:
    writer0 = csv.writer(csvfile0)
    writer0.writerow(headers)

for label_name in os.listdir(file_dir):
    res = xml_target(file_dir + label_name, class_to_ind).tolist()
    
    for i in range(len(res)): # 拆分每个目标的标签值
        res[i].append(label_name) # 去掉'.xml'则标签名只会写入192
        
        with open(csv_path, 'a', newline="") as csvfile0:
            writer0 = csv.writer(csvfile0)
            writer0.writerow(res[i])
