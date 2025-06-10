# -*- coding: utf-8 -*-
"""
Created on Fri May 20 17:26:37 2022

@author: Cheng Yuxuan

Explain: 将txt标签中的目标信息读取到csv中
"""
import os
import csv
import pandas as pd


# 设定xml标签的文件夹与csv文件的保存路径
file_dir = 'D:/AICV-YoloXReDST-ADP/datasets_results/Phase_1/labels/'
csv_path = 'D:/AICV-YoloXReDST-ADP/dataanalysis/COCO_ObjectAnalysis_Phase_1.csv'

# 设定csv文件的抬头
headers=['cls','xcen','ycen','w','h','name']


with open(csv_path,'a',newline="") as csvfile0:
    writer0 = csv.writer(csvfile0)
    writer0.writerow(headers)

for label_name in os.listdir(file_dir):
    list1 = []
    df = pd.read_table(file_dir + label_name, header=None)
    list1 = df.values.tolist() # 列表化标签,有多个目标标签值的列表
    
    for i in range(len(list1)): # 拆分每个目标的标签值
        list2 = list1[i][0].split()
        list2.append(label_name) # 去掉'.txt'则标签名只会写入192
        
        with open(csv_path, 'a', newline="") as csvfile0:
            writer0 = csv.writer(csvfile0)
            writer0.writerow(list2)
