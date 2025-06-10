# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 15:07:36 2022

@author: Cheng Yuxuan Original

Explain: Random Pick certain amount images
"""
import os
import random
import shutil


def check_if_picked(sample, picked_list):
    if sample in picked_list:
        return True
    else:
        return False


def re_random_choice(choice_list, picked_list):
    sample = random.choice(choice_list)
    if check_if_picked(sample, picked_list) == True:
        re_random_choice(choice_list, picked_list)
    else:
        return sample


old_img_path = 'F:/person_train_image/'
new_img_path = 'D:/AICV-YoloXReDST-ADP/datasets_results/images_val/'
old_lab_path = 'F:/person_train_labeltxt/'
new_lab_path = 'D:/AICV-YoloXReDST-ADP/datasets_results/labels_val/'
img_train = 'D:/AICV-YoloXReDST-ADP/datasets_results/images_train/'

img_list = os.listdir(old_img_path) # 所有图片路径
train_list = os.listdir(img_train) # 已抽取训练集

"""
# 冒泡排序,外层迭代驱动器,内层取值比较器
for i in range(len(img_list)):
    for j in range(len(img_list) - i - 1):
        if int(img_list[j][:-4]) > int(img_list[j+1][:-4]):
            img_list[j], img_list[j+1] = img_list[j+1], img_list[j]

print("Sorting Complete")
"""

# 复制1W个图片与标签到新路径
num_list = random.sample(range(len(img_list)), 1000)
for i in num_list:
    if i in train_list:
        i = random.choice(img_list)
    else:
        pass
    shutil.copy(old_img_path + img_list[i], new_img_path + img_list[i]) # old, new
    shutil.copy(old_lab_path + img_list[i][0:-4] + '.txt', new_lab_path + img_list[i][0:-4] + '.txt')

# 检查val是否包含train的图片(此代码需要单独取出测试)
val_list = os.listdir(new_img_path) # 已抽取验证集
for i in val_list:
    if i in train_list:
        print(i)
