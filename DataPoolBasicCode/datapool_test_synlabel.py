# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:32:56 2022

@author: DELL
"""
import cv2
import copy
import torch
import random
import numpy as np

height, width = 1080, 1920


def four_to_one_less_config1_synthetise(img1, img2, img3, img4, lab1, lab2, lab3, lab4, ratio_area, ratio_width_top, ratio_width_bot, ratio_height):
    # 左上图块如果w < h则翻转目标变成w > h
    if img1.shape[1] < img1.shape[0]:
        # 首先将标签翻转90°
        xmin_new = lab1[1] # xmin_new = ymin
        ymin_new = img1.shape[1] - lab1[2] # ymin_new = width - xmax
        xmax_new = lab1[3] # xmax_new = ymin + target_height
        ymax_new = img1.shape[1] - lab1[0] # ymax_new = width - xmax + target_width
        
        lab1[0] = xmin_new
        lab1[1] = ymin_new
        lab1[2] = xmax_new
        lab1[3] = ymax_new
        
        # 再把图片翻转90°
        img1 = np.rot90(img1)
    
    # 右上侧图块如果w > h则翻转目标变成w < h
    if img2.shape[1] > img2.shape[0]:
        # 首先将标签翻转90°
        xmin_new = lab2[1] # xmin_new = ymin
        ymin_new = img2.shape[1] - lab2[2] # ymin_new = width - xmax
        xmax_new = lab2[3] # xmax_new = ymin + target_height
        ymax_new = img2.shape[1] - lab2[0] # ymax_new = width - xmax + target_width
        
        lab2[0] = xmin_new
        lab2[1] = ymin_new
        lab2[2] = xmax_new
        lab2[3] = ymax_new
        
        # 再把图片翻转90°
        img2 = np.rot90(img2)
    
    # cv2.imread读取为[h, w, c], 将其转换为[c, h, w]
    img1 = np.transpose(img1, (2, 0, 1))
    img2 = np.transpose(img2, (2, 0, 1))
    
    # 取得二合一新宽高与缩放模式
    new_h, new_w, mode = height * pow(ratio_area, 0.5), width * pow(ratio_area, 0.5), 'nearest'
    
    # 缩放img1与img2至指定宽高
    divide_img_1 = torch.nn.functional.interpolate(torch.from_numpy(img1.copy()).unsqueeze(0), size=(int(new_h * ratio_height), int(new_w * ratio_width_top)), mode=mode).squeeze(0)
    divide_img_2 = torch.nn.functional.interpolate(torch.from_numpy(img2.copy()).unsqueeze(0), size=(int(new_h * ratio_height), int(new_w * (1-ratio_width_top))), mode=mode).squeeze(0)
    
    # 缩放img1与img2的标签
    ratios_1 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * ratio_width_top), int(new_h * ratio_height)), (img1.shape[2], img1.shape[1]))) # width, height
    ratios_2 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * (1-ratio_width_top)), int(new_h * ratio_height)), (img2.shape[2], img2.shape[1]))) # width, height
    lab1[0] = lab1[0] * ratios_1[0] # xmin * ratio_width
    lab1[2] = lab1[2] * ratios_1[0] # xmax * ratio_width
    lab1[1] = lab1[1] * ratios_1[1] # ymin * ratio_height
    lab1[3] = lab1[3] * ratios_1[1] # ymax * ratio_height
    lab2[0] = lab2[0] * ratios_2[0] # xmin * ratio_width
    lab2[2] = lab2[2] * ratios_2[0] # xmax * ratio_width
    lab2[1] = lab2[1] * ratios_2[1] # ymin * ratio_height
    lab2[3] = lab2[3] * ratios_2[1] # ymax * ratio_height
    
    # 创建二合一全零复制面板并拼接图块,syn_img.shape=[c, h, w]
    syn_img1 = torch.zeros((3, divide_img_1.shape[1], divide_img_1.shape[2]+divide_img_2.shape[2]))
    syn_img1[:3, :divide_img_1.shape[1], :divide_img_1.shape[2]].copy_(divide_img_1)
    syn_img1[:3, :divide_img_1.shape[1], divide_img_1.shape[2]:].copy_(divide_img_2)
    
    # 计算二合一偏移量并调整坐标到新位置
    offsets = [[0.0, 0.0, 0.0, 0.0], [int(new_w * ratio_width_top), 0.0, int(new_w * ratio_width_top), 0.0]] # 上左 上右
    lab1[0] = lab1[0] + offsets[0][0]
    lab1[1] = lab1[1] + offsets[0][1]
    lab1[2] = lab1[2] + offsets[0][2]
    lab1[3] = lab1[3] + offsets[0][3]
    lab2[0] = lab2[0] + offsets[1][0]
    lab2[1] = lab2[1] + offsets[1][1]
    lab2[2] = lab2[2] + offsets[1][2]
    lab2[3] = lab2[3] + offsets[1][3]
    
    temp_img = copy.deepcopy(syn_img1)
    temp_img = np.ascontiguousarray(np.transpose(temp_img.numpy(), (1, 2, 0)))
    # np.transpose会导致数组储存不连续,需要使用np.ascontiguousarray使其内存连续
    
    # 可视化拼贴图片的标签是否与目标匹配,结果正确
    label_tensor = torch.cat((torch.from_numpy(np.array([lab1])), torch.from_numpy(np.array([lab2]))), 0)
    _COLORS = np.array([0.000, 0.447, 0.741]).astype(np.float32).reshape(-1, 3)
    for i in range(len(label_tensor)):
        box = label_tensor[i]
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        color = (_COLORS[0] * 255).astype(np.uint8).tolist()
        cv2.rectangle(temp_img, (x0, y0), (x1, y1), color, 2)
    cv2.imwrite('D:/AICV-YoloXReDST-SGD/syn_img1.jpg', temp_img) # cv2.imwrite reqire [h, w, c]
    
    del temp_img
    
    # 左下图块如果w < h则翻转目标变成w > h
    if img3.shape[1] < img3.shape[0]:
        # 首先将标签翻转90°
        xmin_new = lab3[1] # xmin_new = ymin
        ymin_new = img3.shape[1] - lab3[2] # ymin_new = width - xmax
        xmax_new = lab3[3] # xmax_new = ymin + target_height
        ymax_new = img3.shape[1] - lab3[0] # ymax_new = width - xmax + target_width
        
        lab3[0] = xmin_new
        lab3[1] = ymin_new
        lab3[2] = xmax_new
        lab3[3] = ymax_new
        
        # 再把图片翻转90°
        img3 = np.rot90(img3)
    
    # 右下侧图块如果w > h则翻转目标变成w < h
    if img4.shape[1] > img4.shape[0]:
        # 首先将标签翻转90°
        xmin_new = lab4[1] # xmin_new = ymin
        ymin_new = img4.shape[1] - lab4[2] # ymin_new = width - xmax
        xmax_new = lab4[3] # xmax_new = ymin + target_height
        ymax_new = img4.shape[1] - lab4[0] # ymax_new = width - xmax + target_width
        
        lab4[0] = xmin_new
        lab4[1] = ymin_new
        lab4[2] = xmax_new
        lab4[3] = ymax_new
        
        # 再把图片翻转90°
        img4 = np.rot90(img4)
    
    # cv2.imread读取为[h, w, c], 将其转换为[c, h, w]
    img3 = np.transpose(img3, (2, 0, 1))
    img4 = np.transpose(img4, (2, 0, 1))
    
    # 缩放img3与img4至指定宽高
    divide_img_3 = torch.nn.functional.interpolate(torch.from_numpy(img3.copy()).unsqueeze(0), size=(int(new_h * (1-ratio_height)), int(new_w * ratio_width_bot)), mode=mode).squeeze(0)
    divide_img_4 = torch.nn.functional.interpolate(torch.from_numpy(img4.copy()).unsqueeze(0), size=(int(new_h * (1-ratio_height)), int(new_w * (1-ratio_width_bot))), mode=mode).squeeze(0)
    
    # 缩放img3与img4的标签
    ratios_3 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * ratio_width_bot), int(new_h * (1-ratio_height))), (img3.shape[2], img3.shape[1]))) # width, height
    ratios_4 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * (1-ratio_width_bot)), int(new_h * (1-ratio_height))), (img4.shape[2], img4.shape[1]))) # width, height
    lab3[0] = lab3[0] * ratios_3[0] # xmin * ratio_width
    lab3[2] = lab3[2] * ratios_3[0] # xmax * ratio_width
    lab3[1] = lab3[1] * ratios_3[1] # ymin * ratio_height
    lab3[3] = lab3[3] * ratios_3[1] # ymax * ratio_height
    lab4[0] = lab4[0] * ratios_4[0] # xmin * ratio_width
    lab4[2] = lab4[2] * ratios_4[0] # xmax * ratio_width
    lab4[1] = lab4[1] * ratios_4[1] # ymin * ratio_height
    lab4[3] = lab4[3] * ratios_4[1] # ymax * ratio_height
    
    print(int(new_w * ratio_width_top) + int(new_w * (1-ratio_width_top)))
    print(int(new_w * ratio_width_bot) + int(new_w * (1-ratio_width_bot)))
    
    # 创建二合一全零复制面板并拼接图块,syn_img.shape=[c, h, w]
    syn_img2 = torch.zeros((3, divide_img_3.shape[1], divide_img_3.shape[2]+divide_img_4.shape[2]))
    syn_img2[:3, :divide_img_3.shape[1], :divide_img_3.shape[2]].copy_(divide_img_3)
    syn_img2[:3, :divide_img_3.shape[1], divide_img_3.shape[2]:].copy_(divide_img_4)
    
    # 计算二合一偏移量并调整坐标到新位置
    offsets = [[0.0, 0.0, 0.0, 0.0], [int(new_w * ratio_width_bot), 0.0, int(new_w * ratio_width_bot), 0.0]] # 下左 下右
    lab3[0] = lab3[0] + offsets[0][0]
    lab3[1] = lab3[1] + offsets[0][1]
    lab3[2] = lab3[2] + offsets[0][2]
    lab3[3] = lab3[3] + offsets[0][3]
    lab4[0] = lab4[0] + offsets[1][0]
    lab4[1] = lab4[1] + offsets[1][1]
    lab4[2] = lab4[2] + offsets[1][2]
    lab4[3] = lab4[3] + offsets[1][3]
    
    temp_img = copy.deepcopy(syn_img2)
    temp_img = np.ascontiguousarray(np.transpose(temp_img.numpy(), (1, 2, 0)))
    # np.transpose会导致数组储存不连续,需要使用np.ascontiguousarray使其内存连续
    
    # 可视化拼贴图片的标签是否与目标匹配,结果正确
    label_tensor = torch.cat((torch.from_numpy(np.array([lab3])), torch.from_numpy(np.array([lab4]))), 0)
    _COLORS = np.array([0.000, 0.447, 0.741]).astype(np.float32).reshape(-1, 3)
    for i in range(len(label_tensor)):
        box = label_tensor[i]
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        color = (_COLORS[0] * 255).astype(np.uint8).tolist()
        cv2.rectangle(temp_img, (x0, y0), (x1, y1), color, 2)
    cv2.imwrite('D:/AICV-YoloXReDST-SGD/syn_img2.jpg', temp_img) # cv2.imwrite reqire [h, w, c]
    
    del temp_img
    
    # 创建二合一全零复制面板并拼接图块,syn_img.shape=[c, h, w]
    syn_img3 = torch.zeros((3, syn_img1.shape[1]+syn_img2.shape[1], syn_img1.shape[2] if syn_img1.shape[2] > syn_img2.shape[2] else syn_img2.shape[2]))
    syn_img3[:3, :syn_img1.shape[1], :divide_img_1.shape[2]+divide_img_2.shape[2]].copy_(syn_img1)
    syn_img3[:3, syn_img1.shape[1]:, :divide_img_3.shape[2]+divide_img_4.shape[2]].copy_(syn_img2)
    
    # 计算二合一偏移量并调整坐标到新位置
    offsets = [[0.0, int(new_h * ratio_height), 0.0, int(new_h * ratio_height)], [0.0, int(new_h * ratio_height), 0.0, int(new_h * ratio_height)]] # 下左 下右
    lab3[0] = lab3[0] + offsets[0][0]
    lab3[1] = lab3[1] + offsets[0][1]
    lab3[2] = lab3[2] + offsets[0][2]
    lab3[3] = lab3[3] + offsets[0][3]
    lab4[0] = lab4[0] + offsets[1][0]
    lab4[1] = lab4[1] + offsets[1][1]
    lab4[2] = lab4[2] + offsets[1][2]
    lab4[3] = lab4[3] + offsets[1][3]
    
    # 填充图片至input_size, default=[1080, 1920] [height, width]
    if syn_img3.shape[1] < height: # input_size=[height, width], pad_img=[c, h, w]
        dh = height - syn_img3.shape[1]
        dh /= 2
        pad_top, pad_bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    else:
        pad_top, pad_bottom = 0, 0
    
    if syn_img3.shape[2] < width: # input_size=[height, width], pad_img=[c, h, w]
        dw = width - syn_img3.shape[2]
        dw /= 2
        pad_left, pad_right = int(round(dw - 0.1)), int(round(dw + 0.1))
    else:
        pad_left, pad_right = 0, 0
    
    syn_img3 = cv2.copyMakeBorder(np.transpose(syn_img3.numpy(), (1, 2, 0)), 
                                  pad_top, pad_bottom, pad_left, pad_right, 
                                  cv2.BORDER_CONSTANT, value=(114, 114, 114)) # syn_img = [h, w, c]
    
    # 为坐标加上填充量
    lab1[0], lab1[2] = lab1[0] + pad_left, lab1[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab1[1], lab1[3] = lab1[1] + pad_top, lab1[3] + pad_top # ymin + pad_top, ymax + pad_top
    lab2[0], lab2[2] = lab2[0] + pad_left, lab2[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab2[1], lab2[3] = lab2[1] + pad_top, lab2[3] + pad_top # ymin + pad_top, ymax + pad_top
    lab3[0], lab3[2] = lab3[0] + pad_left, lab3[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab3[1], lab3[3] = lab3[1] + pad_top, lab3[3] + pad_top # ymin + pad_top, ymax + pad_top
    lab4[0], lab4[2] = lab4[0] + pad_left, lab4[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab4[1], lab4[3] = lab4[1] + pad_top, lab4[3] + pad_top # ymin + pad_top, ymax + pad_top
    
    # 可视化拼贴图片的标签是否与目标匹配,结果正确
    label_tensor = torch.cat((torch.from_numpy(np.array([lab1])), torch.from_numpy(np.array([lab2])), torch.from_numpy(np.array([lab3])), torch.from_numpy(np.array([lab4]))), 0)
    _COLORS = np.array([0.000, 0.447, 0.741]).astype(np.float32).reshape(-1, 3)
    for i in range(len(label_tensor)): # [cls, x1, y1, x2, y2]
        box = label_tensor[i]
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        color = (_COLORS[0] * 255).astype(np.uint8).tolist()
        cv2.rectangle(syn_img3, (x0, y0), (x1, y1), color, 2)
    
    cv2.imwrite('D:/AICV-YoloXReDST-SGD/syn_img3.jpg', syn_img3) # cv2.imwrite reqire [h, w, c]


img1 = cv2.imread('D:/AICV-YoloXReDST-SGD/chub_lianyu_845.jpg')
img2 = cv2.imread('D:/AICV-YoloXReDST-SGD/chub_lianyu_845.jpg')
img3 = cv2.imread('D:/AICV-YoloXReDST-SGD/chub_lianyu_845.jpg')
img4 = cv2.imread('D:/AICV-YoloXReDST-SGD/chub_lianyu_845.jpg')

lab1 = [846, 419, 967, 650]
lab2 = [846, 419, 967, 650]
lab3 = [846, 419, 967, 650]
lab4 = [846, 419, 967, 650]

ratio_area = random.uniform(1/2, 1) # 生成[1/2, 1)之间的随机数,用于取得合成图与原图的比例
ratio_width_top = random.uniform(1/2, 3/5) # 生成[1/2, 3/5)之间的随机数,用于左上右上两图块之间的比例
ratio_width_bot = random.uniform(1/2, 3/5)
ratio_height = random.uniform(2/5, 3/5) # 生成[2/5, 3/5)之间的随机数,用于上下三图块之间的比例
print(ratio_area, ratio_width_top, ratio_width_bot, ratio_height)

four_to_one_less_config1_synthetise(img1, img2, img3, img4, lab1, lab2, lab3, lab4, ratio_area, ratio_width_top, ratio_width_bot, ratio_height)