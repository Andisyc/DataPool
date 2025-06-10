# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:22:59 2022

@author: Cheng Yuxuan Original
"""
import cv2
import csv
import torch
import random
import numpy as np
import pandas as pd


img_file = 'F:/VOCtrainval_images/'
save_dir = 'D:/AICV-YoloXReDST-ADP/datasets_result/'


# 三合一解析算法,输入所需目标的尺寸范围并设定三图块的面积比例
def three_to_one_less_config1_analysis(scope, height, width): # scope: 像素数量的范围, 本函数默认scope=[0, 1024], 即MSCOCO定义的小目标
    # 生成拼接图片/原始图片面积随机比值&两图块面积随机比值
    ratio_area = random.uniform(1/2, 1) # 生成[1/2, 1)之间的随机数,用于取得合成图与原图的比例
    ratio_width = random.uniform(1/2, 3/5) # 生成[1/2, 3/5)之间的随机数,用于左上右上两图块之间的比例
    ratio_height = random.uniform(2/5, 3/5) # 生成[2/5, 3/5)之间的随机数,用于上下三图块之间的比例

    # 计算需要抽取的两块拼图的条件,拼图面积最小值,拼图面积/目标面积比例范围
    search_patch_1_low = height * width * ratio_area * 0.8 * ratio_height * ratio_width
    search_patch_2_low = height * width * ratio_area * 0.8 * ratio_height * (1-ratio_width)
    search_patch_3_low = height * width * ratio_area * 0.8 * (1-ratio_height)
    
    if scope[0] == 0: # 得到所需拼图面积与目标面积比例范围
        proportion_1_low = search_patch_1_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_1_high = 'infinite' # 由于最低值为0, 因为比例为无限大
        proportion_2_low = search_patch_2_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_2_high = 'infinite' # 由于最低值为0, 因为比例为无限大
        proportion_3_low = search_patch_3_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_3_high = 'infinite' # 由于最低值为0, 因为比例为无限大
    else:
        proportion_1_low = search_patch_1_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_1_high = search_patch_1_low / scope[0] # 拼图面积 / 目标面积最小值
        proportion_2_low = search_patch_2_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_2_high = search_patch_2_low / scope[0] # 拼图面积 / 目标面积最小值
        proportion_3_low = search_patch_3_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_3_high = search_patch_3_low / scope[0] # 拼图面积 / 目标面积最小值
        
    return (ratio_area, ratio_width, ratio_height, 
           [search_patch_1_low, proportion_1_low, proportion_1_high], 
           [search_patch_2_low, proportion_2_low, proportion_2_high],
           [search_patch_3_low, proportion_3_low, proportion_3_high])


class three_to_one_less_config1_search:
    def __init__(self, list1, list2, list3, csv_path, ratio_area, ratio_width, ratio_height, height, width):
        self.list1 = list1
        self.list2 = list2
        self.list3 = list3
        
        self.csv_path = csv_path
        
        self.ratio_area = ratio_area
        self.ratio_width = ratio_width
        self.ratio_height = ratio_height
        self.height = height
        self.width = width
    
    def do_search(self):
        search_area_1_low, proportion_1_low, proportion_1_high = self.list1
        search_area_2_low, proportion_2_low, proportion_2_high = self.list2
        search_area_3_low, proportion_3_low, proportion_3_high = self.list3
    
        self.potential_area_1, self.potential_area_2, self.potential_area_3 = [], [], []
    
        df = pd.read_table(self.csv_path, header=None)
        list_target = df.values.tolist()
    
        for i in range(len(list_target)):
            list2 = list_target[i][0].split(",")
            list2[0] = list2[0] # xml name
            list2[1] = float(list2[1]) # res xmin
            list2[2] = float(list2[2]) # res ymin
            list2[3] = float(list2[3]) # res xmax
            list2[4] = float(list2[4]) # res ymax
            list2[5] = int(float(list2[5])) # res cls
            list2[6] = int(float(list2[6])) # cut xmin
            list2[7] = int(float(list2[7])) # cut ymin
            list2[8] = int(float(list2[8])) # cut xmax
            list2[9] = int(float(list2[9])) # cut ymax
            list2[10] = float(list2[10]) # ratio target_area
            list2[11] = float(list2[11]) # ratio cutout_area
            list2[12] = float(list2[12]) # ratio ratio_area
        
            # 判断该目标是否满足search_area_1的条件
            if proportion_1_high == 'infinite':
                if list2[12] > proportion_1_low and list2[11] > search_area_1_low:
                    self.potential_area_1.append(list2)
            else:
                if proportion_1_high > list2[12] > proportion_1_low and list2[11] > search_area_1_low:
                    self.potential_area_1.append(list2)
        
            # 判断该目标是否满足search_area_2的条件
            if proportion_2_high == 'infinite':
                if list2[12] > proportion_2_low and list2[11] > search_area_2_low:
                    self.potential_area_2.append(list2)
            else:
                if proportion_2_high > list2[12] > proportion_2_low and list2[11] > search_area_2_low:
                    self.potential_area_2.append(list2)
        
            # 判断该目标是否满足search_area_3的条件
            if proportion_3_high == 'infinite':
                if list2[12] > proportion_3_low and list2[11] > search_area_3_low:
                    self.potential_area_3.append(list2)
            else:
                if proportion_3_high > list2[12] > proportion_3_low and list2[11] > search_area_3_low:
                    self.potential_area_3.append(list2)
        
    
        assert len(self.potential_area_1) != 0, 'No patch match the search condition 1'
        assert len(self.potential_area_2) != 0, 'No patch match the search condition 2'
        assert len(self.potential_area_3) != 0, 'No patch match the search condition 3'
        
        print(len(self.potential_area_1))
        print(len(self.potential_area_2))
        print(len(self.potential_area_3))
        
        # 初始化使用过的图块列表
        self.patch1_used, self.patch2_used, self.patch3_used = [], [], []
        
        # 初始化计数器
        count = 0
        
        # 每个列表中的图片只能使用一次,不能重复出现在多张合成图中
        if len(self.potential_area_1) != 0 and len(self.potential_area_2) != 0 and len(self.potential_area_3) != 0:
            self.i, self.j, self.k = random.choice(self.potential_area_1), random.choice(self.potential_area_2), random.choice(self.potential_area_3)
            while(len(self.potential_area_1) != 0 and len(self.potential_area_2) != 0 and len(self.potential_area_3) != 0):
                # 必须同时满足相互不同且非已用图块才能继续进行
                self.i, self.j, self.k = self.none_same_patch()
                self.i, self.j, self.k, self.patch1_used, self.patch2_used, self.patch3_used = self.none_used_patch()
            
                img1 = cv2.imread(img_file + self.i[0][:-4] + '.jpg')
                img2 = cv2.imread(img_file + self.j[0][:-4] + '.jpg')
                img3 = cv2.imread(img_file + self.k[0][:-4] + '.jpg')
                img1 = img1[self.i[7]:self.i[9], self.i[6]:self.i[8]]
                lab1 = [self.i[1]-self.i[6], self.i[2]-self.i[7], self.i[3]-self.i[6], self.i[4]-self.i[7], self.i[5]]
                img2 = img2[self.j[7]:self.j[9], self.j[6]:self.j[8]]
                lab2 = [self.j[1]-self.j[6], self.j[2]-self.j[7], self.j[3]-self.j[6], self.j[4]-self.j[7], self.j[5]]
                img3 = img3[self.k[7]:self.k[9], self.k[6]:self.k[8]]
                lab3 = [self.k[1]-self.k[6], self.k[2]-self.k[7], self.k[3]-self.k[6], self.k[4]-self.k[7], self.k[5]]
                        
                three_to_one_less_config1_synthetise(img1, img2, img3, lab1, lab2, lab3,
                                                     self.ratio_area, self.ratio_width, self.ratio_height, 
                                                     count, self.height, self.width)
                
                count = count + 1
            
                # 储存&移除旧图块, 抽取新图块
                self.patch1_used, self.potential_area_1, self.i = self.store_remove_pick(self.patch1_used, self.potential_area_1, self.i)
                self.patch2_used, self.potential_area_2, self.j = self.store_remove_pick(self.patch2_used, self.potential_area_2, self.j)
                self.patch3_used, self.potential_area_3, self.k = self.store_remove_pick(self.patch3_used, self.potential_area_3, self.k)
    
    
    def store_remove_pick(self, store_list, remove_list, patch_info):
        store_list.append(patch_info)
        remove_list.remove(patch_info)
        patch_info = random.choice(remove_list)
        
        return store_list, remove_list, patch_info


    def remove_pick(self, remove_list, patch_info):
        remove_list.remove(patch_info)
        patch_info = random.choice(remove_list)
        
        return remove_list, patch_info
        

    def compare_info(self, i, j):
        if (i[0] == j[0] and i[1] == j[1] and i[2] == j[2] and i[3] == j[3] and i[4] == j[4]) == False:
            return True # False说明两列表不等, 不等是我们想要的
        else:
            return False # True说明两列表相等, 相等则我们不想要
    
    def none_same_patch(self):
        # 确保同时使用的四个图块不会存在相同, 如果相同则重新选择图块  
        while((self.compare_info(self.i, self.j) and self.compare_info(self.i, self.k) and self.compare_info(self.j, self.k)) == False):
            self.i, self.j, self.k = random.choice(self.potential_area_1), random.choice(self.potential_area_2), random.choice(self.potential_area_3)
        
        return self.i, self.j, self.k # 成功跳出while循环即说明互不相同
    
    def none_used_patch(self):
        # 确保不会再使用到之前使用过的图块, 如果出现即重新选择图块, 并且删除当前图块
        while(self.i in self.patch2_used or self.i in self.patch3_used):
            self.potential_area_1, self.i = self.remove_pick(self.potential_area_1, self.i)
            self.i, self.j, self.k = self.none_same_patch()
            self.i, self.j, self.k, self.patch1_used, self.patch2_used, self.patch3_used = self.none_used_patch()
        while(self.j in self.patch1_used or self.j in self.patch3_used):
            self.potential_area_2, self.j = self.remove_pick(self.potential_area_2, self.j)
            self.i, self.j, self.k = self.none_same_patch()
            self.i, self.j, self.k, self.patch1_used, self.patch2_used, self.patch3_used = self.none_used_patch()
        while(self.k in self.patch1_used or self.k in self.patch2_used):
            self.potential_area_3, self.k = self.remove_pick(self.potential_area_3, self.k)
            self.i, self.j, self.k = self.none_same_patch()
            self.i, self.j, self.k, self.patch1_used, self.patch2_used, self.patch3_used = self.none_used_patch()

        return self.i, self.j, self.k, self.patch1_used, self.patch2_used, self.patch3_used


def three_to_one_less_config1_synthetise(img1, img2, img3, lab1, lab2, lab3, ratio_area, ratio_width, ratio_height, count, height, width):
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
    divide_img_1 = torch.nn.functional.interpolate(torch.from_numpy(img1.copy()).unsqueeze(0), size=(int(new_h * ratio_height), int(new_w * ratio_width)), mode=mode).squeeze(0)
    divide_img_2 = torch.nn.functional.interpolate(torch.from_numpy(img2.copy()).unsqueeze(0), size=(int(new_h * ratio_height), int(new_w * (1-ratio_width))), mode=mode).squeeze(0)
    
    # 缩放img1与img2的标签
    ratios_1 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * ratio_width), int(new_h * ratio_height)), (img1.shape[2], img1.shape[1]))) # width, height
    ratios_2 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * (1-ratio_width)), int(new_h * ratio_height)), (img2.shape[2], img2.shape[1]))) # width, height
    lab1[0] = lab1[0] * ratios_1[0] # xmin * ratio_width
    lab1[2] = lab1[2] * ratios_1[0] # xmax * ratio_width
    lab1[1] = lab1[1] * ratios_1[1] # ymin * ratio_height
    lab1[3] = lab1[3] * ratios_1[1] # ymax * ratio_height
    lab2[0] = lab2[0] * ratios_2[0] # xmin * ratio_width
    lab2[2] = lab2[2] * ratios_2[0] # xmax * ratio_width
    lab2[1] = lab2[1] * ratios_2[1] # ymin * ratio_height
    lab2[3] = lab2[3] * ratios_2[1] # ymax * ratio_height
    
    # 创建二合一全零复制面板并拼接图块,syn_img.shape=[c, h, w]
    syn_img = torch.zeros((3, divide_img_1.shape[1], divide_img_1.shape[2]+divide_img_2.shape[2]))
    syn_img[:3, :divide_img_1.shape[1], :divide_img_1.shape[2]].copy_(divide_img_1)
    syn_img[:3, :divide_img_1.shape[1], divide_img_1.shape[2]:].copy_(divide_img_2)
    
    # 计算二合一偏移量并调整坐标到新位置
    offsets = [[0.0, 0.0, 0.0, 0.0], [int(new_w * ratio_width), 0.0, int(new_w * ratio_width), 0.0]] # 上左 上右
    lab1[0] = lab1[0] + offsets[0][0]
    lab1[1] = lab1[1] + offsets[0][1]
    lab1[2] = lab1[2] + offsets[0][2]
    lab1[3] = lab1[3] + offsets[0][3]
    lab2[0] = lab2[0] + offsets[1][0]
    lab2[1] = lab2[1] + offsets[1][1]
    lab2[2] = lab2[2] + offsets[1][2]
    lab2[3] = lab2[3] + offsets[1][3]
    
    # 判断二合一合成图宽高以决定如何拼接第三张图块
    if syn_img.shape[2] > syn_img.shape[1]: # 二合一合成图w > h则需要第3个图块w > h
        if img3.shape[0] > img3.shape[1]: # 当第3个图块h > w时翻转图块
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
    
    del xmin_new, ymin_new, xmax_new, ymax_new
    
    # cv2.imread读取为[h, w, c], 将其转换为[c, h, w]
    img3 = np.transpose(img3, (2, 0, 1))
    
    # 缩放img3至指定宽高
    divide_img_3 = torch.nn.functional.interpolate(torch.from_numpy(img3).unsqueeze(0), size=(int(new_h * (1-ratio_height)), int(new_w*ratio_width)+int(new_w*(1-ratio_width))), mode=mode).squeeze(0)
    
    # 缩放img3的标签
    ratios_3 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w*ratio_width)+int(new_w*(1-ratio_width)), int(new_h * (1-ratio_height))), (img3.shape[2], img3.shape[1]))) # width, height
    lab3[0] = lab3[0] * ratios_3[0] # xmin * ratio_width
    lab3[2] = lab3[2] * ratios_3[0] # xmax * ratio_width
    lab3[1] = lab3[1] * ratios_3[1] # ymin * ratio_height
    lab3[3] = lab3[3] * ratios_3[1] # ymax * ratio_height
    
    # 创建三合一全零复制面板并拼接图块,syn_img.shape=[c, h, w]
    temp = syn_img
    syn_img = torch.zeros((3, divide_img_1.shape[1]+divide_img_3.shape[1], divide_img_1.shape[2]+divide_img_2.shape[2]))
    syn_img[:3, :divide_img_1.shape[1], :divide_img_1.shape[2]+divide_img_2.shape[2]].copy_(temp)
    syn_img[:3, divide_img_1.shape[1]:, :divide_img_1.shape[2]+divide_img_2.shape[2]].copy_(divide_img_3)
    del temp
    
    # 计算二合一偏移量并调整坐标到新位置
    offsets = [[0.0, int(new_h * ratio_height), 0.0, int(new_h * ratio_height)]] # 正下
    lab3[0] = lab3[0] + offsets[0][0]
    lab3[1] = lab3[1] + offsets[0][1]
    lab3[2] = lab3[2] + offsets[0][2]
    lab3[3] = lab3[3] + offsets[0][3]
    
    # 填充图片至input_size, default=[1080, 1920] [height, width]
    if syn_img.shape[1] < height: # input_size=[height, width], pad_img=[c, h, w]
        dh = height - syn_img.shape[1]
        dh /= 2
        pad_top, pad_bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    else:
        pad_top, pad_bottom = 0, 0
    
    if syn_img.shape[2] < width: # input_size=[height, width], pad_img=[c, h, w]
        dw = width - syn_img.shape[2]
        dw /= 2
        pad_left, pad_right = int(round(dw - 0.1)), int(round(dw + 0.1))
    else:
        pad_left, pad_right = 0, 0
    
    syn_img = cv2.copyMakeBorder(np.transpose(syn_img.numpy(), (1, 2, 0)), 
                                 pad_top, pad_bottom, pad_left, pad_right, 
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114)) # syn_img = [h, w, c]
    
    # 为坐标加上填充量
    lab1[0], lab1[2] = lab1[0] + pad_left, lab1[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab1[1], lab1[3] = lab1[1] + pad_top, lab1[3] + pad_top # ymin + pad_top, ymax + pad_top
    lab2[0], lab2[2] = lab2[0] + pad_left, lab2[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab2[1], lab2[3] = lab2[1] + pad_top, lab2[3] + pad_top # ymin + pad_top, ymax + pad_top
    lab3[0], lab3[2] = lab3[0] + pad_left, lab3[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab3[1], lab3[3] = lab3[1] + pad_top, lab3[3] + pad_top # ymin + pad_top, ymax + pad_top
    
    cv2.imwrite(save_dir + 'syn_img_' + str(count) + '.jpg', syn_img) # cv2.imwrite reqire [h, w, c]
    # return syn_img # [c, h, w]


# 三合一解析算法,输入所需目标的尺寸范围并设定三图块的面积比例
def three_to_one_less_config2_analysis(scope, height, width): # scope: 像素数量的范围, 本函数默认scope=[0, 1024], 即MSCOCO定义的小目标
    # 生成拼接图片/原始图片面积随机比值&两图块面积随机比值
    ratio_area = random.uniform(1/2, 1) # 生成[1/2, 1)之间的随机数,用于取得合成图与原图的比例
    ratio_width = random.uniform(2/5, 1/2) # 生成[1/2, 3/5)之间的随机数,用于左上右上两图块之间的比例
    ratio_height = random.uniform(1/2, 3/5) # 生成[2/5, 3/5)之间的随机数,用于上下三图块之间的比例
    
    # 计算需要抽取的两块拼图的条件,拼图面积最小值,拼图面积/目标面积比例范围
    search_patch_1_low = height * width * ratio_area * 0.8 * ratio_height * ratio_width
    search_patch_2_low = height * width * ratio_area * 0.8 * (1-ratio_height) * ratio_width
    search_patch_3_low = height * width * ratio_area * 0.8 * (1-ratio_width)
    
    if scope[0] == 0: # 得到所需拼图面积与目标面积比例范围
        proportion_1_low = search_patch_1_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_1_high = 'infinite' # 由于最低值为0, 因为比例为无限大
        proportion_2_low = search_patch_2_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_2_high = 'infinite' # 由于最低值为0, 因为比例为无限大
        proportion_3_low = search_patch_3_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_3_high = 'infinite' # 由于最低值为0, 因为比例为无限大
    else:
        proportion_1_low = search_patch_1_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_1_high = search_patch_1_low / scope[0] # 拼图面积 / 目标面积最小值
        proportion_2_low = search_patch_2_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_2_high = search_patch_2_low / scope[0] # 拼图面积 / 目标面积最小值
        proportion_3_low = search_patch_3_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_3_high = search_patch_3_low / scope[0] # 拼图面积 / 目标面积最小值
        
    return (ratio_area, ratio_width, ratio_height, 
           [search_patch_1_low, proportion_1_low, proportion_1_high], 
           [search_patch_2_low, proportion_2_low, proportion_2_high],
           [search_patch_3_low, proportion_3_low, proportion_3_high])


class three_to_one_less_config2_search:
    def __init__(self, list1, list2, list3, csv_path, ratio_area, ratio_width, ratio_height, height, width):
        self.list1 = list1
        self.list2 = list2
        self.list3 = list3
        
        self.csv_path = csv_path
        
        self.ratio_area = ratio_area
        self.ratio_width = ratio_width
        self.ratio_height = ratio_height
        self.height = height
        self.width = width
        
    def do_search(self):
        search_area_1_low, proportion_1_low, proportion_1_high = self.list1
        search_area_2_low, proportion_2_low, proportion_2_high = self.list2
        search_area_3_low, proportion_3_low, proportion_3_high = self.list3
    
        self.potential_area_1, self.potential_area_2, self.potential_area_3 = [], [], []
        
        df = pd.read_table(self.csv_path, header=None)
        list_target = df.values.tolist()
    
        for i in range(len(list_target)):
            list2 = list_target[i][0].split(",")
            list2[0] = list2[0] # xml name
            list2[1] = float(list2[1]) # res xmin
            list2[2] = float(list2[2]) # res ymin
            list2[3] = float(list2[3]) # res xmax
            list2[4] = float(list2[4]) # res ymax
            list2[5] = int(float(list2[5])) # res cls
            list2[6] = int(float(list2[6])) # cut xmin
            list2[7] = int(float(list2[7])) # cut ymin
            list2[8] = int(float(list2[8])) # cut xmax
            list2[9] = int(float(list2[9])) # cut ymax
            list2[10] = float(list2[10]) # ratio target_area
            list2[11] = float(list2[11]) # ratio cutout_area
            list2[12] = float(list2[12]) # ratio ratio_area
        
            # 判断该目标是否满足search_area_1的条件
            if proportion_1_high == 'infinite':
                if list2[12] > proportion_1_low and list2[11] > search_area_1_low:
                    self.potential_area_1.append(list2)
            else:
                if proportion_1_high > list2[12] > proportion_1_low and list2[11] > search_area_1_low:
                    self.potential_area_1.append(list2)
        
            # 判断该目标是否满足search_area_2的条件
            if proportion_2_high == 'infinite':
                if list2[12] > proportion_2_low and list2[11] > search_area_2_low:
                    self.potential_area_2.append(list2)
            else:
                if proportion_2_high > list2[12] > proportion_2_low and list2[11] > search_area_2_low:
                    self.potential_area_2.append(list2)
        
            # 判断该目标是否满足search_area_3的条件
            if proportion_3_high == 'infinite':
                if list2[12] > proportion_3_low and list2[11] > search_area_3_low:
                    self.potential_area_3.append(list2)
            else:
                if proportion_3_high > list2[12] > proportion_3_low and list2[11] > search_area_3_low:
                    self.potential_area_3.append(list2)
        
        assert len(self.potential_area_1) != 0, 'No patch match the search condition 1'
        assert len(self.potential_area_2) != 0, 'No patch match the search condition 2'
        assert len(self.potential_area_3) != 0, 'No patch match the search condition 3'
        
        print(len(self.potential_area_1))
        print(len(self.potential_area_2))
        print(len(self.potential_area_3))
        
        # 初始化使用过的图块列表
        self.patch1_used, self.patch2_used, self.patch3_used = [], [], []
        
        # 初始化计数器
        count = 0
    
        # 每个列表中的图片只能使用一次,不能重复出现在多张合成图中
        if len(self.potential_area_1) != 0 and len(self.potential_area_2) != 0 and len(self.potential_area_3) != 0:
            self.i, self.j, self.k = random.choice(self.potential_area_1), random.choice(self.potential_area_2), random.choice(self.potential_area_3)
            while(len(self.potential_area_1) != 0 and len(self.potential_area_2) != 0 and len(self.potential_area_3) != 0):
                # 必须同时满足相互不同且非已用图块才能继续进行
                self.i, self.j, self.k = self.none_same_patch()
                self.i, self.j, self.k, self.patch1_used, self.patch2_used, self.patch3_used = self.none_used_patch()
            
                img1 = cv2.imread(img_file + self.i[0][:-4] + '.jpg')
                img2 = cv2.imread(img_file + self.j[0][:-4] + '.jpg')
                img3 = cv2.imread(img_file + self.k[0][:-4] + '.jpg')
                img1 = img1[self.i[7]:self.i[9], self.i[6]:self.i[8]]
                lab1 = [self.i[1]-self.i[6], self.i[2]-self.i[7], self.i[3]-self.i[6], self.i[4]-self.i[7], self.i[5]]
                img2 = img2[self.j[7]:self.j[9], self.j[6]:self.j[8]]
                lab2 = [self.j[1]-self.j[6], self.j[2]-self.j[7], self.j[3]-self.j[6], self.j[4]-self.j[7], self.j[5]]
                img3 = img3[self.k[7]:self.k[9], self.k[6]:self.k[8]]
                lab3 = [self.k[1]-self.k[6], self.k[2]-self.k[7], self.k[3]-self.k[6], self.k[4]-self.k[7], self.k[5]]
                        
                three_to_one_less_config2_synthetise(img1, img2, img3, lab1, lab2, lab3,
                                                     self.ratio_area, self.ratio_width, self.ratio_height, 
                                                     count, self.height, self.width)
                
                count = count + 1
            
                # 储存&移除旧图块, 抽取新图块
                self.patch1_used, self.potential_area_1, self.i = self.store_remove_pick(self.patch1_used, self.potential_area_1, self.i)
                self.patch2_used, self.potential_area_2, self.j = self.store_remove_pick(self.patch2_used, self.potential_area_2, self.j)
                self.patch3_used, self.potential_area_3, self.k = self.store_remove_pick(self.patch3_used, self.potential_area_3, self.k)
    
    
    def store_remove_pick(self, store_list, remove_list, patch_info):
        store_list.append(patch_info)
        remove_list.remove(patch_info)
        patch_info = random.choice(remove_list)
        
        return store_list, remove_list, patch_info


    def remove_pick(self, remove_list, patch_info):
        remove_list.remove(patch_info)
        patch_info = random.choice(remove_list)
        
        return remove_list, patch_info
        

    def compare_info(self, i, j):
        if (i[0] == j[0] and i[1] == j[1] and i[2] == j[2] and i[3] == j[3] and i[4] == j[4]) == False:
            return True # False说明两列表不等, 不等是我们想要的
        else:
            return False # True说明两列表相等, 相等则我们不想要
    
    def none_same_patch(self):
        # 确保同时使用的四个图块不会存在相同, 如果相同则重新选择图块  
        while((self.compare_info(self.i, self.j) and self.compare_info(self.i, self.k) and self.compare_info(self.j, self.k)) == False):
            self.i, self.j, self.k = random.choice(self.potential_area_1), random.choice(self.potential_area_2), random.choice(self.potential_area_3)
        
        return self.i, self.j, self.k # 成功跳出while循环即说明互不相同
    
    def none_used_patch(self):
        # 确保不会再使用到之前使用过的图块, 如果出现即重新选择图块, 并且删除当前图块
        while(self.i in self.patch2_used or self.i in self.patch3_used):
            self.potential_area_1, self.i = self.remove_pick(self.potential_area_1, self.i)
            self.i, self.j, self.k = self.none_same_patch()
            self.i, self.j, self.k, self.patch1_used, self.patch2_used, self.patch3_used = self.none_used_patch()
        while(self.j in self.patch1_used or self.j in self.patch3_used):
            self.potential_area_2, self.j = self.remove_pick(self.potential_area_2, self.j)
            self.i, self.j, self.k = self.none_same_patch()
            self.i, self.j, self.k, self.patch1_used, self.patch2_used, self.patch3_used = self.none_used_patch()
        while(self.k in self.patch1_used or self.k in self.patch2_used):
            self.potential_area_3, self.k = self.remove_pick(self.potential_area_3, self.k)
            self.i, self.j, self.k = self.none_same_patch()
            self.i, self.j, self.k, self.patch1_used, self.patch2_used, self.patch3_used = self.none_used_patch()

        return self.i, self.j, self.k, self.patch1_used, self.patch2_used, self.patch3_used


def three_to_one_less_config2_synthetise(img1, img2, img3, lab1, lab2, lab3, ratio_area, ratio_width, ratio_height, count, height, width):
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
    
    # 左下图块如果w < h则翻转目标变成w > h
    if img2.shape[1] < img2.shape[0]:
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
        img2 = np.rot90(img1)
    
    # cv2.imread读取为[h, w, c], 将其转换为[c, h, w]
    img1 = np.transpose(img1, (2, 0, 1))
    img2 = np.transpose(img2, (2, 0, 1))
    
    # 取得二合一新宽高与缩放模式
    new_h, new_w, mode = height * pow(ratio_area, 0.5), width * pow(ratio_area, 0.5), 'nearest'
    
    # 缩放img1与img2至指定宽高
    divide_img_1 = torch.nn.functional.interpolate(torch.from_numpy(img1.copy()).unsqueeze(0), size=(int(new_h * ratio_height), int(new_w * ratio_width)), mode=mode).squeeze(0)
    divide_img_2 = torch.nn.functional.interpolate(torch.from_numpy(img2.copy()).unsqueeze(0), size=(int(new_h * (1-ratio_height)), int(new_w * ratio_width)), mode=mode).squeeze(0)
    
    # 缩放img1与img2的标签
    ratios_1 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * ratio_width), int(new_h * ratio_height)), (img1.shape[2], img1.shape[1]))) # width, height
    ratios_2 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * ratio_width), int(new_h * (1-ratio_height))), (img2.shape[2], img2.shape[1]))) # width, height
    lab1[0] = lab1[0] * ratios_1[0] # xmin * ratio_width
    lab1[2] = lab1[2] * ratios_1[0] # xmax * ratio_width
    lab1[1] = lab1[1] * ratios_1[1] # ymin * ratio_height
    lab1[3] = lab1[3] * ratios_1[1] # ymax * ratio_height
    lab2[0] = lab2[0] * ratios_2[0] # xmin * ratio_width
    lab2[2] = lab2[2] * ratios_2[0] # xmax * ratio_width
    lab2[1] = lab2[1] * ratios_2[1] # ymin * ratio_height
    lab2[3] = lab2[3] * ratios_2[1] # ymax * ratio_height
    
    # 创建二合一全零复制面板并拼接图块,syn_img.shape=[c, h, w]
    syn_img = torch.zeros((3, divide_img_1.shape[1]+divide_img_2.shape[1], divide_img_1.shape[2]))
    syn_img[:3, :divide_img_1.shape[1], :divide_img_1.shape[2]].copy_(divide_img_1)
    syn_img[:3, divide_img_1.shape[1]:, :divide_img_2.shape[2]].copy_(divide_img_2)
    
    # 计算二合一偏移量并调整坐标到新位置
    offsets = [[0.0, 0.0, 0.0, 0.0], [0.0, int(new_h * ratio_height), 0.0, int(new_h * ratio_height)]] # 左上 左下
    lab1[0] = lab1[0] + offsets[0][0]
    lab1[1] = lab1[1] + offsets[0][1]
    lab1[2] = lab1[2] + offsets[0][2]
    lab1[3] = lab1[3] + offsets[0][3]
    lab2[0] = lab2[0] + offsets[1][0]
    lab2[1] = lab2[1] + offsets[1][1]
    lab2[2] = lab2[2] + offsets[1][2]
    lab2[3] = lab2[3] + offsets[1][3]
    
    # 当第3个图块w > h时翻转图块
    if img3.shape[1] > img3.shape[0]:
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
    
    del xmin_new, ymin_new, xmax_new, ymax_new
    
    # cv2.imread读取为[h, w, c], 将其转换为[c, h, w]
    img3 = np.transpose(img3, (2, 0, 1))
    
    # 缩放img3至指定宽高
    divide_img_3 = torch.nn.functional.interpolate(torch.from_numpy(img3.copy()).unsqueeze(0), size=(int(new_h*ratio_height)+int(new_h*(1-ratio_height)), int(new_w*(1-ratio_width))), mode=mode).squeeze(0)
    
    # 缩放img3的标签
    ratios_3 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w*(1-ratio_width)), int(new_h*ratio_height)+int(new_h*(1-ratio_height))), (img3.shape[2], img3.shape[1]))) # width, height
    lab3[0] = lab3[0] * ratios_3[0] # xmin * ratio_width
    lab3[2] = lab3[2] * ratios_3[0] # xmax * ratio_width
    lab3[1] = lab3[1] * ratios_3[1] # ymin * ratio_height
    lab3[3] = lab3[3] * ratios_3[1] # ymax * ratio_height
    
    # 创建三合一全零复制面板并拼接图块,syn_img.shape=[c, h, w]
    temp = syn_img
    syn_img = torch.zeros((3, divide_img_1.shape[1]+divide_img_2.shape[1], divide_img_1.shape[2]+divide_img_3.shape[2]))
    syn_img[:3, :divide_img_1.shape[1]+divide_img_2.shape[1], :divide_img_1.shape[2]].copy_(temp)
    syn_img[:3, :divide_img_1.shape[1]+divide_img_2.shape[1], divide_img_1.shape[2]:].copy_(divide_img_3)
    del temp
    
    # 计算二合一偏移量并调整坐标到新位置
    offsets = [[int(new_w * ratio_width), 0.0, int(new_w * ratio_width), 0.0]] # 正右
    lab3[0] = lab3[0] + offsets[0][0]
    lab3[1] = lab3[1] + offsets[0][1]
    lab3[2] = lab3[2] + offsets[0][2]
    lab3[3] = lab3[3] + offsets[0][3]
    
    # 填充图片至input_size, default=[1080, 1920] [height, width]
    if syn_img.shape[1] < height: # input_size=[height, width], pad_img=[c, h, w]
        dh = height - syn_img.shape[1]
        dh /= 2
        pad_top, pad_bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    else:
        pad_top, pad_bottom = 0, 0
    
    if syn_img.shape[2] < width: # input_size=[height, width], pad_img=[c, h, w]
        dw = width - syn_img.shape[2]
        dw /= 2
        pad_left, pad_right = int(round(dw - 0.1)), int(round(dw + 0.1))
    else:
        pad_left, pad_right = 0, 0
    
    syn_img = cv2.copyMakeBorder(np.transpose(syn_img.numpy(), (1, 2, 0)), 
                                 pad_top, pad_bottom, pad_left, pad_right, 
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114)) # syn_img = [h, w, c]
    
    # 为坐标加上填充量
    lab1[0], lab1[2] = lab1[0] + pad_left, lab1[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab1[1], lab1[3] = lab1[1] + pad_top, lab1[3] + pad_top # ymin + pad_top, ymax + pad_top
    lab2[0], lab2[2] = lab2[0] + pad_left, lab2[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab2[1], lab2[3] = lab2[1] + pad_top, lab2[3] + pad_top # ymin + pad_top, ymax + pad_top
    lab3[0], lab3[2] = lab3[0] + pad_left, lab3[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab3[1], lab3[3] = lab3[1] + pad_top, lab3[3] + pad_top # ymin + pad_top, ymax + pad_top
    
    cv2.imwrite(save_dir + 'syn_img_' + str(count) + 'jpg', syn_img) # cv2.imwrite reqire [h, w, c]
    # return syn_img # [c, h, w]


# 三合一解析算法,合成图面积大于原图面积,输入所需目标的尺寸范围并设定两图块的面积比例
def three_to_one_more_config1_analysis(scope, height, width): # scope: 像素数量的范围, 本函数默认scope=[0, 1024], 即MSCOCO定义的小目标
    # 生成拼接图片/原始图片面积随机比值&两图块面积随机比值
    ratio_width = random.uniform(1/2, 3/5) # 生成[1/2, 3/5)之间的随机数,用于左上右上两图块之间的比例
    ratio_height = random.uniform(2/5, 3/5) # 生成[2/5, 3/5)之间的随机数,用于上下三图块之间的比例
    
    # 计算需要抽取的两块拼图的条件,拼图面积最小值,拼图面积/目标面积比例范围
    search_patch_1_low = height * width * 0.8 * ratio_height * ratio_width
    search_patch_2_low = height * width * 0.8 * ratio_height * (1-ratio_width)
    search_patch_3_low = height * width * 0.8 * (1-ratio_height)
    
    if scope[0] == 0: # 得到所需拼图面积与目标面积比例范围
        proportion_1_low = search_patch_1_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_1_high = 'infinite' # 由于最低值为0, 因为比例为无限大
        proportion_2_low = search_patch_2_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_2_high = 'infinite' # 由于最低值为0, 因为比例为无限大
        proportion_3_low = search_patch_3_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_3_high = 'infinite' # 由于最低值为0, 因为比例为无限大
    else:
        proportion_1_low = search_patch_1_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_1_high = search_patch_1_low / scope[0] # 拼图面积 / 目标面积最小值
        proportion_2_low = search_patch_2_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_2_high = search_patch_2_low / scope[0] # 拼图面积 / 目标面积最小值
        proportion_3_low = search_patch_3_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_3_high = search_patch_3_low / scope[0] # 拼图面积 / 目标面积最小值
        
    return (ratio_width, ratio_height, 
           [search_patch_1_low, proportion_1_low, proportion_1_high], 
           [search_patch_2_low, proportion_2_low, proportion_2_high],
           [search_patch_3_low, proportion_3_low, proportion_3_high])


class three_to_one_more_config1_search:
    def __init__(self, list1, list2, list3, csv_path, ratio_width, ratio_height, height, width):
        self.list1 = list1
        self.list2 = list2
        self.list3 = list3
        
        self.csv_path = csv_path
        
        self.ratio_width = ratio_width
        self.ratio_height = ratio_height
        self.height = height
        self.width = width
        
    def do_search(self):
        search_area_1_low, proportion_1_low, proportion_1_high = self.list1
        search_area_2_low, proportion_2_low, proportion_2_high = self.list2
        search_area_3_low, proportion_3_low, proportion_3_high = self.list3
    
        self.potential_area_1, self.potential_area_2, self.potential_area_3 = [], [], []
    
        df = pd.read_table(self.csv_path, header=None)
        list_target = df.values.tolist()
    
        for i in range(len(list_target)):
            list2 = list_target[i][0].split(",")
            list2[0] = list2[0] # xml name
            list2[1] = float(list2[1]) # res xmin
            list2[2] = float(list2[2]) # res ymin
            list2[3] = float(list2[3]) # res xmax
            list2[4] = float(list2[4]) # res ymax
            list2[5] = int(float(list2[5])) # res cls
            list2[6] = int(float(list2[6])) # cut xmin
            list2[7] = int(float(list2[7])) # cut ymin
            list2[8] = int(float(list2[8])) # cut xmax
            list2[9] = int(float(list2[9])) # cut ymax
            list2[10] = float(list2[10]) # ratio target_area
            list2[11] = float(list2[11]) # ratio cutout_area
            list2[12] = float(list2[12]) # ratio ratio_area
        
            # 判断该目标是否满足search_area_1的条件
            if proportion_1_high == 'infinite':
                if list2[12] > proportion_1_low and list2[11] > search_area_1_low:
                    self.potential_area_1.append(list2)
            else:
                if proportion_1_high > list2[12] > proportion_1_low and list2[11] > search_area_1_low:
                    self.potential_area_1.append(list2)
        
            # 判断该目标是否满足search_area_2的条件
            if proportion_2_high == 'infinite':
                if list2[12] > proportion_2_low and list2[11] > search_area_2_low:
                    self.potential_area_2.append(list2)
            else:
                if proportion_2_high > list2[12] > proportion_2_low and list2[11] > search_area_2_low:
                    self.potential_area_2.append(list2)
        
            # 判断该目标是否满足search_area_3的条件
            if proportion_3_high == 'infinite':
                if list2[12] > proportion_3_low and list2[11] > search_area_3_low:
                    self.potential_area_3.append(list2)
            else:
                if proportion_3_high > list2[12] > proportion_3_low and list2[11] > search_area_3_low:
                   self.potential_area_3.append(list2)
    
        assert len(self.potential_area_1) != 0, 'No patch match the search condition 1'
        assert len(self.potential_area_2) != 0, 'No patch match the search condition 2'
        assert len(self.potential_area_3) != 0, 'No patch match the search condition 3'
        
        print(len(self.potential_area_1))
        print(len(self.potential_area_2))
        print(len(self.potential_area_3))
    
        # 初始化使用过的图块列表
        self.patch1_used, self.patch2_used, self.patch3_used = [], [], []
        
        # 初始化计数器
        count = 0
        
        # 每个列表中的图片只能使用一次,不能重复出现在多张合成图中
        if len(self.potential_area_1) != 0 and len(self.potential_area_2) != 0 and len(self.potential_area_3) != 0:
            self.i, self.j, self.k = random.choice(self.potential_area_1), random.choice(self.potential_area_2), random.choice(self.potential_area_3)
            while(len(self.potential_area_1) != 0 and len(self.potential_area_2) != 0 and len(self.potential_area_3) != 0):
                # 必须同时满足相互不同且非已用图块才能继续进行
                self.i, self.j, self.k = self.none_same_patch()
                self.i, self.j, self.k, self.patch1_used, self.patch2_used, self.patch3_used = self.none_used_patch()
            
                img1 = cv2.imread(img_file + self.i[0][:-4] + '.jpg')
                img2 = cv2.imread(img_file + self.j[0][:-4] + '.jpg')
                img3 = cv2.imread(img_file + self.k[0][:-4] + '.jpg')
                img1 = img1[self.i[7]:self.i[9], self.i[6]:self.i[8]]
                lab1 = [self.i[1]-self.i[6], self.i[2]-self.i[7], self.i[3]-self.i[6], self.i[4]-self.i[7], self.i[5]]
                img2 = img2[self.j[7]:self.j[9], self.j[6]:self.j[8]]
                lab2 = [self.j[1]-self.j[6], self.j[2]-self.j[7], self.j[3]-self.j[6], self.j[4]-self.j[7], self.j[5]]
                img3 = img3[self.k[7]:self.k[9], self.k[6]:self.k[8]]
                lab3 = [self.k[1]-self.k[6], self.k[2]-self.k[7], self.k[3]-self.k[6], self.k[4]-self.k[7], self.k[5]]
                        
                three_to_one_more_config1_synthetise(img1, img2, img3, lab1, lab2, lab3,
                                                     self.ratio_width, self.ratio_height, 
                                                     count, self.height, self.width)
                
                count = count + 1
            
                # 储存&移除旧图块, 抽取新图块
                self.patch1_used, self.potential_area_1, self.i = self.store_remove_pick(self.patch1_used, self.potential_area_1, self.i)
                self.patch2_used, self.potential_area_2, self.j = self.store_remove_pick(self.patch2_used, self.potential_area_2, self.j)
                self.patch3_used, self.potential_area_3, self.k = self.store_remove_pick(self.patch3_used, self.potential_area_3, self.k)
    
    
    def store_remove_pick(self, store_list, remove_list, patch_info):
        store_list.append(patch_info)
        remove_list.remove(patch_info)
        patch_info = random.choice(remove_list)
        
        return store_list, remove_list, patch_info


    def remove_pick(self, remove_list, patch_info):
        remove_list.remove(patch_info)
        patch_info = random.choice(remove_list)
        
        return remove_list, patch_info
        

    def compare_info(self, i, j):
        if (i[0] == j[0] and i[1] == j[1] and i[2] == j[2] and i[3] == j[3] and i[4] == j[4]) == False:
            return True # False说明两列表不等, 不等是我们想要的
        else:
            return False # True说明两列表相等, 相等则我们不想要
    
    def none_same_patch(self):
        # 确保同时使用的四个图块不会存在相同, 如果相同则重新选择图块  
        while((self.compare_info(self.i, self.j) and self.compare_info(self.i, self.k) and self.compare_info(self.j, self.k)) == False):
            self.i, self.j, self.k = random.choice(self.potential_area_1), random.choice(self.potential_area_2), random.choice(self.potential_area_3)
        
        return self.i, self.j, self.k # 成功跳出while循环即说明互不相同
    
    def none_used_patch(self):
        # 确保不会再使用到之前使用过的图块, 如果出现即重新选择图块, 并且删除当前图块
        while(self.i in self.patch2_used or self.i in self.patch3_used):
            self.potential_area_1, self.i = self.remove_pick(self.potential_area_1, self.i)
            self.i, self.j, self.k = self.none_same_patch()
            self.i, self.j, self.k, self.patch1_used, self.patch2_used, self.patch3_used = self.none_used_patch()
        while(self.j in self.patch1_used or self.j in self.patch3_used):
            self.potential_area_2, self.j = self.remove_pick(self.potential_area_2, self.j)
            self.i, self.j, self.k = self.none_same_patch()
            self.i, self.j, self.k, self.patch1_used, self.patch2_used, self.patch3_used = self.none_used_patch()
        while(self.k in self.patch1_used or self.k in self.patch2_used):
            self.potential_area_3, self.k = self.remove_pick(self.potential_area_3, self.k)
            self.i, self.j, self.k = self.none_same_patch()
            self.i, self.j, self.k, self.patch1_used, self.patch2_used, self.patch3_used = self.none_used_patch()

        return self.i, self.j, self.k, self.patch1_used, self.patch2_used, self.patch3_used


def three_to_one_more_config1_synthetise(img1, img2, img3, lab1, lab2, lab3, ratio_width, ratio_height, count, height, width):
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
    
    # 右侧图块如果w > h则翻转目标变成w < h
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
    new_h, new_w, mode = height, width, 'nearest'
    
    # 缩放img1与img2至指定宽高
    divide_img_1 = torch.nn.functional.interpolate(torch.from_numpy(img1.copy()).unsqueeze(0), size=(int(new_h * ratio_height), int(new_w * ratio_width)), mode=mode).squeeze(0)
    divide_img_2 = torch.nn.functional.interpolate(torch.from_numpy(img2.copy()).unsqueeze(0), size=(int(new_h * ratio_height), int(new_w * (1-ratio_width))), mode=mode).squeeze(0)
    
    # 缩放img1与img2的标签
    ratios_1 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * ratio_width), int(new_h * ratio_height)), (img1.shape[2], img1.shape[1]))) # width, height
    ratios_2 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * (1-ratio_width)), int(new_h * ratio_height)), (img2.shape[2], img2.shape[1]))) # width, height
    lab1[0] = lab1[0] * ratios_1[0] # xmin * ratio_width
    lab1[2] = lab1[2] * ratios_1[0] # xmax * ratio_width
    lab1[1] = lab1[1] * ratios_1[1] # ymin * ratio_height
    lab1[3] = lab1[3] * ratios_1[1] # ymax * ratio_height
    lab2[0] = lab2[0] * ratios_2[0] # xmin * ratio_width
    lab2[2] = lab2[2] * ratios_2[0] # xmax * ratio_width
    lab2[1] = lab2[1] * ratios_2[1] # ymin * ratio_height
    lab2[3] = lab2[3] * ratios_2[1] # ymax * ratio_height
    
    # 创建二合一全零复制面板并拼接图块,syn_img.shape=[c, h, w]
    syn_img = torch.zeros((3, divide_img_1.shape[1], divide_img_1.shape[2]+divide_img_2.shape[2]))
    syn_img[:3, :divide_img_1.shape[1], :divide_img_1.shape[2]].copy_(divide_img_1)
    syn_img[:3, :divide_img_1.shape[1], divide_img_1.shape[2]:].copy_(divide_img_2)
    
    # 计算二合一偏移量并调整坐标到新位置
    offsets = [[0.0, 0.0, 0.0, 0.0], [int(new_w * ratio_width), 0.0, int(new_w * ratio_width), 0.0]] # 上左 上右
    lab1[0] = lab1[0] + offsets[0][0]
    lab1[1] = lab1[1] + offsets[0][1]
    lab1[2] = lab1[2] + offsets[0][2]
    lab1[3] = lab1[3] + offsets[0][3]
    lab2[0] = lab2[0] + offsets[1][0]
    lab2[1] = lab2[1] + offsets[1][1]
    lab2[2] = lab2[2] + offsets[1][2]
    lab2[3] = lab2[3] + offsets[1][3]
    
    # 判断二合一合成图宽高以决定如何拼接第三张图块
    if syn_img.shape[2] > syn_img.shape[1]: # 二合一合成图w > h则需要第3个图块w > h
        if img3.shape[0] > img3.shape[1]: # 当第3个图块h > w时翻转图块
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
    
    del xmin_new, ymin_new, xmax_new, ymax_new
    
    # cv2.imread读取为[h, w, c], 将其转换为[c, h, w]
    img3 = np.transpose(img3, (2, 0, 1))
    
    # 缩放img3至指定宽高
    divide_img_3 = torch.nn.functional.interpolate(torch.from_numpy(img3).unsqueeze(0), size=(int(new_h * (1-ratio_height)), int(new_w*ratio_width)+int(new_w*(1-ratio_width))), mode=mode).squeeze(0)
    
    # 缩放img3的标签
    ratios_3 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w*ratio_width)+int(new_w*(1-ratio_width)), int(new_h * (1-ratio_height))), (img3.shape[2], img3.shape[1]))) # width, height
    lab3[0] = lab3[0] * ratios_3[0] # xmin * ratio_width
    lab3[2] = lab3[2] * ratios_3[0] # xmax * ratio_width
    lab3[1] = lab3[1] * ratios_3[1] # ymin * ratio_height
    lab3[3] = lab3[3] * ratios_3[1] # ymax * ratio_height
    
    # 创建三合一全零复制面板并拼接图块,syn_img.shape=[c, h, w]
    temp = syn_img
    syn_img = torch.zeros((3, divide_img_1.shape[1]+divide_img_3.shape[1], divide_img_1.shape[2]+divide_img_2.shape[2]))
    syn_img[:3, :divide_img_1.shape[1], :divide_img_1.shape[2]+divide_img_2.shape[2]].copy_(temp)
    syn_img[:3, divide_img_1.shape[1]:, :divide_img_1.shape[2]+divide_img_2.shape[2]].copy_(divide_img_3)
    del temp
    
    # 计算二合一偏移量并调整坐标到新位置
    offsets = [[0.0, int(new_h * ratio_height), 0.0, int(new_h * ratio_height)]] # 正下
    lab3[0] = lab3[0] + offsets[0][0]
    lab3[1] = lab3[1] + offsets[0][1]
    lab3[2] = lab3[2] + offsets[0][2]
    lab3[3] = lab3[3] + offsets[0][3]
    
    # 填充图片至input_size, default=[1080, 1920] [height, width]
    if syn_img.shape[1] < height: # input_size=[height, width], pad_img=[c, h, w]
        dh = height - syn_img.shape[1]
        dh /= 2
        pad_top, pad_bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    else:
        pad_top, pad_bottom = 0, 0
    
    if syn_img.shape[2] < width: # input_size=[height, width], pad_img=[c, h, w]
        dw = width - syn_img.shape[2]
        dw /= 2
        pad_left, pad_right = int(round(dw - 0.1)), int(round(dw + 0.1))
    else:
        pad_left, pad_right = 0, 0
    
    syn_img = cv2.copyMakeBorder(np.transpose(syn_img.numpy(), (1, 2, 0)), 
                                 pad_top, pad_bottom, pad_left, pad_right, 
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114)) # syn_img = [h, w, c]
    
    # 为坐标加上填充量
    lab1[0], lab1[2] = lab1[0] + pad_left, lab1[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab1[1], lab1[3] = lab1[1] + pad_top, lab1[3] + pad_top # ymin + pad_top, ymax + pad_top
    lab2[0], lab2[2] = lab2[0] + pad_left, lab2[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab2[1], lab2[3] = lab2[1] + pad_top, lab2[3] + pad_top # ymin + pad_top, ymax + pad_top
    lab3[0], lab3[2] = lab3[0] + pad_left, lab3[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab3[1], lab3[3] = lab3[1] + pad_top, lab3[3] + pad_top # ymin + pad_top, ymax + pad_top
    
    cv2.imwrite(save_dir + 'syn_img_' + str(count) + '.jpg', syn_img) # cv2.imwrite reqire [h, w, c]
    # return syn_img # [c, h, w]


# 三合一解析算法,输入所需目标的尺寸范围并设定三图块的面积比例
def three_to_one_more_config2_analysis(scope, height, width): # scope: 像素数量的范围, 本函数默认scope=[0, 1024], 即MSCOCO定义的小目标
    # 生成拼接图片/原始图片面积随机比值&两图块面积随机比值
    ratio_width = random.uniform(2/5, 1/2) # 生成[1/2, 3/5)之间的随机数,用于左上右上两图块之间的比例
    ratio_height = random.uniform(1/2, 3/5) # 生成[2/5, 3/5)之间的随机数,用于上下三图块之间的比例
    
    # 计算需要抽取的两块拼图的条件,拼图面积最小值,拼图面积/目标面积比例范围
    search_patch_1_low = height * width * 0.8 * ratio_height * ratio_width
    search_patch_2_low = height * width * 0.8 * (1-ratio_height) * ratio_width
    search_patch_3_low = height * width * 0.8 * (1-ratio_width)
    
    if scope[0] == 0: # 得到所需拼图面积与目标面积比例范围
        proportion_1_low = search_patch_1_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_1_high = 'infinite' # 由于最低值为0, 因为比例为无限大
        proportion_2_low = search_patch_2_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_2_high = 'infinite' # 由于最低值为0, 因为比例为无限大
        proportion_3_low = search_patch_3_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_3_high = 'infinite' # 由于最低值为0, 因为比例为无限大
    else:
        proportion_1_low = search_patch_1_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_1_high = search_patch_1_low / scope[0] # 拼图面积 / 目标面积最小值
        proportion_2_low = search_patch_2_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_2_high = search_patch_2_low / scope[0] # 拼图面积 / 目标面积最小值
        proportion_3_low = search_patch_3_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_3_high = search_patch_3_low / scope[0] # 拼图面积 / 目标面积最小值
        
    return (ratio_width, ratio_height, 
           [search_patch_1_low, proportion_1_low, proportion_1_high], 
           [search_patch_2_low, proportion_2_low, proportion_2_high],
           [search_patch_3_low, proportion_3_low, proportion_3_high])


class three_to_one_more_config2_search:
    def __init__(self, list1, list2, list3, csv_path, ratio_width, ratio_height, height, width):
        self.list1 = list1
        self.list2 = list2
        self.list3 = list3
        
        self.csv_path = csv_path
        
        self.ratio_width = ratio_width
        self.ratio_height = ratio_height
        self.height = height
        self.width = width
        
        
    def do_search(self):
        search_area_1_low, proportion_1_low, proportion_1_high = self.list1
        search_area_2_low, proportion_2_low, proportion_2_high = self.list2
        search_area_3_low, proportion_3_low, proportion_3_high = self.list3
    
        self.potential_area_1, self.potential_area_2, self.potential_area_3 = [], [], []
    
        df = pd.read_table(self.csv_path, header=None)
        list_target = df.values.tolist()
    
        for i in range(len(list_target)):
            list2 = list_target[i][0].split(",")
            list2[0] = list2[0] # xml name
            list2[1] = float(list2[1]) # res xmin
            list2[2] = float(list2[2]) # res ymin
            list2[3] = float(list2[3]) # res xmax
            list2[4] = float(list2[4]) # res ymax
            list2[5] = int(float(list2[5])) # res cls
            list2[6] = int(float(list2[6])) # cut xmin
            list2[7] = int(float(list2[7])) # cut ymin
            list2[8] = int(float(list2[8])) # cut xmax
            list2[9] = int(float(list2[9])) # cut ymax
            list2[10] = float(list2[10]) # ratio target_area
            list2[11] = float(list2[11]) # ratio cutout_area
            list2[12] = float(list2[12]) # ratio ratio_area
        
            # 判断该目标是否满足search_area_1的条件
            if proportion_1_high == 'infinite':
                if list2[12] > proportion_1_low and list2[11] > search_area_1_low:
                    self.potential_area_1.append(list2)
            else:
                if proportion_1_high > list2[12] > proportion_1_low and list2[11] > search_area_1_low:
                    self.potential_area_1.append(list2)
        
            # 判断该目标是否满足search_area_2的条件
            if proportion_2_high == 'infinite':
                if list2[12] > proportion_2_low and list2[11] > search_area_2_low:
                    self.potential_area_2.append(list2)
            else:
                if proportion_2_high > list2[12] > proportion_2_low and list2[11] > search_area_2_low:
                    self.potential_area_2.append(list2)
        
            # 判断该目标是否满足search_area_3的条件
            if proportion_3_high == 'infinite':
                if list2[12] > proportion_3_low and list2[11] > search_area_3_low:
                    self.potential_area_3.append(list2)
            else:
                if proportion_3_high > list2[12] > proportion_3_low and list2[11] > search_area_3_low:
                    self.potential_area_3.append(list2)
    
        assert len(self.potential_area_1) != 0, 'No patch match the search condition 1'
        assert len(self.potential_area_2) != 0, 'No patch match the search condition 2'
        assert len(self.potential_area_3) != 0, 'No patch match the search condition 3'
        
        print(len(self.potential_area_1))
        print(len(self.potential_area_2))
        print(len(self.potential_area_3))
    
        # 初始化使用过的图块列表
        self.patch1_used, self.patch2_used, self.patch3_used = [], [], []
        
        # 初始化计数器
        count = 0
    
        # 每个列表中的图片只能使用一次,不能重复出现在多张合成图中
        if len(self.potential_area_1) != 0 and len(self.potential_area_2) != 0 and len(self.potential_area_3) != 0:
            self.i, self.j, self.k = random.choice(self.potential_area_1), random.choice(self.potential_area_2), random.choice(self.potential_area_3)
            while(len(self.potential_area_1) != 0 and len(self.potential_area_2) != 0 and len(self.potential_area_3) != 0):
                # 必须同时满足相互不同且非已用图块才能继续进行
                self.i, self.j, self.k = self.none_same_patch()
                self.i, self.j, self.k, self.patch1_used, self.patch2_used, self.patch3_used = self.none_used_patch()
            
                img1 = cv2.imread(img_file + self.i[0][:-4] + '.jpg')
                img2 = cv2.imread(img_file + self.j[0][:-4] + '.jpg')
                img3 = cv2.imread(img_file + self.k[0][:-4] + '.jpg')
                img1 = img1[self.i[7]:self.i[9], self.i[6]:self.i[8]]
                lab1 = [self.i[1]-self.i[6], self.i[2]-self.i[7], self.i[3]-self.i[6], self.i[4]-self.i[7], self.i[5]]
                img2 = img2[self.j[7]:self.j[9], self.j[6]:self.j[8]]
                lab2 = [self.j[1]-self.j[6], self.j[2]-self.j[7], self.j[3]-self.j[6], self.j[4]-self.j[7], self.j[5]]
                img3 = img3[self.k[7]:self.k[9], self.k[6]:self.k[8]]
                lab3 = [self.k[1]-self.k[6], self.k[2]-self.k[7], self.k[3]-self.k[6], self.k[4]-self.k[7], self.k[5]]
                        
                three_to_one_more_config2_synthetise(img1, img2, img3, lab1, lab2, lab3,
                                                     self.ratio_width, self.ratio_height, 
                                                     count, self.height, self.width)
                
                count = count + 1
            
                # 储存&移除旧图块, 抽取新图块
                self.patch1_used, self.potential_area_1, self.i = self.store_remove_pick(self.patch1_used, self.potential_area_1, self.i)
                self.patch2_used, self.potential_area_2, self.j = self.store_remove_pick(self.patch2_used, self.potential_area_2, self.j)
                self.patch3_used, self.potential_area_3, self.k = self.store_remove_pick(self.patch3_used, self.potential_area_3, self.k)
    
    
    def store_remove_pick(self, store_list, remove_list, patch_info):
        store_list.append(patch_info)
        remove_list.remove(patch_info)
        patch_info = random.choice(remove_list)
        
        return store_list, remove_list, patch_info


    def remove_pick(self, remove_list, patch_info):
        remove_list.remove(patch_info)
        patch_info = random.choice(remove_list)
        
        return remove_list, patch_info
        

    def compare_info(self, i, j):
        if (i[0] == j[0] and i[1] == j[1] and i[2] == j[2] and i[3] == j[3] and i[4] == j[4]) == False:
            return True # False说明两列表不等, 不等是我们想要的
        else:
            return False # True说明两列表相等, 相等则我们不想要
    
    def none_same_patch(self):
        # 确保同时使用的四个图块不会存在相同, 如果相同则重新选择图块  
        while((self.compare_info(self.i, self.j) and self.compare_info(self.i, self.k) and self.compare_info(self.j, self.k)) == False):
            self.i, self.j, self.k = random.choice(self.potential_area_1), random.choice(self.potential_area_2), random.choice(self.potential_area_3)
        
        return self.i, self.j, self.k # 成功跳出while循环即说明互不相同
    
    def none_used_patch(self):
        # 确保不会再使用到之前使用过的图块, 如果出现即重新选择图块, 并且删除当前图块
        while(self.i in self.patch2_used or self.i in self.patch3_used):
            self.potential_area_1, self.i = self.remove_pick(self.potential_area_1, self.i)
            self.i, self.j, self.k = self.none_same_patch()
            self.i, self.j, self.k, self.patch1_used, self.patch2_used, self.patch3_used = self.none_used_patch()
        while(self.j in self.patch1_used or self.j in self.patch3_used):
            self.potential_area_2, self.j = self.remove_pick(self.potential_area_2, self.j)
            self.i, self.j, self.k = self.none_same_patch()
            self.i, self.j, self.k, self.patch1_used, self.patch2_used, self.patch3_used = self.none_used_patch()
        while(self.k in self.patch1_used or self.k in self.patch2_used):
            self.potential_area_3, self.k = self.remove_pick(self.potential_area_3, self.k)
            self.i, self.j, self.k = self.none_same_patch()
            self.i, self.j, self.k, self.patch1_used, self.patch2_used, self.patch3_used = self.none_used_patch()

        return self.i, self.j, self.k, self.patch1_used, self.patch2_used, self.patch3_used


def three_to_one_more_config2_synthetise(img1, img2, img3, lab1, lab2, lab3, ratio_width, ratio_height, count, height, width):
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
    
    # 左下图块如果w < h则翻转目标变成w > h
    if img2.shape[1] < img2.shape[0]:
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
        img2 = np.rot90(img1)
    
    # cv2.imread读取为[h, w, c], 将其转换为[c, h, w]
    img1 = np.transpose(img1, (2, 0, 1))
    img2 = np.transpose(img2, (2, 0, 1))
    
    # 取得二合一新宽高与缩放模式
    new_h, new_w, mode = height, width, 'nearest'
    
    # 缩放img1与img2至指定宽高
    divide_img_1 = torch.nn.functional.interpolate(torch.from_numpy(img1.copy()).unsqueeze(0), size=(int(new_h * ratio_height), int(new_w * ratio_width)), mode=mode).squeeze(0)
    divide_img_2 = torch.nn.functional.interpolate(torch.from_numpy(img2.copy()).unsqueeze(0), size=(int(new_h * (1-ratio_height)), int(new_w * ratio_width)), mode=mode).squeeze(0)
    
    # 缩放img1与img2的标签
    ratios_1 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * ratio_width), int(new_h * ratio_height)), (img1.shape[2], img1.shape[1]))) # width, height
    ratios_2 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * ratio_width), int(new_h * (1-ratio_height))), (img2.shape[2], img2.shape[1]))) # width, height
    lab1[0] = lab1[0] * ratios_1[0] # xmin * ratio_width
    lab1[2] = lab1[2] * ratios_1[0] # xmax * ratio_width
    lab1[1] = lab1[1] * ratios_1[1] # ymin * ratio_height
    lab1[3] = lab1[3] * ratios_1[1] # ymax * ratio_height
    lab2[0] = lab2[0] * ratios_2[0] # xmin * ratio_width
    lab2[2] = lab2[2] * ratios_2[0] # xmax * ratio_width
    lab2[1] = lab2[1] * ratios_2[1] # ymin * ratio_height
    lab2[3] = lab2[3] * ratios_2[1] # ymax * ratio_height
    
    # 创建二合一全零复制面板并拼接图块,syn_img.shape=[c, h, w]
    syn_img = torch.zeros((3, divide_img_1.shape[1]+divide_img_2.shape[1], divide_img_1.shape[2]))
    syn_img[:3, :divide_img_1.shape[1], :divide_img_1.shape[2]].copy_(divide_img_1)
    syn_img[:3, divide_img_1.shape[1]:, :divide_img_2.shape[2]].copy_(divide_img_2)
    
    # 计算二合一偏移量并调整坐标到新位置
    offsets = [[0.0, 0.0, 0.0, 0.0], [0.0, int(new_h * ratio_height), 0.0, int(new_h * ratio_height)]] # 左上 左下
    lab1[0] = lab1[0] + offsets[0][0]
    lab1[1] = lab1[1] + offsets[0][1]
    lab1[2] = lab1[2] + offsets[0][2]
    lab1[3] = lab1[3] + offsets[0][3]
    lab2[0] = lab2[0] + offsets[1][0]
    lab2[1] = lab2[1] + offsets[1][1]
    lab2[2] = lab2[2] + offsets[1][2]
    lab2[3] = lab2[3] + offsets[1][3]
    
    # 当第3个图块w > h时翻转图块
    if img3.shape[1] > img3.shape[0]:
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
    
    del xmin_new, ymin_new, xmax_new, ymax_new
    
    # cv2.imread读取为[h, w, c], 将其转换为[c, h, w]
    img3 = np.transpose(img3, (2, 0, 1))
    
    # 缩放img3至指定宽高
    divide_img_3 = torch.nn.functional.interpolate(torch.from_numpy(img3.copy()).unsqueeze(0), size=(int(new_h*ratio_height)+int(new_h*(1-ratio_height)), int(new_w*(1-ratio_width))), mode=mode).squeeze(0)
    
    # 缩放img3的标签
    ratios_3 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w*(1-ratio_width)), int(new_h*ratio_height)+int(new_h*(1-ratio_height))), (img3.shape[2], img3.shape[1]))) # width, height
    lab3[0] = lab3[0] * ratios_3[0] # xmin * ratio_width
    lab3[2] = lab3[2] * ratios_3[0] # xmax * ratio_width
    lab3[1] = lab3[1] * ratios_3[1] # ymin * ratio_height
    lab3[3] = lab3[3] * ratios_3[1] # ymax * ratio_height
    
    # 创建三合一全零复制面板并拼接图块,syn_img.shape=[c, h, w]
    temp = syn_img
    syn_img = torch.zeros((3, divide_img_1.shape[1]+divide_img_2.shape[1], divide_img_1.shape[2]+divide_img_3.shape[2]))
    syn_img[:3, :divide_img_1.shape[1]+divide_img_2.shape[1], :divide_img_1.shape[2]].copy_(temp)
    syn_img[:3, :divide_img_1.shape[1]+divide_img_2.shape[1], divide_img_1.shape[2]:].copy_(divide_img_3)
    del temp
    
    # 计算二合一偏移量并调整坐标到新位置
    offsets = [[int(new_w * ratio_width), 0.0, int(new_w * ratio_width), 0.0]] # 正右
    lab3[0] = lab3[0] + offsets[0][0]
    lab3[1] = lab3[1] + offsets[0][1]
    lab3[2] = lab3[2] + offsets[0][2]
    lab3[3] = lab3[3] + offsets[0][3]
    
    # 填充图片至input_size, default=[1080, 1920] [height, width]
    if syn_img.shape[1] < height: # input_size=[height, width], pad_img=[c, h, w]
        dh = height - syn_img.shape[1]
        dh /= 2
        pad_top, pad_bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    else:
        pad_top, pad_bottom = 0, 0
    
    if syn_img.shape[2] < width: # input_size=[height, width], pad_img=[c, h, w]
        dw = width - syn_img.shape[2]
        dw /= 2
        pad_left, pad_right = int(round(dw - 0.1)), int(round(dw + 0.1))
    else:
        pad_left, pad_right = 0, 0
    
    syn_img = cv2.copyMakeBorder(np.transpose(syn_img.numpy(), (1, 2, 0)), 
                                 pad_top, pad_bottom, pad_left, pad_right, 
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114)) # syn_img = [h, w, c]
    
    # 为坐标加上填充量
    lab1[0], lab1[2] = lab1[0] + pad_left, lab1[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab1[1], lab1[3] = lab1[1] + pad_top, lab1[3] + pad_top # ymin + pad_top, ymax + pad_top
    lab2[0], lab2[2] = lab2[0] + pad_left, lab2[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab2[1], lab2[3] = lab2[1] + pad_top, lab2[3] + pad_top # ymin + pad_top, ymax + pad_top
    lab3[0], lab3[2] = lab3[0] + pad_left, lab3[2] + pad_left # xmin + pad_left, xmax + pad_left
    lab3[1], lab3[3] = lab3[1] + pad_top, lab3[3] + pad_top # ymin + pad_top, ymax + pad_top
    
    cv2.imwrite(save_dir + 'syn_img_' + str(count) + '.jpg', syn_img) # cv2.imwrite reqire [h, w, c]
    # return syn_img # [c, h, w]