# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:21:51 2022

@author: Cheng Yuxuan Original
"""
import cv2
import copy
import torch
import random
import numpy as np
import pandas as pd

img_file = 'F:/VOCtrainval_images/'
save_dir = 'D:/AICV-YoloXReDST-ADP/datasets_result/'


# 二合一解析算法,合成图面积小于原图面积,输入所需目标的尺寸范围并设定两图块的面积比例
def two_to_one_less_analysis(scope, height, width): # scope: 像素数量的范围, 本函数默认scope=[0, 1024], 即MSCOCO定义的小目标
    # 首先确定两图块拼接面积,
    # 然后设定两图块大小比例,
    # 根据比例设定面积最低值,
    # 搜索图块列表并拼接图块,
    # 多次运行以生成多个图片;
    
    # 生成拼接图片/原始图片面积随机比值&两图块面积随机比值
    ratio_area = random.uniform(1/2, 1) # 生成[1/2, 1)之间的随机数,用于取得合成图与原图的比例
    ratio_divide = random.uniform(1/2, 4/5) # 生成[1/2, 4/5)之间的随机数,用于两图块之间的比例
    
    # 计算两块拼图高低值, Patch1=H × W × [1/2, 1) × [4/5, 1], Patch2=H × W × [1/2, 1) × (1 - [1/5, 1])
    search_patch_1_low = height * width * ratio_area * 0.8 * ratio_divide # Patch1 low end, 可以适当比所需面积小20%
    search_patch_2_low = height * width * ratio_area * 0.8 * (1 - ratio_divide) # Patch2 low end, 可以适当比所需面积小20%
    
    if scope[0] == 0: # 得到所需拼图面积与目标面积比例范围
        proportion_1_low = search_patch_1_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_1_high = 'infinite' # 由于最低值为0, 因为比例为无限大
        proportion_2_low = search_patch_2_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_2_high = 'infinite' # 由于最低值为0, 因为比例为无限大
    else:
        proportion_1_low = search_patch_1_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_1_high = search_patch_1_low / scope[0] # 拼图面积 / 目标面积最小值
        proportion_2_low = search_patch_2_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_2_high = search_patch_2_low / scope[0] # 拼图面积 / 目标面积最小值
    
    return ratio_area, ratio_divide, [search_patch_1_low, proportion_1_low, proportion_1_high], [search_patch_2_low, proportion_2_low, proportion_2_high]


class two_to_one_less_search:
    def __init__(self, list1, list2, csv_path, ratio_area, ratio_divide, height, width):
        self.list1 = list1
        self.list2 = list2
        
        self.csv_path = csv_path
        
        self.ratio_area = ratio_area
        self.ratio_divide = ratio_divide
        self.height = height
        self.width = width
        
    def do_search(self):
        search_area_1_low, proportion_1_low, proportion_1_high = self.list1
        search_area_2_low, proportion_2_low, proportion_2_high = self.list2
    
        self.potential_area_1, self.potential_area_2 = [], []
    
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
            
        assert len(self.potential_area_1) != 0, 'No patch match the search condition 1'
        assert len(self.potential_area_2) != 0, 'No patch match the search condition 2'
        
        print(len(self.potential_area_1))
        print(len(self.potential_area_2))
        
        # 初始化使用过的图块列表
        self.patch1_used, self.patch2_used = [], []
    
        # 初始化计数器
        count = 0
    
        # 每个列表中的图片只能使用一次,不能重复出现在多张合成图中
        if len(self.potential_area_1) != 0 and len(self.potential_area_2) != 0:
            self.i, self.j = random.choice(self.potential_area_1), random.choice(self.potential_area_2)
            while(len(self.potential_area_1) != 0 and len(self.potential_area_2) != 0):
                # 必须同时满足相互不同且非已用图块才能继续进行
                self.i, self.j = self.none_same_patch()
                self.i, self.j self.patch1_used, self.patch2_used = self.none_used_patch()
            
                img1 = cv2.imread(img_file + self.i[0][:-4] + '.jpg')
                img2 = cv2.imread(img_file + self.j[0][:-4] + '.jpg')
                img1 = img1[self.i[7]:self.i[9], self.i[6]:self.i[8]]
                lab1 = [self.i[1]-self.i[6], self.i[2]-self.i[7], self.i[3]-self.i[6], self.i[4]-self.i[7], self.i[5]]
                img2 = img2[self.j[7]:self.j[9], self.j[6]:self.j[8]]
                lab2 = [self.j[1]-self.j[6], self.j[2]-self.j[7], self.j[3]-self.j[6], self.j[4]-self.j[7], self.j[5]]
                        
                three_to_one_less_config1_synthetise(img1, img2, lab1, lab2, self.ratio_area, self.ratio_divide, count, self.height, self.width)
                
                count = count + 1
            
                # 储存&移除旧图块, 抽取新图块
                self.patch1_used, self.potential_area_1, self.i = self.store_remove_pick(self.patch1_used, self.potential_area_1, self.i)
                self.patch2_used, self.potential_area_2, self.j = self.store_remove_pick(self.patch2_used, self.potential_area_2, self.j)
    
    
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
        while(self.compare_info(self.i, self.j) == False):
            self.i, self.j = random.choice(self.potential_area_1), random.choice(self.potential_area_2)
        
        return self.i, self.j # 成功跳出while循环即说明互不相同
    
    def none_used_patch(self):
        # 确保不会再使用到之前使用过的图块, 如果出现即重新选择图块, 并且删除当前图块
        while(self.i in self.patch2_used):
            self.potential_area_1, self.i = self.remove_pick(self.potential_area_1, self.i)
            self.i, self.j, = self.none_same_patch()
            self.i, self.j, self.patch1_used, self.patch2_used = self.none_used_patch()
        while(self.j in self.patch1_used):
            self.potential_area_2, self.j = self.remove_pick(self.potential_area_2, self.j)
            self.i, self.j, = self.none_same_patch()
            self.i, self.j, self.patch1_used, self.patch2_used = self.none_used_patch()

        return self.i, self.j, self.patch1_used, self.patch2_used


def two_to_one_less_synthetise(img1, img2, lab1, lab2, ratio_area, ratio_divide, count, height, width): # 需要理顺图片通道顺序
    # 左侧图块如果w < h则翻转目标变成w > h
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
    
    # 右侧图块如果w > h则翻转目标变成h > w
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
    
    del xmin_new, ymin_new, xmax_new, ymax_new
    
    # cv2.imread读取为[h, w, c], 将其转换为[c, h, w]
    img1 = np.transpose(img1, (2, 0, 1))
    img2 = np.transpose(img2, (2, 0, 1))
    
    # 取得新宽高与缩放模式
    new_h, new_w, mode = height * pow(ratio_area, 0.5), width * pow(ratio_area, 0.5), 'nearest'
    
    # 缩放img1与img2至指定宽高
    divide_img_1 = torch.nn.functional.interpolate(torch.from_numpy(img1.copy()).unsqueeze(0), size=(int(new_h), int(new_w * ratio_divide)), mode=mode).squeeze(0)
    divide_img_2 = torch.nn.functional.interpolate(torch.from_numpy(img2.copy()).unsqueeze(0), size=(int(new_h), int(new_w * (1-ratio_divide))), mode=mode).squeeze(0)
    
    # 缩放img1与img2的标签
    ratios_1 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * ratio_divide), int(new_h)), (img1.shape[2], img1.shape[1]))) # width, height
    ratios_2 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * (1-ratio_divide)), int(new_h)), (img2.shape[2], img2.shape[1]))) # width, height
    lab1[0] = lab1[0] * ratios_1[0] # xmin * ratio_width
    lab1[2] = lab1[2] * ratios_1[0] # xmax * ratio_width
    lab1[1] = lab1[1] * ratios_1[1] # ymin * ratio_height
    lab1[3] = lab1[3] * ratios_1[1] # ymax * ratio_height
    lab2[0] = lab2[0] * ratios_2[0] # xmin * ratio_width
    lab2[2] = lab2[2] * ratios_2[0] # xmax * ratio_width
    lab2[1] = lab2[1] * ratios_2[1] # ymin * ratio_height
    lab2[3] = lab2[3] * ratios_2[1] # ymax * ratio_height
    
    # 创建全零复制面板并拼接图块
    syn_img = torch.zeros((3, divide_img_1.shape[1], divide_img_1.shape[2]+divide_img_2.shape[2]))
    syn_img[:3, :divide_img_1.shape[1], :divide_img_1.shape[2]].copy_(divide_img_1)
    syn_img[:3, :divide_img_1.shape[1], divide_img_1.shape[2]:].copy_(divide_img_2)
    
    # 计算偏移量并调整坐标到新位置
    offsets = [[0.0, 0.0, 0.0, 0.0], [int(new_w * ratio_divide), 0.0, int(new_w * ratio_divide), 0.0]] # 上左 上右
    lab1[0] = lab1[0] + offsets[0][0]
    lab1[1] = lab1[1] + offsets[0][1]
    lab1[2] = lab1[2] + offsets[0][2]
    lab1[3] = lab1[3] + offsets[0][3]
    lab2[0] = lab2[0] + offsets[1][0]
    lab2[1] = lab2[1] + offsets[1][1]
    lab2[2] = lab2[2] + offsets[1][2]
    lab2[3] = lab2[3] + offsets[1][3]
    
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
    
    cv2.imwrite(save_dir + 'syn_img_' + str(count) + '.jpg', syn_img) # cv2.imwrite reqire [h, w, c]
    # return syn_img # [c, h, w]


# 二合一解析算法,合成图面积大于原图面积,输入所需目标的尺寸范围并设定两图块的面积比例
def two_to_one_more_analysis(scope, height, width): # scope: 像素数量的范围, 本函数默认scope=[0, 1024], 即MSCOCO定义的小目标
    # 生成两图块面积随机比值
    ratio_divide = random.uniform(1/2, 4/5) # 生成[1/2, 4/5)之间的随机数,用于两图块之间的比例
    
    # 计算需要抽取的两块拼图的条件,拼图面积最小值,拼图面积/目标面积比例范围
    search_patch_1_low = height * width * ratio_divide * 0.8 # 得到所需拼图面积最低值,可以适当比所需面积小20%
    search_patch_2_low = height * width * (1 - ratio_divide) * 0.8 # 得到所需拼图面积最低值,可以适当比所需面积小20%
    
    if scope[0] == 0: # 得到所需拼图面积与目标面积比例范围
        proportion_1_low = search_patch_1_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_1_high = 'infinite' # 由于最低值为0, 因为比例为无限大
        proportion_2_low = search_patch_2_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_2_high = 'infinite' # 由于最低值为0, 因为比例为无限大
    else:
        proportion_1_low = search_patch_1_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_1_high = search_patch_1_low / scope[0] # 拼图面积 / 目标面积最小值
        proportion_2_low = search_patch_2_low / scope[1] # 拼图面积 / 目标面积最大值
        proportion_2_high = search_patch_2_low / scope[0] # 拼图面积 / 目标面积最小值
    
    return (None, ratio_divide, 
           [search_patch_1_low, proportion_1_low, proportion_1_high], 
           [search_patch_2_low, proportion_2_low, proportion_2_high])
    

class two_to_one_more_search:
    def __init__(self, list1, list2, csv_path, ratio_divide, height, width):
        self.list1 = list1
        self.list2 = list2
        
        self.csv_path = csv_path
        
        self.ratio_divide = ratio_divide
        self.height = height
        self.width = width
        
    def do_search(self):
        search_area_1_low, proportion_1_low, proportion_1_high = self.list1
        search_area_2_low, proportion_2_low, proportion_2_high = self.list2
    
        self.potential_area_1, self.potential_area_2 = [], []
    
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
    
        assert len(self.potential_area_1) != 0, 'No patch match the search condition 1'
        assert len(self.potential_area_2) != 0, 'No patch match the search condition 2'
        
        print(len(self.potential_area_1))
        print(len(self.potential_area_2))
        
        # 初始化使用过的图块列表
        self.patch1_used, self.patch2_used = [], []
        
        # 初始化计数器
        count = 0
    
        # 每个列表中的图片只能使用一次,不能重复出现在多张合成图中
        if len(self.potential_area_1) != 0 and len(self.potential_area_2) != 0:
            self.i, self.j = random.choice(self.potential_area_1), random.choice(self.potential_area_2)
            while(len(self.potential_area_1) != 0 and len(self.potential_area_2) != 0):
                # 必须同时满足相互不同且非已用图块才能继续进行
                self.i, self.j = self.none_same_patch()
                self.i, self.j self.patch1_used, self.patch2_used = self.none_used_patch()
            
                img1 = cv2.imread(img_file + self.i[0][:-4] + '.jpg')
                img2 = cv2.imread(img_file + self.j[0][:-4] + '.jpg')
                img1 = img1[self.i[7]:self.i[9], self.i[6]:self.i[8]]
                lab1 = [self.i[1]-self.i[6], self.i[2]-self.i[7], self.i[3]-self.i[6], self.i[4]-self.i[7], self.i[5]]
                img2 = img2[self.j[7]:self.j[9], self.j[6]:self.j[8]]
                lab2 = [self.j[1]-self.j[6], self.j[2]-self.j[7], self.j[3]-self.j[6], self.j[4]-self.j[7], self.j[5]]
                        
                three_to_one_less_config1_synthetise(img1, img2, lab1, lab2, self.ratio_divide, count, self.height, self.width)
                
                count = count + 1
            
                # 储存&移除旧图块, 抽取新图块
                self.patch1_used, self.potential_area_1, self.i = self.store_remove_pick(self.patch1_used, self.potential_area_1, self.i)
                self.patch2_used, self.potential_area_2, self.j = self.store_remove_pick(self.patch2_used, self.potential_area_2, self.j)
    
    
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
        while(self.compare_info(self.i, self.j) == False):
            self.i, self.j = random.choice(self.potential_area_1), random.choice(self.potential_area_2)
        
        return self.i, self.j # 成功跳出while循环即说明互不相同
    
    def none_used_patch(self):
        # 确保不会再使用到之前使用过的图块, 如果出现即重新选择图块, 并且删除当前图块
        while(self.i in self.patch2_used):
            self.potential_area_1, self.i = self.remove_pick(self.potential_area_1, self.i)
            self.i, self.j, = self.none_same_patch()
            self.i, self.j, self.patch1_used, self.patch2_used = self.none_used_patch()
        while(self.j in self.patch1_used):
            self.potential_area_2, self.j = self.remove_pick(self.potential_area_2, self.j)
            self.i, self.j, = self.none_same_patch()
            self.i, self.j, self.patch1_used, self.patch2_used = self.none_used_patch()

        return self.i, self.j, self.patch1_used, self.patch2_used


def two_to_one_more_synthetise(img1, img2, lab1, lab2, ratio_divide, count, height, width):
    # 左侧图块如果w < h则翻转目标变成h < w
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
    
    # 右侧图块如果w > h则翻转目标变成h > w
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
    
    del xmin_new, ymin_new, xmax_new, ymax_new
    
    # cv2.imread读取为[h, w, c], 将其转换为[c, h, w]
    img1 = np.transpose(img1, (2, 0, 1))
    img2 = np.transpose(img2, (2, 0, 1))
    
    new_h, new_w, mode = height, width, 'nearest'
    
    # 缩放img1与img2至指定宽高
    divide_img_1 = torch.nn.functional.interpolate(torch.from_numpy(img1.copy()).unsqueeze(0), size=(int(new_h), int(new_w * ratio_divide)), mode=mode).squeeze(0)
    divide_img_2 = torch.nn.functional.interpolate(torch.from_numpy(img2.copy()).unsqueeze(0), size=(int(new_h), int(new_w * (1-ratio_divide))), mode=mode).squeeze(0)
    
    # 缩放img1与img2的标签
    ratios_1 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * ratio_divide), int(new_h)), (img1.shape[2], img1.shape[1]))) # width, height
    ratios_2 = tuple(float(s) / float(s_orig) for s, s_orig in zip((int(new_w * (1-ratio_divide)), int(new_h)), (img2.shape[2], img2.shape[1]))) # width, height
    lab1[0] = lab1[0] * ratios_1[0] # xmin * ratio_width
    lab1[2] = lab1[2] * ratios_1[0] # xmax * ratio_width
    lab1[1] = lab1[1] * ratios_1[1] # ymin * ratio_height
    lab1[3] = lab1[3] * ratios_1[1] # ymax * ratio_height
    lab2[0] = lab2[0] * ratios_2[0] # xmin * ratio_width
    lab2[2] = lab2[2] * ratios_2[0] # xmax * ratio_width
    lab2[1] = lab2[1] * ratios_2[1] # ymin * ratio_height
    lab2[3] = lab2[3] * ratios_2[1] # ymax * ratio_height
    
    # 创建全零复制面板并拼接图块
    syn_img = torch.zeros((3, divide_img_1.shape[1], divide_img_1.shape[2]+divide_img_2.shape[2]))
    syn_img[:3, :divide_img_1.shape[1], :divide_img_1.shape[2]].copy_(divide_img_1)
    syn_img[:3, :divide_img_1.shape[1], divide_img_1.shape[2]:].copy_(divide_img_2)
    
    # 计算偏移量并调整坐标到新位置
    offsets = [[0.0, 0.0, 0.0, 0.0], [int(new_w * ratio_divide), 0.0, int(new_w * ratio_divide), 0.0]] # 上左 上右
    lab1[0] = lab1[0] + offsets[0][0]
    lab1[1] = lab1[1] + offsets[0][1]
    lab1[2] = lab1[2] + offsets[0][2]
    lab1[3] = lab1[3] + offsets[0][3]
    lab2[0] = lab2[0] + offsets[1][0]
    lab2[1] = lab2[1] + offsets[1][1]
    lab2[2] = lab2[2] + offsets[1][2]
    lab2[3] = lab2[3] + offsets[1][3]
    
    cv2.imwrite(save_dir + 'syn_img_' + str(count) + '.jpg', syn_img) # cv2.imwrite reqire [h, w, c]
    # return syn_img # [c, h, w]