# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 21:34:56 2023

@author: Pilot Crysi
"""

import os
import cv2
import csv
import copy
import torch
# import torchvision
import numpy as np
# import pandas as pd
import xml.etree.ElementTree as ET

from datapool_analysis_overlap import overlap
from datapool_analysis_overlap import estimate_point


def sever_overlap(results, current, sever_list, second_sever, threshold, while_flag):   
    while_flag = False # 初始化while循环停止符, 当不再有坐标被找到时停止while循环
    for i in range(len(second_sever)): # 遍历二阶严重相交目标
        for j in range(len(results)): # 遍历总坐标矩阵
            if results[j] in sever_list or results[j] in second_sever: # 排除已找到的目标
                continue
            
            # 判断是否严重相交, 当前目标不为LargeObj时不接受LargeObj
            if overlap(second_sever[i][:4], results[j][:4]) > threshold:
                if current[5] != 4 and results[j][5] != 4: # elimate Large Object
                    second_sever.append(results[j])
                    while_flag = True
    
    return second_sever, while_flag


def estimate_position(overlap_obj): # 依据相交边的重叠度判断位置关系, 调用datapool_analysis_overlap.py的estimate_point()
    """
    Analysis two objects position pattern with overlap length of their hgith and
    width, if two object are overlap, there must be at lest one side overlap with
    other object, most situations, two sides are overlap with other object. 
    Therefore, all we need to do is determine the ratio of overlap side and 
    shorter side, which should be more than 3/4 to be considered as overlap
    """
    other_obj, cluster_obj = overlap_obj[0], overlap_obj[1]
    ext_top, ext_bot, ext_lef, ext_rig = False, False, False, False # 目标簇目标从哪个方向兼并其他目标
    
    # 判断tensor1的顶点是否在tensor2中, 返回由四个边是否可扩展的布尔列表[top, bot, lef, rig]
    top, bot, lef, rig = estimate_point(cluster_obj, other_obj)
    
    # 计算宽方向相交长度
    overlap_xmin = max(other_obj[0], cluster_obj[0])
    overlap_xmax = min(other_obj[2], cluster_obj[2])
    overlap_wlen = overlap_xmax - overlap_xmin
    
    # 计算高方向相交长度
    overlap_ymin = max(other_obj[1], cluster_obj[1])
    overlap_ymax = min(other_obj[3], cluster_obj[3])
    overlap_hlen = overlap_ymax - overlap_ymin
    
    # 判断到底是向上下兼并还是向左右兼并
    if overlap_wlen > overlap_hlen: # 向上下兼并, 相交距离大于宽较短的目标的3/4才可兼并
        if overlap_wlen / min((cluster_obj[2] - cluster_obj[0]), other_obj[2] - other_obj[0]) > 3/4:
            if top == False: return 'top'
            if bot == False: return 'bot'
    else: # 向左右兼并, 相交距离大于高较短的目标的3/4才可兼并
        if overlap_hlen / min((cluster_obj[3] - cluster_obj[1]), (other_obj[3] - other_obj[1])) > 3/4:
            if lef == False: return 'lef'
            if rig == False: return 'rig'
    
    return 'none'


# 代码来自datapool_analysis_overlap.py的situation_3()
def sever_cluster(results, current, slight_list, sever_list, threshold):
    # 首先确定严重相交列表中的目标是否还与其他目标严重相交
    second_sever, temp_list, while_flag, rescopy = [], [], True, copy.deepcopy(results)
    
    # 将rescopy和sever_list中的元素全部转成list, 否则if xx in yy这句会报错
    # 当用列表储存np.ndarray时, 用in这个方法搜索列表中的目标是否存在时会报错
    rescopy = rescopy.tolist()
    for i in range(len(sever_list)):
        sever_list[i] = sever_list[i].tolist()
    
    # 首先排除当前目标
    rescopy.remove(current)
    
    # 先排除超大目标, 防止干扰目标簇的模式判断
    # 若当前目标自身为LargeObj, 则不排除超大目标
    if current[5] != 4:
        for i in range(len(sever_list)):
            if sever_list[i][5] == 4:
                temp_list.append(sever_list[i])
    if len(temp_list) != 0:
        for i in temp_list:
            sever_list.remove(i)
    
    # 再筛选出与sever_list严重相交的目标的坐标
    for i in range(len(sever_list)): # 遍历严重相交坐标列表
        for j in range(len(rescopy)): # 遍历总坐标矩阵
            if rescopy[j] in sever_list: # 排除已找到的目标
                continue
            
            # 判断是否严重相交, 当前目标不为LargeObj时不接受LargeObj
            if overlap(sever_list[i][:4], rescopy[j][:4]) > threshold:
                if current[5] != 4 and rescopy[j][5] != 4: # elimate Large Object
                    second_sever.append(rescopy[j])
    
    # 将second_sever进行递归, 寻找其他与其严重相交的目标
    while while_flag == True:
        second_sever, while_flag = sever_overlap(rescopy, current, sever_list, second_sever, threshold, while_flag)
    
    # 取得严重相交的目标构成的大目标的坐标
    xmin, ymin, xmax, ymax = current[0], current[1], current[2], current[3]
    
    if len(sever_list) != 0:
        for i in range(len(sever_list)):
            if xmin > sever_list[i][0]:
                xmin = sever_list[i][0]
            if ymin > sever_list[i][1]:
                ymin = sever_list[i][1]
            if xmax < sever_list[i][2]:
                xmax = sever_list[i][2]
            if ymax < sever_list[i][3]:
                ymax = sever_list[i][3]
        
        if len(second_sever) != 0:
            for i in range(len(second_sever)):
                if xmin > second_sever[i][0]:
                    xmin = second_sever[i][0]
                if ymin > second_sever[i][1]:
                    ymin = second_sever[i][1]
                if xmax < second_sever[i][2]:
                    xmax = second_sever[i][2]
                if ymax < second_sever[i][3]:
                    ymax = second_sever[i][3]
    
    dense_obj = [xmin, ymin, xmax, ymax]
    
    return dense_obj, second_sever


def slight_cluster(results, current, sever_list, second_sever, dense_obj, threshold):
    # current, sever_list, second_sever三者组成了目标簇, 即dense_obj, 现在需要找出与这些目标轻微相交的目标, 并判断边界距离和位置形态
    # 当用列表储存np.ndarray时, 用in这个方法搜索列表中的目标是否存在时会报错, 因此需要将rescopy和sever_list中的元素全部转成list
    temp_list, obj_cluster, rescopy = [], [], copy.deepcopy(results).tolist()
    
    # 收集所有组成目标簇的目标
    obj_cluster.append(current)
    if len(sever_list) != 0:
        for i in range(len(sever_list)):
            obj_cluster.append(sever_list[i])
    if len(second_sever) != 0:
        for i in range(len(second_sever)):
            obj_cluster.append(second_sever[i])
    
    # 去除obj_cluster列表中重复的元素
    for i in obj_cluster:
        if i not in temp_list:
            temp_list.append(i)
    obj_cluster = temp_list
    
    # 排除总坐标矩阵中的LargeObj, sever_list, second_sever已有目标
    if len(obj_cluster) != 0:
        for i in obj_cluster:
            rescopy.remove(i)
    for i in rescopy:
        if i[5] == 4:
            rescopy.remove(i)
    
    temp_list, bool_list, merger_list = [], [], [] # 重新初始化temp_list再利用
    
    # 遍历总坐标矩阵, 查看剩余目标是否存在与目标簇轻微相交
    if len(rescopy) != 0:
        for i in range(len(rescopy)):
            for j in range(len(obj_cluster)):
                if threshold >= overlap(rescopy[i][:4], obj_cluster[j][:4]) > 0:
                    temp_list.append([rescopy[i], obj_cluster[j]]) # 以目标对的形式储存轻微相交的目标
    
    # 评估相交边, 判断位置关系是否合适, 若合适则兼容进目标簇
    for i in range(len(temp_list)):
        bool_list.append(estimate_position(temp_list[i])) # 返回的是字符串: 'top', 'bot', 'lef', 'rig', 'none'
    
    # 根据评估结果兼容合适的轻微相交目标进目标簇
    for i in range(len(temp_list)):
        if bool_list[i] != 'none':
            xmin, ymin = dense_obj[0], dense_obj[1]
            xmax, ymax = dense_obj[2], dense_obj[3]
            if xmin > temp_list[i][0][0]:
                xmin = temp_list[i][0][0]
            if ymin > temp_list[i][0][1]:
                ymin = temp_list[i][0][1]
            if xmax < temp_list[i][0][2]:
                xmax = temp_list[i][0][2]
            if ymax < temp_list[i][0][3]:
                ymax = temp_list[i][0][3]
            
            # 新增加的背景面积大于前景面积时放弃兼并
            if (xmax - xmin) * (ymax - ymin) - (dense_obj[2] - dense_obj[0]) * (dense_obj[3] - dense_obj[1]) \
                 - (temp_list[i][0][2] - temp_list[i][0][0]) * (temp_list[i][0][3] - temp_list[i][0][1]) \
                 > (temp_list[i][0][2] - temp_list[i][0][0]) * (temp_list[i][0][3] - temp_list[i][0][1]):
                    pass
            else:
                merger_list.append(temp_list[i][0])
    
    xmin, ymin, xmax, ymax = dense_obj[0], dense_obj[1], dense_obj[2], dense_obj[3]
    
    for i in range(len(merger_list)):
        if xmin > merger_list[i][0]:
            xmin = merger_list[i][0]
        if ymin > merger_list[i][1]:
            ymin = merger_list[i][1]
        if xmax < merger_list[i][2]:
            xmax = merger_list[i][2]
        if ymax < merger_list[i][3]:
            ymax = merger_list[i][3]
    
    dense_obj[0], dense_obj[1], dense_obj[2], dense_obj[3] = xmin, ymin, xmax, ymax
    
    return dense_obj, merger_list


def expand_analysis(xmin, ymin, xmax, ymax, height, width, temp_list, threshold, expand_ratio):
    # 初始化最终返回的四个目标列表
    expand_leftop, expand_lefbot, expand_rigtop, expand_rigbot = [], [], [], []
    
    # ---------------------------------------------------------------
    
    # 以目标簇的左上点为测试区域的左上点进行扩展
    search_xmin, search_ymin = xmin, ymin
    
    # 扩展距离必须大于已有目标簇的宽高, 若小于这个距离就没必要扩展了
    if xmax - xmin < width * expand_ratio:
        search_xmax = xmin + width * expand_ratio
    else:
        search_xmax = xmax
    if ymax - ymin < height * expand_ratio:
        search_ymax = ymin + height * expand_ratio
    else:
        search_ymax = ymax
    
    expand_area = [search_xmin, search_ymin, search_xmax, search_ymax]
    
    # 测试由上述操作得到的扩展区域中是否存在其他小目标
    # 将原始目标坐标列表中的小目标筛选出来判断相交度即可
    for i in range(len(temp_list)):
        if threshold >= overlap(temp_list[i][:4], expand_area[:4]) > 0:
            expand_leftop.append(temp_list[i]) # 存在相交
        elif overlap(temp_list[i][:4], expand_area[:4]) > threshold:
            expand_leftop.append(temp_list[i]) # 严重相交
    
    # ---------------------------------------------------------------
    
    # 以目标簇的左下点作为测试区域的左下点进行扩展
    search_xmin, search_ymax = xmin, ymax
    
    # 扩展距离必须大于已有目标簇的宽高, 若小于这个距离就没必要扩展了
    if xmax - xmin < width * expand_ratio:
        search_xmax = xmin + width * expand_ratio
    else:
        search_xmax = xmax
    if ymax - ymin < height * expand_ratio:
        search_ymin = ymax - height * expand_ratio
    else:
        search_ymin = ymin
    
    expand_area = [search_xmin, search_ymin, search_xmax, search_ymax]
    
    # 测试由上述操作得到的扩展区域中是否存在其他小目标
    # 将原始目标坐标列表中的小目标筛选出来判断相交度即可
    for i in range(len(temp_list)):
        if threshold >= overlap(temp_list[i][:4], expand_area[:4]) > 0:
            expand_lefbot.append(temp_list[i]) # 存在相交
        elif overlap(temp_list[i][:4], expand_area[:4]) > threshold:
            expand_lefbot.append(temp_list[i]) # 严重相交
    
    # ---------------------------------------------------------------
    
    # 以目标簇的右上点作为测试区域的右上点进行扩展
    search_xmax, search_ymin = xmax, ymin
    
    # 扩展距离必须大于已有目标簇的宽高, 若小于这个距离就没必要扩展了
    if xmax - xmin < width * expand_ratio:
        search_xmin = xmax - width * expand_ratio
    else:
        search_xmin = xmin
    if ymax - ymin < height * expand_ratio:
        search_ymax = ymin + height * expand_ratio
    else:
        search_ymax = ymax
    
    expand_area = [search_xmin, search_ymin, search_xmax, search_ymax]
    
    # 测试由上述操作得到的扩展区域中是否存在其他小目标
    # 将原始目标坐标列表中的小目标筛选出来判断相交度即可
    for i in range(len(temp_list)):
        if threshold >= overlap(temp_list[i][:4], expand_area[:4]) > 0:
            expand_rigtop.append(temp_list[i]) # 存在相交
        elif overlap(temp_list[i][:4], expand_area[:4]) > threshold:
            expand_rigtop.append(temp_list[i]) # 严重相交
    
    # ---------------------------------------------------------------
    
    # 以目标簇的右下点作为测试区域的右上点进行扩展
    search_xmax, search_ymax = xmax, ymax
    
    # 扩展距离必须大于已有目标簇的宽高, 若小于这个距离就没必要扩展了
    if xmax - xmin < width * expand_ratio:
        search_xmin = xmax - width * expand_ratio
    else:
        search_xmin = xmin
    if ymax - ymin < height * expand_ratio:
        search_ymin = ymax - height * expand_ratio
    else:
        search_ymin = ymin
    
    expand_area = [search_xmin, search_ymin, search_xmax, search_ymax]
    
    # 测试由上述操作得到的扩展区域中是否存在其他小目标
    # 将原始目标坐标列表中的小目标筛选出来判断相交度即可
    for i in range(len(temp_list)):
        if threshold >= overlap(temp_list[i][:4], expand_area[:4]) > 0:
            expand_rigbot.append(temp_list[i]) # 存在相交
        elif overlap(temp_list[i][:4], expand_area[:4]) > threshold:
            expand_rigbot.append(temp_list[i]) # 严重相交
    
    # ---------------------------------------------------------------
    
    return expand_leftop, expand_lefbot, expand_rigtop, expand_rigbot
