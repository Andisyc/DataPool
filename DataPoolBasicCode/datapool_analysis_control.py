# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 15:12:25 2022

@author: Pilot Crysi, Disjoint Object Analyze of DataPool
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

from datapool_analysis_disjoint import check_direction1, check_direction2
from datapool_analysis_overlap import situation_2, situation_3, overlap
from datapool_analysis_cluster import sever_cluster, slight_cluster, expand_analysis
from datapool_synthauxiliary_v6 import normlize_index


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

# 设定目标尺寸类别
# OBJ_SIZE = (0 'Tiny', 1 'Small', 2 'LowerM', 3 'UpperM', 4 'Large')


def disjoint(img, res, label_name, save_path, visual, wrcsv):
    # 每个目标都对比一遍其他目标测试两者是否相对,并保存布尔列表
    outer_bool = []
    for i in range(res.shape[0]): # 外层for循环,选定一个目标作为基准
        inner_bool = []
        for j in range(res.shape[0]): # 内层for循环,选定另一个目标与基准对比
            if res[i].tolist() == res[j].tolist(): # 当检测到两目标相同时跳过本轮次
                continue
            else: # 判断目标是否存在于本目标四个对面中,返回列表,其中为四个布尔变量
                inner_bool.append(check_direction1(res[i], res[j]))
        outer_bool.append(inner_bool) # 储存该基准目标与其他所有目标的对比结果
    
    # print(outer_bool)
    
    del inner_bool
    
    # 将每个目标与其他目标的两两测试布尔列表同位置进行或操作,得到该目标四个对面是否存在目标
    four_side = []
    for i in range(len(outer_bool)): # outer_bool列表中对比结果的顺序为xml标签中目标顺序
        top, bottom, left, right = False, False, False, False # 初始化四个方向的布尔变量
        for j in range(len(outer_bool[i])): # 对每个目标的对比结果进行或操作
            top    = top    or outer_bool[i][j][0]
            bottom = bottom or outer_bool[i][j][1]
            left   = left   or outer_bool[i][j][2]
            right  = right  or outer_bool[i][j][3]
        four_side.append([top, bottom, left, right]) # 储存最终对比结果,即该目标四面是否存在空面
    
    # print(four_side)
    
    # 取得每个目标的[上距 上左距 上右距], [下距 下左距 下右距], [左距 左上距 左下距], [右距 右上距 右下距]
    expandinfo = []
    for i in range(len(four_side)):
        list1, list2, list3, list4 = check_direction2(four_side[i], res[i], res, height, width)
        expandinfo.append([list1, list2, list3, list4])

    # 分析目标数据得到多种可能的裁剪方式, 可以适当向外扩展几个像素
    cutinfo = []
    for i in range(len(expandinfo)):
        inner_info = []
        for j in range(len(expandinfo[i])):
            if len(expandinfo[i][j]) == 0:
                continue
            else:
                xcutmin = res[i][0] - expandinfo[i][j][2] # x1 - left
                ycutmin = res[i][1] - expandinfo[i][j][0] # y1 - top
                xcutmax = res[i][2] + expandinfo[i][j][3] # x2 + right
                ycutmax = res[i][3] + expandinfo[i][j][1] # y2 + bottom
                
                # print("\n", i, res[i], expandinfo[i][j], xcutmin, ycutmin, xcutmax, ycutmax)
                
                if xcutmin < 0:
                    xcutmin = 0
                if ycutmin < 0:
                    ycutmin = 0
                if xcutmax > width:
                    xcutmax = width
                if ycutmax > height:
                    ycutmax = height
                
                inner_info.append([xcutmin, ycutmin, xcutmax, ycutmax])
        cutinfo.append(inner_info)
    
    # 计算图块/目标比例方便检索
    ratio = []
    for i in range(res.shape[0]):
        inner_info = []
        for j in range(len(cutinfo[i])):
            target_area = (res[i][2] - res[i][0]) * (res[i][3] - res[i][1])
            cutout_area = (cutinfo[i][j][2] - cutinfo[i][j][0]) * (cutinfo[i][j][3] - cutinfo[i][j][1])
            ratio_area = cutout_area / target_area
            inner_info.append([target_area, cutout_area, ratio_area])
        ratio.append(inner_info)

    del inner_info
    
    # 逐个目标地可视化图块计算是否正确
    if visual == True:
        # label_tensor = torch.from_numpy(res[:, :4]) # 可视化当前目标与所有目标
        for i in range(res.shape[0]):
            img_temp = copy.deepcopy(img)
            for j in range(len(cutinfo[i])):
                # img_temp = copy.deepcopy(img)
                # label_temp = torch.cat((label_tensor, torch.from_numpy(np.array([cutinfo[i][j]]))), 0) # 可视化当前目标与所有目标
                label_temp = torch.from_numpy(np.array([cutinfo[i][j]])) # 只可视化当前目标
                for k in range(len(label_temp)):
                    box = label_temp[k]
                    x0 = int(box[0])
                    y0 = int(box[1])
                    x1 = int(box[2])
                    y1 = int(box[3])
                    color = (_COLORS[i] * 255).astype(np.uint8).tolist()
                    # cv2.rectangle(img_temp, (x0, y0), (x1, y1), color, 2) # cv2 reqire [h, w, c]
                    cv2.rectangle(img_temp, (x0, y0), (x1, y1), color, 2) # cv2 reqire [h, w, c]
                # cv2.imwrite(save_path + str(label_name[:-4]) + '_' + str(i) + '_' + str(j) + '.jpg', img_temp)
            cv2.imwrite(save_path + str(label_name[:-4]) + '_' + str(i) + '.jpg', img_temp)
    
    if wrcsv == True:
        # 将res内所有坐标从np.array转换成list
        if isinstance(res, list) != True:
            res = res.tolist()
        for i in range(len(res)):
            if isinstance(res[i], list) != True:
                res[i] = res[i].tolist()
    
        # 写入csv文件,图片名 目标坐标(xyxycls) 裁剪坐标(xyxy) 目标面积 裁剪面积 图块/目标比例
        for i in range(len(res)):
            for j in range(len(cutinfo[i])):
                list_info = [label_name,
                             res[i][0], res[i][1], res[i][2], res[i][3], res[i][4], 
                             cutinfo[i][j][0], cutinfo[i][j][1], cutinfo[i][j][2], cutinfo[i][j][3], 
                             ratio[i][j][0], ratio[i][j][1], ratio[i][j][2],
                             res[i]]
                with open(save_path, 'a', newline="") as csvfile:
                    writer0 = csv.writer(csvfile)
                    writer0.writerow(list_info)


def onedisjoint(target, res): # 处理存在相交目标的图片中的没有相交的目标
    # 该目标对比其他目标测试两者是否相对,并保存布尔列表
    inner_bool = []
    for i in range(res.shape[0]): # 内层for循环,选定另一个目标与基准对比
        if target.tolist() == res[i].tolist(): # 当检测到两目标相同时跳过本轮次
            continue
        else: # 判断目标是否存在于本目标四个对面中,返回列表,其中为四个布尔变量
            inner_bool.append(check_direction1(target, res[i]))
    
    # 该目标与其他目标的两两测试布尔列表同位置进行或操作,得到该目标四个对面是否存在目标
    top, bottom, left, right = False, False, False, False # 初始化四个方向的布尔变量
    for i in range(len(inner_bool)): # 对该目标的对比结果进行或操作
        top = top or inner_bool[i][0]
        bottom = bottom or inner_bool[i][1]
        left = left or inner_bool[i][2]
        right = right or inner_bool[i][3]
    
    list1, list2, list3, list4 = check_direction2([top, bottom, left, right], target, res, height, width)
    
    return [list1, list2, list3, list4]


def object_cluster(results, current_object, slight_list, sever_list, height, width):
    # 还需要分成两种情况, 占据图片的几乎所有区域, 占据图片的部分区域, 占据图片的小块区域
    # 当目标簇的实际面积超过高阈值时, 即占据图像的几乎所有区域时, 直接判定为整张图片为大目标
    # 当目标簇的实际面积小于高阈值但大于低阈值时, 即占据图像的部分区域时, 排除较为孤立的目标减少冗余背景
    # 当目标簇的实际面积小于低阈值时, 即占据图像的小部分区域时, 向外寻找其他小目标组成不相交的目标簇
    # 首先需要确定目标簇, 再分析目标簇的平均面积, 针对孤立值进行处理, 再判断目标簇的面积, 最后进行图块区域的判定
    
    # 将current_object转换成列表, 防止无法使用in这个操作, 并初始化各种参数
    rescopy, current_object, count, dense_obj, exinfo, temp_list = copy.deepcopy(results).tolist(), current_object.tolist(), 0, [], [], []
    
    # 当前目标为大目标时, 图片上不存在其他大目标则放弃此轮裁剪
    if current_object[5] == 4:
        count = count + 1
        for i in range(len(rescopy)):
            if current_object == rescopy[i]:
                continue
            else:
                if rescopy[i][5] == 4:
                    count = count + 1
    
    # 确定严重相交的目标簇坐标: 确定相连的所有目标
    if count == 0 or count > 1: # 当前目标不为大目标或图片存在多个大目标时进行分析
        dense_obj, second_sever = sever_cluster(results, current_object, slight_list, sever_list, threshold)
        dense_obj, merger_list = slight_cluster(results, current_object, sever_list, second_sever, dense_obj, threshold)
        
        # 将current_object, sever_list, merger_list中包含的目标从results中排除
        if current_object in rescopy:
            rescopy.remove(current_object)
        if len(sever_list) != 0:
            for i in range(len(sever_list)):
                rescopy.remove(sever_list[i])
        if len(merger_list) != 0:
            for i in range(len(merger_list)):
                if merger_list[i] in rescopy:
                    rescopy.remove(merger_list[i])
        
        # 以目标簇为核心目标判断轻微相交和严重相交
        list0, list1 = [], []
        for i in range(len(rescopy)):
            if overlap(rescopy[i][:4], dense_obj[:4]) == 0:
                list0.append(rescopy[i]) # 无相交
            elif threshold >= overlap(rescopy[i][:4], dense_obj[:4]) >= 0:
                list1.append(rescopy[i]) # 存在相交
        
        # 分析得到目标簇后送入situation_2中得到扩张距离, situation_2参数为current_obj, list_obj, height, width
        # list_obj包含3个子列表, 无相交, 轻微相交, 严重相交, 因此需要重新处理当前列表成目标簇的无相交, 轻微相交列表
        exinfo = situation_2(dense_obj, [list0, list1], height, width)
    else: # count == 1说明整张图片只有当前目标一个大目标, 因此放弃此轮分析
        dense_obj, exinfo = [], []
    
    return dense_obj, exinfo


def outreach_cluster(results, current_object, none_list, slight_list, height, width):
    # 将current_object转换成列表, 防止无法使用in这个操作, 并初始化各种参数
    rescopy, current_object = copy.deepcopy(results).tolist(), current_object.tolist()
    expand_ratio, dense_obj, exinfo, temp_list = 1/3, [], [], []
    
    # 当前目标必须是Small Object或者LowerM, 向外扩展的目标也必须是Small Object或者LowerM
    # 并且最大扩展范围应当在宽高的三分之一到二分之一之内, 不能太远, 若这个范围存在小目标, 则将其
    # 当做目标簇拼接, 实际上没有与任何目标相交的小目标也可以这样分析, 得到一个合适的目标簇图块
    
    # 首先将轻微相交的目标组成大目标, 只取Slight_list中的小目标
    xmin, ymin, xmax, ymax = current_object[0], current_object[1], current_object[2], current_object[3]
    for i in range(len(slight_list)):
        if slight_list[i][5] == 0 or slight_list[i][5] == 1 or slight_list[i][5] == 2:
            if xmin > slight_list[i][0]:
                xmin = slight_list[i][0]
            if ymin > slight_list[i][1]:
                ymin = slight_list[i][1]
            if xmax < slight_list[i][2]:
                xmax = slight_list[i][2]
            if ymax < slight_list[i][3]:
                ymax = slight_list[i][3]
    dense_obj = [xmin, ymin, xmax, ymax]
    
    # 再提取出Tiny, Small, LowerM的坐标方便后续计算
    for i in range(len(rescopy)):
        if rescopy[i][5] == 0 or rescopy[i][5] == 1 or rescopy[i][5] == 2:
            temp_list.append(rescopy[i])
    
    # 以当前大目标左上点, 左下点, 右上点, 右下点向反方向扩展宽高各5/12的距离查看是否存在小目标
    # 这种方式即虚构出一个可行的大目标范围, 以此测量这个范围内的小目标是否合适, 若没兼并的目标
    # 就算了, 因为可能已经能兼并到较多小目标了, 没必要再兼并很远的小目标, 相隔较远的小目标本身
    # 就应该属于不同的目标簇, 兼并过程中目标簇必须不含UpperM或者Large, 当遇到U或L时就止步于此
    expand_leftop, expand_lefbot, expand_rigtop, expand_rigbot = expand_analysis(xmin, ymin, xmax, ymax, height, width, temp_list, threshold, expand_ratio)
    
    # 遍历收集到的目标列表, 组装出新的目标簇, 注意先前已经排除了大目标, 所以只会兼并小目标
    for i in range(len(expand_leftop)):
        if xmin > expand_leftop[i][0]:
            xmin = expand_leftop[i][0]
        if ymin > expand_leftop[i][1]:
            ymin = expand_leftop[i][1]
        if xmax < expand_leftop[i][2]:
            xmax = expand_leftop[i][2]
        if ymax < expand_leftop[i][3]:
            ymax = expand_leftop[i][3]
    
    for i in range(len(expand_lefbot)):
        if xmin > expand_lefbot[i][0]:
            xmin = expand_lefbot[i][0]
        if ymin > expand_lefbot[i][1]:
            ymin = expand_lefbot[i][1]
        if xmax < expand_lefbot[i][2]:
            xmax = expand_lefbot[i][2]
        if ymax < expand_lefbot[i][3]:
            ymax = expand_lefbot[i][3]
    
    for i in range(len(expand_rigtop)):
        if xmin > expand_rigtop[i][0]:
            xmin = expand_rigtop[i][0]
        if ymin > expand_rigtop[i][1]:
            ymin = expand_rigtop[i][1]
        if xmax < expand_rigtop[i][2]:
            xmax = expand_rigtop[i][2]
        if ymax < expand_rigtop[i][3]:
            ymax = expand_rigtop[i][3]
    
    for i in range(len(expand_rigbot)):
        if xmin > expand_rigbot[i][0]:
            xmin = expand_rigbot[i][0]
        if ymin > expand_rigbot[i][1]:
            ymin = expand_rigbot[i][1]
        if xmax < expand_rigbot[i][2]:
            xmax = expand_rigbot[i][2]
        if ymax < expand_rigbot[i][3]:
            ymax = expand_rigbot[i][3]
    
    dense_obj = [xmin, ymin, xmax, ymax]
    
    # 将以上四个列表中包含的目标从总目标列表中排除方便判断目标簇与其他目标的相交情况
    if current_object in rescopy:
        rescopy.remove(current_object)
    if len(expand_leftop) != 0:
        for i in range(len(expand_leftop)):
            if expand_leftop[i] in rescopy:
                rescopy.remove(expand_leftop[i])
    if len(expand_lefbot) != 0:
        for i in range(len(expand_lefbot)):
            if expand_lefbot[i] in rescopy:
                rescopy.remove(expand_lefbot[i])
    if len(expand_rigtop) != 0:
        for i in range(len(expand_rigtop)):
            if expand_rigtop[i] in rescopy:
                rescopy.remove(expand_rigtop[i])
    if len(expand_rigbot) != 0:
        for i in range(len(expand_rigbot)):
            if expand_rigbot[i] in rescopy:
                rescopy.remove(expand_rigbot[i])
    
    # 以目标簇为核心目标判断轻微相交和严重相交
    list0, list1 = [], []
    for i in range(len(rescopy)):
        if overlap(rescopy[i][:4], dense_obj[:4]) == 0:
            list0.append(rescopy[i]) # 无相交
        elif threshold >= overlap(rescopy[i][:4], dense_obj[:4]) >= 0:
            list1.append(rescopy[i]) # 存在相交
    
    exinfo = situation_2(dense_obj, [list0, list1], height, width)
    
    return dense_obj, exinfo


def intersect(img, res, label_name, save_path, visual, wrcsv):
    # 判断目标尺寸并拼接进坐标矩阵中
    res = area_analysis(img, res, label_name, save_path, False)
    
    # 初始化列表: 所有目标的重叠情况列表, 被看作为一个整体的目标坐标
    list_obj, results_whole = [], copy.deepcopy(res).tolist()
    
    # 分析每个目标与其他目标的模式, 完全无相交, 存在相交但不多, 严重相交
    for i in range(res.shape[0]):
        list1, list2, list3 = [], [], [] # 完全无相交, 小于0.1, 大于0.1
        for j in range(res.shape[0]):
            if res[i].tolist() == res[j].tolist():
                continue
            else:
                if overlap(res[i][:4], res[j][:4]) == 0: # torchvision.ops.box_iou(torch.from_numpy(res[i][:4]).unsqueeze(0), torch.from_numpy(res[j][:4]).unsqueeze(0)) == 0:
                    list1.append(res[j]) # 无相交
                elif threshold >= overlap(res[i][:4], res[j][:4]) > 0: # 0.1 >= torchvision.ops.box_iou(torch.from_numpy(res[i][:4]).unsqueeze(0), torch.from_numpy(res[j][:4]).unsqueeze(0)) > 0:
                    list2.append(res[j]) # 存在相交
                elif overlap(res[i][:4], res[j][:4]) > threshold: # torchvision.ops.box_iou(torch.from_numpy(res[i][:4]).unsqueeze(0), torch.from_numpy(res[j][:4]).unsqueeze(0)) > 0.1:
                    list3.append(res[j]) # 严重相交
        list_obj.append([list1, list2, list3]) # 为每个目标都生成这么三个列表
    
    expandinfo = [] # expandinfo用于储存当前目标四个方向的扩张距离, results_whole用于储存大目标坐标
    # 分别对每个目标进行处理, 即处理其与其他目标的相交信息
    
    for i in range(len(list_obj)):
        # 当该目标与其他目标无重叠时
        if len(list_obj[i][0]) == len(res) - 1:
            # print("no intersect", i)
            # expandinfo.append(onedisjoint(res[i], res)) # 只需要返回单个目标的扩展信息即可
            dense_obj, exinfo = outreach_cluster(res, res[i], list_obj[i][0], list_obj[i][1], height, width)
            expandinfo.append(exinfo)
            results_whole[i][:4] = np.array(dense_obj) # 实际上是用大目标替换列表中的初始目标
        
        # 当该目标与其他目标重叠均不大于阈值时
        if len(list_obj[i][2]) == 0 and len(list_obj[i][1]) != 0:
            # print("some intersect", i)
            if res[i][5] == 0 or res[i][5] == 1 or res[i][5] == 2: # 当目标为Tiny, Small, LowerM时进行扩展分析
                dense_obj, exinfo = outreach_cluster(res, res[i], list_obj[i][0], list_obj[i][1], height, width)
                expandinfo.append(exinfo)
                results_whole[i][:4] = np.array(dense_obj) # 实际上是用大目标替换列表中的初始目标
            else: # 当目标为UpperM或Large时不进行扩展分析, 对其施加歧视
                expandinfo.append(situation_2(res[i], list_obj[i], height, width))
        
        # 当该目标与其他目标存在严重重叠时, 计算是否可以成为目标簇
        if len(list_obj[i][2]) != 0:
            # print("Objects Cluster", i)
            dense_obj, exinfo = object_cluster(res, res[i], list_obj[i][1], list_obj[i][2], height, width)
            expandinfo.append(exinfo)
            results_whole[i][:4] = np.array(dense_obj) # 实际上是用大目标替换列表中的初始目标
            
        """
        # 当该目标存在重叠大于阈值的其他目标时
        if len(list_obj[i][2]) != 0:
            # print("sever intersect", i)
            dense_obj, exinfo = situation_3(res[i], list_obj[i], res, height, width) # exinfo是向外扩张的距离
            expandinfo.append(exinfo)
            results_whole[i][:4] = np.array(dense_obj)
            if list_obj[i][2] != 1:
                list_obj[i][2].append(res[i]) # 该目标自己也算被大目标包含的目标之一
                results_multi[i] = list_obj[i][2] # 将这个大目标包含的所有目标全部赋予results_multi[i]
            else:
                if list_obj[i][2][0][0] <= res[i][0] < res[i][2] <= list_obj[i][2][0][2] and  list_obj[i][2][0][1] <= res[i][1] < res[i][3] <= list_obj[i][2][0][3]:
                    pass # 当该目标被其他目标包含且没有其他目标存在时, 该目标向外扩展10个像素, 因此只包括自己
                else:
                    list_obj[i][2].append(res[i]) # 该目标自己也算被大目标包含的目标之一
                    results_multi[i] = list_obj[i][2] # 将这个大目标包含的所有目标全部赋予results_multi[i]
        """
    
    assert len(res) == len(expandinfo), "length of object list and expandinfo list are not equal"
    
    # 分析目标数据得到多种可能的裁剪方式, 每边适当向外扩展10个像素--此处获得扩展信息准备切割
    cutinfo = []
    for i in range(len(res)):
        if expandinfo[i] != [] and isinstance(expandinfo[i][0], list) == True: # 当该目标扩展信息有多组时
            cut_temp = []
            for j in range(len(expandinfo[i])): # 拆解这个有多组扩展参数的列表
                if len(expandinfo[i][j]) == 0: # 当然也可能里面存在空列表
                    continue
                else: # 的确存在扩展参数而不是空列表时
                    cut_temp.append(computecoordinate(results_whole[i], expandinfo[i][j]))
            if len(cut_temp) == 1:
                cut_temp = cut_temp[0]
            cutinfo.append(cut_temp)
        else: # 扩展信息只有一组时
            if expandinfo[i] == []:
                cutinfo.append([])
            else:
                cutinfo.append(computecoordinate(results_whole[i], expandinfo[i]))
    
    assert len(results_whole) == len(expandinfo), "missing object"
    
    # 此时results_whole内部元素是np.array, 因此需要将results_whole的np.array转换成列表才能排除其中的空列表
    for i in range(len(results_whole)):
        if type(results_whole[i]) != list:
            results_whole[i] = results_whole[i].tolist()
    
    # 将results_whole中长度不等于6的元素移除, 由于将空列表赋予元素中的前4位, 因此会产生长度为2的元素
    pop_list = []
    for i in range(len(results_whole)):
        if len(results_whole[i]) != 6:
            pop_list.append(results_whole[i])
    if len(pop_list) != 0:
        for i in pop_list:
            if i in results_whole:
                results_whole.remove(i)
    
    # 移除列表中的空列表
    for i in range(len(cutinfo)):
        if cutinfo[i] == []:
            cutinfo[i] = 'occ'
    while 'occ' in cutinfo:
        cutinfo.remove('occ')
    for i in range(len(results_whole)):
        if results_whole[i] == []:
            results_whole[i] = 'occ'
    while 'occ' in results_whole:
        results_whole.remove('occ')
    
    # 倒序列表和正序列表不相等, 因此无法对比列表, 必须对比np.array
    # 但np.array无法使用if xx in yy操作, 因此外层储存器必须为list
    # a为np.arr, b也为np.arr, b包含a但其a'乱序, 则if a.all() in b操作可行
    # 但执行该操作需要将[np.arr]转换成np.arr(np.arr), 再逆转转换拼接元素
    # SideNote: np.delete(arr, index, axis=0)中axis=0时结果不会降维
    
    # 将储存大目标和目标簇的列表重新numpy矩阵化
    for i in range(len(results_whole)):
        results_whole[i] = np.array(results_whole[i])
    for i in range(len(cutinfo)):
        cutinfo[i] = np.array(cutinfo[i])
    
    # 筛选出切割坐标列表中相同的坐标
    temp_list, pop_list = [], []
    for i in range(len(cutinfo)):
        if retrieval_list(cutinfo[i], temp_list) == False:
            temp_list.append(cutinfo[i])
        else:
            pop_list.append(i) # 这个i是重复元素的序号
    
    cutinfo, temp_list = temp_list, []
    
    # 排除大目标列表与目标簇列表内的相同元素
    if len(pop_list) != 0:
        for i in pop_list:
            results_whole[i] = 'occ'
    while 'occ' in results_whole:
        results_whole.remove('occ')
    
    # 去除每组坐标末尾的目标尺寸标识, 防止合成代码解析坐标时出现错误
    # 此时results_whole是列表, 内部元素是np.array, 需要转成list
    # 判断单个内部元素是否由多个目标组成
    for i in range(len(results_whole)):
        results_whole[i] = results_whole[i].tolist()
    for i in range(len(results_whole)):
        if is_float(results_whole[i][0]):
            results_whole[i] = results_whole[i][0:5]
        else:
            for j in range(len(results_whole[i])):
                results_whole[i][j] = results_whole[i][j][0:5]
    
    # -------------------------------------------------------------------------
    """
    # 逐个目标地可视化图块计算是否正确
    if visual == True:
        for i in range(len(results_whole)):
            img_temp = copy.deepcopy(img)
            label_temp = torch.from_numpy(np.array([results_whole[i]])) # 只可视化当前目标, 此句与上句不能同时存在, 否则会覆盖上句
            for k in range(len(label_temp)):
                box = label_temp[k]
                x0 = int(box[0])
                y0 = int(box[1])
                x1 = int(box[2])
                y1 = int(box[3])
                color = (_COLORS[k] * 255).astype(np.uint8).tolist()
                cv2.rectangle(img_temp, (x0, y0), (x1, y1), color, 2) # cv2 reqire [h, w, c]
            cv2.imwrite(save_path + str(label_name[0:-4]) + '_ObjCluster_' + str(i) + '.jpg', img_temp)
    """
    # -------------------------------------------------------------------------
    
    # 计算大目标内包含的目标簇, 采用这种方式而不是从头就开始收集目标簇坐标
    # 是因为前段排除相同元素的代码会导致results_multi排除掉正确元素
    # 注意此时results_whole外部为列表, 内部元素为np.array, 并且还需要
    # 去除每组坐标末尾的目标尺寸标识, 防止合成代码解析坐标时出现错误
    results_multi = []
    for i in range(len(results_whole)):
        temp_list = []
        for j in range(res.shape[0]):
            if results_whole[i][0] <= res[j][0] and results_whole[i][1] <= res[j][1] and results_whole[i][2] >= res[j][2] and results_whole[i][3] >= res[j][3]:
                temp_list.append(res[j][0:5])
        results_multi.append(temp_list)
    
    assert len(cutinfo) == len(results_whole), "something is wrong"
    
    # 计算图块/目标比例方便检索
    ratio, ratio_temp = [], []
    for i in range(len(cutinfo)):
        if isinstance(cutinfo[i][0], float) == True:
            target_area = (results_whole[i][2] - results_whole[i][0]) * (results_whole[i][3] - results_whole[i][1]) # 目标区域坐标
            cutout_area = (cutinfo[i][2] - cutinfo[i][0]) * (cutinfo[i][3] - cutinfo[i][1]) # 图块区域坐标
            ratio_area = cutout_area / target_area # 目标区域与图块区域的面积比例
            ratio.append([target_area, cutout_area, ratio_area])
        else:
            for j in range(len(cutinfo[i])):
                target_area = (results_whole[i][2] - results_whole[i][0]) * (results_whole[i][3] - results_whole[i][1]) # 目标区域坐标
                cutout_area = (cutinfo[i][j][2] - cutinfo[i][j][0]) * (cutinfo[i][j][3] - cutinfo[i][j][1]) # 图块区域坐标
                ratio_area = cutout_area / target_area # 目标区域与图块区域的面积比例
                ratio_temp.append([target_area, cutout_area, ratio_area])
            ratio.append(ratio_temp)
            ratio_temp = []
    
    # 逐个目标地可视化图块计算是否正确
    if visual == True:
        label_tensor = torch.from_numpy(res[:, :4]) # 取得所有目标坐标, 用于在所有目标基础上可视化当前目标
        for i in range(len(cutinfo)):
            if isinstance(cutinfo[i][0], float) == True:
                img_temp = copy.deepcopy(img)
                label_temp = torch.cat((label_tensor, torch.from_numpy(np.array([cutinfo[i]]))), 0) # 在所有目标基础上可视化当前目标
                # label_temp = torch.from_numpy(np.array([cutinfo[i]])) # 只可视化当前目标, 此句与上句不能同时存在, 否则会覆盖上句
                for k in range(len(label_temp)):
                    box = label_temp[k]
                    x0 = int(box[0])
                    y0 = int(box[1])
                    x1 = int(box[2])
                    y1 = int(box[3])
                    if k >= 18:
                        ind = normlize_index(k)
                    else:
                        ind = k
                    color = (_COLORS[ind] * 255).astype(np.uint8).tolist()
                    cv2.rectangle(img_temp, (x0, y0), (x1, y1), color, 2) # cv2 reqire [h, w, c]
                cv2.imwrite(save_path + str(label_name[0:-4]) + '_' + str(i) + '.jpg', img_temp)
            else:
                for j in range(len(cutinfo[i])):
                    img_temp = copy.deepcopy(img)
                    label_temp = torch.cat((label_tensor, torch.from_numpy(np.array([cutinfo[i][j]]))), 0) # 在所有目标基础上可视化当前目标
                    # label_temp = torch.from_numpy(np.array([cutinfo[i][j]])) # 只可视化当前目标, 此句与上句不能同时存在, 否则会覆盖上句
                    for k in range(len(label_temp)):
                        box = label_temp[k]
                        x0 = int(box[0])
                        y0 = int(box[1])
                        x1 = int(box[2])
                        y1 = int(box[3])
                        if k >= 18:
                            ind = normlize_index(k)
                        else:
                            ind = k
                        color = (_COLORS[ind] * 255).astype(np.uint8).tolist()
                        cv2.rectangle(img_temp, (x0, y0), (x1, y1), color, 2) # cv2 reqire [h, w, c]
                    cv2.imwrite(save_path + str(label_name[0:-4]) + '_' + str(i) + '_' + str(j) + '.jpg', img_temp)
    
    # 检测目标区域是否小于切割区域(不可删除此检测代码)
    for i in range(len(results_whole)):
        if len(cutinfo[i].shape) > 1:
            for j in range(len(cutinfo[i])):
                assert results_whole[i][0] >= cutinfo[i][j][0], f"{label_name} xmin went wrong {results_whole}, {cutinfo}"
                assert results_whole[i][1] >= cutinfo[i][j][1], f"{label_name} ymin went wrong {results_whole}, {cutinfo}"
                assert results_whole[i][2] <= cutinfo[i][j][2], f"{label_name} xmax went wrong {results_whole}, {cutinfo}"
                assert results_whole[i][3] <= cutinfo[i][j][3], f"{label_name} ymax went wrong {results_whole}, {cutinfo}"
        else:
            assert results_whole[i][0] >= cutinfo[i][0], f"{label_name} xmin went wrong {results_whole}, {cutinfo}"
            assert results_whole[i][1] >= cutinfo[i][1], f"{label_name} ymin went wrong {results_whole}, {cutinfo}"
            assert results_whole[i][2] <= cutinfo[i][2], f"{label_name} xmax went wrong {results_whole}, {cutinfo}"
            assert results_whole[i][3] <= cutinfo[i][3], f"{label_name} ymax went wrong {results_whole}, {cutinfo}"

    # 将坐标转换成list保存进csv中
    if wrcsv == True:
        # 将results_multi的所有元素全部转成list
        for i in range(len(results_multi)):
            for j in range(len(results_multi[i])):
                if isinstance(results_multi[i][j], float) == True:
                    pass
                elif isinstance(results_multi[i][j], list) != True:
                    results_multi[i][j] = results_multi[i][j].tolist()
    
        # 写入csv文件: 图片名, 目标坐标(xyxycls), 裁剪坐标(xyxy), 目标面积, 裁剪面积, 图块/目标比例
        # 写入csv文件: image name, xmin, ymin, xmax, ymax, cls, cutxmin, cutymin, cutxmax, cutymax, obj area, cut area, ratio=obj_area / cut_area
        # new csv文件: csv文件名: image name, 单/多目标flag ('Single', 'Multi'), 
        for i in range(len(cutinfo)):
            if isinstance(cutinfo[i][0], float) == True:
                list_info = [label_name, # 图片名称
                             results_whole[i][0], results_whole[i][1], results_whole[i][2], results_whole[i][3], results_whole[i][4], # 目标整体坐标
                             cutinfo[i][0], cutinfo[i][1], cutinfo[i][2], cutinfo[i][3], # 图块切割坐标
                             ratio[i][0], ratio[i][1], ratio[i][2], 
                             results_multi[i]] # 目标面积, 图块面积, 两者比例
                with open(save_path, 'a', newline="") as csvfile:
                    writer0 = csv.writer(csvfile)
                    writer0.writerow(list_info)
            else:
                for j in range(len(cutinfo[i])):
                    list_info = [label_name, # 图片名称
                                 results_whole[i][0], results_whole[i][1], results_whole[i][2], results_whole[i][3], results_whole[i][4], # 目标整体坐标
                                 cutinfo[i][j][0], cutinfo[i][j][1], cutinfo[i][j][2], cutinfo[i][j][3], # 图块切割坐标
                                 ratio[i][j][0], ratio[i][j][1], ratio[i][j][2], 
                                 results_multi[i]] # 目标面积, 图块面积, 两者比例
                    with open(save_path, 'a', newline="") as csvfile:
                        writer0 = csv.writer(csvfile)
                        writer0.writerow(list_info)


# 输入np.array和包含多个np.array的list, 判断该arr是否在list中
def retrieval_list(array1, list1):
    for i in range(len(list1)):
        if array1.shape[0] == list1[i].shape[0]:
            if (array1 == list1[i]).all():
                return True
    return False


# 通过float()函数的转换成功与否测试参数是否为浮点数
def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


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
    
    # 将目标尺寸参数拼接到总坐标矩阵
    res = res.tolist()
    for i in range(len(res)):
        res[i].append(obj_size[i])
    res = np.array(res)
    
    return res

# -----------------------------------------------------------------------------

def computecoordinate(coor, info):
    xcutmin = coor[0] - info[2] # x1 - left
    ycutmin = coor[1] - info[0] # y1 - top
    xcutmax = coor[2] + info[3] # x2 + right
    ycutmax = coor[3] + info[1] # y2 + bottom
    
    if xcutmin < 0:
        xcutmin = 0
    if ycutmin < 0:
        ycutmin = 0
    if xcutmax > width:
        xcutmax = width
    if ycutmax > height:
        ycutmax = height

    return [xcutmin, ycutmin, xcutmax, ycutmax]


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
    # 设定相交阈值
    global threshold
    threshold = 0.2
    
    # 可视化标签确认分割正确还是写入csv文件
    # state, visual, wrcsv = 0, True, False # 初级测试: 单目标, 可视化, 不写入
    # state, visual, wrcsv = 1, True, False # 高级测试: 多目标, 可视化, 不写入
    state, visual, wrcsv = 2, False, True # 合成测试: 多目标, 不可视, 写入坐标
    
    # 将科学计数法转换为数字
    np.set_printoptions(suppress=True)
    
    if state == 0:
        # 设定单张图片与标签的DeBug测试文件夹路径 Single image test
        img_path = 'D:/AICV-DSTRethink/Code-DataPoolTest&Results/singleimg/'
        label_path = 'D:/AICV-DSTRethink/Code-DataPoolTest&Results/singlelab/'
        test_path = 'D:/AICV-DSTRethink/Code-DataPoolTest&Results/singleres/'
    elif state == 1:
        # 设定多张图片与标签的批量测试文件夹路径 Multi images test
        img_path = 'F:/VOCtrainval_images/' # D:/AICV-DSTRethink/Code-DataPoolTest&Results/vocimages/
        label_path = 'F:/VOCtrainval_xml/' # D:/AICV-DSTRethink/Code-DataPoolTest&Results/voclabels/
        test_path = 'D:/AICV-DSTRethink/Code-DataPoolTest&Results/vocresults/'
    elif state == 2:
        # 设定正式计算目标边界的数据集与csv文件路径 Formal Comuptation for Patches
        img_path = 'D:/AICV-DSTRethink/Code-DataPoolTest&Results/singleimg/' # D:/AICV-DSTRethink/Code-DataPoolTest&Results/vocdataset/images/ F:/VOCtrainval_images/
        label_path = 'D:/AICV-DSTRethink/Code-DataPoolTest&Results/singlelab/' # D:/AICV-DSTRethink/Code-DataPoolTest&Results/vocdataset/labels/ F:/VOCtrainval_xml/
        csv_path = 'D:/AICV-DSTRethink/Code-DataPoolTest&Results/singletest.csv' # VOC_0712trainval_MultiObject.csv / VOC_0712trainval_ObjectInfo.csv
    
    # 测试路径与csv文件保存路径不能同时存在, 但同时存在时会选择csv_path
    if 'test_path' in dir():
        save_path = locals()['test_path']
    if 'csv_path' in dir():
        save_path = locals()['csv_path']
    
    for label_name in os.listdir(label_path):
        # 设定图片路径并读取图片宽高, [xcen, ycen, w, h] to [xmin, ymin, xmax, ymax]
        img = cv2.imread(img_path + label_name[0:-4] + '.jpg')
        height, width = img.shape[0], img.shape[1]
        
        # 读取txt文件或xml文件中的目标坐标
        if label_name[-4:] == '.txt':
            res = xywh_to_xyxy(txt_target(label_path + label_name, height, width))
        elif label_name[-4:] == '.xml':
            res, height, width = xml_target(label_path + label_name, class_to_ind)
        
        # print(res)
        # print("------------------------")
        
        label_flag = 0 # 初始化label_flag
        
        # 当前只需要intersect()即可, disjoint()已经不再被需要
        intersect(img, res, label_name, save_path, visual, wrcsv)
        """
        # 如果直接将只有一个目标的图片送入后续会产生问题
        if len(res) == 1:
            disjoint(img, res, label_name, save_path, visual, wrcsv)
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
        
        # 检测该图片是否存在重叠, 需要调用哪套方案
        if label_flag == 1:
            intersect(img, res, label_name, save_path, visual, wrcsv)
        else:
            disjoint(img, res, label_name, save_path, visual, wrcsv)
        """