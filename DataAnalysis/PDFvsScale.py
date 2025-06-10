# -*- coding: utf-8 -*-
"""
Created on Sat May 14 21:31:08 2022

@author: Modified by Cheng Yuxuan
         https://blog.csdn.net/weixin_35757704/article/details/119541386 
         https://github.com/scipy/scipy/blob/v1.8.0/scipy/stats/_stats_py.py
        
Explain: 通过设定直方图参数取得数据集的目标分布
"""
# import math
# import scipy
import warnings
import numpy as np
import pandas as pd
# import seaborn as sns
from scipy import stats
# import statsmodels.api as sm
import matplotlib.pyplot as plt
from collections import namedtuple


HistogramResult = namedtuple('HistogramResult', ('count', 'lowerlimit', 'binsize', 'extrapoints'))

def histogram(a, numbins=10, defaultlimits=None, weights=None, printextras=False):
    a = np.ravel(a)
    if defaultlimits is None:
        if a.size == 0:
            # handle empty arrays. Undetermined range, so use 0-1.
            defaultlimits = (0, 1)
        else:
            # no range given, so use values in `a`
            data_min = a.min()
            data_max = a.max()
            # Have bins extend past min and max values slightly
            s = (data_max - data_min) / (2. * (numbins - 1.))
            defaultlimits = (data_min - s, data_max + s)

    # use numpy's histogram method to compute bins
    hist, bin_edges = np.histogram(a, bins=numbins, range=defaultlimits, weights=weights)
    
    # hist are not always floats, convert to keep with old output
    hist = np.array(hist, dtype=float)
    # fixed width for bins is assumed, as numpy's histogram gives
    # fixed width bins for int values for 'bins'
    binsize = bin_edges[1] - bin_edges[0]
    # calculate number of extra points
    extrapoints = len([v for v in a if defaultlimits[0] > v or v > defaultlimits[1]])
    if extrapoints > 0 and printextras:
        warnings.warn("Points outside given histogram range = %s" % extrapoints)

    return HistogramResult(hist, defaultlimits[0], binsize, extrapoints), bin_edges


def get_data_frequency(csv_path='D:/AICV-YoloXReDST-ADP/dataanalysis/COCO_ObjectAnalysis_Phase_1.csv'):
    # 导入数据
    csv_path = csv_path

    list1 = []
    area = []

    # 选择需要导入的类别的CSV文件名称
    df = pd.read_table(csv_path, header=0)
    list1 = df.values.tolist()

    small_object_1, small_object_2, small_object_3 = 0, 0, 0
    medium_object_1, medium_object_2 = 0, 0
    large_object_1, large_object_2 = 0, 0

    # 提取单个类别标签中的目标面积
    for i in range(len(list1)):
        list2 = list1[i][0].split(",")
        # list2[0] = int(list2[0]) # list2[0]=cls
        list2[1] = float(list2[1]) # list2[1]=xmin
        list2[2] = float(list2[2]) # list2[2]=ymin
        list2[3] = float(list2[3]) # list2[3]=xmax
        list2[4] = float(list2[4]) # list2[4]=ymax
    
        if 0 <= list2[3]*512 * list2[4]*320 <= 12 * 12:
            small_object_1 = small_object_1 + 1
            
        if 12 * 12 <= list2[3]*512 * list2[4]*320 <= 21 * 21:
            small_object_2 = small_object_2 + 1
        
        if 21 * 21 <= list2[3]*512 * list2[4]*320 <= 32 * 32:
            small_object_3 = small_object_3 + 1
        
        if 32 * 32 <= list2[3]*512 * list2[4]*320 <= 54 * 54:
            medium_object_1 = medium_object_1 + 1
        
        if 54 * 54 <= list2[3]*512 * list2[4]*320 <= 96 * 96:
            medium_object_2 = medium_object_2 + 1
        
        if 96 * 96 <= list2[3]*512 * list2[4]*320 <= 166 * 166:
            large_object_1 = large_object_1 + 1
            
        if 166 * 166 <= list2[3]*512 * list2[4]*320:
            large_object_2 = large_object_2 + 1
    
        area.append(list2[3]*512 * list2[4]*320) # txt标签中为xywh
        # area.append((list2[3]-list2[1]) * (list2[4]-list2[2])) # xml标签中为xyxy格式

    # numpy矩阵化列表
    area = np.array(area) # max(area) = 262144 = 512 * 512
    
    res_freq = stats.relfreq(area, numbins=100, defaultreallimits=(0, max(area))) # len(area) 10 100
    _, bin_edges = histogram(np.asanyarray(area), 100, defaultlimits=(0, max(area))) # h是矩阵,l、b、e是单个数字
    
    y_axis = [small_object_1, small_object_2, small_object_3, 
              medium_object_1, medium_object_2, 
              large_object_1, large_object_2]
    
    return res_freq, area, bin_edges, y_axis

def get_data_distributions(res_freq, area):
    # 设定y轴矩阵: PDF离散概率密度函数, CDF离散累积分布函数
    # pdf_value = res_freq.frequency # stats.relfreq中进行了归一化操作,因此全是小数
    pdf_value2 = res_freq.frequency * area.shape[0]
    # cdf_value = np.cumsum(res_freq.frequency) # np.cumsum指定轴每个值为之前所有值加上自己的和
    """
    print("y-axis")
    for i in range(len(pdf_value2)):
        print(pdf_value2[i])
    """
    return pdf_value2

def plot_data_hist(res_freq, area, bin_edges, pdf_value2, y_axis):
    # 设定x轴矩阵: np.linspace(start, stop, number) np.logspace(start, stop, number)
    x = np.linspace(res_freq.binsize / 2, res_freq.binsize * res_freq.frequency.size - res_freq.binsize / 2, res_freq.frequency.size)
    # x = res_freq.lowerlimit + np.linspace(0, res_freq.binsize * res_freq.frequency.size, res_freq.frequency.size)
    # print("res_freq.lowerlimit: ", res_freq.lowerlimit) # min(area) - res_freq.binsize / 2
    # print("res_freq.binsize: ", res_freq.binsize) # np.histogram内部使用np.linspace(first, last, num+1)
    # print("res_freq.extrapoints: ", res_freq.extrapoints)
    # print("res_freq.frequency.size: ", res_freq.frequency.size) # 柱子的数量,上文自己设定的值
    # print("min: ", min(area)) # 数据集内目标的最小值
    # print("res_freq.binsize * res_freq.frequency.size: ", res_freq.binsize * res_freq.frequency.size)
    
    # 抽样函数,子数据集依据此函数曲线从主数据集中抽样
    # y2 = [55500000 / a - 50 for a in x]
    
    """
    print("bin_edges") # 图像中每个柱子的取值区间
    for i in range(len(bin_edges)):
        print(bin_edges[i])
        # print(bin_edges[i] - res_freq.binsize / 2) # 图像中每个柱子取值区间的中值
    
    print("x-axis") # 图像中每个柱子的中值
    for i in range(len(x)):
        print(int(x[i]), ": ", pdf_value2[i])
        
    print("res_freq.binsize: ", res_freq.binsize) # 间隔长度
    
    total = 0 # 抽样函数在对应尺度下所有目标数量
    for i in range(len(y2)):
        total = total + y2[i]
    print("total number of targets: ", round(total))
    """
    x = [10, 20, 30, 40, 50, 60, 70]
    x_index = ['0-12*12', '12*12-21*21', '21*21-32*32', '32*32-54*54', '54*54-96*96', '96*96-166*166', '166*166-∞']
    pdf_value2 = y_axis
    print(y_axis[0])
    
    # 画出两张图像
    fig1 = plt.figure(figsize=(16, 8)) # (32, 16)会比(16, 8)更清晰
    ax1 = fig1.add_subplot(121) # 这个值别动
    ax1.bar(x, pdf_value2, width=1) # width=res_freq.binsize
    ax1.set_xlabel("Scale")
    ax1.set_ylabel("Number")
    _ = plt.xticks(x, x_index) # 显示标签
    
    plt.show()


if __name__ == '__main__':
    res_freq, area, bin_edges, y_axis = get_data_frequency()
    pdf_value2 = get_data_distributions(res_freq, area)
    plot_data_hist(res_freq, area, bin_edges, pdf_value2, y_axis)