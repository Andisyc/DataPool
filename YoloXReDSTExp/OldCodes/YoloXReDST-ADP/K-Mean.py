# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 21:30:46 2022

@author: DELL
"""

import matplotlib.pyplot as plt
import numpy as np


# 1、先列出一列数组
y = np.array([[2,3],[2,2],[3,4],[1,2],[9,8],[8,8],[8,7],[9,9],[1,5],[2,4],[7.9,7],[8.9,9],[2,1],[7,9],[9,7],[8,8],[9,7],[8,8.5]])


# 2，方便画图
x_scatter = [data[0] for data in y]
y_scatter = [data[1] for data in y]


# 3，分类 0类，1类
k = [0, 1, 2, 3]


# 3，先给出两个点
y_center = np.array([[7,7], [9,9]], dtype = np.float64)
y_center_new = np.copy(y_center)


# 4，用于判断是否退出
flag = True


# 5，用于后者分类
y_res = np.zeros(len(y))


#6，用于判断是否退出
tmp = 0

while flag and tmp<10:
    tmp += 1
    
    for i in range(len(y)):             # y里面的点数
        item = y[i]                     # 二维数组里面的一维数组
        d0 = (item[0] - y_center[0][0]) ** 2 + (item[1] - y_center[0][1]) ** 2
        
        d1 = (item[0] - y_center[1][0]) ** 2 + (item[1] - y_center[1][1]) ** 2
        
        y_res[i] = 0 if d0>d1 else 1      # 测距分类

    y_res_like_0 = [[i,i] for i in y_res] # 二维列表，里面非0即1
    temp_center = y * y_res_like_0          # 乘0得0，乘1得1            ####关键，到后面中心点不会动的原因是，y中分类已经分的固定了，每次计算用都是固定的几个数
    y_center_new[0] = np.sum(temp_center, axis=0) / np.sum(y_res) # x坐标求和，y坐标求和，以及得到的y_res(1的求和)

    y_res_like_1 = [[1-i,1-i] for i in y_res] # 二维列表，里面非0即1
    temp_center = y * y_res_like_1              # 乘0得0，乘1得1            ####关键，到后面中心点不会动的原因是，y中分类已经分的固定了，每次计算用都是固定的几个数
    y_center_new[1] = np.sum(temp_center, axis=0) / (len(y_res) - np.sum(y_res)) #y_res的总数减去1的总数等于0的总数


    if(y_center != y_center_new).any():      # 判断前后两次中心点是否相同
        y_center = y_center_new
    else:
        flag = False                        # 相同直接退出

    # 7画图
    plt.scatter(x_scatter, y_scatter, c='blue', marker='.')
    plt.scatter([y_center[0][0], y_center[1][0]], [y_center[0][1], y_center[1][1]], c="red", s=100, marker='*')
    plt.title("K-means")
    plt.show()

