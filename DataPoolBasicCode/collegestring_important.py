# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:57:36 2022

@author: DELL
"""
import pandas as pd
from ast import literal_eval


def divide_str(string):
    string = list(string)                                             # 列表化字符串, 方便后续通过枚举的方式处理字符串
    if string[0] == "[" and string[1] == "[":                         # 当字符串有双层[时, 去除外层[
        string.pop(0)
    if string[len(string)-1] == "]" and string[len(string)-2] == "]": # 当字符串有双层]时, 去除外层]
        string.pop(len(string)-1)
    coor_list, list_temp, num_temp, count = [], [], 0, 0              # coor_list外层列表, list_temp内层列表, num_temp数字首字符, count计数器
    for i in range(len(string)):
        if string[i] == '[':                          # 当字符为[时, 初始化内层列表, 数字首字符, 计数器
            list_temp, num_temp, count = [], string[i+1], i+1
        elif string[i] != ' ' and string[i] != ']':   # 字符不为空格同时前字符不为]时, 拼接数字
            if count == i:
                pass
            else:
                num_temp = num_temp + string[i]
        elif string[i] == ' ' and string[i-1] != ']': # 字符为空格且前字符不为]时, 拼接组装好的数字成列表, 并重新初始化
            list_temp.append(float(num_temp))
            num_temp, count = string[i+1], i+1
        elif string[i] == ' ' and string[i-1] == ']': # 字符为空格且前字符为]时, 跳过本轮次, 因为是两个内层列表中的空格
            pass
        elif string[i] == ']':                        # 字符为]时, 拼接组装好的列表进外层列表
            list_temp.append(float(num_temp))
            coor_list.append(list_temp)
    
    return coor_list


csv_path = 'D:/AICV-DSTRethink/DataPool/test.csv'
df = pd.read_table(csv_path, header=None)
list_target = df.values.tolist()
for i in range(len(list_target)):
    print("\n")
    print(list_target[i])
    list2 = list_target[i][0].split(",")
    list2[0] = list2[0] # xml name
    list2[1] = float(list2[1]) # object xmin
    list2[2] = float(list2[2]) # object ymin
    list2[3] = float(list2[3]) # object xmax
    list2[4] = float(list2[4]) # object ymax
    list2[5] = int(float(list2[5])) # object cls
    list2[6] = int(float(list2[6])) # cutarea xmin
    list2[7] = int(float(list2[7])) # cutarea ymin
    list2[8] = int(float(list2[8])) # cutarea xmax
    list2[9] = int(float(list2[9])) # cutarea ymax
    list2[10] = float(list2[10]) # area target
    list2[11] = float(list2[11]) # area cutout
    list2[12] = float(list2[12]) # target/cutout
    
    print("name: ", list2[0])
    print("object: ", list2[1], list2[2], list2[3], list2[4], list2[5])
    print("cutout: ", list2[6], list2[7], list2[8], list2[9])
    print("ratio: ", list2[10], list2[11], list2[12])
    
    if list2[13] != str(0):
        temp = list2[13]
        for i in range(14, len(list2)):
            temp = temp + list2[i]
        list2[13] = literal_eval(temp)
        print("list2[13]: ", list2[13])
        list2[13] = divide_str(list2[13])
        print("list2[13]: ", list2[13], type(list2[13]))
        