# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 20:19:29 2023

@author: Crysi
"""

import cv2
import copy
import torch
import numpy as np

xml_head = '''<annotation>
    <folder>VOC2022</folder>
    <filename>{}</filename>.
    <source>
        <database>The VOC2022 Datasets</database>
        <annotation>PASCAL VOC2022</annotation>
        <image>flickr</image>
        <flickrid>325991873</flickrid>
    </source>
    <owner>
        <flickrid>null</flickrid>
        <name>null</name>
    </owner>    
    <size>
        <width>{}</width>
        <height>{}</height>
        <depth>{}</depth>
    </size>
    <segmented>0</segmented>
    '''

xml_obj = '''
    <object>        
        <name>{}</name>
        <pose>Rear</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{}</xmin>
            <ymin>{}</ymin>
            <xmax>{}</xmax>
            <ymax>{}</ymax>
        </bndbox>
    </object>
    '''

xml_end = '''
</annotation>'''

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


def xml_label_generate(height, width, label_num, xml_path):
    label_str = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", 
                 "cat", "chair", "cow", "diningtable", "dog", "horse",  "motorbike", 
                 "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")
    
    # 创建xml的head与目标格
    head, obj = xml_head.format(str(os.path.basename(xml_path[:-4])), str(width), str(height), str(3)), ''
    
    for i in range(len(label_num)):
        obj += xml_obj.format(label_str[int(label_num[i][4])], 
                              int(label_num[i][0]), 
                              int(label_num[i][1]), 
                              int(label_num[i][2]), 
                              int(label_num[i][3]))

    # 打开xml文件,写入各种信息
    with open(xml_path, 'w') as f_xml:
        f_xml.write(head + obj + xml_end)


def correct_distortion_patch1(width, height, img, lab, cluster, count):
    # print("+++++++++++")
    # print("count: ", count)
    # print("img.shape: ", img.shape, " ", width, " ", height)
    
    # 空位大于图块, 填充图块
    if (width / img.shape[1]) >= 1 and (height / img.shape[0]) >= 1:
        # print("into section 1") 
        pad_top, pad_lef = int(height - img.shape[0]), int(width - img.shape[1])
        img = cv2.copyMakeBorder(img, pad_top, 0, pad_lef, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        lab[0], lab[2], lab[1], lab[3] = lab[0] + pad_lef, lab[2] + pad_lef, lab[1] + pad_top, lab[3] + pad_top
        if len(cluster) != 0:
            for i in range(len(cluster)):
                cluster[i][0], cluster[i][1] = cluster[i][0] + pad_lef, cluster[i][1] + pad_top
                cluster[i][2], cluster[i][3] = cluster[i][2] + pad_lef, cluster[i][3] + pad_top
    
    # 空位小于图块, 切割图块
    elif (width / img.shape[1]) <= 1 and (height / img.shape[0]) <= 1:
        # print("into section 2")
        top, bot, lef, rig = lab[1], img.shape[0] - lab[3], lab[0], img.shape[1] - lab[2]
        if bot < 0: bot = 0
        if rig < 0: rig = 0
        need_cut_w, need_cut_h = img.shape[1] - width, img.shape[0] - height
        new_img_xmin, new_img_xmax = need_cut_w * (lef/(lef+rig)), img.shape[1] - need_cut_w * (rig/(lef+rig))
        new_img_ymin, new_img_ymax = need_cut_h * (top/(top+bot)), img.shape[0] - need_cut_h * (bot/(top+bot))
        img = img[int(new_img_ymin): int(new_img_ymax), int(new_img_xmin): int(new_img_xmax)]
        lab[0], lab[2] = lab[0] - new_img_xmin, lab[2] - new_img_xmin
        lab[1], lab[3] = lab[1] - new_img_ymin, lab[3] - new_img_ymin
        if len(cluster) != 0:
            for i in range(len(cluster)):
                cluster[i][0], cluster[i][1] = cluster[i][0] - new_img_xmin, cluster[i][1] - new_img_ymin
                cluster[i][2], cluster[i][3] = cluster[i][2] - new_img_xmin, cluster[i][3] - new_img_ymin
    
    # 空位宽大于图块, 空位高小于图块, 切割图块高, 填充图块宽
    elif (width / img.shape[1]) >= 1 and (height / img.shape[0]) <= 1:
        # print("into section 3")
        top, bot = lab[1], img.shape[0] - lab[3]
        if bot < 0: bot = 0
        need_cut_h = img.shape[0] - height
        new_img_ymin, new_img_ymax = need_cut_h * (top/(top+bot)), img.shape[0] - need_cut_h * (bot/(top+bot))
        img = img[int(new_img_ymin): int(new_img_ymax), 0: int(img.shape[1])]
        lab[1], lab[3] = lab[1] - new_img_ymin, lab[3] - new_img_ymin
        
        pad_lef = int(width - img.shape[1])
        img = cv2.copyMakeBorder(img, 0, 0, pad_lef, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        lab[0], lab[2] = lab[0] + pad_lef, lab[2] + pad_lef
        
        if len(cluster) != 0:
            for i in range(len(cluster)):
                cluster[i][1], cluster[i][3] = cluster[i][1] - new_img_ymin, cluster[i][3] - new_img_ymin
                cluster[i][0], cluster[i][2] = cluster[i][0] + pad_lef, cluster[i][2] + pad_lef
    
    # 空位宽小于图块, 空位高大于图块, 切割图块宽, 填充图块高
    elif (width / img.shape[1]) <= 1 and (height / img.shape[0]) >= 1:
        # print("into section 4")
        lef, rig = lab[0], img.shape[1] - lab[2]
        if rig < 0: rig = 0
        need_cut_w = img.shape[1] - width
        new_img_xmin, new_img_xmax = need_cut_w * (lef/(lef+rig)), img.shape[1] - need_cut_w * (rig/(lef+rig))
        img = img[0: int(img.shape[0]), int(new_img_xmin): int(new_img_xmax)]
        lab[0], lab[2] = lab[0] - new_img_xmin, lab[2] - new_img_xmin
        
        pad_top = int(height - img.shape[0])
        img = cv2.copyMakeBorder(img, pad_top, 0, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        lab[1], lab[3] = lab[1] + pad_top, lab[3] + pad_top
        
        if len(cluster) != 0:
            for i in range(len(cluster)):
                cluster[i][0], cluster[i][2] = cluster[i][0] - new_img_xmin, cluster[i][2] - new_img_xmin
                cluster[i][1], cluster[i][3] = cluster[i][1] + pad_top, cluster[i][3] + pad_top
        
    # print("----------")
    return img, lab, cluster


def correct_distortion_patch2(width, height, img, lab, cluster, count): # ): # 
    # print("+++++++++++")
    # print("count: ", count)
    # print("img.shape: ", img.shape, " ", width, " ", height)
    
    # 空位大于图块, 填充图块
    if (width / img.shape[1]) >= 1 and (height / img.shape[0]) >= 1:
        # print("into section 1") 
        pad_top, pad_rig = int(height - img.shape[0]), int(width - img.shape[1])
        img = cv2.copyMakeBorder(img, pad_top, 0, 0, pad_rig, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        lab[1], lab[3] = lab[1] + pad_top, lab[3] + pad_top
        if len(cluster) != 0:
            for i in range(len(cluster)):
                cluster[i][1], cluster[i][3] = cluster[i][1] + pad_top, cluster[i][3] + pad_top
    
    # 空位小于图块, 切割图块
    elif (width / img.shape[1]) <= 1 and (height / img.shape[0]) <= 1:
        # print("into section 2")
        top, bot, lef, rig = lab[1], img.shape[0] - lab[3], lab[0], img.shape[1] - lab[2]
        if bot < 0: bot = 0
        if rig < 0: rig = 0
        need_cut_w, need_cut_h = img.shape[1] - width, img.shape[0] - height
        new_img_xmin, new_img_xmax = need_cut_w * (lef/(lef+rig)), img.shape[1] - need_cut_w * (rig/(lef+rig))
        new_img_ymin, new_img_ymax = need_cut_h * (top/(top+bot)), img.shape[0] - need_cut_h * (bot/(top+bot))
        img = img[int(new_img_ymin): int(new_img_ymax), int(new_img_xmin): int(new_img_xmax)]
        lab[0], lab[2] = lab[0] - new_img_xmin, lab[2] - new_img_xmin
        lab[1], lab[3] = lab[1] - new_img_ymin, lab[3] - new_img_ymin
        if len(cluster) != 0:
            for i in range(len(cluster)):
                cluster[i][0], cluster[i][2] = cluster[i][0] - new_img_xmin, cluster[i][2] - new_img_xmin
                cluster[i][1], cluster[i][3] = cluster[i][1] - new_img_ymin, cluster[i][3] - new_img_ymin
    
    # 空位宽大于图块, 空位高小于图块, 切割图块高, 填充图块宽
    elif (width / img.shape[1]) >= 1 and (height / img.shape[0]) <= 1:
        # print("into section 3")
        top, bot = lab[1], img.shape[0] - lab[3]
        if bot < 0: bot = 0
        need_cut_h = img.shape[0] - height
        new_img_ymin, new_img_ymax = need_cut_h * (top/(top+bot)), img.shape[0] - need_cut_h * (bot/(top+bot))
        img = img[int(new_img_ymin): int(new_img_ymax), 0: int(img.shape[1])]
        lab[1], lab[3] = lab[1] - new_img_ymin, lab[3] - new_img_ymin
        if len(cluster) != 0:
            for i in range(len(cluster)):
                cluster[i][1], cluster[i][3] = cluster[i][1] - new_img_ymin, cluster[i][3] - new_img_ymin
        
        pad_rig = int(width - img.shape[1])
        img = cv2.copyMakeBorder(img, 0, 0, 0, pad_rig, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    # 空位宽小于图块, 空位高大于图块, 切割图块宽, 填充图块高
    elif (width / img.shape[1]) <= 1 and (height / img.shape[0]) >= 1:
        # print("into section 4")
        lef, rig = lab[0], img.shape[1] - lab[2]
        if rig < 0: rig = 0
        need_cut_w = img.shape[1] - width
        new_img_xmin, new_img_xmax = need_cut_w * (lef/(lef+rig)), img.shape[1] - need_cut_w * (rig/(lef+rig))
        img = img[0: int(img.shape[0]), int(new_img_xmin): int(new_img_xmax)]
        lab[0], lab[2] = lab[0] - new_img_xmin, lab[2] - new_img_xmin
        
        pad_top = int(height - img.shape[0])
        img = cv2.copyMakeBorder(img, pad_top, 0, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        lab[1], lab[3] = lab[1] + pad_top, lab[3] + pad_top
        
        if len(cluster) != 0:
            for i in range(len(cluster)):
                cluster[i][0], cluster[i][2] = cluster[i][0] - new_img_xmin, cluster[i][2] - new_img_xmin
                cluster[i][1], cluster[i][3] = cluster[i][1] + pad_top, cluster[i][3] + pad_top
        
    # print("----------")
    return img, lab, cluster


def correct_distortion_patch3(width, height, img, lab, cluster, count): # ): # 
    # print("+++++++++++")
    # print("count: ", count)
    # print("img.shape: ", img.shape, " ", width, " ", height)
    
    # 空位大于图块, 填充图块
    if (width / img.shape[1]) >= 1 and (height / img.shape[0]) >= 1:
        # print("into section 1")
        pad_bot, pad_lef = int(height - img.shape[0]), int(width - img.shape[1])
        img = cv2.copyMakeBorder(img, 0, pad_bot, pad_lef, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        lab[0], lab[2] = lab[0] + pad_lef, lab[2] + pad_lef
        if len(cluster) != 0:
            for i in range(len(cluster)):
                cluster[i][0], cluster[i][2] = cluster[i][0] + pad_lef, cluster[i][2] + pad_lef
    
    # 空位小于图块, 切割图块
    elif (width / img.shape[1]) <= 1 and (height / img.shape[0]) <= 1:
        # print("into section 2")
        top, bot, lef, rig = lab[1], img.shape[0] - lab[3], lab[0], img.shape[1] - lab[2]
        if bot < 0: bot = 0
        if rig < 0: rig = 0
        need_cut_w, need_cut_h = img.shape[1] - width, img.shape[0] - height
        new_img_xmin, new_img_xmax = need_cut_w * (lef/(lef+rig)), img.shape[1] - need_cut_w * (rig/(lef+rig))
        new_img_ymin, new_img_ymax = need_cut_h * (top/(top+bot)), img.shape[0] - need_cut_h * (bot/(top+bot))
        img = img[int(new_img_ymin): int(new_img_ymax), int(new_img_xmin): int(new_img_xmax)]
        lab[0], lab[2] = lab[0] - new_img_xmin, lab[2] - new_img_xmin
        lab[1], lab[3] = lab[1] - new_img_ymin, lab[3] - new_img_ymin
        if len(cluster) != 0:
            for i in range(len(cluster)):
                cluster[i][0], cluster[i][2] = cluster[i][0] - new_img_xmin, cluster[i][2] - new_img_xmin
                cluster[i][1], cluster[i][3] = cluster[i][1] - new_img_ymin, cluster[i][3] - new_img_ymin
    
    # 空位宽大于图块, 空位高小于图块, 切割图块高, 填充图块宽
    elif (width / img.shape[1]) >= 1 and (height / img.shape[0]) <= 1:
        # print("into section 3")
        top, bot = lab[1], img.shape[0] - lab[3]
        if bot < 0: bot = 0
        need_cut_h = img.shape[0] - height
        new_img_ymin, new_img_ymax = need_cut_h * (top/(top+bot)), img.shape[0] - need_cut_h * (bot/(top+bot))
        img = img[int(new_img_ymin): int(new_img_ymax), 0: int(img.shape[1])]
        lab[1], lab[3] = lab[1] - new_img_ymin, lab[3] - new_img_ymin
        
        pad_lef = int(width - img.shape[1])
        img = cv2.copyMakeBorder(img, 0, 0, pad_lef, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        lab[0], lab[2] = lab[0] + pad_lef, lab[2] + pad_lef
        
        if len(cluster) != 0:
            for i in range(len(cluster)):
                cluster[i][0], cluster[i][2] = cluster[i][0] + pad_lef, cluster[i][2] + pad_lef
                cluster[i][1], cluster[i][3] = cluster[i][1] - new_img_ymin, cluster[i][3] - new_img_ymin
    
    # 空位宽小于图块, 空位高大于图块, 切割图块宽, 填充图块高
    elif (width / img.shape[1]) <= 1 and (height / img.shape[0]) >= 1:
        # print("into section 4")
        lef, rig = lab[0], img.shape[1] - lab[2]
        if rig < 0: rig = 0
        need_cut_w = img.shape[1] - width
        new_img_xmin, new_img_xmax = need_cut_w * (lef/(lef+rig)), img.shape[1] - need_cut_w * (rig/(lef+rig))
        img = img[0: int(img.shape[0]), int(new_img_xmin): int(new_img_xmax)]
        lab[0], lab[2] = lab[0] - new_img_xmin, lab[2] - new_img_xmin
        
        pad_bot = int(height - img.shape[0])
        img = cv2.copyMakeBorder(img, 0, pad_bot, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        if len(cluster) != 0:
            for i in range(len(cluster)):
                cluster[i][0], cluster[i][2] = cluster[i][0] - new_img_xmin, cluster[i][2] - new_img_xmin
        
    # print("----------")
    return img, lab, cluster


def correct_distortion_patch4(width, height, img, lab, cluster, count): # ): # 
    # print("+++++++++++")
    # print("count: ", count)
    # print("img.shape: ", img.shape, " ", width, " ", height)
    
    # 空位大于图块, 填充图块
    if (width / img.shape[1]) >= 1 and (height / img.shape[0]) >= 1:
        # print("into section 1")
        pad_bot, pad_rig = int(height - img.shape[0]), int(width - img.shape[1])
        img = cv2.copyMakeBorder(img, 0, pad_bot, 0, pad_rig, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    # 空位小于图块, 切割图块
    elif (width / img.shape[1]) <= 1 and (height / img.shape[0]) <= 1:
        # print("into section 2")
        top, bot, lef, rig = lab[1], img.shape[0] - lab[3], lab[0], img.shape[1] - lab[2]
        if bot < 0: bot = 0
        if rig < 0: rig = 0
        need_cut_w, need_cut_h = img.shape[1] - width, img.shape[0] - height
        new_img_xmin, new_img_xmax = need_cut_w * (lef/(lef+rig)), img.shape[1] - need_cut_w * (rig/(lef+rig))
        new_img_ymin, new_img_ymax = need_cut_h * (top/(top+bot)), img.shape[0] - need_cut_h * (bot/(top+bot))
        img = img[int(new_img_ymin): int(new_img_ymax), int(new_img_xmin): int(new_img_xmax)]
        lab[0], lab[2] = lab[0] - new_img_xmin, lab[2] - new_img_xmin
        lab[1], lab[3] = lab[1] - new_img_ymin, lab[3] - new_img_ymin
        if len(cluster) != 0:
            for i in range(len(cluster)):
                cluster[i][0], cluster[i][2] = cluster[i][0] - new_img_xmin, cluster[i][2] - new_img_xmin
                cluster[i][1], cluster[i][3] = cluster[i][1] - new_img_ymin, cluster[i][3] - new_img_ymin
    
    # 空位宽大于图块, 空位高小于图块, 切割图块高, 填充图块宽
    elif (width / img.shape[1]) >= 1 and (height / img.shape[0]) <= 1:
        # print("into section 3")
        top, bot = lab[1], img.shape[0] - lab[3]
        if bot < 0: bot = 0
        need_cut_h = img.shape[0] - height
        new_img_ymin, new_img_ymax = need_cut_h * (top/(top+bot)), img.shape[0] - need_cut_h * (bot/(top+bot))
        img = img[int(new_img_ymin): int(new_img_ymax), 0: int(img.shape[1])]
        lab[1], lab[3] = lab[1] - new_img_ymin, lab[3] - new_img_ymin
        
        pad_rig = int(width - img.shape[1])
        img = cv2.copyMakeBorder(img, 0, 0, 0, pad_rig, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        if len(cluster) != 0:
            for i in range(len(cluster)):
                cluster[i][1], cluster[i][3] = cluster[i][1] - new_img_ymin, cluster[i][3] - new_img_ymin
    
    # 空位宽小于图块, 空位高大于图块, 切割图块宽, 填充图块高
    elif (width / img.shape[1]) <= 1 and (height / img.shape[0]) >= 1:
        # print("into section 4")
        lef, rig = lab[0], img.shape[1] - lab[2]
        if rig < 0: rig = 0
        need_cut_w = img.shape[1] - width
        new_img_xmin, new_img_xmax = need_cut_w * (lef/(lef+rig)), img.shape[1] - need_cut_w * (rig/(lef+rig))
        img = img[0: int(img.shape[0]), int(new_img_xmin): int(new_img_xmax)]
        lab[0], lab[2] = lab[0] - new_img_xmin, lab[2] - new_img_xmin
        
        pad_bot = int(height - img.shape[0])
        img = cv2.copyMakeBorder(img, 0, pad_bot, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        if len(cluster) != 0:
            for i in range(len(cluster)):
                cluster[i][0], cluster[i][2] = cluster[i][0] - new_img_xmin, cluster[i][2] - new_img_xmin
        
    # print("----------")
    return img, lab, cluster


def shrink_object_single(img, lab, scope): # img.shape=[h, w, c]
    w, h = lab[2] - lab[0], lab[3] - lab[1]
    x = pow(scope[1] * w / h, 0.5) # 保持原有宽高比, 计算得到缩放后的目标宽度: x/y = w/h, x*y=32×32, y=32×32/x, x*x/(32×32)=w/h
    ratio_shrink = x / w # new / ori
    img = np.transpose(img, (2, 0, 1)) # [h, w, c]转换为[c, h, w]
    img_temp = torch.nn.functional.interpolate(torch.from_numpy(img.copy()).unsqueeze(0), 
                                                                size=(int(img.shape[1] * ratio_shrink), int(img.shape[2] * ratio_shrink)), 
                                                                mode='nearest').squeeze(0).numpy()
    lab[0], lab[1], lab[2], lab[3] = lab[0] * ratio_shrink, lab[1] * ratio_shrink, lab[2] * ratio_shrink, lab[3] * ratio_shrink
    img_temp = np.transpose(img_temp, (1, 2, 0)) # [c, h, w]转换为[h, w, c]
    
    return img_temp, lab


def shrink_object_cluster(img, lab, cluster, mean, scope): # img.shape=[h, w, c]
    # w, h = lab[2] - lab[0], lab[3] - lab[1]
    # x = pow(scope[1] * w / h, 0.5)
    ratio_shrink = scope[1] / mean
    img = np.transpose(img, (2, 0, 1)) # [h, w, c]转换为[c, h, w]
    img_temp = torch.nn.functional.interpolate(torch.from_numpy(img.copy()).unsqueeze(0), 
                                                                size=(int(img.shape[1] * ratio_shrink), int(img.shape[2] * ratio_shrink)), 
                                                                mode='nearest').squeeze(0).numpy()
    lab[0], lab[1], lab[2], lab[3] = lab[0] * ratio_shrink, lab[1] * ratio_shrink, lab[2] * ratio_shrink, lab[3] * ratio_shrink
    for i in range(len(cluster)):
        cluster[i][0], cluster[i][1] = cluster[i][0] * ratio_shrink, cluster[i][1] * ratio_shrink
        cluster[i][2], cluster[i][3] = cluster[i][2] * ratio_shrink, cluster[i][3] * ratio_shrink
    img_temp = np.transpose(img_temp, (1, 2, 0)) # [c, h, w]转换为[h, w, c]
    
    return img_temp, lab, cluster


def check_objectsize(width, height, img, lab, cluster): # [h, w, c]
    if (lab[3] - lab[1]) > height and (lab[2] - lab[0]) > width: # h > height, w > width
        x, y = lab[2] - lab[0], lab[3] - lab[1]
        ratio_shrink = min(width / x, height / y)
        img = np.transpose(img, (2, 0, 1)) # [h, w, c]转换为[c, h, w]
        img = torch.nn.functional.interpolate(torch.from_numpy(img.copy()).unsqueeze(0), 
                                              size=(int(img.shape[1] * ratio_shrink), int(img.shape[2] * ratio_shrink)), 
                                              mode='nearest').squeeze(0).numpy()
        lab[0], lab[1], lab[2], lab[3] = lab[0] * ratio_shrink, lab[1] * ratio_shrink, lab[2] * ratio_shrink, lab[3] * ratio_shrink
        if len(cluster) != 0:
            for i in range(len(cluster)):
                cluster[i][0], cluster[i][1] = cluster[i][0] * ratio_shrink, cluster[i][1] * ratio_shrink
                cluster[i][2], cluster[i][3] = cluster[i][2] * ratio_shrink, cluster[i][3] * ratio_shrink
        img = np.transpose(img, (1, 2, 0)) # [c, h, w]转换为[h, w, c]
        
    if (lab[3] - lab[1]) <= height and (lab[2] - lab[0]) > width: # h <= height, w > width
        x = lab[2] - lab[0]
        ratio_shrink = width / x
        img = np.transpose(img, (2, 0, 1)) # [h, w, c]转换为[c, h, w]
        img = torch.nn.functional.interpolate(torch.from_numpy(img.copy()).unsqueeze(0), 
                                              size=(int(img.shape[1] * ratio_shrink), int(img.shape[2] * ratio_shrink)), 
                                              mode='nearest').squeeze(0).numpy()
        lab[0], lab[1], lab[2], lab[3] = lab[0] * ratio_shrink, lab[1] * ratio_shrink, lab[2] * ratio_shrink, lab[3] * ratio_shrink
        if len(cluster) != 0:
            for i in range(len(object_cluster)):
                cluster[i][0], cluster[i][1] = cluster[i][0] * ratio_shrink, cluster[i][1] * ratio_shrink
                cluster[i][2], cluster[i][3] = cluster[i][2] * ratio_shrink, cluster[i][3] * ratio_shrink
        img = np.transpose(img, (1, 2, 0)) # [c, h, w]转换为[h, w, c]
        
    if (lab[3] - lab[1]) > height and (lab[2] - lab[0]) <= width: # h > height, w <= width
        y = lab[3] - lab[1]
        ratio_shrink = height / y
        img = np.transpose(img, (2, 0, 1)) # [h, w, c]转换为[c, h, w]
        img = torch.nn.functional.interpolate(torch.from_numpy(img.copy()).unsqueeze(0), 
                                              size=(int(img.shape[1] * ratio_shrink), int(img.shape[2] * ratio_shrink)), 
                                              mode='nearest').squeeze(0).numpy()
        lab[0], lab[1], lab[2], lab[3] = lab[0] * ratio_shrink, lab[1] * ratio_shrink, lab[2] * ratio_shrink, lab[3] * ratio_shrink
        if len(cluster) != 0:
            for i in range(len(object_cluster)):
                cluster[i][0], cluster[i][1] = cluster[i][0] * ratio_shrink, cluster[i][1] * ratio_shrink
                cluster[i][2], cluster[i][3] = cluster[i][2] * ratio_shrink, cluster[i][3] * ratio_shrink
        img = np.transpose(img, (1, 2, 0)) # [c, h, w]转换为[h, w, c]
        
    if (lab[3] - lab[1]) <= height and (lab[2] - lab[0]) <= width: # h <= height, w<= width
        pass

    return img, lab, cluster


def check_objectorient(width, height, img, lab, cluster):
    if int(width) > int(height): # require w > h
        # if top-left patch w < h then rotate 90°
        if img.shape[1] < img.shape[0]:
            # 首先将标签翻转90°
            lab = filp_label_90(img, lab)
            
            # 再翻转目标簇中的标签
            if len(cluster) != 0:
                for i in range(len(cluster)):
                    cluster[i] = filp_label_90(img, cluster[i])
            
            # 再把图片翻转90°
            img = np.rot90(img)
    else: # require w < h
        # if top-left patch w > h then rotate 90°
        if img.shape[1] > img.shape[0]:
            # 首先将标签翻转90°
            lab = filp_label_90(img, lab)
            
            # 再翻转目标簇中的标签
            if len(cluster) != 0:
                for i in range(len(cluster)):
                    cluster[i] = filp_label_90(img, cluster[i])
            
            # 再把图片翻转90°
            img = np.rot90(img)
    
    return img, lab, cluster


def calculate_cluster_meanarea(cluster):
    area = 0 # 初始化面积和参数
    for i in range(len(cluster)):
        area = area + (cluster[i][2] - cluster[i][0]) * (cluster[i][3] - cluster[i][1])
    mean = area / len(cluster)
    
    return mean


def filp_label_90(img, lab):
    xmin_new = lab[1] # xmin_new = ymin
    ymin_new = img.shape[1] - lab[2] # ymin_new = width - xmax
    xmax_new = lab[3] # xmax_new = ymin + target_height
    ymax_new = img.shape[1] - lab[0] # ymax_new = width - xmax + target_width
    lab[0] = xmin_new
    lab[1] = ymin_new
    lab[2] = xmax_new
    lab[3] = ymax_new
    return lab


