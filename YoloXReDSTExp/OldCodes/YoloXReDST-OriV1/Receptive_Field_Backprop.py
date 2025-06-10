# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 17:49:56 2022
@author: ChengYuxuan
Function: Receptive Field with Backpropagation
"""
import cv2
import torch
import imutils
import numpy as np
import torch.nn as nn
from collections import namedtuple
from torchvision import transforms
# from FullyConvolutionalResnet18 import FullyConvolutionalResnet18
# from Yolov5s_selfcustom import Yolov5s
from yolox.models import YOLOXERF, YOLOPAFPN, YOLOXHead

def backprop_receptive_field(image, model):
    model = model.train() # 将网络置于训练模式
    for module in model.modules():
        print(module)
        try:
            nn.init.constant_(module.weight, 0.05) # inference overflows with ones
            nn.init.zeros_(module.bias)
            nn.init.zeros_(module.running_mean)
            nn.init.ones_(module.running_var)
        except:
            pass
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            module.eval() # 将网络置于评估模式

    input = torch.ones_like(image, requires_grad=True)
    out = model(input) # out.shape: [1,1000,3,8]

    grad = torch.zeros_like(out, requires_grad=True) # 默认shape为[1,1000,3,8]
    
    a,b = grad.shape[2],grad.shape[3]
    a = round(a / 2)
    b = round(b / 2)

    with torch.no_grad(): grad[0, 0, a, b] = 1 # 将最大概率处置为1
    # grad是全零矩阵,可能具体用哪一层切片的中心置为1都没关系,所以这里使用第1层切片
    out.backward(gradient=grad) # out是全零矩阵输入网络后得到的输出,并不为零
    gradient_of_input = input.grad[0, 0].data.numpy() # 取得第1个矩阵切片
    gradient_of_input = gradient_of_input / np.amax(gradient_of_input) # 归一化
    
    # Counting the number of activated pixels 计算激活像素面积
    W = gradient_of_input.shape[0] # 如果W与H反过来就会报错,说明顺序正确
    H = gradient_of_input.shape[1]
    count = 0
    for i in range(0,W):
        for j in range(0,H):
            if gradient_of_input[i][j] > 0:
                count = count + 1
    # print(count)
    
    return gradient_of_input

def find_rect(activations): # 取得激活区域的边框坐标
    Rect = namedtuple('Rect', 'x1 y1 x2 y2') # 创建一个只有key而无value的空元组

    # Dilate and erode the activations to remove grid-like artifacts 去除网格状条纹
    kernel = np.ones((5, 5), np.uint8)
    activations = cv2.dilate(activations, kernel=kernel) # 卷积核中如果存在白色那么整个卷积核全是白色
    activations = cv2.erode(activations, kernel=kernel) # 卷积核中哪种颜色的比例大那么整个卷积核全是此颜色
    # 可以使用cv2.imshow("erode", activations),cv2.waitKey(0),cv2.destroyAllWindows()查看原始、膨胀、腐蚀图像

    # Binarize the activations 二值化激活图像
    _, activations = cv2.threshold(activations, 0.25, 1, type=cv2.THRESH_BINARY) # 阈值,二值化图像
    activations = activations.astype(np.uint8).copy()

    # Find the countour of the binary blob
    contours, _ = cv2.findContours(activations, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    # 返回能表示轮廓的点的集合&轮廓数量,必须输入黑白图像,即二值化图像,首先将原始图像变为灰度图再变为二值化图像
    # cv2.findContours查找物体的轮廓,contours是元组,里面就1个元素即numpy三维矩阵,eg[214,1,2]
    
    # Find bounding box around the object.
    rect = cv2.boundingRect(contours[0])
    # cv2.boundingRect即用最小的矩形将形状包裹起来,输入轮廓点集合,返回边角坐标,一般与cv2.findContours连用
    # rect是元组,里面每个元素全是数字,感觉可以用列表的方式储存数字,但这里就是用了元组

    return Rect(rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]) # (x1=839, y1=391, x2=1142, y2=690)


def normalize(activations): # 数据归一化: zhuanlan.zhihu.com/p/424518359, https://zhuanlan.zhihu.com/p/76682561
    activations = activations - np.min(activations[:])
    activations = activations / np.max(activations[:])
    return activations # 归一化以去除极端值影响


def visualize_activations(image, activations, show_bounding_rect=False):
    activations = normalize(activations)

    # 将原图与激活矩阵叠加到一起,注意激活矩阵只取了1层,因此这里需要叠加三层对应原图的三个通道
    activations_multichannel = np.stack([activations, activations, activations], axis=2)
    masked_image = (image * activations_multichannel).astype(np.uint8)

    if show_bounding_rect:
        rect = find_rect(activations) # 取得激活区域的边框坐标
        cv2.rectangle(masked_image, (rect.x1, rect.y1), (rect.x2, rect.y2), color=(0, 0, 255), thickness=2)
        # 原始方框,通过OpenCV的cv2.findContours与cv2.boundingRect取得左上和右下坐标
        x1 = round(0.918*rect.x1 + 0.082*rect.x2)
        x2 = round(0.918*rect.x2 + 0.082*rect.x1)
        y1 = round(0.918*rect.y1 + 0.082*rect.y2)
        y2 = round(0.918*rect.y2 + 0.082*rect.y1)
        cv2.rectangle(masked_image, (x1, y1),(x2, y2), color=(0, 0, 255), thickness=2)
        # 方框面积为原始方框面积的0.7,有效感受野内的有效目标大小
        x11 = round(0.85*rect.x1 + 0.15*rect.x2)
        x22 = round(0.15*rect.x1 + 0.85*rect.x2)
        y11 = round(0.85*rect.y1 + 0.15*rect.y2)
        y22 = round(0.15*rect.y1 + 0.85*rect.y2)
        cv2.rectangle(masked_image, (x11, y11),(x22, y22), color=(0, 0, 255), thickness=2)
        # 方框边长为原始方框边长的0.7,有效感受野内的真正有效目标大小
        
        print("The Optimum Target W & H: ", x22-x11, y22-y11)
        
    return masked_image

def main(image_path, model):
    # Read the image
    image = cv2.imread(image_path)
    origin_image = image
    
    # Convert original image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToTensor(), # Convert image to tensor.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # Subtract mean
                             std=[0.229, 0.224, 0.225]) # Divide by standard deviation
    ])
    
    image = transform(image)
    image = image.unsqueeze(0)

    receptive_field_map = backprop_receptive_field(image, model) # 取得输入层激活像素矩阵
    
    # Display the images
    displayWidth = 640 # 设置展示时图片宽度
    # cv2.imshow("Original Image", imutils.resize(image,width=displayWidth))
    cv2.imshow("receptive_field_max_activation", imutils.resize(visualize_activations(origin_image, receptive_field_map, show_bounding_rect=True),
                                                                width=displayWidth))
    cv2.waitKey(0)
    cv2.destroyAllWindows() # 结束程序按q键即可

if __name__ == "__main__":
    # 图片路径与模型选择
    image_path = "D:\\AICV-TestFile2\\a.jpg" # "D:\\AICV-VisReceptF\\Receptive-Field-With-Backprop\\camel.jpg" # 
    # model = FullyConvolutionalResnet18() # 载入的只是空网络结构而已
    # model = Yolov5s()
    
    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(1.00, 1.00, in_channels=in_channels, act="silu") # 0.33, 0.50
    model = YOLOXERF(backbone)
    
    main(image_path, model)