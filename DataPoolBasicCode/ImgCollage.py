# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 15:08:50 2022
@author: Cheng Yuxuan
"""
import os
import cv2
import copy
import random
import torch
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

class AnnotationTransform(object):

    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = np.empty((0, 5))
        for obj in target.iter("object"):
            difficult = obj.find("difficult")
            if difficult is not None:
                difficult = int(difficult.text) == 1
            else:
                difficult = False
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.strip()
            bbox = obj.find("bndbox")

            pts = ["xmin", "ymin", "xmax", "ymax"]
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text)) - 1
                # scale height or width
                # cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            
            if name == 'mask':
                name = 'face_mask'
            if name == 'face_nask':
                name = 'face_mask'
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        width = int(target.find("size").find("width").text)
        height = int(target.find("size").find("height").text)
        img_info = (height, width)

        return res, img_info

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

def to_image_list_synthesize_4(tensors, targets, input_size, size_divisible=0):
    # tensors = transposed_info[0] # batch经过list(zip(*x))处理后的transposed_info是1个列表,其中有1个元组
    if isinstance(tensors, (tuple, list)): # 判断tensors是否属于tuple或list tensors[i].shape=(3, 640, 640)
        # targets = transposed_info[1] # targets[i].shape=(120, 5)
        # img_ids = transposed_info[2] # ((321, 500), (333, 500), (320, 499), (220, 331), (334, 499), (220, 331), (321, 500), (333, 500))

        # synthesize data:
        assert len(tensors) % 4 == 0, 'len(tensor) % 4 != 0, could not be synthesized ! uneven'
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors])) # 取得最大图片的尺寸,(3, 320, 498) c h w
        
        # TODO Ideally, just remove this and let me model handle arbitrary input sizs
        if size_divisible > 0:
            import math
            
            # 将最大宽高的图片尺寸缩放为步长的倍数
            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride) # math.ceil返回输入值的上整数
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride) # math.ceil返回输入值的上整数
            max_size = tuple(max_size)

        batch_shape = (len(tensors)//4,) + max_size # 将两个元组拼接成1个元组
        syn_batched_imgs = torch.from_numpy(tensors[0]).new(*batch_shape).zero_() # syn_batched_imgs.shape = [1, 3, 320, 512]
        # 创造1个batch_shape类型的空张量,并且每处都赋予0值,tensors[0].new()创建1个无值的张量,啥张量后跟new都行
        # 但new()需要输入参数,并且输入的参数不能是列表,因此需要用*batch_shape方式来去除列表
        
        syn_imgs = []
        syn_targets = []
        with torch.no_grad():
            for idx, pad_img in enumerate(syn_batched_imgs): # idx = 0, pad_img.shape = [3, 320, 512], 因为len(tensors)//4==1
                # currently suppose first w then h
                new_h, new_w = max_size[1]//2, max_size[2]//2 # 缩放后尺寸是最大宽高的一半

                # NOTE: interpolate api require first h then w ! interpolate()作用为缩放图片, [c, h, w]
                mode = 'nearest' # squeeze(0)去除第1维
                topLeftImg = torch.nn.functional.interpolate(torch.from_numpy(tensors[idx*4]).unsqueeze(0),size=(new_h, new_w),mode=mode).squeeze(0)
                topRightImg = torch.nn.functional.interpolate(torch.from_numpy(tensors[idx*4+1]).unsqueeze(0),size=(new_h, new_w),mode=mode).squeeze(0)
                bottomLeftImg = torch.nn.functional.interpolate(torch.from_numpy(tensors[idx*4+2]).unsqueeze(0),size=(new_h, new_w),mode=mode).squeeze(0)
                bottomRightImg = torch.nn.functional.interpolate(torch.from_numpy(tensors[idx*4+3]).unsqueeze(0),size=(new_h, new_w),mode=mode).squeeze(0)
                """
                print("pad_img: ", pad_img.shape)
                print("topLeftImg: ", tensors[idx*4].shape, topLeftImg.shape)
                print("topRightImg: ", tensors[idx*4+1].shape, topRightImg.shape)
                print("bottomLeftImg: ", tensors[idx*4+2].shape, bottomLeftImg.shape)
                print("bottomRightImg: ", tensors[idx*4+3].shape, bottomRightImg.shape)
                """
                c = topLeftImg.shape[0] # 取得缩放后图片的通道数
                assert c == topRightImg.shape[0] and c == bottomLeftImg.shape[0] and c == bottomRightImg.shape[0] # 确定缩放后子图片通道数相等
                
                # 当pad_img的宽高不是new_w×2&new_h×2时拼贴会报错
                if topRightImg.shape[1] * 2 != pad_img.shape[1] or topRightImg.shape[2] * 2 != pad_img.shape[2]:
                    pad_img = torch.nn.functional.interpolate(pad_img.unsqueeze(0), size=(new_h * 2, new_w * 2), mode=mode).squeeze(0)
                
                # 将四张缩放后的子图片拼接成一张图片,画个图就很清晰了
                pad_img[:c, :topLeftImg.shape[1], :topLeftImg.shape[2]].copy_(topLeftImg)
                pad_img[:c, :topRightImg.shape[1], topLeftImg.shape[2]:].copy_(topRightImg)
                pad_img[:c, topLeftImg.shape[1]:, :bottomLeftImg.shape[2]].copy_(bottomLeftImg)
                pad_img[:c, topRightImg.shape[1]:, topLeftImg.shape[2]:].copy_(bottomRightImg)
                
                # cv2.imwrite("D:/AICV-YoloXReGPU/abc.jpg", np.transpose(pad_img.numpy(), (1, 2, 0)))
                
                # resize each of four sub-imgs into (new_h, new_w) scale
                # resize api require first w then h ! (120, 5) 120个[cls, x, y, w, h]
                topLeftBL = resize(torch.from_numpy(targets[idx*4]), (tensors[idx*4].shape[2], tensors[idx*4].shape[1]), (new_w, new_h))
                topRightBL = resize(torch.from_numpy(targets[idx*4+1]), (tensors[idx*4+1].shape[2], tensors[idx*4+1].shape[1]), (new_w, new_h))
                bottomLeftBL = resize(torch.from_numpy(targets[idx*4+2]), (tensors[idx*4+2].shape[2], tensors[idx*4+2].shape[1]), (new_w, new_h))
                bottomRightBL = resize(torch.from_numpy(targets[idx*4+3]), (tensors[idx*4+3].shape[2], tensors[idx*4+3].shape[1]), (new_w, new_h))
                
                # 计算四张图片上得到新目标所需的偏移值
                offsets = [torch.Tensor([0.0,0.0,0.0,0.0]), torch.Tensor([new_w,0.0,new_w,0.0]), torch.Tensor([0.0,new_h,0.0,new_h]), torch.Tensor([new_w,new_h,new_w,new_h])]
                
                # append offsets to box coordinates except for topLeftBL 调整GT框坐标到新位置
                topLeftBL = compute_tensor(topLeftBL, offsets[0])
                topRightBL = compute_tensor(topRightBL, offsets[1])
                bottomLeftBL = compute_tensor(bottomLeftBL, offsets[2])
                bottomRightBL = compute_tensor(bottomRightBL, offsets[3])
                
                # 从xyxy还原成xywh格式
                topLeftBL = xyxy_to_xywh(topLeftBL)
                topRightBL = xyxy_to_xywh(topRightBL)
                bottomLeftBL = xyxy_to_xywh(bottomLeftBL)
                bottomRightBL = xyxy_to_xywh(bottomRightBL)
                
                # 填充图片至input_size, default=[640,640] [height, width]
                if pad_img.shape[1] < input_size[0]: # input_size=[height, width], pad_img=[c, h, w]
                    dh = input_size[0] - pad_img.shape[1]
                    dh /= 2
                    pad_top, pad_bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
                else:
                    pad_top, pad_bottom = 0, 0
                
                if pad_img.shape[2] < input_size[1]: # input_size=[height, width], pad_img=[c, h, w]
                    dw = input_size[1] - pad_img.shape[2]
                    dw /= 2
                    pad_left, pad_right = int(round(dw - 0.1)), int(round(dw + 0.1))
                else:
                    pad_left, pad_right = 0, 0
                
                pad_img = cv2.copyMakeBorder(np.transpose(pad_img.numpy(), (1, 2, 0)), pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
                
                # cv2.imwrite("D:/AICV-YoloXReDST-Orig/after_pad_img.jpg", pad_img)
                
                pad_img = torch.from_numpy(np.transpose(pad_img, (2, 0, 1)))
                
                # 保存图片到列表中,最后拼接成批量
                syn_imgs.append(pad_img.unsqueeze(0))
                
                # 根据填充过程相应地移动边框坐标
                topLeftBL = add_tensor(topLeftBL, pad_left, pad_top)
                topRightBL = add_tensor(topRightBL, pad_left, pad_top)
                bottomLeftBL = add_tensor(bottomLeftBL, pad_left, pad_top)
                bottomRightBL = add_tensor(bottomRightBL, pad_left, pad_top)
                
                
                topLeft = xywh_to_xyxy_syn(copy.deepcopy(topLeftBL))
                topRight = xywh_to_xyxy_syn(copy.deepcopy(topRightBL))
                bottomLeft = xywh_to_xyxy_syn(copy.deepcopy(bottomLeftBL))
                bottomRight = xywh_to_xyxy_syn(copy.deepcopy(bottomRightBL))
                
                # 可视化拼贴图片的标签是否与目标匹配
                temp_img = np.transpose(copy.deepcopy(pad_img).numpy(), (1, 2, 0)).copy()
                # cv2.imwrite('/mnt/yoloxredstorig/synthesis/syn_img_' + str(random.randint(0, 100000)) + '.jpg', temp_img)
                label_tensor = torch.cat((topLeft,  topRight, bottomLeft, bottomRight), 0)
                _COLORS = np.array([0.000, 0.447, 0.741]).astype(np.float32).reshape(-1, 3)
                for i in range(len(label_tensor)):
                    box = label_tensor[i]
                    x0 = int(box[1])
                    y0 = int(box[2])
                    x1 = int(box[3])
                    y1 = int(box[4])
                    color = (_COLORS[0] * 255).astype(np.uint8).tolist()
                    cv2.rectangle(temp_img, (x0, y0), (x1, y1), color, 2)
                cv2.imwrite('E:/AICV-TestFile5/res/syn_img_' + str(random.randint(0, 100000)) + '.jpg', temp_img) # cv2.imwrite reqire [h, w, c]
                # print("already done!")
                
                topLeft = xyxy_to_xywh(copy.deepcopy(topLeftBL))
                topRight = xyxy_to_xywh(copy.deepcopy(topRightBL))
                bottomLeft = xyxy_to_xywh(copy.deepcopy(bottomLeftBL))
                bottomRight = xyxy_to_xywh(copy.deepcopy(bottomRightBL))
                
                """
                # 添加0值行变为shape=(120, 5)
                syn_bbox = torch.cat((topLeftBL, topRightBL, bottomLeftBL, bottomRightBL), dim=0)
                zero = torch.tensor([[0., 0., 0., 0., 0.]])
                for i in range(120 - syn_bbox.shape[0]):
                    syn_bbox = torch.cat((syn_bbox, zero),dim=0)
                del zero
                syn_targets.append(syn_bbox.unsqueeze(0))
                """
        """
        # 检查ID数量是否也为4的倍数
        assert len(img_ids)%4 == 0
        
        # 拼接合成目标与合成标签为batch张量
        syn_imgs = torch.cat(syn_imgs, dim=0)
        syn_targets = torch.cat(syn_targets, dim=0)
        
        return syn_imgs, syn_targets
        """
    else:
        raise TypeError("Unsupported type for to_image_list: {}".format(type(tensors)))


def resize(targets, sizeori, sizenew): # 输入xywh返回xyxy
    # 去掉标签张量中的0值行
    temp = []
    for i in range(targets.shape[0]):
        if targets[i][3]!=0 and targets[i][4]!=0:
            temp.append(targets[i])
    targets = torch.stack(temp, dim=0)
    del temp

    # 取得新宽高与旧宽高的比例元组
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(sizenew, sizeori))

    # 当宽高新旧比例不等时需要分别操作
    ratio_width, ratio_height = ratios
    for i in range(targets.shape[0]):
        xmin = targets[i][1] - targets[i][3]/2
        ymin = targets[i][2] - targets[i][4]/2
        xmax = targets[i][1] + targets[i][3]/2
        ymax = targets[i][2] + targets[i][4]/2
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        targets[i][1] = scaled_xmin
        targets[i][2] = scaled_ymin
        targets[i][3] = scaled_xmax
        targets[i][4] = scaled_ymax
    
    return targets


def compute_tensor(tensor1, tensor2):
    for i in range(tensor1.shape[0]):
        tensor1[i][1:] = tensor1[i][1:] + tensor2
    
    return tensor1


def add_tensor(tensor1, pad_left, pad_top): # 专门用于为xcen&ycen添加填充量
    for i in range(tensor1.shape[0]):
        tensor1[i][1] = tensor1[i][1] + pad_left
        tensor1[i][2] = tensor1[i][2] + pad_top

    return tensor1


def xyxy_to_xywh(tensor): # 输入xyxy返回xywh
    for i in range(tensor.shape[0]):
        scaled_xcen = (tensor[i][3] + tensor[i][1]) / 2
        scaled_ycen = (tensor[i][4] + tensor[i][2]) / 2
        scaled_w = tensor[i][3] - tensor[i][1]
        scaled_h = tensor[i][4] - tensor[i][2]
        tensor[i][1] = scaled_xcen
        tensor[i][2] = scaled_ycen
        tensor[i][3] = scaled_w
        tensor[i][4] = scaled_h
    
    return tensor


def xywh_to_xyxy_syn(tensor): # [cls, x1, y1, x2, y2]
    for i in range(tensor.shape[0]):
        x1 = tensor[i][1] - tensor[i][3] / 2
        y1 = tensor[i][2] - tensor[i][4] / 2
        x2 = tensor[i][1] + tensor[i][3] / 2
        y2 = tensor[i][2] + tensor[i][4] / 2
        tensor[i][1] = x1
        tensor[i][2] = y1
        tensor[i][3] = x2
        tensor[i][4] = y2
        
    return tensor

def xywh_to_xyxy_lab(tensor): # [x1, y1, x2, y2, cls]
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

if __name__ == "__main__":
    # 将科学计数法转换为数字
    np.set_printoptions(suppress=True)

    # 设定并封装类别
    """
    VOC_CLASSES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
                   "car", "cat", "chair", "cow", "diningtable", "dog", "horse", 
                   "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")
    """
    VOC_CLASSES = ("person",)
    class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
    
    path_image = 'E:/AICV-TestFile5/img/'
    path_label = 'E:/AICV-TestFile5/lab/'
    total_img = os.listdir(path_image)
    readinlabel = AnnotationTransform()
    tensors, targets, img_size = [], [], (512, 512)
    
    for i in range(len(total_img)):
        img = cv2.imread(path_image + total_img[i]) # 读取图片本身
        height, width = img.shape[0], img.shape[1]
        # r = min(img_size[0] / img.shape[0], img_size[1] / img.shape[1])
        # img = cv2.resize(img,(int(img.shape[1] * r), int(img.shape[0] * r)),interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        
        # label, _ = readinlabel(ET.parse(path_label + total_img[i][0:-4] + '.xml').getroot())
        # height, width = _
        
        # 读取txt文件中的目标坐标
        res = txt_target(path_label + total_img[i][:-4] + '.txt', height, width)
        
        # [xcen, ycen, w, h] to [xmin, ymin, xmax, ymax]
        label = xywh_to_xyxy_lab(res)
        
        # r = min(img_size[0] / height, img_size[1] / width)
        # label[:, :4] *= r
        
        
        # 可视化拼贴图片的标签是否与目标匹配
        temp_img = copy.deepcopy(img)
        _COLORS = np.array([0.000, 0.447, 0.741]).astype(np.float32).reshape(-1, 3)
        for i in range(len(label)):
            box = label[i]
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            color = (_COLORS[0] * 255).astype(np.uint8).tolist()
            cv2.rectangle(temp_img, (x0, y0), (x1, y1), color, 2)
        cv2.imwrite('E:/AICV-TestFile5/res/syn_img_' + str(random.randint(0, 100000)) + '.jpg', temp_img) # cv2.imwrite reqire [h, w, c]
        
        
        # 缩放后填充前, xyxy2xywh+填充标签
        img = np.transpose(img, (2, 0, 1))
        boxes = label[:, :4].copy()
        classes = label[:, 4].copy()
        for i in range(boxes.shape[0]):
            scaled_xcen = (boxes[i][2] + boxes[i][0]) / 2
            scaled_ycen = (boxes[i][3] + boxes[i][1]) / 2
            scaled_w = boxes[i][2] - boxes[i][0]
            scaled_h = boxes[i][3] - boxes[i][1]
            boxes[i][0] = scaled_xcen
            boxes[i][1] = scaled_ycen
            boxes[i][2] = scaled_w
            boxes[i][3] = scaled_h
        classes = np.expand_dims(classes, 1)
        targets_t = np.hstack((classes, boxes))
        label = np.zeros((120, 5))
        label[range(len(targets_t))[: 120]] = targets_t[: 120]
        label = np.ascontiguousarray(label, dtype=np.float32)
    
        tensors.append(img)
        targets.append(label)
    
    to_image_list_synthesize_4(tensors, targets, img_size)
    