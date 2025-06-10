# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 19:51:29 2023

@author: Pilot Crysi, DataPool Synthetise Control
"""

# import cv2

from datapool_synmain4to1_v6 import four_to_one_less_config1_analysis, four_to_one_less_config1_search


if __name__ == "__main__":
    # 设定target_info路径
    csv_path = 'D:/AICV-DSTRethink/Code-DataPoolTest&Results/VOC_0712trainval_ObjectInfo.csv' # VOC_0712trainval_SingleMultiMix.csv / VOC_0712trainval_MultiObject.csv / 
    save_path = 'D:/AICV-DSTRethink/Code-DataPoolTest&Results/overlapresults_Syn/'
    
    # 取得图片宽高
    # img = cv2.imread('D:/AICV-YoloXReDST-SGD/000000000049.jpg') # cv2.imread [h, w, c]
    # height, width = img.shape[0], img.shape[1]
    extre_small, scope, height, width = 10 * 10, [0, 4096], 512, 512 # 32 × 32 = 1024, 64 × 64 = 4096, 96 × 96 = 9216
    
    # 四合一构型1增强, 生成图小于原图
    ratio_width_top, ratio_width_bot, ratio_height = [1/3, 2/3], [1/3, 2/3], [1/3, 2/3]
    for i in ratio_height:
        for j in ratio_width_top:
            for k in ratio_width_bot:
                if i == 1/3 and j == 1/3 and k == 1/3:
                    name = 'hig_lef_lef'
                if i == 1/3 and j == 1/3 and k == 2/3:
                    name = 'hig_lef_rig'
                if i == 1/3 and j == 2/3 and k == 1/3:
                    name = 'hig_rig_lef'
                if i == 1/3 and j == 2/3 and k == 2/3:
                    name = 'hig_rig_rig'
                if i == 2/3 and j == 1/3 and k == 1/3:
                    name = 'low_lef_lef'
                if i == 2/3 and j == 1/3 and k == 2/3:
                    name = 'low_lef_rig'
                if i == 2/3 and j == 2/3 and k == 1/3:
                    name = 'low_rig_lef'
                if i == 2/3 and j == 2/3 and k == 2/3:
                    name = 'low_rig_rig'
                list1, list2, list3, list4 = four_to_one_less_config1_analysis(scope, height, width, i, j, k)
                syn_tool_4lc1 = four_to_one_less_config1_search(list1, list2, list3, list4, csv_path, j, k, i, height, width)
                syn_tool_4lc1.do_search(save_path + 'h' + str(round(i,2)) + '_wt' + str(round(j,2)) + '_wb' + str(round(k,2)) + '/' + name + '_', scope, extre_small)
    
    """
    # 四合一构型, 均匀分割图块, 生成图小于原图
    ratio_width_top, ratio_width_bot, ratio_height, name = 1/2, 1/2, 1/2, 'uniform'
    list1, list2, list3, list4 = four_to_one_less_config1_analysis(scope, height, width, ratio_height, ratio_width_top, ratio_width_bot)
    syn_tool_4lc1 = four_to_one_less_config1_search(list1, list2, list3, list4, csv_path, ratio_width_top, ratio_width_bot, ratio_height, height, width)
    syn_tool_4lc1.do_search(save_path + 'h' + str(round(1/2,2)) + '_wt' + str(round(1/2,2)) + '_wb' + str(round(1/2,2)) + '/' + name + '_', scope)
    """