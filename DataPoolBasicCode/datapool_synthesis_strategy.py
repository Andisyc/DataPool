# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 15:55:49 2022

@author: Cheng Yuxuan Original
"""

# 严格合成策略: 同一张图片中不存在重复图块，不使用已经用过的图块
"""
        # 初始化使用过的图块列表
        self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used = [], [], [], []
        
        # 初始化合成计数器和递归停止符
        count, self.isStop = 0, False
        
        # 每个列表中的图片只能使用一次,不能重复出现在多张合成图中
        if len(self.potential_area_1) != 0 and len(self.potential_area_2) != 0 and len(self.potential_area_3) != 0 and len(self.potential_area_4) != 0:
            self.i, self.j = random.choice(self.potential_area_1), random.choice(self.potential_area_2)
            self.k, self.v = random.choice(self.potential_area_3), random.choice(self.potential_area_4)
            while(len(self.potential_area_1) != 0 and len(self.potential_area_2) != 0 and len(self.potential_area_3) != 0 and len(self.potential_area_4) != 0):
                # 必须同时满足相互不同且非已用图块才能继续进行
                self.i, self.j, self.k, self.v = self.none_same_patch()
                self.i, self.j, self.k, self.v, self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used = self.none_used_patch()
                
                # 判断列表是否抽空而跳出循环
                if self.isStop == True:
                    break
                
                # 读取并切割图片,修改标签
                img1 = cv2.imread(img_file + self.i[0][:-4] + '.jpg')
                img2 = cv2.imread(img_file + self.j[0][:-4] + '.jpg')
                img3 = cv2.imread(img_file + self.k[0][:-4] + '.jpg')
                img4 = cv2.imread(img_file + self.v[0][:-4] + '.jpg')
                img1 = img1[self.i[7]:self.i[9], self.i[6]:self.i[8]]
                lab1 = [self.i[1]-self.i[6], self.i[2]-self.i[7], self.i[3]-self.i[6], self.i[4]-self.i[7], self.i[5]]
                img2 = img2[self.j[7]:self.j[9], self.j[6]:self.j[8]]
                lab2 = [self.j[1]-self.j[6], self.j[2]-self.j[7], self.j[3]-self.j[6], self.j[4]-self.j[7], self.j[5]]
                img3 = img3[self.k[7]:self.k[9], self.k[6]:self.k[8]]
                lab3 = [self.k[1]-self.k[6], self.k[2]-self.k[7], self.k[3]-self.k[6], self.k[4]-self.k[7], self.k[5]]
                img4 = img4[self.v[7]:self.v[9], self.v[6]:self.v[8]]
                lab4 = [self.v[1]-self.v[6], self.v[2]-self.v[7], self.v[3]-self.v[6], self.v[4]-self.v[7], self.v[5]]
                
                # 调用合成函数合成图片
                four_to_one_less_config1_synthetise(img1, img2, img3, img4, lab1, lab2, lab3, lab4, 
                                                    self.ratio_width_top, self.ratio_width_bot, self.ratio_height, # self.ratio_area, 
                                                    count, self.height, self.width, save_dir)
                
                count = count + 1
                
                # 储存&移除旧图块, 抽取新图块
                self.patch1_used, self.potential_area_1, self.i = self.store_remove_pick(self.patch1_used, self.potential_area_1, self.i)
                self.patch2_used, self.potential_area_2, self.j = self.store_remove_pick(self.patch2_used, self.potential_area_2, self.j)
                self.patch3_used, self.potential_area_3, self.k = self.store_remove_pick(self.patch3_used, self.potential_area_3, self.k)
                self.patch4_used, self.potential_area_4, self.v = self.store_remove_pick(self.patch4_used, self.potential_area_4, self.v)
                
                # 判断列表是否抽空而跳出循环
                if self.isStop == True:
                    break
    
        print("Synthesis Complete")

    def store_remove_pick(self, store_list, remove_list, patch_info):
        store_list.append(patch_info)
        remove_list.remove(patch_info)
        if len(remove_list) == 0:
            self.isStop = True
            return store_list, remove_list, patch_info
        patch_info = random.choice(remove_list)
        
        return store_list, remove_list, patch_info
    
    def remove_pick(self, remove_list, patch_info):
        if len(remove_list) == 0:
            return remove_list, patch_info, True
        remove_list.remove(patch_info)
        if len(remove_list) == 0:
            return remove_list, patch_info, True
        patch_info = random.choice(remove_list)
        
        return remove_list, patch_info, False
    
    def compare_info(self, i, j):
        if (i[0] == j[0] and i[1] == j[1] and i[2] == j[2] and i[3] == j[3] and i[4] == j[4]) == False:
            return True # False说明两列表不等, 不等是我们想要的
        else:
            return False # True说明两列表相等, 相等则我们不想要
    
    def none_same_patch(self):
        # 确保同时使用的四个图块不会存在相同, 如果相同则重新选择图块
        while((self.compare_info(self.i, self.j) and self.compare_info(self.i, self.k) and self.compare_info(self.i, self.v) and \
               self.compare_info(self.j, self.k) and self.compare_info(self.k, self.v)) == False):
            self.i, self.j, self.k, self.v = random.choice(self.potential_area_1), random.choice(self.potential_area_2), random.choice(self.potential_area_3), random.choice(self.potential_area_4)
        
        return self.i, self.j, self.k, self.v # 成功跳出while循环即说明互不相同
    
    def none_used_patch(self):
        # 确保不会再使用到之前使用过的图块, 如果出现即重新选择图块, 并且删除当前图块
        while(self.i in self.patch2_used or self.i in self.patch3_used or self.i in self.patch4_used):
            self.potential_area_1, self.i, self.isStop = self.remove_pick(self.potential_area_1, self.i)
            if self.isStop == True:
                break
            self.i, self.j, self.k, self.v = self.none_same_patch()
            self.i, self.j, self.k, self.v, self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used = self.none_used_patch()
            if self.isStop == True:
                break
        while(self.j in self.patch1_used or self.j in self.patch3_used or self.j in self.patch4_used):
            self.potential_area_2, self.j, self.isStop = self.remove_pick(self.potential_area_2, self.j)
            if self.isStop == True:
                break
            self.i, self.j, self.k, self.v = self.none_same_patch()
            self.i, self.j, self.k, self.v, self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used = self.none_used_patch()
            if self.isStop == True:
                break
        while(self.k in self.patch1_used or self.k in self.patch2_used or self.k in self.patch4_used):
            self.potential_area_3, self.k, self.isStop = self.remove_pick(self.potential_area_3, self.k)
            if self.isStop == True:
                break
            self.i, self.j, self.k, self.v = self.none_same_patch()
            self.i, self.j, self.k, self.v, self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used = self.none_used_patch()
            if self.isStop == True:
                break
        while(self.v in self.patch1_used or self.v in self.patch2_used or self.v in self.patch3_used):
            self.potential_area_4, self.v, self.isStop = self.remove_pick(self.potential_area_4, self.v)
            if self.isStop == True:
                break
            self.i, self.j, self.k, self.v = self.none_same_patch()
            self.i, self.j, self.k, self.v, self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used = self.none_used_patch()
            if self.isStop == True:
                break

        return self.i, self.j, self.k, self.v, self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used
"""


# 中等合成策略: 同一张图片中不存在重复图块，可以使用已经用过的图块
"""
        # 初始化使用过的图块列表
        self.patch1_used, self.patch2_used, self.patch3_used, self.patch4_used = [], [], [], []
        
        # 初始化合成计数器和递归停止符
        count, self.isStop = 0, False
        
        # 每个列表中的图片只能使用一次,不能重复出现在多张合成图中
        if len(self.potential_area_1) != 0 and len(self.potential_area_2) != 0 and len(self.potential_area_3) != 0 and len(self.potential_area_4) != 0:
            self.i, self.j = random.choice(self.potential_area_1), random.choice(self.potential_area_2)
            self.k, self.v = random.choice(self.potential_area_3), random.choice(self.potential_area_4)
            while(len(self.potential_area_1) != 0 or len(self.potential_area_2) != 0 or len(self.potential_area_3) != 0 or len(self.potential_area_4) != 0):
                # 必须同时满足相互不同且非已用图块才能继续进行
                self.i, self.j, self.k, self.v = self.none_same_patch()
                
                # 读取并切割图片,修改标签
                img1 = cv2.imread(img_file + self.i[0][:-4] + '.jpg')
                img2 = cv2.imread(img_file + self.j[0][:-4] + '.jpg')
                img3 = cv2.imread(img_file + self.k[0][:-4] + '.jpg')
                img4 = cv2.imread(img_file + self.v[0][:-4] + '.jpg')
                img1 = img1[self.i[7]:self.i[9], self.i[6]:self.i[8]]
                lab1 = [self.i[1]-self.i[6], self.i[2]-self.i[7], self.i[3]-self.i[6], self.i[4]-self.i[7], self.i[5]]
                img2 = img2[self.j[7]:self.j[9], self.j[6]:self.j[8]]
                lab2 = [self.j[1]-self.j[6], self.j[2]-self.j[7], self.j[3]-self.j[6], self.j[4]-self.j[7], self.j[5]]
                img3 = img3[self.k[7]:self.k[9], self.k[6]:self.k[8]]
                lab3 = [self.k[1]-self.k[6], self.k[2]-self.k[7], self.k[3]-self.k[6], self.k[4]-self.k[7], self.k[5]]
                img4 = img4[self.v[7]:self.v[9], self.v[6]:self.v[8]]
                lab4 = [self.v[1]-self.v[6], self.v[2]-self.v[7], self.v[3]-self.v[6], self.v[4]-self.v[7], self.v[5]]
                
                # 调用合成函数合成图片
                four_to_one_less_config1_synthetise(img1, img2, img3, img4, lab1, lab2, lab3, lab4, 
                                                    self.ratio_width_top, self.ratio_width_bot, self.ratio_height, # self.ratio_area, 
                                                    count, self.height, self.width, save_dir)
                
                count = count + 1
                
                # 储存旧图块, 抽取新图块
                self.patch1_used, self.potential_area_1, self.i = self.store_remove_pick(self.patch1_used, self.potential_area_1, self.i)
                self.patch2_used, self.potential_area_2, self.j = self.store_remove_pick(self.patch2_used, self.potential_area_2, self.j)
                self.patch3_used, self.potential_area_3, self.k = self.store_remove_pick(self.patch3_used, self.potential_area_3, self.k)
                self.patch4_used, self.potential_area_4, self.v = self.store_remove_pick(self.patch4_used, self.potential_area_4, self.v)
    
        print("Synthesis Complete")
    
    def store_remove_pick(self, store_list, pick_list, patch_info):
        # 储存已用图块,但不储存复用图块
        if patch_info in store_list:
            pass
        else:
            store_list.append(patch_info)
        
        # 确认待抽列表是否为0
        if len(pick_list) == 0:
            pass
        else:
            if patch_info in pick_list:
                pick_list.remove(patch_info)
            else:
                pass
        
        # 待抽列表可能移除后为0,因此分为两次判断
        if len(pick_list) == 0:
            patch_info = random.choice(store_list)
        else:
            prob = random.choice(range(100))
            if prob < 30: # 30%概率取已用图块
                patch_info = random.choice(store_list)
            else: # 70%概率取未用图块
                patch_info = random.choice(pick_list)
        
        return store_list, pick_list, patch_info
    
    def compare_info(self, i, j):
        if (i[0] == j[0] and i[1] == j[1] and i[2] == j[2] and i[3] == j[3] and i[4] == j[4]) == False:
            return True # False说明两列表不等, 不等是我们想要的
        else:
            return False # True说明两列表相等, 相等则我们不想要
    
    def none_same_patch(self):
        # 确保同时使用的四个图块不会存在相同, 如果相同则重新选择图块
        while((self.compare_info(self.i, self.j) and self.compare_info(self.i, self.k) and self.compare_info(self.i, self.v) and \
               self.compare_info(self.j, self.k) and self.compare_info(self.k, self.v)) == False):
            if len(self.potential_area_1) == 0:
                list1 = self.patch1_used
            else:
                list1 = self.potential_area_1
            if len(self.potential_area_2) == 0:
                list2 = self.patch2_used
            else:
                list2 = self.potential_area_2
            if len(self.potential_area_3) == 0:
                list3 = self.patch3_used
            else:
                list3 = self.potential_area_3
            if len(self.potential_area_4) == 0:
                list4 = self.patch4_used
            else:
                list4 = self.potential_area_4
            
            self.i, self.j, self.k, self.v = random.choice(list1), random.choice(list2), random.choice(list3), random.choice(list4)
        
        return self.i, self.j, self.k, self.v # 成功跳出while循环即说明互不相同
"""

# 宽松合成策略: 同一张图片中存在重复图块，可以使用已经用过的图块

