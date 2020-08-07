# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 10:08:19 2020

@author: Ken Zhou
"""

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np 
import math
import os
import pandas as pd
from PIL import Image
'''
for i in range(0,46,2):
    for j in range(0,80,2):
        plt.figure(figsize=(1,1))
        plt.rcParams['figure.dpi'] = 256
        x1 = [10,128,246]
        y1 = [0+i,128,0+i]
        # plotting the line 1 points 
        plt.axis('off')
        plt.plot(x1, y1, color='black')
        # line 2 points
        
        x2 = [30+j,128,230-j]
        y2 = [246,129,246]
        # plotting the line 2 points 
        plt.plot(x2, y2, color='black')
        plt.savefig(r'D:\representation_syn\testing_img\test_img_{}_{}.jpg'.format(i,j))

'''
def all_files_under(path, extension=None, append_path=True, sort=False):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname) for fname in os.listdir(path) if fname.endswith(extension)]
    
    if sort:
        filenames = sorted(filenames)
    
    return filenames

train_list = all_files_under(r'D:\representation_syn\training_img')
'''
label_list = all_files_under(r'D:\representation_syn\training_label')


test=pd.DataFrame(data=train_list)#数据有三列，列名分别为one,two,three
test_1=pd.DataFrame(data=label_list)#数据有三列，列名分别为one,two,three
print(test)
print(test_1)
test.to_csv(r'D:\representation_syn\train_files.csv',encoding='gbk')
test_1.to_csv(r'D:\representation_syn\label.csv',encoding='gbk')
'''

for index, img_path in enumerate(train_list):
    #print(index)
    #print(img_path)
    img = Image.open(img_path)
    img2 = img.rotate(180)   # 自定义旋转度数
    #img2 = img2.resize((400, 400))   # 改变图片尺寸
    img2.save(r'D:\representation_syn\testing_img\test_img_180_{}.jpg'.format(index))




