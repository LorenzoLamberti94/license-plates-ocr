#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python preprocess.py -image=preprocessing_example/ccpd -dir_train=preprocessing_example/train -dir_valid=preprocessing_example/valid
"""
Created on May 27 10:17:20 2020

@author: Lorenzo Lamberti
"""
import os
# Set working directory
os.chdir('/home/lamberti/work/2_steps_license_plates/lp_recognition') 

from imutils import paths
import numpy as np
import cv2
import os
import argparse
import random
import pandas as pd
from glob import glob

# openvino_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<Anhui>', '<Beijing>', '<Chongqing>', '<Fujian>', '<Gansu>', '<Guangdong>', '<Guangxi>', '<Guizhou>', '<Hainan>', '<Hebei>', '<Heilongjiang>', '<Henan>', '<HongKong>', '<Hubei>', '<Hunan>', '<InnerMongolia>', '<Jiangsu>', '<Jiangxi>', '<Jilin>', '<Liaoning>', '<Macau>', '<Ningxia>', '<Qinghai>', '<Shaanxi>', '<Shandong>', '<Shanghai>', '<Shanxi>', '<Sichuan>', '<Tianjin>', '<Tibet>', '<Xinjiang>', '<Yunnan>', '<Zhejiang>', '<police>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_'])

# original_synthetic=[ '<Beijing>', '<Chongqing>', '<Fujian>', '<Gansu>', '<Guangdong>', '<Guangxi>', '<Guizhou>', '<Hainan>', '<Hebei>', '<Heilongjiang>', '<Henan>', '<HongKong>', '<Hubei>', '<Hunan>', '<InnerMongolia>', '<Jiangsu>', '<Jiangxi>', '<Jilin>', '<Liaoning>', '<Macau>', '<Ningxia>', '<Qinghai>', '<Shaanxi>', '<Shandong>', '<Shanxi>', '<Sichuan>', '<Tianjin>', '<Tibet>', '<Xinjiang>', '<Yunnan>', '<Zhejiang>', '<police>']
# mancanti = [ '<HongKong>', '<Macau>']
# symbols=   [  '港'     ,    '澳' ]
# corrispondenza
# ['<Anhui>', '<Shanghai>','<Tianjin>', '<Chongqing >', '<Hebei >', '<Shanxi>', '<InnerMongolia>', '<Liaoning>', '<Jilin>', '<Heilongjiang>', '<Jiangsu>', '<Zhejiang>', '<Beijing>', '<Fujian>', '<Jiangxi>', '<Shandong>', '<Henan>', '<Hubei>', '<Hunan>', '<Guangdong>', '<Guangxi>' , '<Hainan>', '<Sichuan>' , '<Guizhou>', '<Yunnan>' , '<Tibet>', '<Shaanxi>' ,'<Gansu>' , '<Qinghai>', '<Ningxia>' ,'<Xinjiang>' , '<police>', '<school>' ,  '<_>' ]
# ["皖",       "沪",           "津",      "渝",           "冀",       "晋",           "蒙",           "辽",           "吉",       "黑",           "苏",           "浙",       "京",        "闽",      "赣",           "鲁",      "豫",       "鄂",       "湘",        "粤",       "桂",        "琼",          "川",             "贵",       "云",      "藏",      "陕",       "甘",       "青",       "宁",           "新",        "警",         "学",       "O"]


provinces = ['<Anhui>', '<Shanghai>','<Tianjin>', '<Chongqing>', '<Hebei>', '<Shanxi>', '<InnerMongolia>', '<Liaoning>', '<Jilin>', '<Heilongjiang>', '<Jiangsu>', '<Zhejiang>', '<Beijing>', '<Fujian>', '<Jiangxi>', '<Shandong>', '<Henan>', '<Hubei>', '<Hunan>', '<Guangdong>', '<Guangxi>' , '<Hainan>', '<Sichuan>' , '<Guizhou>', '<Yunnan>' , '<Tibet>', '<Shaanxi>' ,'<Gansu>' , '<Qinghai>', '<Ningxia>' ,'<Xinjiang>' , '<police>', '<school>' ,  '_' ]
# provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


parser = argparse.ArgumentParser(description='crop the licence plate from original image')
parser.add_argument("-image", help='image path', default='CCPD_dataset/', type=str)
parser.add_argument("-save_dir", help='save directory', default='cropped', type=str)
args = parser.parse_args()

# make list of all the paths of the images to be cropped
img_paths= glob('/home/lamberti/work/Dataset/ccpd_dataset/ccpd_base/*')
num_of_images = print(len(img_paths))
#shuffle
random.shuffle(img_paths)

save_dir = args.save_dir

print('image data processing is kicked off...')
print("%d images in total" % num_of_images )

idx = 0
dataframe_list=[]

for i in range(len(img_paths)):
    filename = img_paths[i]
    
    basename = os.path.basename(filename)
    imgname, suffix = os.path.splitext(basename)
    imgname_split = imgname.split('-')
    rec_x1y1 = imgname_split[2].split('_')[0].split('&')
    rec_x2y2 = imgname_split[2].split('_')[1].split('&')  
    x1, y1, x2, y2 = int(rec_x1y1[0]), int(rec_x1y1[1]), int(rec_x2y2[0]), int(rec_x2y2[1])
    w = int(x2 - x1 + 1.0)
    h = int(y2 - y1 + 1.0)
    
    img = cv2.imread(filename)
    img_crop = np.zeros((h, w, 3))
    img_crop = img[y1:y2+1, x1:x2+1, :]
#    img_crop = cv2.resize(img_crop, (94, 24), interpolation=cv2.INTER_LINEAR)
    
    pre_label = imgname_split[4].split('_')
    lb = "" # label
    lb += provinces[int(pre_label[0])]
    lb += alphabets[int(pre_label[1])]
    for label in pre_label[2:]:
        lb += ads[int(label)]
        
    idx += 1 
    
    if idx % 100 == 0:
        print("%d / %d images done" % (idx, num_of_images) )
    
    image_path = save_dir + '/' + '{0:06d}'.format(idx) + suffix   # increasing numbers as image names
    # image_path = save_dir + '/' + lb + suffix   # GT label as image names  
    cv2.imwrite(image_path, img_crop)

    dataframe_list.append([image_path, lb])


print('image data processing done, wrote %d images' % (idx))
print('... starting creation of txt file with all the paths and GT labels')

#create dataframe
header = ['image_path','GT_label']
df = pd.DataFrame(dataframe_list, columns = header)
print(df)
# Shuffle rows
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv('lprnet_annot.csv', columns=header, sep = " ", index=False, header=False)

print('Correctly created .txt file with all paths and GT labels.')

# initialize list of lists 
# data = [['tom', 10], ['nick', 15], ['juli', 14]] 
# Create the pandas DataFrame 
# df = pd.DataFrame(data, columns = ['Name', 'Age'])
# print(df)
