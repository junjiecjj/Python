#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 18:51:21 2022

https://mp.weixin.qq.com/s?__biz=MzA5MzUxMzg5NA==&mid=2453846525&idx=1&sn=764bcc7ff2571e17730bd971cb83802c&chksm=87eaa834b09d212211d5fcf96f0da456b17842ab52a236f47eea8caa1b7b12c2257cdef2268a&mpshare=1&scene=1&srcid=0310XqgioQzBm0LJbnGR29IN&sharer_sharetime=1647653070417&sharer_shareid=0d5c82ce3c8b7c8f30cc9a686416d4a8&exportkey=AQ0KVTBeF%2BQk0gEycWBAX58%3D&acctmode=0&pass_ticket=0vFGKoIMy%2B4HFp%2B0mSPDzyOp9z18Rzr4q2tIa0pnNQ88otF6K%2FaI5VWhIBOdDxOj&wx_header=0#rd

一行 Python 代码实现并行
@author: jack
"""

import os 
import PIL 

from multiprocessing import Pool 
from PIL import Image

SIZE = (75,75)
SAVE_DIRECTORY = 'thumbs'

def get_image_paths(folder):
    return (os.path.join(folder, f) 
            for f in os.listdir(folder) 
            if 'jpeg' in f)

def create_thumbnail(filename): 
    im = Image.open(filename)
    im.thumbnail(SIZE, Image.ANTIALIAS)
    base, fname = os.path.split(filename) 
    save_path = os.path.join(base, SAVE_DIRECTORY, fname)
    im.save(save_path)

def main1():
    folder = os.path.abspath(
        '11_18_2013_R000_IQM_Big_Sur_Mon__e10d1958e7b766c3e840')
    os.mkdir(os.path.join(folder, SAVE_DIRECTORY))

    images = get_image_paths(folder)

    for image in images:
        create_thumbnail(Image)    

def main2():
    folder = os.path.abspath(
        '11_18_2013_R000_IQM_Big_Sur_Mon__e10d1958e7b766c3e840')
    os.mkdir(os.path.join(folder, SAVE_DIRECTORY))

    images = get_image_paths(folder)

    pool = Pool()
    pool.map(create_thumbnail, images)
    pool.close()
    pool.join()


if __name__ == '__main__':
    main1()
    main2()
    
    
    
    
    
    
    
    
    
    
    
    