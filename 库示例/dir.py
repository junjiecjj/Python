#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 22:46:50 2022

@author: jack
"""


import os

print ('***获取当前目录***')
print( os.getcwd())
print (os.path.abspath(os.path.dirname(__file__)))

print( '***获取上级目录***')
print (os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
print (os.path.abspath(os.path.dirname(os.getcwd())))
print (os.path.abspath(os.path.join(os.getcwd(), "..")))

print ('***获取上上级目录***')
print( os.path.abspath(os.path.join(os.getcwd(), "../..")))


import sys, os

print(f"__file__ = {__file__}")    #当前.py文件的位置
print(f"os.path.abspath(__file__) = {os.path.abspath(__file__)}\n")  #返回当前.py文件的绝对路径
print(f"os.path.dirname(os.path.abspath(__file__)) = {os.path.dirname(os.path.abspath(__file__))}\n")   #当前文件的绝对路径目录，不包括当前 *.py 部分，即只到该文件目录
print(f"os.path.dirname(os.path.dirname(os.path.abspath(__file__))) = {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}\n") #返回文件本身目录的上层目录    
print(f"os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) = {os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}\n")  #每多一层，即再上一层目录

print(f"os.path.realpath(__file__) = {os.path.realpath(__file__)}\n")   #当前文件的真实地址
print(f"os.path.dirname(os.path.realpath(__file__)) = {os.path.dirname(os.path.realpath(__file__))}\n") # 当前文件夹的路径

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)   #将目录或路径加入搜索路径

print(__name__)