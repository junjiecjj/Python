#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 19:46:46 2022

@author: jack
"""


# choices：参数值只能从几个选项里面选择
# file-name: choices.py
import argparse

def get_parser():
    parser = argparse.ArgumentParser( description='choices demo')
    parser.add_argument('-arch', required=True, choices=['alexnet', 'vgg'])

    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print('the arch of CNN is {}'.format(args.arch))
    print(f"args.arch = {args.arch}")
    
    
    
"""
❯ python3 argparse4.py  -arch  vgg
the arch of CNN is vgg
args.arch = vgg


usage: argparse4.py [-h] -arch {alexnet,vgg}
argparse4.py: error: argument -arch: invalid choice: 'vasa' (choose from 'alexnet', 'vgg')







"""
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    