#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 13:51:48 2022

@author: jack
"""


import argparse
def basic_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_mode', type=str, default= 'unaligned', help='chooses how datasets are loaded')
    parser.add_argument('--mode', type=str, default='test', help='test mode')
    return parser

def data_options(parser):
    parser.add_argument('--lr', type=str, default='0.0001', help='learning rate')
    return parser

if __name__ == '__main__':
    parser = basic_options()
    opt, unparsed = parser.parse_known_args()
    print(opt)
    print(unparsed)
    parser = data_options(parser)
    opt = parser.parse_args()
    print(opt)




# python argparse8.py --data_mode alignedaaaa --mode trainnnn  --lr 0.0002
# python argparse8.py
#该例子说明了在一开始仅导入了basic_options()选项时，多余出来的–lr选项会被保存起来，不会报错，直到接下来导入了data_options(parser)之后再将其赋值，这时候如果我们传入一个没有配置的选项，它在中间的时候也会保存起来，但是最后就会报错：



# python test_data.py --data_mode aligned --lr 0.0002 --no_clue True





