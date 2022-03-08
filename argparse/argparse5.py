#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 19:49:24 2022

@author: jack


使用方式如下：

parser.add_argument('-name', nargs=x)
其中x的候选值和含义如下：

值  含义
N   参数的绝对个数（例如：3）
'?'   0或1个参数
'*'   0或所有参数
'+'   所有，并且至少一个参数



"""

# file-name: nargs.py
import argparse

def get_parser():
    parser = argparse.ArgumentParser(
        description='nargs demo')
    parser.add_argument('-name', required=True, nargs='+')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    names = ', '.join(args.name)
    print('Hello to {}'.format(names))
    
    # $ python argparse5.py -name A B C