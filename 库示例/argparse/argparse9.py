#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 15:06:35 2022

@author: jack
"""

import argparse




parser = argparse.ArgumentParser()
parser.add_argument('--fff', action='store_true' )
parser.add_argument('--foo', action='store_true', default = False,)
parser.add_argument('--foa', action='store_true', default = True,)
parser.add_argument('--bbb', action='store_false' )
parser.add_argument('--bar', action='store_false', default = False,)
parser.add_argument('--baz', action='store_false', default = True,)
parser.add_argument('--data_train', type=str, default='DIV2K+CAF10+MINST', help='train dataset name')


args = parser.parse_args()
args, unparsed = parser.parse_known_args()
args.data_train = args.data_train.split('+')

print(f"args = {args}")


for arg in vars(args):
    print(f"{arg} ------> {vars(args)[arg]}")


"""
❯ python argparse9.py 
args = Namespace(foo=False, foa=False, bar=True, baz=True)
foo ------> False
foa ------> False
bar ------> True
baz ------> True
data_train ------> ['DIV2K', 'CAF10', 'MINST']


❯ python argparse9.py --foo --bar
args = Namespace(foo=True, foa=False, bar=False, baz=True)
foo ------> True
foa ------> False
bar ------> False
baz ------> True
data_train ------> ['DIV2K', 'CAF10', 'MINST']


"""