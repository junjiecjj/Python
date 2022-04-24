#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 15:06:35 2022

@author: jack
"""

import argparse




parser = argparse.ArgumentParser()
parser.add_argument('--foo', action='store_true')
parser.add_argument('--foa', action='store_true')
parser.add_argument('--bar', action='store_false')
parser.add_argument('--baz', action='store_false')


args = parser.parse_args()
args, unparsed = parser.parse_known_args()


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



❯ python argparse9.py --foo --bar
args = Namespace(foo=True, foa=False, bar=False, baz=True)
foo ------> True
foa ------> False
bar ------> False
baz ------> True



"""