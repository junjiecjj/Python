#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 19:12:21 2022

@author: 
     
     choices：参数值只能从几个选项里面选择
     
     
"""

import argparse



parser = argparse.ArgumentParser(description="calculate X to the power of Y")
parser.add_argument("square", type=int,
                    help="display a square of a given number")
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], default=1,
                    help="increase output verbosity")
args = parser.parse_args()
answer = args.square**2
if args.verbosity == 2:
    print( "the square of {} equals {}".format(args.square, answer))
elif args.verbosity == 1:
    print("{}^2 == {}".format(args.square, answer))
else:
    print(answer)
    
    
    
    
"""

$ python3 argparse6.py 8
8^2 == 64

$ python3 argparse6.py 8 -v 0
64

$ python3 argparse6.py 8 -v 1
8^2 == 64

$ python3 argparse6.py 8 -v 2
the square of 8 equals 64

$ python3 argparse6.py 8 -v 3
usage: argparse6.py [-h] [-v {0,1,2}] square
argparse6.py: error: argument -v/--verbosity: invalid choice: 3 (choose from 0, 1, 2)


"""