#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 19:12:21 2022

@author: jack
"""

import argparse



parser = argparse.ArgumentParser(description="calculate X to the power of Y")
group = parser.add_mutually_exclusive_group() #定义了一个互斥组
group.add_argument("-v", "--verbose", action="store_true") #在互斥组中添加了 -v 和 -q 两个参数，
group.add_argument("-q", "--quiet", action="store_true")
parser.add_argument("x", type=int, help="the base")
parser.add_argument("y", type=int, help="the exponent")
parser.add_argument('--data_train', type=str, default='DIV2K+CAF10+MINST', help='train dataset name')

args = parser.parse_args()
args.data_train = args.data_train.split('+')

answer = args.x**args.y

if args.quiet:
    print(answer)
elif args.verbose:
    print("{} to the power {} equals {}".format(args.x, args.y, answer))
else:
    print("{}^{} == {}".format(args.x, args.y, answer))
    

"""
$ python3 argparse7.py -h


$ python3 argparse7.py 4 2
4^2 == 16
$ python3 argparse7.py 4 2 -q
16
$ python3 argparse7.py 4 2 -v
4 to the power 2 equals 16

$ python3 argparse7.py 4 2 -vq
usage: prog.py [-h] [-v | -q] x y
prog.py: error: argument -q/--quiet: not allowed with argument -v/--verbose

$ python3 argparse7.py 4 2 -v --quiet
usage: prog.py [-h] [-v | -q] x y
prog.py: error: argument -q/--quiet: not allowed with argument -v/--

"""


















