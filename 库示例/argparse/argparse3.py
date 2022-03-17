#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 19:45:46 2022

@author: jack
"""

#name: square.py
import argparse

def get_parser():
    parser = argparse.ArgumentParser(
        description='Calculate square of a given number')
    parser.add_argument('-number', type=int,default=3)

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    res = args.number ** 2
    print('square of {} is {}'.format(args.number, res))



#$ python argparse3.py -number 5
# python  argparse3.py
