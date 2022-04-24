#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 19:12:21 2022

@author: jack
"""

import argparse



parser = argparse.ArgumentParser(description="calculate X to the power of Y")
parser.add_argument('--data_train', type=str, default='DIV2K+CAF10+MINST', help='train dataset name')

args = parser.parse_args()
args.data_train = args.data_train.split('+')
