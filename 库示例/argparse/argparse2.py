#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 19:22:40 2022

@author: jack
"""

import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('--name',type=str,  default='Great')
    
    return parser




if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    name = args.name
    print('Hello {}'.format(name))

