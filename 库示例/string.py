#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 11:19:03 2022

@author: jack
"""


# 1
name = 'leo'
n=37
s = '{} has {} message.'.format(name,n)
print(s)

# 2
name = 'leo'
n=37
s = '{name} has {n} message.' 
print(s.format_map(vars()))


name = 'jack'
n = 43
print(s.format_map(vars()))





# 
























