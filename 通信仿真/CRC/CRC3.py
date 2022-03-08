#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 18:23:39 2022

https://www.programminghunter.com/article/8166864580/

@author: jack
"""

def crc16(x, invert):
    a = 0xFFFF
    b = 0xA001
    for byte in x:
        a ^= ord(byte)
        for i in range(8):
            last = a % 2
            a >>= 1
            if last == 1:
                a ^= b
    s = hex(a).upper()
    
    return s[4:6]+s[2:4] if invert == True else s[2:4]+s[4:6]

print(crc16("012345678", True))
print(crc16("012345678", False))
print(crc16("010600010017", True))