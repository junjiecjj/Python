#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 21:08:38 2023

@author: jack
"""


class  FourWay_List(object):
    def __init__(self):
        self.m_row_no = 0
        self.m_col_no = 0
        self.m_alpha  = [0, 0]
        self.m_beta   = [0, 0]
        self.m_v2c    = [0, 0]
        self.m_c2v    = [0, 0]

        self.left     = None
        self.right    = None
        self.up       = None
        self.down     = None

        return



a = FourWay_List()
a.m_row_no = 6
a.m_col_no = 7
aa = FourWay_List()
aa.m_row_no = 12
aa.m_col_no = 3

a.right = aa


b = FourWay_List()
b.m_row_no = 4
b.m_col_no = 9

L = []

L.append(a)
L.append(b)


